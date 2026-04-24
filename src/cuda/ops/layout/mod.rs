//! Layout and memory operations for GPU tensors

use super::cache::OpsKernelCache;
use crate::cuda::context::{CudaError, IconnxCudaContext};
use crate::cuda::memory_pool::GpuMemoryPool;
use crate::cuda::tensor::GpuTensor;

/// Prepend an op-name prefix to a dtype-accessor error so the resulting
/// message reads e.g. `"Transpose: data_f32 called on int64 tensor"`
/// instead of the accessor's raw
/// `"data_f32 called on int64 tensor"`. Introduced for WS-2 / YOLOS-tiny
/// diagnostic narrowing: when a layout op is handed a non-f32 tensor
/// (typical for shape-info Int64 tensors flowing through Reshape /
/// Concat / …), the error now tells us *which* op to extend next,
/// rather than leaving a grep across ~30 `data_f32()` call sites to
/// figure out where the failure came from.
///
/// Only rewrites `CudaError::Kernel` variants; other variants pass
/// through unchanged so we never mask a non-dtype error behind a
/// misleading op-name prefix.
fn op_ctx<T>(op: &str, r: Result<T, CudaError>) -> Result<T, CudaError> {
    r.map_err(|e| match e {
        CudaError::Kernel(msg) => CudaError::Kernel(format!("{op}: {msg}")),
        other => other,
    })
}

/// GPU Shape operator: Returns tensor dimensions as Int64 tensor
///
/// Takes any tensor and returns a 1D Int64 tensor containing its shape.
/// For example, a tensor with shape [2, 3, 4] returns Int64([2, 3, 4]).
pub fn gpu_shape(ctx: &IconnxCudaContext, input: &GpuTensor) -> Result<GpuTensor, CudaError> {
    // Get shape as i64 values
    let shape_values: Vec<i64> = input.shape().iter().map(|&d| d as i64).collect();
    let ndim = shape_values.len();

    // Create Int64 tensor with shape values
    GpuTensor::from_host_i64(ctx, &shape_values, vec![ndim])
}

/// GPU ConstantOfShape operator: Creates a tensor filled with a constant value
///
/// Takes a 1D shape tensor and fills a new tensor of that shape with a constant.
/// The shape tensor must be Int64. Returns a Float32 tensor (most common use case).
/// NOTE: This version does D2H transfer. Use gpu_constant_of_shape_direct when shape is known.
#[allow(dead_code)]
pub fn gpu_constant_of_shape(
    ctx: &IconnxCudaContext,
    shape_tensor: &GpuTensor,
    value: f32,
) -> Result<GpuTensor, CudaError> {
    // Get shape from GPU - shape_tensor is 1D Int64
    let shape_values = shape_tensor.to_host_i64(ctx)?;
    let target_shape: Vec<usize> = shape_values.iter().map(|&d| d as usize).collect();

    // Create tensor filled with the value
    let total_elements: usize = target_shape.iter().product();
    let data = vec![value; total_elements];
    GpuTensor::from_host_f32(ctx, &data, target_shape)
}

/// GPU ConstantOfShape with pre-resolved shape - uses GPU fill kernel (no D2H transfer)
pub fn gpu_constant_of_shape_direct(
    ctx: &IconnxCudaContext,
    cache: &OpsKernelCache,
    pool: &mut GpuMemoryPool,
    target_shape: Vec<usize>,
    value: f32,
) -> Result<GpuTensor, CudaError> {
    let total_elements: usize = target_shape.iter().product();

    if total_elements == 0 {
        return pool.get_tensor_f32(ctx, target_shape);
    }

    // Allocate output tensor (uninitialized would be ideal, but zeros is safe)
    let mut output = pool.get_tensor_f32(ctx, target_shape)?;

    // Use fill kernel to set all values.
    // SAFETY: fill_kernel takes (float* out, float value, size_t total_elements).
    let kernel = unsafe {
        cache.module().typed_kernel::<(
            &mut garboard::DeviceSlice<'_, f32>,
            f32,
            usize,
        )>("fill_kernel")
    }
    .map_err(|e| CudaError::Kernel(format!("fill_kernel lookup: {}", e)))?;

    let config = garboard::LaunchConfig::for_num_elems(total_elements as u32);

    let out_slice = op_ctx("ConstantOfShape", output.data_f32_mut())?;
    kernel
        .launch(
            ctx.garboard_stream(),
            &config,
            (out_slice, value, total_elements),
        )
        .map_err(|e| CudaError::Kernel(format!("fill_kernel launch failed: {}", e)))?;

    Ok(output)
}

/// Int64 variant of [`gpu_constant_of_shape_direct`].
///
/// ONNX ConstantOfShape's output dtype is determined by the `value`
/// attribute's dtype. When that attribute is Int64 — common after a
/// Shape op whose output is Int64 by spec — this path allocates an
/// Int64 output tensor and launches `fill_i64_kernel` instead of the
/// f32 path. YOLOS-tiny's ViT embedding interpolation hit this case.
pub fn gpu_constant_of_shape_direct_i64(
    ctx: &IconnxCudaContext,
    cache: &OpsKernelCache,
    pool: &mut GpuMemoryPool,
    target_shape: Vec<usize>,
    value: i64,
) -> Result<GpuTensor, CudaError> {
    let total_elements: usize = target_shape.iter().product();

    if total_elements == 0 {
        return pool.get_tensor_i64(ctx, target_shape);
    }

    let mut output = pool.get_tensor_i64(ctx, target_shape)?;

    // SAFETY: fill_i64_kernel takes (long long* out, long long value,
    // size_t total_elements). CUDA's `long long` is int64 on all
    // supported toolchains.
    let kernel = unsafe {
        cache.module().typed_kernel::<(
            &mut garboard::DeviceSlice<'_, i64>,
            i64,
            usize,
        )>("fill_i64_kernel")
    }
    .map_err(|e| CudaError::Kernel(format!("fill_i64_kernel lookup: {}", e)))?;

    let config = garboard::LaunchConfig::for_num_elems(total_elements as u32);

    kernel
        .launch(
            ctx.garboard_stream(),
            &config,
            (output.data_i64_mut()?, value, total_elements),
        )
        .map_err(|e| CudaError::Kernel(format!("fill_i64_kernel launch failed: {}", e)))?;

    Ok(output)
}

/// GPU 2D Transpose: [M, N] -> [N, M]
pub fn gpu_transpose_2d(
    ctx: &IconnxCudaContext,
    cache: &OpsKernelCache,
    pool: &mut GpuMemoryPool,
    input: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    let shape = input.shape();
    assert_eq!(shape.len(), 2, "transpose_2d requires 2D tensor");

    let rows = shape[0];
    let cols = shape[1];
    let total = rows * cols;

    let mut output = pool.get_tensor_f32(ctx, vec![cols, rows])?;

    // SAFETY: transpose_2d_kernel takes (float* out, const float* inp,
    //   size_t rows, size_t cols).
    let kernel = unsafe {
        cache.module().typed_kernel::<(
            &mut garboard::DeviceSlice<'_, f32>,
            &garboard::DeviceSlice<'_, f32>,
            usize,
            usize,
        )>("transpose_2d_kernel")
    }
    .map_err(|e| CudaError::Kernel(format!("transpose_2d_kernel lookup: {}", e)))?;

    let config = garboard::LaunchConfig::for_num_elems(total as u32);

    let out_slice = op_ctx("Transpose", output.data_f32_mut())?;
    let in_slice = op_ctx("Transpose", input.data_f32())?;
    kernel
        .launch(
            ctx.garboard_stream(),
            &config,
            (out_slice, in_slice, rows, cols),
        )
        .map_err(|e| CudaError::Kernel(format!("transpose_2d launch failed: {}", e)))?;

    Ok(output)
}

/// GPU N-dimensional Transpose with permutation
///
/// Takes a tensor and a permutation vector, returns the transposed tensor.
/// For example, with shape [2, 3, 4] and perm [2, 0, 1]:
/// - Output shape is [4, 2, 3]
/// - Element at [i, j, k] in output comes from [j, k, i] in input
pub fn gpu_transpose_nd(
    ctx: &IconnxCudaContext,
    cache: &OpsKernelCache,
    pool: &mut GpuMemoryPool,
    input: &GpuTensor,
    perm: &[i64],
) -> Result<GpuTensor, CudaError> {
    let in_shape = input.shape();
    let ndim = in_shape.len();

    if perm.len() != ndim {
        return Err(CudaError::Kernel(format!(
            "Transpose perm length {} doesn't match tensor ndim {}",
            perm.len(),
            ndim
        )));
    }

    if ndim > 6 {
        return Err(CudaError::Kernel(format!(
            "Transpose supports up to 6 dimensions, got {}",
            ndim
        )));
    }

    // Convert and validate permutation
    let perm_usize: Vec<usize> = perm
        .iter()
        .map(|&p| {
            if p < 0 {
                (ndim as i64 + p) as usize
            } else {
                p as usize
            }
        })
        .collect();

    // Verify perm is a valid permutation
    let mut seen = vec![false; ndim];
    for &p in &perm_usize {
        if p >= ndim || seen[p] {
            return Err(CudaError::Kernel(format!(
                "Invalid permutation {:?} for ndim {}",
                perm, ndim
            )));
        }
        seen[p] = true;
    }

    // Calculate output shape by permuting input shape
    let out_shape: Vec<usize> = perm_usize.iter().map(|&p| in_shape[p]).collect();

    // Calculate input strides (row-major)
    let mut in_strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        in_strides[i] = in_strides[i + 1] * in_shape[i + 1];
    }

    // Calculate output strides (row-major)
    let mut out_strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
    }

    let total_elements = input.len();

    // Upload arrays to GPU
    let in_shape_gpu = ctx.htod_usize(in_shape)?;
    let in_strides_gpu = ctx.htod_usize(&in_strides)?;
    let out_strides_gpu = ctx.htod_usize(&out_strides)?;
    let perm_gpu = ctx.htod_usize(&perm_usize)?;

    let config = garboard::LaunchConfig::for_num_elems(total_elements as u32);

    // Dispatch based on input dtype
    match input {
        GpuTensor::Float32 { .. } => {
            let mut output = pool.get_tensor_f32(ctx, out_shape)?;
            // SAFETY: transpose_general_kernel takes (float* out, const float* inp,
            //   const size_t* in_shape, const size_t* in_strides,
            //   const size_t* out_strides, const size_t* perm,
            //   size_t ndim, size_t total_elements).
            let kernel = unsafe {
                cache.module().typed_kernel::<(
                    &mut garboard::DeviceSlice<'_, f32>,
                    &garboard::DeviceSlice<'_, f32>,
                    &garboard::DeviceSlice<'_, u64>,
                    &garboard::DeviceSlice<'_, u64>,
                    &garboard::DeviceSlice<'_, u64>,
                    &garboard::DeviceSlice<'_, u64>,
                    usize,
                    usize,
                )>("transpose_general_kernel")
            }
            .map_err(|e| CudaError::Kernel(format!("transpose_general_kernel lookup: {}", e)))?;
            let out_slice = op_ctx("Transpose", output.data_f32_mut())?;
            let in_slice = op_ctx("Transpose", input.data_f32())?;
            kernel
                .launch(
                    ctx.garboard_stream(),
                    &config,
                    (
                        out_slice,
                        in_slice,
                        &in_shape_gpu,
                        &in_strides_gpu,
                        &out_strides_gpu,
                        &perm_gpu,
                        ndim,
                        total_elements,
                    ),
                )
                .map_err(|e| {
                    CudaError::Kernel(format!("transpose_general launch failed: {}", e))
                })?;
            Ok(output)
        }
        GpuTensor::Int64 { .. } => {
            let mut output = pool.get_tensor_i64(ctx, out_shape)?;
            // SAFETY: transpose_general_i64_kernel signature mirrors transpose_general_kernel
            // with `long long*` (i64) data pointers in place of `float*`.
            let kernel = unsafe {
                cache.module().typed_kernel::<(
                    &mut garboard::DeviceSlice<'_, i64>,
                    &garboard::DeviceSlice<'_, i64>,
                    &garboard::DeviceSlice<'_, u64>,
                    &garboard::DeviceSlice<'_, u64>,
                    &garboard::DeviceSlice<'_, u64>,
                    &garboard::DeviceSlice<'_, u64>,
                    usize,
                    usize,
                )>("transpose_general_i64_kernel")
            }
            .map_err(|e| {
                CudaError::Kernel(format!("transpose_general_i64_kernel lookup: {}", e))
            })?;
            kernel
                .launch(
                    ctx.garboard_stream(),
                    &config,
                    (
                        output.data_i64_mut()?,
                        input.data_i64()?,
                        &in_shape_gpu,
                        &in_strides_gpu,
                        &out_strides_gpu,
                        &perm_gpu,
                        ndim,
                        total_elements,
                    ),
                )
                .map_err(|e| {
                    CudaError::Kernel(format!("transpose_general_i64 launch failed: {}", e))
                })?;
            Ok(output)
        }
        GpuTensor::Int32 { .. } => {
            let mut output = pool.get_tensor_i32(ctx, out_shape)?;
            // SAFETY: transpose_general_i32_kernel signature mirrors transpose_general_kernel
            // with `int*` (i32) data pointers in place of `float*`. Garboard's
            // `TypedKernel` requires the kernel's parameter types to match the
            // Rust-side tuple, so we compile a dedicated i32 symbol rather than
            // reusing the f32 kernel.
            let kernel = unsafe {
                cache.module().typed_kernel::<(
                    &mut garboard::DeviceSlice<'_, i32>,
                    &garboard::DeviceSlice<'_, i32>,
                    &garboard::DeviceSlice<'_, u64>,
                    &garboard::DeviceSlice<'_, u64>,
                    &garboard::DeviceSlice<'_, u64>,
                    &garboard::DeviceSlice<'_, u64>,
                    usize,
                    usize,
                )>("transpose_general_i32_kernel")
            }
            .map_err(|e| {
                CudaError::Kernel(format!("transpose_general_i32_kernel lookup: {}", e))
            })?;
            kernel
                .launch(
                    ctx.garboard_stream(),
                    &config,
                    (
                        output.data_i32_mut()?,
                        input.data_i32()?,
                        &in_shape_gpu,
                        &in_strides_gpu,
                        &out_strides_gpu,
                        &perm_gpu,
                        ndim,
                        total_elements,
                    ),
                )
                .map_err(|e| {
                    CudaError::Kernel(format!("transpose_general_i32 launch failed: {}", e))
                })?;
            Ok(output)
        }
    }
}

/// GPU Where: out = condition ? x : y
///
/// Supports Float32, Int64, and Int32 for x/y tensors (condition is always Float32).
/// Output dtype matches x/y dtype.
pub fn gpu_where(
    ctx: &IconnxCudaContext,
    cache: &OpsKernelCache,
    pool: &mut GpuMemoryPool,
    condition: &GpuTensor,
    x: &GpuTensor,
    y: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    let n = condition.len();
    assert_eq!(x.len(), n, "x length mismatch");
    assert_eq!(y.len(), n, "y length mismatch");

    let dtype = x.dtype();
    if y.dtype() != dtype {
        return Err(CudaError::Kernel(format!(
            "Where: x and y must have same dtype, got {} and {}",
            dtype.name(),
            y.dtype().name()
        )));
    }

    let config = garboard::LaunchConfig::for_num_elems(n as u32);

    // Dispatch based on x/y dtype
    match dtype {
        crate::cuda::tensor::DType::Float32 => {
            let mut output = pool.get_tensor_f32(ctx, condition.shape().to_vec())?;
            // SAFETY: where_kernel takes (float* out, const float* condition,
            //   const float* x, const float* y, size_t n).
            let kernel = unsafe {
                cache.module().typed_kernel::<(
                    &mut garboard::DeviceSlice<'_, f32>,
                    &garboard::DeviceSlice<'_, f32>,
                    &garboard::DeviceSlice<'_, f32>,
                    &garboard::DeviceSlice<'_, f32>,
                    usize,
                )>("where_kernel")
            }
            .map_err(|e| CudaError::Kernel(format!("where_kernel lookup: {}", e)))?;
            let out_slice = op_ctx("Where", output.data_f32_mut())?;
            let cond_slice = op_ctx("Where", condition.data_f32())?;
            let x_slice = op_ctx("Where", x.data_f32())?;
            let y_slice = op_ctx("Where", y.data_f32())?;
            kernel
                .launch(
                    ctx.garboard_stream(),
                    &config,
                    (out_slice, cond_slice, x_slice, y_slice, n),
                )
                .map_err(|e| CudaError::Kernel(format!("where launch failed: {}", e)))?;
            Ok(output)
        }
        crate::cuda::tensor::DType::Int64 => {
            let mut output = pool.get_tensor_i64(ctx, condition.shape().to_vec())?;
            // SAFETY: where_i64_kernel takes (long long* out, const float* condition,
            //   const long long* x, const long long* y, size_t n).
            let kernel = unsafe {
                cache.module().typed_kernel::<(
                    &mut garboard::DeviceSlice<'_, i64>,
                    &garboard::DeviceSlice<'_, f32>,
                    &garboard::DeviceSlice<'_, i64>,
                    &garboard::DeviceSlice<'_, i64>,
                    usize,
                )>("where_i64_kernel")
            }
            .map_err(|e| CudaError::Kernel(format!("where_i64_kernel lookup: {}", e)))?;
            let cond_slice = op_ctx("Where", condition.data_f32())?;
            kernel
                .launch(
                    ctx.garboard_stream(),
                    &config,
                    (
                        output.data_i64_mut()?,
                        cond_slice,
                        x.data_i64()?,
                        y.data_i64()?,
                        n,
                    ),
                )
                .map_err(|e| CudaError::Kernel(format!("where_i64 launch failed: {}", e)))?;
            Ok(output)
        }
        crate::cuda::tensor::DType::Int32 => {
            let mut output = pool.get_tensor_i32(ctx, condition.shape().to_vec())?;
            // SAFETY: where_i32_kernel takes (int* out, const float* condition,
            //   const int* x, const int* y, size_t n).
            let kernel = unsafe {
                cache.module().typed_kernel::<(
                    &mut garboard::DeviceSlice<'_, i32>,
                    &garboard::DeviceSlice<'_, f32>,
                    &garboard::DeviceSlice<'_, i32>,
                    &garboard::DeviceSlice<'_, i32>,
                    usize,
                )>("where_i32_kernel")
            }
            .map_err(|e| CudaError::Kernel(format!("where_i32_kernel lookup: {}", e)))?;
            let cond_slice = op_ctx("Where", condition.data_f32())?;
            kernel
                .launch(
                    ctx.garboard_stream(),
                    &config,
                    (
                        output.data_i32_mut()?,
                        cond_slice,
                        x.data_i32()?,
                        y.data_i32()?,
                        n,
                    ),
                )
                .map_err(|e| CudaError::Kernel(format!("where_i32 launch failed: {}", e)))?;
            Ok(output)
        }
    }
}

/// GPU Gather along axis 0
///
/// ONNX Gather semantics for axis=0:
/// - data: [N, D1, D2, ...] (input tensor)
/// - indices: [I1, I2, ...] (indices tensor with arbitrary shape)
/// - output: [I1, I2, ..., D1, D2, ...] (indices shape + data shape[1:])
///
/// For example, embedding lookup:
/// - data (embedding): [vocab_size, embed_dim]
/// - indices (tokens): [batch, seq_len]
/// - output: [batch, seq_len, embed_dim]
///
/// Supports Float32 and Int64 input tensors
/// Handles negative indices (Python/ONNX convention: -1 = last element)
pub fn gpu_gather(
    ctx: &IconnxCudaContext,
    cache: &OpsKernelCache,
    pool: &mut GpuMemoryPool,
    input: &GpuTensor,
    indices: &[i64],
    indices_shape: &[usize], // Shape of the indices tensor
) -> Result<GpuTensor, CudaError> {
    let data_shape = input.shape();
    let dim0_size = data_shape.first().copied().unwrap_or(1);
    let slice_size: usize = if data_shape.len() > 1 {
        data_shape[1..].iter().product()
    } else {
        1
    };
    let num_indices = indices.len();

    // Upload indices to GPU
    let indices_gpu = ctx.htod_i64(indices)?;

    // Output shape = indices_shape + data_shape[1:]
    let mut out_shape: Vec<usize> = indices_shape.to_vec();
    if data_shape.len() > 1 {
        out_shape.extend_from_slice(&data_shape[1..]);
    }

    let total = num_indices * slice_size;
    let config = garboard::LaunchConfig::for_num_elems(total as u32);

    // Dispatch based on input dtype
    match input {
        GpuTensor::Float32 { .. } => {
            let mut output = pool.get_tensor_f32(ctx, out_shape)?;
            // SAFETY: gather_kernel takes (float* out, const float* inp,
            //   const long long* indices, size_t num_indices, size_t slice_size,
            //   size_t dim0_size).
            let kernel = unsafe {
                cache.module().typed_kernel::<(
                    &mut garboard::DeviceSlice<'_, f32>,
                    &garboard::DeviceSlice<'_, f32>,
                    &garboard::DeviceSlice<'_, i64>,
                    usize,
                    usize,
                    usize,
                )>("gather_kernel")
            }
            .map_err(|e| CudaError::Kernel(format!("gather_kernel lookup: {}", e)))?;
            let out_slice = op_ctx("Gather", output.data_f32_mut())?;
            let in_slice = op_ctx("Gather", input.data_f32())?;
            kernel
                .launch(
                    ctx.garboard_stream(),
                    &config,
                    (
                        out_slice,
                        in_slice,
                        &indices_gpu,
                        num_indices,
                        slice_size,
                        dim0_size,
                    ),
                )
                .map_err(|e| CudaError::Kernel(format!("gather launch failed: {}", e)))?;
            Ok(output)
        }
        GpuTensor::Int64 { .. } => {
            let mut output = pool.get_tensor_i64(ctx, out_shape)?;
            // SAFETY: gather_i64_kernel takes (long long* out, const long long* inp,
            //   const long long* indices, size_t num_indices, size_t slice_size,
            //   size_t dim0_size).
            let kernel = unsafe {
                cache.module().typed_kernel::<(
                    &mut garboard::DeviceSlice<'_, i64>,
                    &garboard::DeviceSlice<'_, i64>,
                    &garboard::DeviceSlice<'_, i64>,
                    usize,
                    usize,
                    usize,
                )>("gather_i64_kernel")
            }
            .map_err(|e| CudaError::Kernel(format!("gather_i64_kernel lookup: {}", e)))?;
            kernel
                .launch(
                    ctx.garboard_stream(),
                    &config,
                    (
                        output.data_i64_mut()?,
                        input.data_i64()?,
                        &indices_gpu,
                        num_indices,
                        slice_size,
                        dim0_size,
                    ),
                )
                .map_err(|e| CudaError::Kernel(format!("gather_i64 launch failed: {}", e)))?;
            Ok(output)
        }
        _ => Err(CudaError::Kernel(format!(
            "Gather: unsupported input dtype {}",
            input.dtype().name()
        ))),
    }
}

/// GPU Gather with GPU-resident indices (avoids D2H + H2D round trip)
/// indices_tensor must be Int64 and already on GPU
pub fn gpu_gather_from_gpu(
    ctx: &IconnxCudaContext,
    cache: &OpsKernelCache,
    pool: &mut GpuMemoryPool,
    input: &GpuTensor,
    indices_tensor: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    let data_shape = input.shape();
    let indices_shape = indices_tensor.shape();
    let dim0_size = data_shape.first().copied().unwrap_or(1);
    let slice_size: usize = if data_shape.len() > 1 {
        data_shape[1..].iter().product()
    } else {
        1
    };
    let num_indices = indices_tensor.len();

    // Output shape = indices_shape + data_shape[1:]
    let mut out_shape: Vec<usize> = indices_shape.to_vec();
    if data_shape.len() > 1 {
        out_shape.extend_from_slice(&data_shape[1..]);
    }

    let total = num_indices * slice_size;
    let config = garboard::LaunchConfig::for_num_elems(total as u32);

    // Dispatch based on input dtype
    match input {
        GpuTensor::Float32 { .. } => {
            let mut output = pool.get_tensor_f32(ctx, out_shape)?;
            // SAFETY: gather_kernel takes (float* out, const float* inp,
            //   const long long* indices, size_t num_indices, size_t slice_size,
            //   size_t dim0_size).
            let kernel = unsafe {
                cache.module().typed_kernel::<(
                    &mut garboard::DeviceSlice<'_, f32>,
                    &garboard::DeviceSlice<'_, f32>,
                    &garboard::DeviceSlice<'_, i64>,
                    usize,
                    usize,
                    usize,
                )>("gather_kernel")
            }
            .map_err(|e| CudaError::Kernel(format!("gather_kernel lookup: {}", e)))?;
            let out_slice = op_ctx("Gather", output.data_f32_mut())?;
            let in_slice = op_ctx("Gather", input.data_f32())?;
            kernel
                .launch(
                    ctx.garboard_stream(),
                    &config,
                    (
                        out_slice,
                        in_slice,
                        indices_tensor.data_i64()?,
                        num_indices,
                        slice_size,
                        dim0_size,
                    ),
                )
                .map_err(|e| CudaError::Kernel(format!("gather launch failed: {}", e)))?;
            Ok(output)
        }
        GpuTensor::Int64 { .. } => {
            let mut output = pool.get_tensor_i64(ctx, out_shape)?;
            // SAFETY: gather_i64_kernel takes (long long* out, const long long* inp,
            //   const long long* indices, size_t num_indices, size_t slice_size,
            //   size_t dim0_size).
            let kernel = unsafe {
                cache.module().typed_kernel::<(
                    &mut garboard::DeviceSlice<'_, i64>,
                    &garboard::DeviceSlice<'_, i64>,
                    &garboard::DeviceSlice<'_, i64>,
                    usize,
                    usize,
                    usize,
                )>("gather_i64_kernel")
            }
            .map_err(|e| CudaError::Kernel(format!("gather_i64_kernel lookup: {}", e)))?;
            kernel
                .launch(
                    ctx.garboard_stream(),
                    &config,
                    (
                        output.data_i64_mut()?,
                        input.data_i64()?,
                        indices_tensor.data_i64()?,
                        num_indices,
                        slice_size,
                        dim0_size,
                    ),
                )
                .map_err(|e| CudaError::Kernel(format!("gather_i64 launch failed: {}", e)))?;
            Ok(output)
        }
        _ => Err(CudaError::Kernel(format!(
            "Gather: unsupported input dtype {}",
            input.dtype().name()
        ))),
    }
}

/// GPU Slice: Extract contiguous slice
pub fn gpu_slice_contiguous(
    ctx: &IconnxCudaContext,
    cache: &OpsKernelCache,
    pool: &mut GpuMemoryPool,
    input: &GpuTensor,
    start: usize,
    length: usize,
    out_shape: Vec<usize>,
) -> Result<GpuTensor, CudaError> {
    assert!(start + length <= input.len(), "Slice out of bounds");

    let mut output = pool.get_tensor_f32(ctx, out_shape)?;

    // SAFETY: slice_kernel takes (float* out, const float* inp,
    //   size_t start_offset, size_t total_elements).
    let kernel = unsafe {
        cache.module().typed_kernel::<(
            &mut garboard::DeviceSlice<'_, f32>,
            &garboard::DeviceSlice<'_, f32>,
            usize,
            usize,
        )>("slice_kernel")
    }
    .map_err(|e| CudaError::Kernel(format!("slice_kernel lookup: {}", e)))?;

    let config = garboard::LaunchConfig::for_num_elems(length as u32);

    let out_slice = op_ctx("Slice", output.data_f32_mut())?;
    let in_slice = op_ctx("Slice", input.data_f32())?;
    kernel
        .launch(
            ctx.garboard_stream(),
            &config,
            (out_slice, in_slice, start, length),
        )
        .map_err(|e| CudaError::Kernel(format!("slice launch failed: {}", e)))?;

    Ok(output)
}

/// GPU Copy: Simple memory copy (Identity op)
pub fn gpu_copy(
    ctx: &IconnxCudaContext,
    cache: &OpsKernelCache,
    pool: &mut GpuMemoryPool,
    input: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    let n = input.len();

    let mut output = pool.get_tensor_f32(ctx, input.shape().to_vec())?;

    // SAFETY: copy_kernel takes (float* out, const float* inp, size_t n).
    let kernel = unsafe {
        cache.module().typed_kernel::<(
            &mut garboard::DeviceSlice<'_, f32>,
            &garboard::DeviceSlice<'_, f32>,
            usize,
        )>("copy_kernel")
    }
    .map_err(|e| CudaError::Kernel(format!("copy_kernel lookup: {}", e)))?;

    let config = garboard::LaunchConfig::for_num_elems(n as u32);

    let out_slice = op_ctx("Copy", output.data_f32_mut())?;
    let in_slice = op_ctx("Copy", input.data_f32())?;
    kernel
        .launch(
            ctx.garboard_stream(),
            &config,
            (out_slice, in_slice, n),
        )
        .map_err(|e| CudaError::Kernel(format!("copy launch failed: {}", e)))?;

    Ok(output)
}

/// GPU Concat: Concatenate tensors along an axis
/// Supports Float32, Int64, and Int32 tensors
pub fn gpu_concat(
    ctx: &IconnxCudaContext,
    cache: &OpsKernelCache,
    pool: &mut GpuMemoryPool,
    inputs: &[&GpuTensor],
    axis: i64,
) -> Result<GpuTensor, CudaError> {
    if inputs.is_empty() {
        return Err(CudaError::Kernel(
            "Concat requires at least one input".into(),
        ));
    }

    let ndim = inputs[0].ndim();
    let dtype = inputs[0].dtype();
    let axis = if axis < 0 {
        (ndim as i64 + axis) as usize
    } else {
        axis as usize
    };

    // Validate shapes and dtypes - all dims except axis must match, dtypes must match
    for (i, inp) in inputs.iter().enumerate().skip(1) {
        if inp.ndim() != ndim {
            return Err(CudaError::Kernel(format!(
                "Concat: ndim mismatch at input {}: {} vs {}",
                i,
                inp.ndim(),
                ndim
            )));
        }
        if inp.dtype() != dtype {
            return Err(CudaError::Kernel(format!(
                "Concat: dtype mismatch at input {}: {} vs {}",
                i,
                inp.dtype().name(),
                dtype.name()
            )));
        }
        for d in 0..ndim {
            if d != axis && inp.shape()[d] != inputs[0].shape()[d] {
                return Err(CudaError::Kernel(format!(
                    "Concat: shape mismatch at dim {} of input {}",
                    d, i
                )));
            }
        }
    }

    // Calculate output shape
    let mut out_shape = inputs[0].shape().to_vec();
    let out_axis_size: usize = inputs.iter().map(|t| t.shape()[axis]).sum();
    out_shape[axis] = out_axis_size;

    // Calculate outer_size (product of dims before axis) and inner_size (product of dims after axis)
    let outer_size: usize = if axis > 0 {
        out_shape[..axis].iter().product()
    } else {
        1
    };
    let inner_size: usize = if axis + 1 < ndim {
        out_shape[axis + 1..].iter().product()
    } else {
        1
    };

    // Dispatch based on dtype
    match dtype {
        crate::cuda::tensor::DType::Float32 => {
            let mut output = pool.get_tensor_f32(ctx, out_shape.clone())?;
            // SAFETY: concat_kernel takes (float* out, const float* inp,
            //   size_t inp_offset, size_t inp_axis_size, size_t outer_size,
            //   size_t inner_size, size_t out_axis_size,
            //   size_t total_inp_elements).
            let kernel = unsafe {
                cache.module().typed_kernel::<(
                    &mut garboard::DeviceSlice<'_, f32>,
                    &garboard::DeviceSlice<'_, f32>,
                    usize,
                    usize,
                    usize,
                    usize,
                    usize,
                    usize,
                )>("concat_kernel")
            }
            .map_err(|e| CudaError::Kernel(format!("concat_kernel lookup: {}", e)))?;

            let mut inp_offset: usize = 0;
            for inp in inputs {
                let inp_axis_size = inp.shape()[axis];
                let total_inp_elements = inp.len();

                // Skip empty inputs to avoid invalid launch config
                if total_inp_elements == 0 {
                    inp_offset += inp_axis_size;
                    continue;
                }

                let config = garboard::LaunchConfig::for_num_elems(total_inp_elements as u32);

                let out_slice = op_ctx("Concat", output.data_f32_mut())?;
                let in_slice = op_ctx("Concat", inp.data_f32())?;
                kernel
                    .launch(
                        ctx.garboard_stream(),
                        &config,
                        (
                            out_slice,
                            in_slice,
                            inp_offset,
                            inp_axis_size,
                            outer_size,
                            inner_size,
                            out_axis_size,
                            total_inp_elements,
                        ),
                    )
                    .map_err(|e| CudaError::Kernel(format!("concat launch failed: {}", e)))?;

                inp_offset += inp_axis_size;
            }
            Ok(output)
        }
        crate::cuda::tensor::DType::Int64 => {
            let mut output = pool.get_tensor_i64(ctx, out_shape.clone())?;
            // SAFETY: concat_i64_kernel mirrors concat_kernel with `long long*`
            // (i64) data pointers in place of `float*`.
            let kernel = unsafe {
                cache.module().typed_kernel::<(
                    &mut garboard::DeviceSlice<'_, i64>,
                    &garboard::DeviceSlice<'_, i64>,
                    usize,
                    usize,
                    usize,
                    usize,
                    usize,
                    usize,
                )>("concat_i64_kernel")
            }
            .map_err(|e| CudaError::Kernel(format!("concat_i64_kernel lookup: {}", e)))?;

            let mut inp_offset: usize = 0;
            for inp in inputs {
                let inp_axis_size = inp.shape()[axis];
                let total_inp_elements = inp.len();

                // Skip empty inputs to avoid invalid launch config
                if total_inp_elements == 0 {
                    inp_offset += inp_axis_size;
                    continue;
                }

                let config = garboard::LaunchConfig::for_num_elems(total_inp_elements as u32);

                kernel
                    .launch(
                        ctx.garboard_stream(),
                        &config,
                        (
                            output.data_i64_mut()?,
                            inp.data_i64()?,
                            inp_offset,
                            inp_axis_size,
                            outer_size,
                            inner_size,
                            out_axis_size,
                            total_inp_elements,
                        ),
                    )
                    .map_err(|e| CudaError::Kernel(format!("concat_i64 launch failed: {}", e)))?;

                inp_offset += inp_axis_size;
            }
            Ok(output)
        }
        crate::cuda::tensor::DType::Int32 => {
            let mut output = pool.get_tensor_i32(ctx, out_shape.clone())?;
            // SAFETY: concat_i32_kernel mirrors concat_kernel with `int*` (i32)
            // data pointers in place of `float*`.
            let kernel = unsafe {
                cache.module().typed_kernel::<(
                    &mut garboard::DeviceSlice<'_, i32>,
                    &garboard::DeviceSlice<'_, i32>,
                    usize,
                    usize,
                    usize,
                    usize,
                    usize,
                    usize,
                )>("concat_i32_kernel")
            }
            .map_err(|e| CudaError::Kernel(format!("concat_i32_kernel lookup: {}", e)))?;

            let mut inp_offset: usize = 0;
            for inp in inputs {
                let inp_axis_size = inp.shape()[axis];
                let total_inp_elements = inp.len();

                // Skip empty inputs to avoid invalid launch config
                if total_inp_elements == 0 {
                    inp_offset += inp_axis_size;
                    continue;
                }

                let config = garboard::LaunchConfig::for_num_elems(total_inp_elements as u32);

                kernel
                    .launch(
                        ctx.garboard_stream(),
                        &config,
                        (
                            output.data_i32_mut()?,
                            inp.data_i32()?,
                            inp_offset,
                            inp_axis_size,
                            outer_size,
                            inner_size,
                            out_axis_size,
                            total_inp_elements,
                        ),
                    )
                    .map_err(|e| CudaError::Kernel(format!("concat_i32 launch failed: {}", e)))?;

                inp_offset += inp_axis_size;
            }
            Ok(output)
        }
    }
}

/// GPU Expand: Broadcast tensor to larger shape
///
/// Supports Float32, Int64, and Int32 tensors.
pub fn gpu_expand(
    ctx: &IconnxCudaContext,
    cache: &OpsKernelCache,
    pool: &mut GpuMemoryPool,
    input: &GpuTensor,
    target_shape: Vec<usize>,
) -> Result<GpuTensor, CudaError> {
    let inp_shape = input.shape();
    let ndim = target_shape.len().max(inp_shape.len());
    let dtype = input.dtype();

    // Pad input shape with 1s on the left if needed
    let mut padded_inp_shape = vec![1usize; ndim];
    let offset = ndim - inp_shape.len();
    for (i, &dim) in inp_shape.iter().enumerate() {
        padded_inp_shape[offset + i] = dim;
    }

    // Pad target shape with 1s on the left if needed
    let mut padded_target_shape = vec![1usize; ndim];
    let target_offset = ndim - target_shape.len();
    for (i, &dim) in target_shape.iter().enumerate() {
        padded_target_shape[target_offset + i] = dim;
    }

    // ONNX Expand uses bidirectional broadcasting:
    // Output dim = max(inp_dim, target_dim) when one of them is 1
    // Validate and compute actual output shape
    let mut out_shape = Vec::with_capacity(ndim);
    for (i, (&inp_dim, &target_dim)) in padded_inp_shape
        .iter()
        .zip(padded_target_shape.iter())
        .enumerate()
    {
        if inp_dim == target_dim {
            out_shape.push(inp_dim);
        } else if inp_dim == 1 {
            out_shape.push(target_dim);
        } else if target_dim == 1 {
            out_shape.push(inp_dim);
        } else {
            return Err(CudaError::Kernel(format!(
                "Expand: incompatible shapes at dim {}: {} vs {}",
                i, inp_dim, target_dim
            )));
        }
    }

    let total_elements: usize = out_shape.iter().product();

    // Calculate strides
    let out_strides = compute_strides(&out_shape);
    let inp_strides = compute_strides(&padded_inp_shape);

    // Upload shape arrays to GPU
    let out_shape_gpu = ctx.htod_usize(&out_shape)?;
    let out_strides_gpu = ctx.htod_usize(&out_strides)?;
    let inp_shape_gpu = ctx.htod_usize(&padded_inp_shape)?;
    let inp_strides_gpu = ctx.htod_usize(&inp_strides)?;

    let config = garboard::LaunchConfig::for_num_elems(total_elements as u32);

    // Dispatch based on dtype
    match dtype {
        crate::cuda::tensor::DType::Float32 => {
            let mut output = pool.get_tensor_f32(ctx, out_shape.clone())?;
            // SAFETY: expand_kernel takes (float* out, const float* inp,
            //   const size_t* out_shape, const size_t* out_strides,
            //   const size_t* inp_shape, const size_t* inp_strides,
            //   size_t ndim, size_t total_elements).
            let kernel = unsafe {
                cache.module().typed_kernel::<(
                    &mut garboard::DeviceSlice<'_, f32>,
                    &garboard::DeviceSlice<'_, f32>,
                    &garboard::DeviceSlice<'_, u64>,
                    &garboard::DeviceSlice<'_, u64>,
                    &garboard::DeviceSlice<'_, u64>,
                    &garboard::DeviceSlice<'_, u64>,
                    usize,
                    usize,
                )>("expand_kernel")
            }
            .map_err(|e| CudaError::Kernel(format!("expand_kernel lookup: {}", e)))?;
            let out_slice = op_ctx("Expand", output.data_f32_mut())?;
            let in_slice = op_ctx("Expand", input.data_f32())?;
            kernel
                .launch(
                    ctx.garboard_stream(),
                    &config,
                    (
                        out_slice,
                        in_slice,
                        &out_shape_gpu,
                        &out_strides_gpu,
                        &inp_shape_gpu,
                        &inp_strides_gpu,
                        ndim,
                        total_elements,
                    ),
                )
                .map_err(|e| CudaError::Kernel(format!("expand launch failed: {}", e)))?;
            Ok(output)
        }
        crate::cuda::tensor::DType::Int64 => {
            let mut output = pool.get_tensor_i64(ctx, out_shape.clone())?;
            // SAFETY: expand_i64_kernel mirrors expand_kernel with `long long*`
            // (i64) data pointers in place of `float*`.
            let kernel = unsafe {
                cache.module().typed_kernel::<(
                    &mut garboard::DeviceSlice<'_, i64>,
                    &garboard::DeviceSlice<'_, i64>,
                    &garboard::DeviceSlice<'_, u64>,
                    &garboard::DeviceSlice<'_, u64>,
                    &garboard::DeviceSlice<'_, u64>,
                    &garboard::DeviceSlice<'_, u64>,
                    usize,
                    usize,
                )>("expand_i64_kernel")
            }
            .map_err(|e| CudaError::Kernel(format!("expand_i64_kernel lookup: {}", e)))?;
            kernel
                .launch(
                    ctx.garboard_stream(),
                    &config,
                    (
                        output.data_i64_mut()?,
                        input.data_i64()?,
                        &out_shape_gpu,
                        &out_strides_gpu,
                        &inp_shape_gpu,
                        &inp_strides_gpu,
                        ndim,
                        total_elements,
                    ),
                )
                .map_err(|e| CudaError::Kernel(format!("expand_i64 launch failed: {}", e)))?;
            Ok(output)
        }
        crate::cuda::tensor::DType::Int32 => {
            let mut output = pool.get_tensor_i32(ctx, out_shape.clone())?;
            // SAFETY: expand_i32_kernel mirrors expand_kernel with `int*`
            // (i32) data pointers in place of `float*`.
            let kernel = unsafe {
                cache.module().typed_kernel::<(
                    &mut garboard::DeviceSlice<'_, i32>,
                    &garboard::DeviceSlice<'_, i32>,
                    &garboard::DeviceSlice<'_, u64>,
                    &garboard::DeviceSlice<'_, u64>,
                    &garboard::DeviceSlice<'_, u64>,
                    &garboard::DeviceSlice<'_, u64>,
                    usize,
                    usize,
                )>("expand_i32_kernel")
            }
            .map_err(|e| CudaError::Kernel(format!("expand_i32_kernel lookup: {}", e)))?;
            kernel
                .launch(
                    ctx.garboard_stream(),
                    &config,
                    (
                        output.data_i32_mut()?,
                        input.data_i32()?,
                        &out_shape_gpu,
                        &out_strides_gpu,
                        &inp_shape_gpu,
                        &inp_strides_gpu,
                        ndim,
                        total_elements,
                    ),
                )
                .map_err(|e| CudaError::Kernel(format!("expand_i32 launch failed: {}", e)))?;
            Ok(output)
        }
    }
}

/// GPU Pad: Pad tensor with constant or reflect mode
/// Uses optimized scalar kernels for 3D and 4D tensors (avoids H2D transfers)
/// Supports "constant", "reflect", and "edge" padding modes
pub fn gpu_pad(
    ctx: &IconnxCudaContext,
    cache: &OpsKernelCache,
    pool: &mut GpuMemoryPool,
    input: &GpuTensor,
    pads: &[i64],
    value: f32,
    mode: &str,
) -> Result<GpuTensor, CudaError> {
    let inp_shape = input.shape();
    let ndim = inp_shape.len();

    // pads is [begin_0, begin_1, ..., end_0, end_1, ...]
    if pads.len() != ndim * 2 {
        return Err(CudaError::Kernel(format!(
            "Pad: expected {} pad values, got {}",
            ndim * 2,
            pads.len()
        )));
    }

    // Calculate output shape and collect pad values
    let mut out_shape = Vec::with_capacity(ndim);
    let mut pads_begin = Vec::with_capacity(ndim);
    for i in 0..ndim {
        let pad_begin = pads[i] as usize;
        let pad_end = pads[ndim + i] as usize;
        pads_begin.push(pad_begin);
        out_shape.push(inp_shape[i] + pad_begin + pad_end);
    }

    let total_out_elements: usize = out_shape.iter().product();
    let mut output = pool.get_tensor_f32(ctx, out_shape.clone())?;

    // Calculate strides
    let out_strides = compute_strides(&out_shape);
    let inp_strides = compute_strides(inp_shape);

    let config = garboard::LaunchConfig::for_num_elems(total_out_elements as u32);

    // Select kernel based on mode
    match mode {
        "reflect" => {
            // Reflect padding: reflect values from input edges
            // For 3D tensor with pads [0, 0, 10, 0, 0, 10]:
            //   - Dimension 0: no padding
            //   - Dimension 1: reflect 10 values at start and end
            //   - Dimension 2: no padding
            gpu_pad_reflect(
                ctx,
                cache,
                &mut output,
                input,
                inp_shape,
                &out_shape,
                &inp_strides,
                &out_strides,
                &pads_begin,
                total_out_elements,
                config,
            )?;
        }
        "edge" => {
            // Edge padding: replicate edge values
            gpu_pad_edge(
                ctx,
                cache,
                &mut output,
                input,
                inp_shape,
                &out_shape,
                &inp_strides,
                &out_strides,
                &pads_begin,
                total_out_elements,
                config,
            )?;
        }
        _ => {
            // Default: constant padding
            gpu_pad_constant(
                ctx,
                cache,
                &mut output,
                input,
                inp_shape,
                &out_shape,
                &inp_strides,
                &out_strides,
                &pads_begin,
                value,
                total_out_elements,
                config,
            )?;
        }
    }

    Ok(output)
}

/// GPU Pad with constant value (original implementation)
#[allow(clippy::too_many_arguments)] // Pad params are genuinely independent and grouping would obscure intent
fn gpu_pad_constant(
    ctx: &IconnxCudaContext,
    cache: &OpsKernelCache,
    output: &mut GpuTensor,
    input: &GpuTensor,
    inp_shape: &[usize],
    out_shape: &[usize],
    inp_strides: &[usize],
    out_strides: &[usize],
    pads_begin: &[usize],
    value: f32,
    total_out_elements: usize,
    config: garboard::LaunchConfig,
) -> Result<(), CudaError> {
    let ndim = inp_shape.len();

    match ndim {
        3 => {
            // Pack 15 size_t params: inp_d[3], out_d[3], inp_s[3], out_s[3], pad_b[3].
            let params: Vec<usize> = [
                &inp_shape[..3],
                &out_shape[..3],
                &inp_strides[..3],
                &out_strides[..3],
                &pads_begin[..3],
            ]
            .concat();
            let params_gpu = ctx.htod_usize(&params)?;

            // SAFETY: pad_3d_scalar_kernel takes
            // (float* out, const float* inp, const size_t* params,
            //  float pad_value, size_t total_out_elements).
            let kernel = unsafe {
                cache.module().typed_kernel::<(
                    &mut garboard::DeviceSlice<'_, f32>,
                    &garboard::DeviceSlice<'_, f32>,
                    &garboard::DeviceSlice<'_, u64>,
                    f32,
                    usize,
                )>("pad_3d_scalar_kernel")
            }
            .map_err(|e| CudaError::Kernel(format!("pad_3d_scalar_kernel lookup: {}", e)))?;

            let out_slice = op_ctx("Pad", output.data_f32_mut())?;
            let in_slice = op_ctx("Pad", input.data_f32())?;
            kernel
                .launch(
                    ctx.garboard_stream(),
                    &config,
                    (out_slice, in_slice, &params_gpu, value, total_out_elements),
                )
                .map_err(|e| CudaError::Kernel(format!("pad_3d launch failed: {}", e)))?;
        }
        4 => {
            // Pack 20 size_t params: inp_d[4], out_d[4], inp_s[4], out_s[4], pad_b[4].
            let params: Vec<usize> = [
                &inp_shape[..4],
                &out_shape[..4],
                &inp_strides[..4],
                &out_strides[..4],
                &pads_begin[..4],
            ]
            .concat();
            let params_gpu = ctx.htod_usize(&params)?;

            // SAFETY: pad_4d_scalar_kernel takes
            // (float* out, const float* inp, const size_t* params,
            //  float pad_value, size_t total_out_elements).
            let kernel = unsafe {
                cache.module().typed_kernel::<(
                    &mut garboard::DeviceSlice<'_, f32>,
                    &garboard::DeviceSlice<'_, f32>,
                    &garboard::DeviceSlice<'_, u64>,
                    f32,
                    usize,
                )>("pad_4d_scalar_kernel")
            }
            .map_err(|e| CudaError::Kernel(format!("pad_4d_scalar_kernel lookup: {}", e)))?;

            let out_slice = op_ctx("Pad", output.data_f32_mut())?;
            let in_slice = op_ctx("Pad", input.data_f32())?;
            kernel
                .launch(
                    ctx.garboard_stream(),
                    &config,
                    (out_slice, in_slice, &params_gpu, value, total_out_elements),
                )
                .map_err(|e| CudaError::Kernel(format!("pad_4d launch failed: {}", e)))?;
        }
        _ => {
            // Fall back to generic kernel with H2D transfers for other dimensions
            let inp_shape_vec: Vec<usize> = inp_shape.to_vec();
            let inp_shape_gpu = ctx.htod_usize(&inp_shape_vec)?;
            let out_shape_gpu = ctx.htod_usize(out_shape)?;
            let inp_strides_gpu = ctx.htod_usize(inp_strides)?;
            let out_strides_gpu = ctx.htod_usize(out_strides)?;
            let pads_begin_gpu = ctx.htod_usize(pads_begin)?;

            // SAFETY: pad_kernel takes (float* out, const float* inp,
            //   const size_t* inp_shape, const size_t* out_shape,
            //   const size_t* inp_strides, const size_t* out_strides,
            //   const size_t* pads_begin, size_t ndim,
            //   float pad_value, size_t total_out_elements).
            let kernel = unsafe {
                cache.module().typed_kernel::<(
                    &mut garboard::DeviceSlice<'_, f32>,
                    &garboard::DeviceSlice<'_, f32>,
                    &garboard::DeviceSlice<'_, u64>,
                    &garboard::DeviceSlice<'_, u64>,
                    &garboard::DeviceSlice<'_, u64>,
                    &garboard::DeviceSlice<'_, u64>,
                    &garboard::DeviceSlice<'_, u64>,
                    usize,
                    f32,
                    usize,
                )>("pad_kernel")
            }
            .map_err(|e| CudaError::Kernel(format!("pad_kernel lookup: {}", e)))?;

            let out_slice = op_ctx("Pad", output.data_f32_mut())?;
            let in_slice = op_ctx("Pad", input.data_f32())?;
            kernel
                .launch(
                    ctx.garboard_stream(),
                    &config,
                    (
                        out_slice,
                        in_slice,
                        &inp_shape_gpu,
                        &out_shape_gpu,
                        &inp_strides_gpu,
                        &out_strides_gpu,
                        &pads_begin_gpu,
                        ndim,
                        value,
                        total_out_elements,
                    ),
                )
                .map_err(|e| CudaError::Kernel(format!("pad launch failed: {}", e)))?;
        }
    }

    Ok(())
}

/// GPU Pad with reflect mode
/// Reflects values at boundaries: [a, b, c, d] with pad 2 -> [c, b, a, b, c, d, c, b]
#[allow(clippy::too_many_arguments)] // Pad params are genuinely independent
fn gpu_pad_reflect(
    ctx: &IconnxCudaContext,
    cache: &OpsKernelCache,
    output: &mut GpuTensor,
    input: &GpuTensor,
    inp_shape: &[usize],
    out_shape: &[usize],
    inp_strides: &[usize],
    out_strides: &[usize],
    pads_begin: &[usize],
    total_out_elements: usize,
    config: garboard::LaunchConfig,
) -> Result<(), CudaError> {
    let ndim = inp_shape.len();

    match ndim {
        3 => {
            // Pack 15 size_t params: inp_d[3], out_d[3], inp_s[3], out_s[3], pad_b[3].
            let params: Vec<usize> = [
                &inp_shape[..3],
                &out_shape[..3],
                &inp_strides[..3],
                &out_strides[..3],
                &pads_begin[..3],
            ]
            .concat();
            let params_gpu = ctx.htod_usize(&params)?;

            // SAFETY: pad_reflect_3d_kernel takes
            // (float* out, const float* inp, const size_t* params,
            //  size_t total_out_elements).
            let kernel = unsafe {
                cache.module().typed_kernel::<(
                    &mut garboard::DeviceSlice<'_, f32>,
                    &garboard::DeviceSlice<'_, f32>,
                    &garboard::DeviceSlice<'_, u64>,
                    usize,
                )>("pad_reflect_3d_kernel")
            }
            .map_err(|e| CudaError::Kernel(format!("pad_reflect_3d_kernel lookup: {}", e)))?;

            let out_slice = op_ctx("Pad", output.data_f32_mut())?;
            let in_slice = op_ctx("Pad", input.data_f32())?;
            kernel
                .launch(
                    ctx.garboard_stream(),
                    &config,
                    (out_slice, in_slice, &params_gpu, total_out_elements),
                )
                .map_err(|e| CudaError::Kernel(format!("pad_reflect_3d launch failed: {}", e)))?;
        }
        _ => {
            // Fallback: use generic reflect kernel with H2D transfers
            let inp_shape_vec: Vec<usize> = inp_shape.to_vec();
            let inp_shape_gpu = ctx.htod_usize(&inp_shape_vec)?;
            let out_shape_gpu = ctx.htod_usize(out_shape)?;
            let inp_strides_gpu = ctx.htod_usize(inp_strides)?;
            let out_strides_gpu = ctx.htod_usize(out_strides)?;
            let pads_begin_gpu = ctx.htod_usize(pads_begin)?;

            // SAFETY: pad_reflect_kernel takes (float* out, const float* inp,
            //   const size_t* inp_shape, const size_t* out_shape,
            //   const size_t* inp_strides, const size_t* out_strides,
            //   const size_t* pads_begin, size_t ndim,
            //   size_t total_out_elements).
            let kernel = unsafe {
                cache.module().typed_kernel::<(
                    &mut garboard::DeviceSlice<'_, f32>,
                    &garboard::DeviceSlice<'_, f32>,
                    &garboard::DeviceSlice<'_, u64>,
                    &garboard::DeviceSlice<'_, u64>,
                    &garboard::DeviceSlice<'_, u64>,
                    &garboard::DeviceSlice<'_, u64>,
                    &garboard::DeviceSlice<'_, u64>,
                    usize,
                    usize,
                )>("pad_reflect_kernel")
            }
            .map_err(|e| CudaError::Kernel(format!("pad_reflect_kernel lookup: {}", e)))?;

            let out_slice = op_ctx("Pad", output.data_f32_mut())?;
            let in_slice = op_ctx("Pad", input.data_f32())?;
            kernel
                .launch(
                    ctx.garboard_stream(),
                    &config,
                    (
                        out_slice,
                        in_slice,
                        &inp_shape_gpu,
                        &out_shape_gpu,
                        &inp_strides_gpu,
                        &out_strides_gpu,
                        &pads_begin_gpu,
                        ndim,
                        total_out_elements,
                    ),
                )
                .map_err(|e| CudaError::Kernel(format!("pad_reflect launch failed: {}", e)))?;
        }
    }

    Ok(())
}

/// GPU Pad with edge mode
/// Replicates edge values: [a, b, c, d] with pad 2 -> [a, a, a, b, c, d, d, d]
#[allow(clippy::too_many_arguments)] // Pad params are genuinely independent
fn gpu_pad_edge(
    ctx: &IconnxCudaContext,
    cache: &OpsKernelCache,
    output: &mut GpuTensor,
    input: &GpuTensor,
    inp_shape: &[usize],
    out_shape: &[usize],
    inp_strides: &[usize],
    out_strides: &[usize],
    pads_begin: &[usize],
    total_out_elements: usize,
    config: garboard::LaunchConfig,
) -> Result<(), CudaError> {
    let ndim = inp_shape.len();

    // Fallback: use generic edge kernel with H2D transfers
    let inp_shape_vec: Vec<usize> = inp_shape.to_vec();
    let inp_shape_gpu = ctx.htod_usize(&inp_shape_vec)?;
    let out_shape_gpu = ctx.htod_usize(out_shape)?;
    let inp_strides_gpu = ctx.htod_usize(inp_strides)?;
    let out_strides_gpu = ctx.htod_usize(out_strides)?;
    let pads_begin_gpu = ctx.htod_usize(pads_begin)?;

    // SAFETY: pad_edge_kernel takes (float* out, const float* inp,
    //   const size_t* inp_shape, const size_t* out_shape,
    //   const size_t* inp_strides, const size_t* out_strides,
    //   const size_t* pads_begin, size_t ndim,
    //   size_t total_out_elements).
    let kernel = unsafe {
        cache.module().typed_kernel::<(
            &mut garboard::DeviceSlice<'_, f32>,
            &garboard::DeviceSlice<'_, f32>,
            &garboard::DeviceSlice<'_, u64>,
            &garboard::DeviceSlice<'_, u64>,
            &garboard::DeviceSlice<'_, u64>,
            &garboard::DeviceSlice<'_, u64>,
            &garboard::DeviceSlice<'_, u64>,
            usize,
            usize,
        )>("pad_edge_kernel")
    }
    .map_err(|e| CudaError::Kernel(format!("pad_edge_kernel lookup: {}", e)))?;

    let out_slice = op_ctx("Pad", output.data_f32_mut())?;
    let in_slice = op_ctx("Pad", input.data_f32())?;
    kernel
        .launch(
            ctx.garboard_stream(),
            &config,
            (
                out_slice,
                in_slice,
                &inp_shape_gpu,
                &out_shape_gpu,
                &inp_strides_gpu,
                &out_strides_gpu,
                &pads_begin_gpu,
                ndim,
                total_out_elements,
            ),
        )
        .map_err(|e| CudaError::Kernel(format!("pad_edge launch failed: {}", e)))?;

    Ok(())
}

/// GPU ScatterND: Scatter updates into output at given indices
#[allow(dead_code)] // Will be used when ScatterND op is integrated
pub fn gpu_scatter_nd(
    ctx: &IconnxCudaContext,
    cache: &OpsKernelCache,
    pool: &mut GpuMemoryPool,
    data: &GpuTensor,
    indices: &[i64],
    updates: &GpuTensor,
    indices_shape: &[usize],
) -> Result<GpuTensor, CudaError> {
    let data_shape = data.shape();
    let ndim = data_shape.len();
    let index_depth = indices_shape[indices_shape.len() - 1];

    // Calculate number of updates and slice size
    let num_updates: usize = indices_shape[..indices_shape.len() - 1].iter().product();
    let slice_size: usize = data_shape[index_depth..].iter().product();

    // Start with a copy of data
    let mut output = gpu_copy(ctx, cache, pool, data)?;

    // Upload indices and shape to GPU
    let indices_gpu = ctx.htod_i64(indices)?;
    let shape_vec: Vec<usize> = data_shape.to_vec();
    let shape_gpu = ctx.htod_usize(&shape_vec)?;

    // SAFETY: scatter_nd_kernel takes (float* out, const float* updates,
    //   const long long* indices, const size_t* shape,
    //   size_t num_updates, size_t slice_size, size_t index_depth, size_t ndim).
    let kernel = unsafe {
        cache.module().typed_kernel::<(
            &mut garboard::DeviceSlice<'_, f32>,
            &garboard::DeviceSlice<'_, f32>,
            &garboard::DeviceSlice<'_, i64>,
            &garboard::DeviceSlice<'_, u64>,
            usize,
            usize,
            usize,
            usize,
        )>("scatter_nd_kernel")
    }
    .map_err(|e| CudaError::Kernel(format!("scatter_nd_kernel lookup: {}", e)))?;

    let total = num_updates * slice_size;
    let config = garboard::LaunchConfig::for_num_elems(total as u32);

    let out_slice = op_ctx("ScatterND", output.data_f32_mut())?;
    let upd_slice = op_ctx("ScatterND", updates.data_f32())?;
    kernel
        .launch(
            ctx.garboard_stream(),
            &config,
            (
                out_slice,
                upd_slice,
                &indices_gpu,
                &shape_gpu,
                num_updates,
                slice_size,
                index_depth,
                ndim,
            ),
        )
        .map_err(|e| CudaError::Kernel(format!("scatter_nd launch failed: {}", e)))?;

    Ok(output)
}

/// Coordinate transformation mode for Resize
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResizeCoordMode {
    /// asymmetric: input_coord = output_coord / scale
    Asymmetric = 0,
    /// half_pixel: input_coord = (output_coord + 0.5) / scale - 0.5
    #[default]
    HalfPixel = 1,
}

/// Nearest mode for rounding in Resize
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResizeNearestMode {
    /// floor: always round down
    Floor = 0,
    /// round_prefer_floor: round to nearest, ties go to floor (ONNX default)
    #[default]
    RoundPreferFloor = 1,
}

/// Interpolation mode for Resize
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResizeMode {
    /// nearest: use nearest neighbor interpolation
    #[default]
    Nearest = 0,
    /// linear: use linear/bilinear/trilinear interpolation
    Linear = 1,
}

/// GPU Resize: Nearest-neighbor or linear interpolation
///
/// Resizes tensor using scale factors for each dimension.
/// Supports nearest-neighbor and linear (bilinear/trilinear) interpolation.
/// Uses optimized scalar kernels for 3D and 4D tensors (avoids H2D transfers).
///
/// # Arguments
/// * `coord_mode` - Coordinate transformation mode (asymmetric or half_pixel)
/// * `nearest_mode` - Rounding mode for integer conversion (floor or round_prefer_floor), only used for nearest mode
/// * `mode` - Interpolation mode (nearest or linear)
#[allow(clippy::too_many_arguments)] // Resize takes independent mode/coord/scale params
pub fn gpu_resize(
    ctx: &IconnxCudaContext,
    cache: &OpsKernelCache,
    pool: &mut GpuMemoryPool,
    input: &GpuTensor,
    scales: &[f32],
    coord_mode: ResizeCoordMode,
    nearest_mode: ResizeNearestMode,
    mode: ResizeMode,
) -> Result<GpuTensor, CudaError> {
    let inp_shape = input.shape();
    let ndim = inp_shape.len();

    if scales.len() != ndim {
        return Err(CudaError::Kernel(format!(
            "Resize: scales length {} doesn't match ndim {}",
            scales.len(),
            ndim
        )));
    }

    // Calculate output shape from scales
    let out_shape: Vec<usize> = inp_shape
        .iter()
        .zip(scales.iter())
        .map(|(&dim, &scale)| (dim as f32 * scale).round() as usize)
        .collect();

    let total_out_elements: usize = out_shape.iter().product();
    let mut output = pool.get_tensor_f32(ctx, out_shape.clone())?;

    // Calculate strides
    let out_strides = compute_strides(&out_shape);
    let inp_shape_vec: Vec<usize> = inp_shape.to_vec();
    let inp_strides = compute_strides(&inp_shape_vec);

    let config = garboard::LaunchConfig::for_num_elems(total_out_elements as u32);

    // Mode flags for kernel
    let coord_mode_int: i32 = coord_mode as i32;
    let nearest_mode_int: i32 = nearest_mode as i32;

    match mode {
        ResizeMode::Linear => {
            // Linear interpolation
            match ndim {
                3 => {
                    // SAFETY: resize_linear_3d_kernel takes (float* out, const float* inp,
                    //   size_t inp_d0, size_t inp_d1, size_t inp_d2,
                    //   size_t out_d0, size_t out_d1, size_t out_d2,
                    //   size_t inp_s0, size_t inp_s1, size_t inp_s2,
                    //   float scale0, float scale1, float scale2,
                    //   size_t total_out_elements, int coord_mode).
                    let kernel = unsafe {
                        cache.module().typed_kernel::<(
                            &mut garboard::DeviceSlice<'_, f32>,
                            &garboard::DeviceSlice<'_, f32>,
                            usize, usize, usize,
                            usize, usize, usize,
                            usize, usize, usize,
                            f32, f32, f32,
                            usize, i32,
                        )>("resize_linear_3d_kernel")
                    }
                    .map_err(|e| {
                        CudaError::Kernel(format!("resize_linear_3d_kernel lookup: {}", e))
                    })?;

                    kernel
                        .launch(
                            ctx.garboard_stream(),
                            &config,
                            (
                                op_ctx("Resize", output.data_f32_mut())?,
                                op_ctx("Resize", input.data_f32())?,
                                inp_shape[0], inp_shape[1], inp_shape[2],
                                out_shape[0], out_shape[1], out_shape[2],
                                inp_strides[0], inp_strides[1], inp_strides[2],
                                scales[0], scales[1], scales[2],
                                total_out_elements,
                                coord_mode_int,
                            ),
                        )
                        .map_err(|e| {
                            CudaError::Kernel(format!("resize_linear_3d launch failed: {}", e))
                        })?;
                }
                4 => {
                    // Pack [inp_d0..3, out_d0..3, inp_s0..3] = 12 params.
                    let params: Vec<usize> = [
                        &inp_shape[..4],
                        &out_shape[..4],
                        &inp_strides[..4],
                    ]
                    .concat();
                    let params_gpu = ctx.htod_usize(&params)?;

                    let kernel = unsafe {
                        cache.module().typed_kernel::<(
                            &mut garboard::DeviceSlice<'_, f32>,
                            &garboard::DeviceSlice<'_, f32>,
                            &garboard::DeviceSlice<'_, u64>,
                            f32, f32, f32, f32,
                            usize, i32,
                        )>("resize_linear_4d_kernel")
                    }
                    .map_err(|e| CudaError::Kernel(format!("resize_linear_4d lookup: {}", e)))?;

                    kernel
                        .launch(
                            ctx.garboard_stream(),
                            &config,
                            (
                                op_ctx("Resize", output.data_f32_mut())?,
                                op_ctx("Resize", input.data_f32())?,
                                &params_gpu,
                                scales[0], scales[1], scales[2], scales[3],
                                total_out_elements,
                                coord_mode_int,
                            ),
                        )
                        .map_err(|e| {
                            CudaError::Kernel(format!("resize_linear_4d launch failed: {}", e))
                        })?;
                }
                _ => {
                    return Err(CudaError::Kernel(format!(
                        "Linear interpolation not implemented for {}D tensors",
                        ndim
                    )));
                }
            }
        }
        ResizeMode::Nearest => {
            // Nearest-neighbor interpolation
            match ndim {
                3 => {
                    // Pack [inp_d0..2, out_d0..2, inp_s0..2, out_s0..2] = 12 params.
                    let params: Vec<usize> = [
                        &inp_shape[..3],
                        &out_shape[..3],
                        &inp_strides[..3],
                        &out_strides[..3],
                    ]
                    .concat();
                    let params_gpu = ctx.htod_usize(&params)?;

                    let kernel = unsafe {
                        cache.module().typed_kernel::<(
                            &mut garboard::DeviceSlice<'_, f32>,
                            &garboard::DeviceSlice<'_, f32>,
                            &garboard::DeviceSlice<'_, u64>,
                            f32, f32, f32,
                            usize, i32, i32,
                        )>("resize_3d_scalar_kernel")
                    }
                    .map_err(|e| CudaError::Kernel(format!("resize_3d_scalar lookup: {}", e)))?;

                    kernel
                        .launch(
                            ctx.garboard_stream(),
                            &config,
                            (
                                op_ctx("Resize", output.data_f32_mut())?,
                                op_ctx("Resize", input.data_f32())?,
                                &params_gpu,
                                scales[0], scales[1], scales[2],
                                total_out_elements,
                                coord_mode_int,
                                nearest_mode_int,
                            ),
                        )
                        .map_err(|e| {
                            CudaError::Kernel(format!("resize_3d launch failed: {}", e))
                        })?;
                }
                4 => {
                    // Pack [inp_d0..3, out_d0..3, inp_s0..3, out_s0..3] = 16 params.
                    let params: Vec<usize> = [
                        &inp_shape[..4],
                        &out_shape[..4],
                        &inp_strides[..4],
                        &out_strides[..4],
                    ]
                    .concat();
                    let params_gpu = ctx.htod_usize(&params)?;

                    let kernel = unsafe {
                        cache.module().typed_kernel::<(
                            &mut garboard::DeviceSlice<'_, f32>,
                            &garboard::DeviceSlice<'_, f32>,
                            &garboard::DeviceSlice<'_, u64>,
                            f32, f32, f32, f32,
                            usize, i32, i32,
                        )>("resize_4d_scalar_kernel")
                    }
                    .map_err(|e| CudaError::Kernel(format!("resize_4d_scalar lookup: {}", e)))?;

                    kernel
                        .launch(
                            ctx.garboard_stream(),
                            &config,
                            (
                                op_ctx("Resize", output.data_f32_mut())?,
                                op_ctx("Resize", input.data_f32())?,
                                &params_gpu,
                                scales[0], scales[1], scales[2], scales[3],
                                total_out_elements,
                                coord_mode_int,
                                nearest_mode_int,
                            ),
                        )
                        .map_err(|e| {
                            CudaError::Kernel(format!("resize_4d launch failed: {}", e))
                        })?;
                }
                _ => {
                    // Fall back to generic kernel with H2D transfers for other dimensions
                    let inp_shape_gpu = ctx.htod_usize(&inp_shape_vec)?;
                    let out_shape_gpu = ctx.htod_usize(&out_shape)?;
                    let inp_strides_gpu = ctx.htod_usize(&inp_strides)?;
                    let out_strides_gpu = ctx.htod_usize(&out_strides)?;
                    let scales_gpu = ctx.htod(scales)?;

                    // SAFETY: resize_nearest_kernel takes (float* out, const float* inp,
                    //   const size_t* inp_shape, const size_t* out_shape,
                    //   const size_t* inp_strides, const size_t* out_strides,
                    //   const float* scales, size_t ndim, size_t total_out_elements,
                    //   int coord_mode, int nearest_mode).
                    let kernel = unsafe {
                        cache.module().typed_kernel::<(
                            &mut garboard::DeviceSlice<'_, f32>,
                            &garboard::DeviceSlice<'_, f32>,
                            &garboard::DeviceSlice<'_, u64>,
                            &garboard::DeviceSlice<'_, u64>,
                            &garboard::DeviceSlice<'_, u64>,
                            &garboard::DeviceSlice<'_, u64>,
                            &garboard::DeviceSlice<'_, f32>,
                            usize, usize, i32, i32,
                        )>("resize_nearest_kernel")
                    }
                    .map_err(|e| CudaError::Kernel(format!("resize_nearest_kernel lookup: {}", e)))?;

                    kernel
                        .launch(
                            ctx.garboard_stream(),
                            &config,
                            (
                                op_ctx("Resize", output.data_f32_mut())?,
                                op_ctx("Resize", input.data_f32())?,
                                &inp_shape_gpu,
                                &out_shape_gpu,
                                &inp_strides_gpu,
                                &out_strides_gpu,
                                &scales_gpu,
                                ndim,
                                total_out_elements,
                                coord_mode_int,
                                nearest_mode_int,
                            ),
                        )
                        .map_err(|e| {
                            CudaError::Kernel(format!("resize launch failed: {}", e))
                        })?;
                }
            }
        }
    }

    Ok(output)
}

/// GPU NonZero: Find indices of non-zero elements
///
/// Returns a tensor of shape (ndim, num_nonzero) containing indices of non-zero elements.
/// This uses CPU fallback since output size is data-dependent.
pub fn gpu_nonzero(ctx: &IconnxCudaContext, input: &GpuTensor) -> Result<GpuTensor, CudaError> {
    // NonZero has dynamic output size - use CPU fallback
    // Copy input to CPU
    let data = input.to_host_f32(ctx)?;
    let shape = input.shape();
    let ndim = shape.len();

    // Calculate strides for multi-dim index computation
    let strides: Vec<usize> = (0..ndim)
        .map(|i| shape[i + 1..].iter().product::<usize>().max(1))
        .collect();

    // Find all non-zero element indices
    let mut all_indices: Vec<Vec<i64>> = vec![Vec::new(); ndim];

    for (flat_idx, &value) in data.iter().enumerate() {
        if value != 0.0 {
            let mut temp_idx = flat_idx;
            for dim in 0..ndim {
                all_indices[dim].push((temp_idx / strides[dim]) as i64);
                temp_idx %= strides[dim];
            }
        }
    }

    let num_nonzero = all_indices.first().map(|v| v.len()).unwrap_or(0);

    // Build output: shape (ndim, num_nonzero)
    // Output is stored row-major: [dim0_indices..., dim1_indices..., ...]
    let mut output_data = Vec::with_capacity(ndim * num_nonzero);
    for dim_indices in &all_indices {
        output_data.extend_from_slice(dim_indices);
    }

    // ONNX NonZero returns Int64
    if num_nonzero == 0 {
        GpuTensor::from_host_i64(ctx, &[], vec![ndim, 0])
    } else {
        GpuTensor::from_host_i64(ctx, &output_data, vec![ndim, num_nonzero])
    }
}

/// Compute row-major strides from shape
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// GPU Slice N-D: Full ONNX Slice implementation
///
/// ONNX Slice inputs:
/// - data: Input tensor
/// - starts: 1-D tensor of starting indices (negative wraps from end)
/// - ends: 1-D tensor of ending indices (negative wraps from end, exclusive)
/// - axes: Optional 1-D tensor specifying which axes to slice (default: 0..len(starts))
/// - steps: Optional 1-D tensor of step sizes (default: 1 for each axis)
///
/// This implementation:
/// - Supports up to 8 dimensions
/// - Handles negative indices (wrapping from end)
/// - Handles negative steps (reverse slicing)
/// - Clamps out-of-bounds indices
/// - Supports Float32, Int64, and Int32 tensors
#[allow(clippy::too_many_arguments)] // Slice takes independent starts/ends/axes/steps arrays
pub fn gpu_slice_nd(
    ctx: &IconnxCudaContext,
    cache: &OpsKernelCache,
    pool: &mut GpuMemoryPool,
    input: &GpuTensor,
    starts: &[i64],
    ends: &[i64],
    axes: Option<&[i64]>,
    steps: Option<&[i64]>,
) -> Result<GpuTensor, CudaError> {
    let input_shape = input.shape();
    let ndim = input_shape.len();

    // Default axes: [0, 1, ..., len(starts)-1]
    let default_axes: Vec<i64> = (0..starts.len() as i64).collect();
    let axes = axes.unwrap_or(&default_axes);

    // Default steps: all 1s
    let default_steps: Vec<i64> = vec![1; starts.len()];
    let steps = steps.unwrap_or(&default_steps);

    // Validate inputs
    if starts.len() != ends.len() || starts.len() != axes.len() || starts.len() != steps.len() {
        return Err(CudaError::Kernel(
            "Slice: starts, ends, axes, steps must have same length".into(),
        ));
    }

    // Compute effective starts/ends/steps for each dimension
    // For dimensions not in axes, start=0, end=dim_size, step=1
    let mut effective_starts = vec![0i64; ndim];
    let mut effective_ends: Vec<i64> = input_shape.iter().map(|&s| s as i64).collect();
    let mut effective_steps = vec![1i64; ndim];

    for i in 0..axes.len() {
        // Normalize axis (handle negative)
        let axis = if axes[i] < 0 {
            (ndim as i64 + axes[i]) as usize
        } else {
            axes[i] as usize
        };

        if axis >= ndim {
            return Err(CudaError::Kernel(format!(
                "Slice: axis {} out of bounds for ndim {}",
                axis, ndim
            )));
        }

        let dim_size = input_shape[axis] as i64;
        let step = if steps[i] == 0 {
            return Err(CudaError::Kernel("Slice: step cannot be 0".into()));
        } else {
            steps[i]
        };

        // Normalize start (handle negative)
        let mut start = if starts[i] < 0 {
            (dim_size + starts[i]).max(0)
        } else {
            starts[i].min(dim_size)
        };

        // Normalize end (handle negative)
        let mut end = if ends[i] < 0 {
            (dim_size + ends[i]).max(0)
        } else {
            ends[i].min(dim_size)
        };

        // For negative step, swap start/end logic
        if step < 0 {
            // When step is negative, we iterate from start down to end+1
            // Clamp: start must be <= dim_size-1, end must be >= -1
            start = start.min(dim_size - 1);
            end = end.max(-1);
        } else {
            // Clamp for positive step
            start = start.max(0);
            end = end.max(0);
        }

        effective_starts[axis] = start;
        effective_ends[axis] = end;
        effective_steps[axis] = step;
    }

    // Compute output shape
    let mut out_shape = Vec::with_capacity(ndim);
    for d in 0..ndim {
        let start = effective_starts[d];
        let end = effective_ends[d];
        let step = effective_steps[d];

        let size = if step > 0 {
            ((end - start + step - 1) / step).max(0) as usize
        } else {
            // Negative step: iterate from start down to end+1
            ((start - end - step - 1) / (-step)).max(0) as usize
        };
        out_shape.push(size);
    }

    let total_out_elements: usize = out_shape.iter().product();

    let dtype = input.dtype();

    if total_out_elements == 0 {
        return match dtype {
            crate::cuda::tensor::DType::Float32 => pool.get_tensor_f32(ctx, out_shape),
            crate::cuda::tensor::DType::Int64 => pool.get_tensor_i64(ctx, out_shape),
            crate::cuda::tensor::DType::Int32 => pool.get_tensor_i32(ctx, out_shape),
        };
    }

    // Compute strides
    let inp_strides = compute_strides(input_shape);
    let out_strides = compute_strides(&out_shape);

    let config = garboard::LaunchConfig::for_num_elems(total_out_elements as u32);

    // Use optimized scalar-argument kernels for 3D and 4D tensors (avoids H2D transfers)
    match (ndim, dtype) {
        // 3D Float32 - optimized path
        (3, crate::cuda::tensor::DType::Float32) => {
            let mut output = pool.get_tensor_f32(ctx, out_shape.clone())?;
            // SAFETY: slice_3d_scalar_kernel takes (float* out, const float* inp,
            //   size_t inp_stride0, size_t inp_stride1, size_t inp_stride2,
            //   size_t out_stride0, size_t out_stride1, size_t out_stride2,
            //   long long start0, long long start1, long long start2,
            //   long long step0, long long step1, long long step2,
            //   size_t total_out_elements).
            let kernel = unsafe {
                cache.module().typed_kernel::<(
                    &mut garboard::DeviceSlice<'_, f32>,
                    &garboard::DeviceSlice<'_, f32>,
                    u64, u64, u64,
                    u64, u64, u64,
                    i64, i64, i64,
                    i64, i64, i64,
                    usize,
                )>("slice_3d_scalar_kernel")
            }
            .map_err(|e| {
                CudaError::Kernel(format!("slice_3d_scalar_kernel lookup: {}", e))
            })?;

            kernel
                .launch(
                    ctx.garboard_stream(),
                    &config,
                    (
                        op_ctx("Slice", output.data_f32_mut())?,
                        op_ctx("Slice", input.data_f32())?,
                        inp_strides[0] as u64,
                        inp_strides[1] as u64,
                        inp_strides[2] as u64,
                        out_strides[0] as u64,
                        out_strides[1] as u64,
                        out_strides[2] as u64,
                        effective_starts[0],
                        effective_starts[1],
                        effective_starts[2],
                        effective_steps[0],
                        effective_steps[1],
                        effective_steps[2],
                        total_out_elements,
                    ),
                )
                .map_err(|e| {
                    CudaError::Kernel(format!("slice_3d_scalar launch failed: {}", e))
                })?;
            Ok(output)
        }
        // 3D Int64 - optimized path
        (3, crate::cuda::tensor::DType::Int64) => {
            let mut output = pool.get_tensor_i64(ctx, out_shape.clone())?;
            // SAFETY: slice_3d_i64_scalar_kernel mirrors slice_3d_scalar_kernel
            // with `long long*` (i64) data pointers in place of `float*`.
            let kernel = unsafe {
                cache.module().typed_kernel::<(
                    &mut garboard::DeviceSlice<'_, i64>,
                    &garboard::DeviceSlice<'_, i64>,
                    u64, u64, u64,
                    u64, u64, u64,
                    i64, i64, i64,
                    i64, i64, i64,
                    usize,
                )>("slice_3d_i64_scalar_kernel")
            }
            .map_err(|e| {
                CudaError::Kernel(format!("slice_3d_i64_scalar_kernel lookup: {}", e))
            })?;

            kernel
                .launch(
                    ctx.garboard_stream(),
                    &config,
                    (
                        output.data_i64_mut()?,
                        input.data_i64()?,
                        inp_strides[0] as u64,
                        inp_strides[1] as u64,
                        inp_strides[2] as u64,
                        out_strides[0] as u64,
                        out_strides[1] as u64,
                        out_strides[2] as u64,
                        effective_starts[0],
                        effective_starts[1],
                        effective_starts[2],
                        effective_steps[0],
                        effective_steps[1],
                        effective_steps[2],
                        total_out_elements,
                    ),
                )
                .map_err(|e| {
                    CudaError::Kernel(format!("slice_3d_i64_scalar launch failed: {}", e))
                })?;
            Ok(output)
        }
        // 4D Float32 - optimized path (packed params).
        (4, crate::cuda::tensor::DType::Float32) => {
            let mut output = pool.get_tensor_f32(ctx, out_shape.clone())?;

            // strides: [inp_s0..3, out_s0..3] = 8 values.
            let strides_packed: Vec<usize> = [&inp_strides[..4], &out_strides[..4]].concat();
            let strides_gpu = ctx.htod_usize(&strides_packed)?;
            // slice_params: [start0..3, step0..3] = 8 values.
            let slice_packed: Vec<i64> = [&effective_starts[..4], &effective_steps[..4]].concat();
            let slice_gpu = ctx.htod_i64(&slice_packed)?;

            let kernel = unsafe {
                cache.module().typed_kernel::<(
                    &mut garboard::DeviceSlice<'_, f32>,
                    &garboard::DeviceSlice<'_, f32>,
                    &garboard::DeviceSlice<'_, u64>,
                    &garboard::DeviceSlice<'_, i64>,
                    usize,
                )>("slice_4d_scalar_kernel")
            }
            .map_err(|e| CudaError::Kernel(format!("slice_4d_scalar lookup: {}", e)))?;

            kernel
                .launch(
                    ctx.garboard_stream(),
                    &config,
                    (
                        op_ctx("Slice", output.data_f32_mut())?,
                        op_ctx("Slice", input.data_f32())?,
                        &strides_gpu,
                        &slice_gpu,
                        total_out_elements,
                    ),
                )
                .map_err(|e| {
                    CudaError::Kernel(format!("slice_4d_scalar launch failed: {}", e))
                })?;
            Ok(output)
        }
        // 4D Int64 - optimized path (packed params).
        (4, crate::cuda::tensor::DType::Int64) => {
            let mut output = pool.get_tensor_i64(ctx, out_shape.clone())?;

            let strides_packed: Vec<usize> = [&inp_strides[..4], &out_strides[..4]].concat();
            let strides_gpu = ctx.htod_usize(&strides_packed)?;
            let slice_packed: Vec<i64> = [&effective_starts[..4], &effective_steps[..4]].concat();
            let slice_gpu = ctx.htod_i64(&slice_packed)?;

            let kernel = unsafe {
                cache.module().typed_kernel::<(
                    &mut garboard::DeviceSlice<'_, i64>,
                    &garboard::DeviceSlice<'_, i64>,
                    &garboard::DeviceSlice<'_, u64>,
                    &garboard::DeviceSlice<'_, i64>,
                    usize,
                )>("slice_4d_i64_scalar_kernel")
            }
            .map_err(|e| CudaError::Kernel(format!("slice_4d_i64_scalar lookup: {}", e)))?;

            kernel
                .launch(
                    ctx.garboard_stream(),
                    &config,
                    (
                        output.data_i64_mut()?,
                        input.data_i64()?,
                        &strides_gpu,
                        &slice_gpu,
                        total_out_elements,
                    ),
                )
                .map_err(|e| {
                    CudaError::Kernel(format!("slice_4d_i64_scalar launch failed: {}", e))
                })?;
            Ok(output)
        }
        // Fallback: use general N-D kernel with GPU arrays (for 1D, 2D, 5D+)
        (_, crate::cuda::tensor::DType::Float32) => {
            // Upload arrays to GPU (only for fallback path)
            let inp_strides_gpu = ctx.htod_usize(&inp_strides)?;
            let out_strides_gpu = ctx.htod_usize(&out_strides)?;
            let starts_gpu = ctx.htod_i64(&effective_starts)?;
            let steps_gpu = ctx.htod_i64(&effective_steps)?;

            let mut output = pool.get_tensor_f32(ctx, out_shape.clone())?;
            // SAFETY: slice_nd_kernel takes (float* out, const float* inp,
            //   const size_t* inp_strides, const size_t* out_strides,
            //   const long long* starts, const long long* steps,
            //   size_t ndim, size_t total_out_elements).
            let kernel = unsafe {
                cache.module().typed_kernel::<(
                    &mut garboard::DeviceSlice<'_, f32>,
                    &garboard::DeviceSlice<'_, f32>,
                    &garboard::DeviceSlice<'_, u64>,
                    &garboard::DeviceSlice<'_, u64>,
                    &garboard::DeviceSlice<'_, i64>,
                    &garboard::DeviceSlice<'_, i64>,
                    usize,
                    usize,
                )>("slice_nd_kernel")
            }
            .map_err(|e| CudaError::Kernel(format!("slice_nd_kernel lookup: {}", e)))?;

            kernel
                .launch(
                    ctx.garboard_stream(),
                    &config,
                    (
                        op_ctx("Slice", output.data_f32_mut())?,
                        op_ctx("Slice", input.data_f32())?,
                        &inp_strides_gpu,
                        &out_strides_gpu,
                        &starts_gpu,
                        &steps_gpu,
                        ndim,
                        total_out_elements,
                    ),
                )
                .map_err(|e| CudaError::Kernel(format!("slice_nd launch failed: {}", e)))?;
            Ok(output)
        }
        (_, crate::cuda::tensor::DType::Int64) => {
            // Upload arrays to GPU (only for fallback path)
            let inp_strides_gpu = ctx.htod_usize(&inp_strides)?;
            let out_strides_gpu = ctx.htod_usize(&out_strides)?;
            let starts_gpu = ctx.htod_i64(&effective_starts)?;
            let steps_gpu = ctx.htod_i64(&effective_steps)?;

            let mut output = pool.get_tensor_i64(ctx, out_shape.clone())?;
            // SAFETY: slice_nd_i64_kernel mirrors slice_nd_kernel with `long long*`
            // (i64) data pointers in place of `float*`.
            let kernel = unsafe {
                cache.module().typed_kernel::<(
                    &mut garboard::DeviceSlice<'_, i64>,
                    &garboard::DeviceSlice<'_, i64>,
                    &garboard::DeviceSlice<'_, u64>,
                    &garboard::DeviceSlice<'_, u64>,
                    &garboard::DeviceSlice<'_, i64>,
                    &garboard::DeviceSlice<'_, i64>,
                    usize,
                    usize,
                )>("slice_nd_i64_kernel")
            }
            .map_err(|e| CudaError::Kernel(format!("slice_nd_i64_kernel lookup: {}", e)))?;

            kernel
                .launch(
                    ctx.garboard_stream(),
                    &config,
                    (
                        output.data_i64_mut()?,
                        input.data_i64()?,
                        &inp_strides_gpu,
                        &out_strides_gpu,
                        &starts_gpu,
                        &steps_gpu,
                        ndim,
                        total_out_elements,
                    ),
                )
                .map_err(|e| CudaError::Kernel(format!("slice_nd_i64 launch failed: {}", e)))?;
            Ok(output)
        }
        (_, crate::cuda::tensor::DType::Int32) => {
            // Upload arrays to GPU (only for fallback path)
            let inp_strides_gpu = ctx.htod_usize(&inp_strides)?;
            let out_strides_gpu = ctx.htod_usize(&out_strides)?;
            let starts_gpu = ctx.htod_i64(&effective_starts)?;
            let steps_gpu = ctx.htod_i64(&effective_steps)?;

            let mut output = pool.get_tensor_i32(ctx, out_shape.clone())?;
            // SAFETY: slice_nd_i32_kernel mirrors slice_nd_kernel with `int*`
            // (i32) data pointers in place of `float*`.
            let kernel = unsafe {
                cache.module().typed_kernel::<(
                    &mut garboard::DeviceSlice<'_, i32>,
                    &garboard::DeviceSlice<'_, i32>,
                    &garboard::DeviceSlice<'_, u64>,
                    &garboard::DeviceSlice<'_, u64>,
                    &garboard::DeviceSlice<'_, i64>,
                    &garboard::DeviceSlice<'_, i64>,
                    usize,
                    usize,
                )>("slice_nd_i32_kernel")
            }
            .map_err(|e| CudaError::Kernel(format!("slice_nd_i32_kernel lookup: {}", e)))?;

            kernel
                .launch(
                    ctx.garboard_stream(),
                    &config,
                    (
                        output.data_i32_mut()?,
                        input.data_i32()?,
                        &inp_strides_gpu,
                        &out_strides_gpu,
                        &starts_gpu,
                        &steps_gpu,
                        ndim,
                        total_out_elements,
                    ),
                )
                .map_err(|e| CudaError::Kernel(format!("slice_nd_i32 launch failed: {}", e)))?;
            Ok(output)
        }
    }
}

#[cfg(test)]
mod tests;
