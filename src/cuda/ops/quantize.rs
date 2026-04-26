//! WS-4 quantization GPU ops.
//!
//! Today: `DequantizeLinear` (M4.4) and `DynamicQuantizeLinear` (M4.5).
//! `MatMulInteger` (M4.6) will land alongside these.

use super::cache::OpsKernelCache;
use crate::cuda::context::{CudaError, IconnxCudaContext};
use crate::cuda::memory_pool::GpuMemoryPool;
use crate::cuda::tensor::{DType, GpuTensor};
use garboard::{DeviceSlice, LaunchConfig};

/// GPU `DequantizeLinear`: `out = (x - zp) * scale`, output FP32.
///
/// `scale` must be FP32. `zp`, when supplied, must share dtype with `x`.
/// Per-tensor (scalar `scale`) is signalled by `axis_size = 1` and
/// `inner = n` so the per-element axis lookup collapses to index 0.
///
/// `zp = None` is materialized as a zero buffer of length `axis_size` in
/// `x`'s dtype before launch; the kernel never observes a null device
/// pointer. This trades one tiny allocation per call for a uniform kernel
/// signature across the present/absent cases — substantially simpler than
/// either threading optionality through the kernel ABI or maintaining
/// duplicate "no-zp" kernel variants.
pub fn gpu_dequantize_linear(
    ctx: &IconnxCudaContext,
    cache: &OpsKernelCache,
    pool: &mut GpuMemoryPool,
    x: &GpuTensor,
    scale: &GpuTensor,
    zp: Option<&GpuTensor>,
    axis: usize,
) -> Result<GpuTensor, CudaError> {
    let shape: Vec<usize> = x.shape().to_vec();
    let n = x.len();

    // Detect per-tensor (scalar scale or single-element 1-D) vs per-axis.
    let scale_shape = scale.shape();
    let scale_len = scale.len();
    let per_tensor = scale_shape.is_empty() || (scale_shape.len() == 1 && scale_len == 1);

    // Reduce per-element axis lookup to index 0 in the per-tensor case by
    // setting axis_size = 1 and inner = n. Otherwise compute the strides
    // from the input shape.
    let (axis_size, inner): (usize, usize) = if per_tensor {
        (1usize, n.max(1))
    } else {
        let ndim = shape.len();
        if axis >= ndim {
            return Err(CudaError::Kernel(format!(
                "DequantizeLinear: axis {} out of range for ndim {}",
                axis, ndim
            )));
        }
        let axis_size = shape[axis];
        if scale_len != axis_size {
            return Err(CudaError::Kernel(format!(
                "DequantizeLinear: per-axis scale length {} doesn't match \
                 input shape[{}] = {}",
                scale_len, axis, axis_size
            )));
        }
        let inner: usize = shape[axis + 1..].iter().product::<usize>().max(1);
        (axis_size, inner)
    };

    // `scale` must be FP32 by spec.
    if scale.dtype() != DType::Float32 {
        return Err(CudaError::Kernel(format!(
            "DequantizeLinear: scale must be Float32, got {}",
            scale.dtype().name()
        )));
    }

    // `zp` must share dtype and shape with `scale` when supplied. ONNX
    // DequantizeLinear requires zp.dtype == x.dtype and zp.shape ==
    // scale.shape; silently coercing one would let a broken graph slip
    // past iconnx that ORT would reject.
    if let Some(zp_t) = zp {
        if zp_t.dtype() != x.dtype() {
            return Err(CudaError::Kernel(format!(
                "DequantizeLinear: zero_point dtype {} must match input \
                 dtype {}",
                zp_t.dtype().name(),
                x.dtype().name()
            )));
        }
        if zp_t.shape() != scale.shape() {
            return Err(CudaError::Kernel(format!(
                "DequantizeLinear: zero_point shape {:?} must match scale \
                 shape {:?}",
                zp_t.shape(),
                scale.shape()
            )));
        }
    }

    let mut output = pool.get_tensor_f32(ctx, shape)?;
    let config = LaunchConfig::for_num_elems(n as u32);

    match x.dtype() {
        DType::Int8 => {
            // Synthesize a zeros zp on demand so the kernel signature is
            // uniform. Length = axis_size; per-tensor falls through with
            // axis_size = 1.
            let zp_owned: Option<GpuTensor> = match zp {
                Some(t) => {
                    // Broadcast a scalar zp to length `axis_size` if needed.
                    if !per_tensor && t.len() == 1 {
                        let host = t.to_host_i8(ctx)?;
                        let buf = vec![host[0]; axis_size];
                        Some(GpuTensor::from_host_i8(ctx, &buf, vec![axis_size])?)
                    } else {
                        None
                    }
                }
                None => Some(GpuTensor::zeros_i8(ctx, vec![axis_size])?),
            };
            // zp_owned is `Some` when we synthesized a zeros buffer (zp = None)
            // or broadcast a scalar zp to axis_size; `None` means the caller's
            // zp already has axis_size elements and is used directly.
            let zp_ref: &GpuTensor = zp_owned.as_ref().or(zp).expect("zp resolved");
            // SAFETY: signature matches `dequantize_linear_i8_kernel`.
            let kernel = unsafe {
                cache.module().typed_kernel::<(
                    &mut DeviceSlice<'_, f32>,
                    &DeviceSlice<'_, i8>,
                    &DeviceSlice<'_, f32>,
                    &DeviceSlice<'_, i8>,
                    usize,
                    usize,
                    usize,
                )>("dequantize_linear_i8_kernel")
            }
            .map_err(|e| {
                CudaError::Kernel(format!("dequantize_linear_i8_kernel lookup: {}", e))
            })?;
            kernel
                .launch(
                    ctx.garboard_stream(),
                    &config,
                    (
                        output.data_f32_mut()?,
                        x.data_i8()?,
                        scale.data_f32()?,
                        zp_ref.data_i8()?,
                        n,
                        axis_size,
                        inner,
                    ),
                )
                .map_err(|e| {
                    CudaError::Kernel(format!("dequantize_linear_i8_kernel launch: {}", e))
                })?;
            Ok(output)
        }
        DType::UInt8 => {
            let zp_owned: Option<GpuTensor> = match zp {
                Some(t) => {
                    if !per_tensor && t.len() == 1 {
                        let host = t.to_host_u8(ctx)?;
                        let buf = vec![host[0]; axis_size];
                        Some(GpuTensor::from_host_u8(ctx, &buf, vec![axis_size])?)
                    } else {
                        None
                    }
                }
                None => Some(GpuTensor::zeros_u8(ctx, vec![axis_size])?),
            };
            // zp_owned is `Some` when we synthesized a zeros buffer (zp = None)
            // or broadcast a scalar zp to axis_size; `None` means the caller's
            // zp already has axis_size elements and is used directly.
            let zp_ref: &GpuTensor = zp_owned.as_ref().or(zp).expect("zp resolved");
            // SAFETY: signature matches `dequantize_linear_u8_kernel`.
            let kernel = unsafe {
                cache.module().typed_kernel::<(
                    &mut DeviceSlice<'_, f32>,
                    &DeviceSlice<'_, u8>,
                    &DeviceSlice<'_, f32>,
                    &DeviceSlice<'_, u8>,
                    usize,
                    usize,
                    usize,
                )>("dequantize_linear_u8_kernel")
            }
            .map_err(|e| {
                CudaError::Kernel(format!("dequantize_linear_u8_kernel lookup: {}", e))
            })?;
            kernel
                .launch(
                    ctx.garboard_stream(),
                    &config,
                    (
                        output.data_f32_mut()?,
                        x.data_u8()?,
                        scale.data_f32()?,
                        zp_ref.data_u8()?,
                        n,
                        axis_size,
                        inner,
                    ),
                )
                .map_err(|e| {
                    CudaError::Kernel(format!("dequantize_linear_u8_kernel launch: {}", e))
                })?;
            Ok(output)
        }
        DType::Int32 => {
            let zp_owned: Option<GpuTensor> = match zp {
                Some(t) => {
                    if !per_tensor && t.len() == 1 {
                        let host = t.to_host_i32(ctx)?;
                        let buf = vec![host[0]; axis_size];
                        Some(GpuTensor::from_host_i32(ctx, &buf, vec![axis_size])?)
                    } else {
                        None
                    }
                }
                None => {
                    let zeros = ctx.alloc_zeros::<i32>(axis_size)?;
                    Some(GpuTensor::new_i32(zeros, vec![axis_size]))
                }
            };
            // zp_owned is `Some` when we synthesized a zeros buffer (zp = None)
            // or broadcast a scalar zp to axis_size; `None` means the caller's
            // zp already has axis_size elements and is used directly.
            let zp_ref: &GpuTensor = zp_owned.as_ref().or(zp).expect("zp resolved");
            // SAFETY: signature matches `dequantize_linear_i32_kernel`.
            let kernel = unsafe {
                cache.module().typed_kernel::<(
                    &mut DeviceSlice<'_, f32>,
                    &DeviceSlice<'_, i32>,
                    &DeviceSlice<'_, f32>,
                    &DeviceSlice<'_, i32>,
                    usize,
                    usize,
                    usize,
                )>("dequantize_linear_i32_kernel")
            }
            .map_err(|e| {
                CudaError::Kernel(format!("dequantize_linear_i32_kernel lookup: {}", e))
            })?;
            kernel
                .launch(
                    ctx.garboard_stream(),
                    &config,
                    (
                        output.data_f32_mut()?,
                        x.data_i32()?,
                        scale.data_f32()?,
                        zp_ref.data_i32()?,
                        n,
                        axis_size,
                        inner,
                    ),
                )
                .map_err(|e| {
                    CudaError::Kernel(format!("dequantize_linear_i32_kernel launch: {}", e))
                })?;
            Ok(output)
        }
        other => Err(CudaError::Kernel(format!(
            "DequantizeLinear: input dtype {} unsupported (must be Int8, \
             UInt8, or Int32)",
            other.name()
        ))),
    }
}

/// GPU `DynamicQuantizeLinear`: per-tensor dynamic quantization to UINT8.
///
/// Returns three tensors `[y, y_scale, y_zero_point]`:
///   * `y` — UINT8, same shape as `x`.
///   * `y_scale` — Float32 scalar (rank-0).
///   * `y_zero_point` — UInt8 scalar (rank-0).
///
/// Algorithm (matching ONNX reference and the CPU forward):
///   1. Scan `x` for `(min, max)`, with both initialized to 0 so the
///      computed `y_zero_point` can always represent zero.
///   2. `y_scale = (max - min) / 255` (in f64 for boundary parity with
///      ONNX reference's float64-promoted intermediates).
///   3. `intermediate_zp = round_ties_even(0 - min / y_scale)`.
///   4. `y_zero_point = clamp(intermediate_zp, 0, 255) as u8`.
///   5. `y[i] = saturate(round_ties_even(x[i] / y_scale) + y_zero_point)`.
///
/// Min/max reduction is **host-side** today: this MVP downloads the input
/// after a stream sync, computes the scalars on CPU, and quantizes back on
/// GPU via `dynamic_quantize_linear_quantize_kernel`. For DistilBERT-INT8's
/// ~768-element activation tensors this is well below the perf ceiling
/// where a two-pass GPU reduction would help; the kernel signature was
/// chosen so a future GPU reduction can be slotted in without changing the
/// quantize kernel ABI (it consumes `(scale, zp)` from device buffers).
///
/// Edge case: when `min == max`, ONNX semantics yield `y_scale = 0`,
/// `y_zero_point = 0`, and `y[i] = 0`. We return zero-filled buffers
/// directly without launching the quantize kernel.
pub fn gpu_dynamic_quantize_linear(
    ctx: &IconnxCudaContext,
    x: &GpuTensor,
) -> Result<Vec<GpuTensor>, CudaError> {
    if x.dtype() != DType::Float32 {
        return Err(CudaError::Kernel(format!(
            "DynamicQuantizeLinear: input must be Float32, got {}",
            x.dtype().name()
        )));
    }
    let shape = x.shape().to_vec();
    let n = x.len();

    // Host-side min/max scan. Promote to f64 for the scale/zp computation
    // to match the CPU forward, which uses f64 throughout for round-half-
    // to-even parity with the ONNX reference (numpy promotes Python's `0.0`
    // to f64, dragging the whole expression up). A pure-f32 path here
    // shifts boundary ties by ~1 ulp — e.g. 2 / (4/255)_f32 evaluates to
    // 127.49999 not 127.5, so the f32 banker's-rounding lands on 127
    // instead of 128. The kernel takes scale as f64 to preserve the
    // precision through the per-element quantize.
    let host_x = x.to_host_f32(ctx)?;
    let mut min_x = 0.0_f64;
    let mut max_x = 0.0_f64;
    for &v in host_x.iter() {
        if v.is_nan() {
            continue;
        }
        let vd = v as f64;
        if vd < min_x {
            min_x = vd;
        }
        if vd > max_x {
            max_x = vd;
        }
    }

    let qmin: f64 = 0.0;
    let qmax: f64 = 255.0;

    let (y_scale_f64, y_zero_point): (f64, u8) = if min_x == max_x {
        // ORT-convention sentinel: scale = 0, zp = 0, y = all zeros.
        (0.0, 0_u8)
    } else {
        let scale = (max_x - min_x) / (qmax - qmin);
        let zp_f = (qmin - min_x / scale).round_ties_even();
        let zp = zp_f.clamp(qmin, qmax) as u8;
        (scale, zp)
    };
    // Output scale is f32 per ONNX spec; internal precision stays f64 so
    // the kernel and CPU forward agree on tied boundaries.
    let y_scale_f32 = y_scale_f64 as f32;

    // Compute the quantization on host using the f64 scale, then upload
    // the result. iconnx's GpuTensor enum has no Float64 variant today,
    // so passing an f64 scale through an NVRTC kernel would bypass the
    // GpuTensor abstraction; doing the per-element quantize on the host
    // keeps CPU/GPU bit-identical at tied boundaries (per the WS-4 spec
    // round-half-to-even parity risk) and avoids that detour. For
    // DistilBERT-INT8's ~768-element activation tensors this is well
    // below the perf threshold where a true GPU quantize-pass would
    // matter; replace with a Float64-aware kernel ABI (or a two-pass
    // reduction + quantize) when activations get big enough to feel the
    // host round-trip.
    let y_data: Vec<u8> = if y_scale_f64 == 0.0 {
        vec![0_u8; n]
    } else {
        host_x
            .iter()
            .map(|&xi| {
                let q = ((xi as f64) / y_scale_f64).round_ties_even()
                    + y_zero_point as f64;
                q.clamp(qmin, qmax) as u8
            })
            .collect()
    };
    let y = GpuTensor::from_host_u8(ctx, &y_data, shape.clone())?;
    let scale_out = GpuTensor::from_host_f32(ctx, &[y_scale_f32], vec![])?;
    let zp_out = GpuTensor::from_host_u8(ctx, &[y_zero_point], vec![])?;

    Ok(vec![y, scale_out, zp_out])
}

/// GPU `MatMulInteger`: integer GEMM, output `Int32`.
///
/// 2-D inputs only (M4.6 MVP — batched MatMul is out of scope, since
/// DistilBERT-INT8 unrolls batches at the IR level). `a` is `[M, K]`,
/// `b` is `[K, N]`, output is `[M, N]`. Each operand can be `Int8` or
/// `UInt8`; the wrapper dispatches to one of the four NVRTC kernel
/// variants — `s8s8`, `u8s8`, `s8u8`, `u8u8`.
///
/// `a_zp` / `b_zp` are optional. When `None`, a 1-element zero buffer in
/// the operand's dtype is synthesized so the kernel signature stays
/// uniform (matches the M4.4 `DequantizeLinear` convention; nullable
/// device pointers are not part of the typed_kernel ABI). When
/// `Some(t)`, the wrapper validates dtype matches the operand and shape
/// is scalar (rank 0 or single-element rank-1). Per-axis zp is rejected
/// — the CPU forward enforces the same restriction.
///
/// Returns `Err(CudaError::Kernel(...))` for any invalid input —
/// non-2-D shapes, mismatched K, unsupported dtype combo, malformed zp.
/// No silent fallbacks.
pub fn gpu_matmul_integer(
    ctx: &IconnxCudaContext,
    cache: &OpsKernelCache,
    pool: &mut GpuMemoryPool,
    a: &GpuTensor,
    b: &GpuTensor,
    a_zp: Option<&GpuTensor>,
    b_zp: Option<&GpuTensor>,
) -> Result<GpuTensor, CudaError> {
    // Shape validation — 2-D only.
    if a.shape().len() != 2 || b.shape().len() != 2 {
        return Err(CudaError::Kernel(format!(
            "MatMulInteger: 2-D inputs required (got a:{:?}, b:{:?}); \
             batched is M4.7+ scope",
            a.shape(),
            b.shape()
        )));
    }
    let m = a.shape()[0];
    let k_dim = a.shape()[1];
    if b.shape()[0] != k_dim {
        return Err(CudaError::Kernel(format!(
            "MatMulInteger: K mismatch — a is [{},{}], b is [{},{}]",
            m,
            k_dim,
            b.shape()[0],
            b.shape()[1]
        )));
    }
    let n = b.shape()[1];
    let out_n = m * n;

    // Validate operand dtypes.
    if !matches!(a.dtype(), DType::Int8 | DType::UInt8) {
        return Err(CudaError::Kernel(format!(
            "MatMulInteger: a dtype {} unsupported (must be Int8 or UInt8)",
            a.dtype().name()
        )));
    }
    if !matches!(b.dtype(), DType::Int8 | DType::UInt8) {
        return Err(CudaError::Kernel(format!(
            "MatMulInteger: b dtype {} unsupported (must be Int8 or UInt8)",
            b.dtype().name()
        )));
    }

    // Materialize zps as 1-element device buffers in the operand dtype.
    // Mirror DequantizeLinear's "no nullable pointers in typed_kernel
    // signatures" stance: synthesize a 1-element zero buffer when zp is
    // absent. When zp is present, validate dtype/shape and reject
    // multi-element (per-axis) zp — DistilBERT does not exercise that
    // path, and CPU forward enforces the same scalar-only restriction.
    let a_zp_owned = resolve_scalar_zp(ctx, a_zp, a.dtype(), "a")?;
    let b_zp_owned = resolve_scalar_zp(ctx, b_zp, b.dtype(), "b")?;

    let mut output = pool.get_tensor_i32(ctx, vec![m, n])?;
    let config = LaunchConfig::for_num_elems(out_n as u32);

    match (a.dtype(), b.dtype()) {
        (DType::Int8, DType::Int8) => {
            // SAFETY: signature matches `matmul_integer_s8s8_kernel`.
            let kernel = unsafe {
                cache.module().typed_kernel::<(
                    &mut DeviceSlice<'_, i32>,
                    &DeviceSlice<'_, i8>,
                    &DeviceSlice<'_, i8>,
                    &DeviceSlice<'_, i8>,
                    &DeviceSlice<'_, i8>,
                    usize,
                    usize,
                    usize,
                )>("matmul_integer_s8s8_kernel")
            }
            .map_err(|e| {
                CudaError::Kernel(format!("matmul_integer_s8s8_kernel lookup: {}", e))
            })?;
            kernel
                .launch(
                    ctx.garboard_stream(),
                    &config,
                    (
                        output.data_i32_mut()?,
                        a.data_i8()?,
                        b.data_i8()?,
                        a_zp_owned.data_i8()?,
                        b_zp_owned.data_i8()?,
                        m,
                        k_dim,
                        n,
                    ),
                )
                .map_err(|e| {
                    CudaError::Kernel(format!("matmul_integer_s8s8_kernel launch: {}", e))
                })?;
        }
        (DType::UInt8, DType::Int8) => {
            // SAFETY: signature matches `matmul_integer_u8s8_kernel`.
            let kernel = unsafe {
                cache.module().typed_kernel::<(
                    &mut DeviceSlice<'_, i32>,
                    &DeviceSlice<'_, u8>,
                    &DeviceSlice<'_, i8>,
                    &DeviceSlice<'_, u8>,
                    &DeviceSlice<'_, i8>,
                    usize,
                    usize,
                    usize,
                )>("matmul_integer_u8s8_kernel")
            }
            .map_err(|e| {
                CudaError::Kernel(format!("matmul_integer_u8s8_kernel lookup: {}", e))
            })?;
            kernel
                .launch(
                    ctx.garboard_stream(),
                    &config,
                    (
                        output.data_i32_mut()?,
                        a.data_u8()?,
                        b.data_i8()?,
                        a_zp_owned.data_u8()?,
                        b_zp_owned.data_i8()?,
                        m,
                        k_dim,
                        n,
                    ),
                )
                .map_err(|e| {
                    CudaError::Kernel(format!("matmul_integer_u8s8_kernel launch: {}", e))
                })?;
        }
        (DType::Int8, DType::UInt8) => {
            // SAFETY: signature matches `matmul_integer_s8u8_kernel`.
            let kernel = unsafe {
                cache.module().typed_kernel::<(
                    &mut DeviceSlice<'_, i32>,
                    &DeviceSlice<'_, i8>,
                    &DeviceSlice<'_, u8>,
                    &DeviceSlice<'_, i8>,
                    &DeviceSlice<'_, u8>,
                    usize,
                    usize,
                    usize,
                )>("matmul_integer_s8u8_kernel")
            }
            .map_err(|e| {
                CudaError::Kernel(format!("matmul_integer_s8u8_kernel lookup: {}", e))
            })?;
            kernel
                .launch(
                    ctx.garboard_stream(),
                    &config,
                    (
                        output.data_i32_mut()?,
                        a.data_i8()?,
                        b.data_u8()?,
                        a_zp_owned.data_i8()?,
                        b_zp_owned.data_u8()?,
                        m,
                        k_dim,
                        n,
                    ),
                )
                .map_err(|e| {
                    CudaError::Kernel(format!("matmul_integer_s8u8_kernel launch: {}", e))
                })?;
        }
        (DType::UInt8, DType::UInt8) => {
            // SAFETY: signature matches `matmul_integer_u8u8_kernel`.
            let kernel = unsafe {
                cache.module().typed_kernel::<(
                    &mut DeviceSlice<'_, i32>,
                    &DeviceSlice<'_, u8>,
                    &DeviceSlice<'_, u8>,
                    &DeviceSlice<'_, u8>,
                    &DeviceSlice<'_, u8>,
                    usize,
                    usize,
                    usize,
                )>("matmul_integer_u8u8_kernel")
            }
            .map_err(|e| {
                CudaError::Kernel(format!("matmul_integer_u8u8_kernel lookup: {}", e))
            })?;
            kernel
                .launch(
                    ctx.garboard_stream(),
                    &config,
                    (
                        output.data_i32_mut()?,
                        a.data_u8()?,
                        b.data_u8()?,
                        a_zp_owned.data_u8()?,
                        b_zp_owned.data_u8()?,
                        m,
                        k_dim,
                        n,
                    ),
                )
                .map_err(|e| {
                    CudaError::Kernel(format!("matmul_integer_u8u8_kernel launch: {}", e))
                })?;
        }
        // The dtype validation at the top of the function rejects every
        // non-Int8/UInt8 pairing before we get here, so this branch is
        // unreachable.
        (a_dt, b_dt) => {
            return Err(CudaError::Kernel(format!(
                "MatMulInteger: unsupported dtype combo a={}, b={} (must be \
                 Int8 or UInt8)",
                a_dt.name(),
                b_dt.name()
            )));
        }
    }

    Ok(output)
}

/// Resolve an optional scalar zp to a 1-element device buffer in the
/// operand's dtype.
///
/// When `zp` is `None`, allocates a 1-element zeros buffer. When
/// `Some(t)`, validates dtype matches `operand_dtype` and shape is
/// scalar (rank 0 or single-element rank-1) — per-axis zp is rejected
/// to keep the M4.6 GPU kernel ABI simple. Returns the zp buffer ready
/// for direct kernel consumption.
fn resolve_scalar_zp(
    ctx: &IconnxCudaContext,
    zp: Option<&GpuTensor>,
    operand_dtype: DType,
    label: &str,
) -> Result<GpuTensor, CudaError> {
    match zp {
        None => match operand_dtype {
            DType::Int8 => GpuTensor::zeros_i8(ctx, vec![1]),
            DType::UInt8 => GpuTensor::zeros_u8(ctx, vec![1]),
            other => Err(CudaError::Kernel(format!(
                "MatMulInteger: cannot synthesize zp for operand dtype {}",
                other.name()
            ))),
        },
        Some(t) => {
            if t.dtype() != operand_dtype {
                return Err(CudaError::Kernel(format!(
                    "MatMulInteger: {}_zero_point dtype {} must match \
                     operand dtype {}",
                    label,
                    t.dtype().name(),
                    operand_dtype.name()
                )));
            }
            let shape = t.shape();
            let is_scalar = shape.is_empty() || (shape.len() == 1 && shape[0] == 1);
            if !is_scalar {
                return Err(CudaError::Kernel(format!(
                    "MatMulInteger: {}_zero_point must be scalar (got shape \
                     {:?}); per-axis zp is not supported in M4.6",
                    label, shape
                )));
            }
            // Caller's zp is already a 1-element buffer in the right
            // dtype; rematerialize to a fresh contiguous 1-element buffer
            // so the kernel always sees a length-1 device slice. Cheaper
            // than threading shape-aware indexing through the kernel ABI.
            match operand_dtype {
                DType::Int8 => {
                    let host = t.to_host_i8(ctx)?;
                    GpuTensor::from_host_i8(ctx, &host[..1], vec![1])
                }
                DType::UInt8 => {
                    let host = t.to_host_u8(ctx)?;
                    GpuTensor::from_host_u8(ctx, &host[..1], vec![1])
                }
                other => Err(CudaError::Kernel(format!(
                    "MatMulInteger: cannot rematerialize zp for operand dtype {}",
                    other.name()
                ))),
            }
        }
    }
}
