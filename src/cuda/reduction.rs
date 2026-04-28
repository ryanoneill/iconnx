//! GPU Reduction Kernels via garboard NVRTC + TypedKernel.
//!
//! Provides GPU-accelerated reduction operations (sum, mean, softmax,
//! layernorm). These operations reduce data along specified axes.

use garboard::{DeviceSlice, LaunchConfig, Module, Program};

use super::context::{CudaError, IconnxCudaContext};
use super::memory_pool::GpuMemoryPool;
use super::tensor::GpuTensor;

/// Compiled reduction kernels loaded as a garboard `Module`.
pub struct ReductionKernelCache {
    module: Module<'static>,
}

impl ReductionKernelCache {
    /// Compile all reduction kernels and create the cache.
    pub fn new(ctx: &IconnxCudaContext) -> Result<Self, CudaError> {
        let program =
            Program::compile_for_device(REDUCTION_KERNELS, ctx.garboard_device(), &[])
                .map_err(|e| CudaError::Kernel(format!("NVRTC compilation failed: {}", e)))?;

        let module = ctx
            .garboard_device()
            .load_module(&program)
            .map_err(|e| CudaError::Kernel(format!("Module load failed: {}", e)))?;

        for name in KERNEL_NAMES {
            module.function(name).map_err(|e| {
                CudaError::Kernel(format!("Failed to load kernel '{}': {}", name, e))
            })?;
        }

        Ok(Self { module })
    }

    pub(crate) fn module(&self) -> &Module<'static> {
        &self.module
    }
}

const KERNEL_NAMES: &[&str] = &[
    "reduce_sum_kernel",
    "reduce_sum_axis_kernel",
    "reduce_mean_kernel",
    "reduce_max_kernel",
    "softmax_kernel",
    "layer_norm_kernel",
    // WS-3 M3.4 sub-3: FP16 IO with f32 accumulator. Pure-FP16
    // accumulation would overflow on Whisper attention's K=1500
    // reduction; the f32 accumulator pattern matches gemm_fp16_acc32.
    "reduce_mean_f16_kernel",
    "softmax_f16_kernel",
    // WS-3.5 Y(2) sub-2c: BF16 IO with f32 accumulator. Same f32
    // accumulator strategy as the FP16 path — BF16's wider exponent
    // (8-bit, FP32-range) means accumulation overflow is less likely
    // than FP16, but the coarser 7-bit mantissa makes pure-BF16
    // accumulation lose precision rapidly. f32 accumulator is the
    // correctness-stable choice. LayerNorm is NEW for BF16 (no FP16
    // LayerNorm kernel exists today — see commit body for context).
    "reduce_mean_bf16_kernel",
    "softmax_bf16_kernel",
    "layer_norm_bf16_kernel",
];

/// Reduction CUDA kernel source code.
///
/// `CUDART_INF_F` comes from `math_constants.h`, auto-included by
/// garboard's `compile_for_device`. `<cuda_fp16.h>` is needed for the
/// WS-3 M3.4 sub-3 FP16 reductions; `<cuda_bf16.h>` is needed for the
/// WS-3.5 Y(2) sub-2c BF16 reductions. Both are supplied by NVRTC's
/// default include path.
const REDUCTION_KERNELS: &str = r#"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Block-level reduction using shared memory
// Each block reduces its portion, then atomically adds to output

// Simple full reduction (sum all elements to a single value)
extern "C" __global__ void reduce_sum_kernel(
    float* out,
    const float* inp,
    size_t n
) {
    extern __shared__ float sdata[];

    size_t tid = threadIdx.x;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load into shared memory
    sdata[tid] = (i < n) ? inp[i] : 0.0f;
    __syncthreads();

    // Reduction in shared memory
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0) {
        atomicAdd(out, sdata[0]);
    }
}

// Reduce along last axis: input[..., axis_size] -> output[...]
// outer_size = product of all dims before axis
// axis_size = size of axis being reduced
extern "C" __global__ void reduce_sum_axis_kernel(
    float* out,
    const float* inp,
    size_t outer_size,
    size_t axis_size
) {
    size_t outer_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (outer_idx < outer_size) {
        float sum = 0.0f;
        const float* row = inp + outer_idx * axis_size;
        for (size_t i = 0; i < axis_size; i++) {
            sum += row[i];
        }
        out[outer_idx] = sum;
    }
}

// Reduce mean along last axis
extern "C" __global__ void reduce_mean_kernel(
    float* out,
    const float* inp,
    size_t outer_size,
    size_t axis_size
) {
    size_t outer_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (outer_idx < outer_size) {
        float sum = 0.0f;
        const float* row = inp + outer_idx * axis_size;
        for (size_t i = 0; i < axis_size; i++) {
            sum += row[i];
        }
        out[outer_idx] = sum / (float)axis_size;
    }
}

// Reduce max along last axis (for softmax)
extern "C" __global__ void reduce_max_kernel(
    float* out,
    const float* inp,
    size_t outer_size,
    size_t axis_size
) {
    size_t outer_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (outer_idx < outer_size) {
        float max_val = -CUDART_INF_F;
        const float* row = inp + outer_idx * axis_size;
        for (size_t i = 0; i < axis_size; i++) {
            if (row[i] > max_val) max_val = row[i];
        }
        out[outer_idx] = max_val;
    }
}

// Numerically stable softmax along last axis
// softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x)))
extern "C" __global__ void softmax_kernel(
    float* out,
    const float* inp,
    size_t outer_size,
    size_t axis_size
) {
    size_t outer_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (outer_idx < outer_size) {
        const float* in_row = inp + outer_idx * axis_size;
        float* out_row = out + outer_idx * axis_size;

        // Find max for numerical stability
        float max_val = -CUDART_INF_F;
        for (size_t i = 0; i < axis_size; i++) {
            if (in_row[i] > max_val) max_val = in_row[i];
        }

        // Compute exp(x - max) and sum
        float sum = 0.0f;
        for (size_t i = 0; i < axis_size; i++) {
            float e = expf(in_row[i] - max_val);
            out_row[i] = e;
            sum += e;
        }

        // Normalize
        for (size_t i = 0; i < axis_size; i++) {
            out_row[i] /= sum;
        }
    }
}

// Layer normalization: y = (x - mean) / sqrt(var + eps) * gamma + beta
// Operates on last axis
extern "C" __global__ void layer_norm_kernel(
    float* out,
    const float* inp,
    const float* gamma,  // scale, size = axis_size
    const float* beta,   // bias, size = axis_size
    size_t outer_size,
    size_t axis_size,
    float eps
) {
    size_t outer_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (outer_idx < outer_size) {
        const float* in_row = inp + outer_idx * axis_size;
        float* out_row = out + outer_idx * axis_size;

        // Compute mean
        float mean = 0.0f;
        for (size_t i = 0; i < axis_size; i++) {
            mean += in_row[i];
        }
        mean /= (float)axis_size;

        // Compute variance
        float var = 0.0f;
        for (size_t i = 0; i < axis_size; i++) {
            float diff = in_row[i] - mean;
            var += diff * diff;
        }
        var /= (float)axis_size;

        // Normalize and scale
        float inv_std = rsqrtf(var + eps);
        for (size_t i = 0; i < axis_size; i++) {
            float normalized = (in_row[i] - mean) * inv_std;
            out_row[i] = normalized * gamma[i] + beta[i];
        }
    }
}

// =============================================================================
// WS-3 M3.4 sub-3 FP16 reductions: FP16 IO with f32 accumulator.
//
// The reduction loop accumulates in float to avoid f16 overflow on long
// axes (Whisper attention reduces over K=1500 keys; the worst-case
// activation magnitude * sequence length easily exceeds f16's 65504
// max). Inputs are loaded via `__half2float`, intermediates stay in
// float, only the final per-element write rounds back via
// `__float2half`. This mirrors the gemm_fp16_acc32 precision strategy.
// =============================================================================

extern "C" __global__ void reduce_mean_f16_kernel(
    __half* out,
    const __half* inp,
    size_t outer_size,
    size_t axis_size
) {
    size_t outer_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (outer_idx < outer_size) {
        float sum = 0.0f;
        const __half* row = inp + outer_idx * axis_size;
        for (size_t i = 0; i < axis_size; i++) {
            sum += __half2float(row[i]);
        }
        out[outer_idx] = __float2half(sum / (float)axis_size);
    }
}

// Numerically stable softmax along last axis, FP16 IO + f32 accumulator.
//   softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x)))
// The exp + sum pass and the normalize pass both use `out_row` as a
// scratch buffer, so we keep an inner f32 store and only convert back
// at the final divide — minimizes f16 round-trip ULP.
extern "C" __global__ void softmax_f16_kernel(
    __half* out,
    const __half* inp,
    size_t outer_size,
    size_t axis_size
) {
    size_t outer_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (outer_idx < outer_size) {
        const __half* in_row = inp + outer_idx * axis_size;
        __half* out_row = out + outer_idx * axis_size;

        // Find max (in float — comparing in float is faster than __hgt
        // and avoids any sNaN propagation issues with Whisper's logits).
        float max_val = -CUDART_INF_F;
        for (size_t i = 0; i < axis_size; i++) {
            float v = __half2float(in_row[i]);
            if (v > max_val) max_val = v;
        }

        // Exp(x - max) — stage the f32 result into out_row (round-trip
        // through f16 once so the value survives the second pass; this
        // costs ULP but avoids a second device-memory scratch).
        float sum = 0.0f;
        for (size_t i = 0; i < axis_size; i++) {
            float e = expf(__half2float(in_row[i]) - max_val);
            out_row[i] = __float2half(e);
            sum += e;
        }

        // Normalize. Read the staged f16, divide in f32, round once.
        for (size_t i = 0; i < axis_size; i++) {
            float e = __half2float(out_row[i]);
            out_row[i] = __float2half(e / sum);
        }
    }
}

// =============================================================================
// WS-3.5 Y(2) sub-2c BF16 reductions: BF16 IO with f32 accumulator.
//
// Same f32-accumulator pattern as the FP16 reductions above. BF16's
// 8-bit exponent (FP32-range) makes overflow less likely than FP16, but
// its 7-bit mantissa loses precision faster — pure-BF16 accumulation
// would alias adjacent activations on long axes. f32 accumulator is the
// correctness-stable choice, mirroring gemm_bf16_acc32.
//
// LayerNorm scale/bias are typed as `const float*` (f32) — no FP16
// LayerNorm kernel exists today to mirror, and the principled choice
// for a wider-exponent dtype's affine params is f32 (matches the f32
// LayerNorm kernel above; ONNX's LayerNormalization spec doesn't
// require scale/bias dtype = data dtype).
// =============================================================================

extern "C" __global__ void reduce_mean_bf16_kernel(
    __nv_bfloat16* out,
    const __nv_bfloat16* inp,
    size_t outer_size,
    size_t axis_size
) {
    size_t outer_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (outer_idx < outer_size) {
        float sum = 0.0f;
        const __nv_bfloat16* row = inp + outer_idx * axis_size;
        for (size_t i = 0; i < axis_size; i++) {
            sum += __bfloat162float(row[i]);
        }
        out[outer_idx] = __float2bfloat16_rn(sum / (float)axis_size);
    }
}

// Numerically stable softmax along last axis, BF16 IO + f32 accumulator.
//   softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x)))
// Stages exp(x - max) into out_row in BF16 (one round-trip ULP) so the
// normalize pass can read it back without a second device-memory scratch.
extern "C" __global__ void softmax_bf16_kernel(
    __nv_bfloat16* out,
    const __nv_bfloat16* inp,
    size_t outer_size,
    size_t axis_size
) {
    size_t outer_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (outer_idx < outer_size) {
        const __nv_bfloat16* in_row = inp + outer_idx * axis_size;
        __nv_bfloat16* out_row = out + outer_idx * axis_size;

        // Find max in float (faster than __hgt and avoids any sNaN
        // propagation issues with attention logits).
        float max_val = -CUDART_INF_F;
        for (size_t i = 0; i < axis_size; i++) {
            float v = __bfloat162float(in_row[i]);
            if (v > max_val) max_val = v;
        }

        // Exp(x - max) — stage f32 result into out_row (one round-trip
        // through bf16 so the value survives the second pass).
        float sum = 0.0f;
        for (size_t i = 0; i < axis_size; i++) {
            float e = expf(__bfloat162float(in_row[i]) - max_val);
            out_row[i] = __float2bfloat16_rn(e);
            sum += e;
        }

        // Normalize. Read the staged bf16, divide in f32, round once.
        for (size_t i = 0; i < axis_size; i++) {
            float e = __bfloat162float(out_row[i]);
            out_row[i] = __float2bfloat16_rn(e / sum);
        }
    }
}

// LayerNorm BF16: y = (x - mean) / sqrt(var + eps) * gamma + beta
// Inputs/outputs BF16; mean/var/scale/bias all f32 internally.
extern "C" __global__ void layer_norm_bf16_kernel(
    __nv_bfloat16* out,
    const __nv_bfloat16* inp,
    const float* gamma,  // scale, size = axis_size (f32 — see comment block above)
    const float* beta,   // bias, size = axis_size (f32 — see comment block above)
    size_t outer_size,
    size_t axis_size,
    float eps
) {
    size_t outer_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (outer_idx < outer_size) {
        const __nv_bfloat16* in_row = inp + outer_idx * axis_size;
        __nv_bfloat16* out_row = out + outer_idx * axis_size;

        // Mean in f32 — accumulate via __bfloat162float lifts.
        float mean = 0.0f;
        for (size_t i = 0; i < axis_size; i++) {
            mean += __bfloat162float(in_row[i]);
        }
        mean /= (float)axis_size;

        // Variance in f32.
        float var = 0.0f;
        for (size_t i = 0; i < axis_size; i++) {
            float d = __bfloat162float(in_row[i]) - mean;
            var += d * d;
        }
        var /= (float)axis_size;
        float inv_std = rsqrtf(var + eps);

        // Normalize, scale, bias — all f32 then round once on store.
        for (size_t i = 0; i < axis_size; i++) {
            float normalized = (__bfloat162float(in_row[i]) - mean) * inv_std;
            float scaled = normalized * gamma[i] + beta[i];
            out_row[i] = __float2bfloat16_rn(scaled);
        }
    }
}
"#;

// =============================================================================
// Per-signature launch helpers
// =============================================================================

/// Unary f32 reduction along an axis: `(out, in, outer_size, axis_size)`.
/// Covers reduce_sum_axis, reduce_mean, reduce_max, softmax.
fn launch_axis_reduction(
    ctx: &IconnxCudaContext,
    kernels: &ReductionKernelCache,
    name: &'static str,
    out: &mut DeviceSlice<'static, f32>,
    inp: &DeviceSlice<'static, f32>,
    outer_size: usize,
    axis_size: usize,
) -> Result<(), CudaError> {
    // SAFETY: kernel signature is
    // `(float* out, const float* inp, size_t outer_size, size_t axis_size)`.
    let kernel = unsafe {
        kernels.module().typed_kernel::<(
            &mut DeviceSlice<'_, f32>,
            &DeviceSlice<'_, f32>,
            usize,
            usize,
        )>(name)
    }
    .map_err(|e| CudaError::Kernel(format!("{} lookup failed: {}", name, e)))?;

    kernel
        .launch(
            ctx.garboard_stream(),
            &LaunchConfig::for_num_elems(outer_size as u32),
            (out, inp, outer_size, axis_size),
        )
        .map_err(|e| CudaError::Kernel(format!("{} launch failed: {}", name, e)))
}

/// Unary BF16 axis reduction: `(out, in, outer_size, axis_size)`.
/// Covers reduce_mean_bf16 and softmax_bf16. The C kernel parameter
/// type is `__nv_bfloat16*`, ABI-compatible with `half::bf16`.
fn launch_axis_reduction_bf16(
    ctx: &IconnxCudaContext,
    kernels: &ReductionKernelCache,
    name: &'static str,
    out: &mut DeviceSlice<'static, half::bf16>,
    inp: &DeviceSlice<'static, half::bf16>,
    outer_size: usize,
    axis_size: usize,
) -> Result<(), CudaError> {
    // SAFETY: kernel signature is
    // `(__nv_bfloat16* out, const __nv_bfloat16* inp, size_t outer_size, size_t axis_size)`.
    let kernel = unsafe {
        kernels.module().typed_kernel::<(
            &mut DeviceSlice<'_, half::bf16>,
            &DeviceSlice<'_, half::bf16>,
            usize,
            usize,
        )>(name)
    }
    .map_err(|e| CudaError::Kernel(format!("{} lookup failed: {}", name, e)))?;

    kernel
        .launch(
            ctx.garboard_stream(),
            &LaunchConfig::for_num_elems(outer_size as u32),
            (out, inp, outer_size, axis_size),
        )
        .map_err(|e| CudaError::Kernel(format!("{} launch failed: {}", name, e)))
}

/// Unary FP16 axis reduction: `(out, in, outer_size, axis_size)`.
/// Covers reduce_mean_f16 and softmax_f16. The C kernel parameter type
/// is `__half*`, ABI-compatible with `half::f16`.
fn launch_axis_reduction_f16(
    ctx: &IconnxCudaContext,
    kernels: &ReductionKernelCache,
    name: &'static str,
    out: &mut DeviceSlice<'static, half::f16>,
    inp: &DeviceSlice<'static, half::f16>,
    outer_size: usize,
    axis_size: usize,
) -> Result<(), CudaError> {
    // SAFETY: kernel signature is
    // `(__half* out, const __half* inp, size_t outer_size, size_t axis_size)`.
    let kernel = unsafe {
        kernels.module().typed_kernel::<(
            &mut DeviceSlice<'_, half::f16>,
            &DeviceSlice<'_, half::f16>,
            usize,
            usize,
        )>(name)
    }
    .map_err(|e| CudaError::Kernel(format!("{} lookup failed: {}", name, e)))?;

    kernel
        .launch(
            ctx.garboard_stream(),
            &LaunchConfig::for_num_elems(outer_size as u32),
            (out, inp, outer_size, axis_size),
        )
        .map_err(|e| CudaError::Kernel(format!("{} launch failed: {}", name, e)))
}

// =============================================================================
// Reduction Operations
// =============================================================================

/// GPU ReduceSum: Sum all elements to a single value.
pub fn gpu_reduce_sum_all(
    ctx: &IconnxCudaContext,
    kernels: &ReductionKernelCache,
    pool: &mut GpuMemoryPool,
    input: &GpuTensor,
) -> Result<f32, CudaError> {
    let n = input.len();
    let mut out = pool.get_tensor_f32(ctx, vec![1])?;

    let block_size = 256u32;
    let grid_size = (n as u32).div_ceil(block_size);
    let shared_mem = block_size * std::mem::size_of::<f32>() as u32;

    let config = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: shared_mem,
    };

    // SAFETY: reduce_sum_kernel signature is
    // `(float* out, const float* inp, size_t n)`.
    let kernel = unsafe {
        kernels.module().typed_kernel::<(
            &mut DeviceSlice<'_, f32>,
            &DeviceSlice<'_, f32>,
            usize,
        )>("reduce_sum_kernel")
    }
    .map_err(|e| CudaError::Kernel(format!("reduce_sum_kernel lookup failed: {}", e)))?;

    kernel
        .launch(
            ctx.garboard_stream(),
            &config,
            (out.data_f32_mut()?, input.data_f32()?, n),
        )
        .map_err(|e| CudaError::Kernel(format!("reduce_sum_kernel launch failed: {}", e)))?;

    let result = out.to_host_f32(ctx)?;
    Ok(result[0])
}

/// GPU ReduceSum along last axis: input[..., N] -> output[...].
pub fn gpu_reduce_sum_axis(
    ctx: &IconnxCudaContext,
    kernels: &ReductionKernelCache,
    pool: &mut GpuMemoryPool,
    input: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    let shape = input.shape();
    if shape.is_empty() {
        return Err(CudaError::Kernel("Cannot reduce empty tensor".into()));
    }

    let axis_size = *shape.last().unwrap();
    let outer_size: usize = shape[..shape.len() - 1].iter().product();

    if outer_size == 0 {
        return pool.get_tensor_f32(ctx, vec![1]);
    }

    let out_shape: Vec<usize> = if shape.len() > 1 {
        shape[..shape.len() - 1].to_vec()
    } else {
        vec![1]
    };
    let mut out = pool.get_tensor_f32(ctx, out_shape)?;

    launch_axis_reduction(
        ctx,
        kernels,
        "reduce_sum_axis_kernel",
        out.data_f32_mut()?,
        input.data_f32()?,
        outer_size,
        axis_size,
    )?;

    Ok(out)
}

/// GPU ReduceMean along last axis: input[..., N] -> output[...].
///
/// Supports Float32 and Float16. The Float16 path uses a kernel with
/// FP16 IO and an f32 accumulator (WS-3 M3.4 sub-3) so long axes don't
/// overflow.
pub fn gpu_reduce_mean_axis(
    ctx: &IconnxCudaContext,
    kernels: &ReductionKernelCache,
    pool: &mut GpuMemoryPool,
    input: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    let shape = input.shape();
    if shape.is_empty() {
        return Err(CudaError::Kernel("Cannot reduce empty tensor".into()));
    }

    let axis_size = *shape.last().unwrap();
    let outer_size: usize = shape[..shape.len() - 1].iter().product();

    let out_shape: Vec<usize> = if shape.len() > 1 {
        shape[..shape.len() - 1].to_vec()
    } else {
        vec![1]
    };

    match input.dtype() {
        super::tensor::DType::Float32 => {
            if outer_size == 0 {
                return pool.get_tensor_f32(ctx, vec![1]);
            }
            let mut out = pool.get_tensor_f32(ctx, out_shape)?;
            launch_axis_reduction(
                ctx,
                kernels,
                "reduce_mean_kernel",
                out.data_f32_mut()?,
                input.data_f32()?,
                outer_size,
                axis_size,
            )?;
            Ok(out)
        }
        super::tensor::DType::Float16 => {
            if outer_size == 0 {
                return pool.get_tensor_f16(ctx, vec![1]);
            }
            let mut out = pool.get_tensor_f16(ctx, out_shape)?;
            launch_axis_reduction_f16(
                ctx,
                kernels,
                "reduce_mean_f16_kernel",
                out.data_f16_mut()?,
                input.data_f16()?,
                outer_size,
                axis_size,
            )?;
            Ok(out)
        }
        super::tensor::DType::BFloat16 => {
            if outer_size == 0 {
                return pool.get_tensor_bf16(ctx, vec![1]);
            }
            let mut out = pool.get_tensor_bf16(ctx, out_shape)?;
            launch_axis_reduction_bf16(
                ctx,
                kernels,
                "reduce_mean_bf16_kernel",
                out.data_bf16_mut()?,
                input.data_bf16()?,
                outer_size,
                axis_size,
            )?;
            Ok(out)
        }
        dt => Err(CudaError::Kernel(format!(
            "ReduceMean: unsupported dtype {}",
            dt.name()
        ))),
    }
}

/// GPU Softmax along last axis (numerically stable).
///
/// Supports Float32 and Float16. The Float16 path uses a kernel with
/// FP16 IO and an f32 accumulator (WS-3 M3.4 sub-3); Whisper attention
/// reduces over K=1500 keys, where pure-FP16 accumulation would
/// overflow.
pub fn gpu_softmax(
    ctx: &IconnxCudaContext,
    kernels: &ReductionKernelCache,
    pool: &mut GpuMemoryPool,
    input: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    let shape = input.shape();
    if shape.is_empty() {
        return Err(CudaError::Kernel("Cannot softmax empty tensor".into()));
    }

    let axis_size = *shape.last().unwrap();
    let outer_size: usize = shape[..shape.len() - 1].iter().product();
    let outer_size = if outer_size == 0 { 1 } else { outer_size };

    match input.dtype() {
        super::tensor::DType::Float32 => {
            let mut out = pool.get_tensor_f32(ctx, shape.to_vec())?;
            launch_axis_reduction(
                ctx,
                kernels,
                "softmax_kernel",
                out.data_f32_mut()?,
                input.data_f32()?,
                outer_size,
                axis_size,
            )?;
            Ok(out)
        }
        super::tensor::DType::Float16 => {
            let mut out = pool.get_tensor_f16(ctx, shape.to_vec())?;
            launch_axis_reduction_f16(
                ctx,
                kernels,
                "softmax_f16_kernel",
                out.data_f16_mut()?,
                input.data_f16()?,
                outer_size,
                axis_size,
            )?;
            Ok(out)
        }
        super::tensor::DType::BFloat16 => {
            let mut out = pool.get_tensor_bf16(ctx, shape.to_vec())?;
            launch_axis_reduction_bf16(
                ctx,
                kernels,
                "softmax_bf16_kernel",
                out.data_bf16_mut()?,
                input.data_bf16()?,
                outer_size,
                axis_size,
            )?;
            Ok(out)
        }
        dt => Err(CudaError::Kernel(format!(
            "Softmax: unsupported dtype {}",
            dt.name()
        ))),
    }
}

/// GPU LayerNormalization along last axis.
/// `y = (x - mean) / sqrt(var + eps) * gamma + beta`.
pub fn gpu_layer_norm(
    ctx: &IconnxCudaContext,
    kernels: &ReductionKernelCache,
    pool: &mut GpuMemoryPool,
    input: &GpuTensor,
    gamma: &GpuTensor, // scale weights, shape = [axis_size]
    beta: &GpuTensor,  // bias weights, shape = [axis_size]
    eps: f32,
) -> Result<GpuTensor, CudaError> {
    let shape = input.shape();
    if shape.is_empty() {
        return Err(CudaError::Kernel("Cannot layer_norm empty tensor".into()));
    }

    let axis_size = *shape.last().unwrap();
    let outer_size: usize = shape[..shape.len() - 1].iter().product();
    let outer_size = if outer_size == 0 { 1 } else { outer_size };

    assert_eq!(gamma.len(), axis_size, "gamma must match axis size");
    assert_eq!(beta.len(), axis_size, "beta must match axis size");

    match input.dtype() {
        super::tensor::DType::Float32 => {
            let mut out = pool.get_tensor_f32(ctx, shape.to_vec())?;

            // SAFETY: layer_norm_kernel signature is
            // `(float* out, const float* inp, const float* gamma, const float* beta,
            //   size_t outer_size, size_t axis_size, float eps)`.
            let kernel = unsafe {
                kernels.module().typed_kernel::<(
                    &mut DeviceSlice<'_, f32>,
                    &DeviceSlice<'_, f32>,
                    &DeviceSlice<'_, f32>,
                    &DeviceSlice<'_, f32>,
                    usize,
                    usize,
                    f32,
                )>("layer_norm_kernel")
            }
            .map_err(|e| {
                CudaError::Kernel(format!("layer_norm_kernel lookup failed: {}", e))
            })?;

            kernel
                .launch(
                    ctx.garboard_stream(),
                    &LaunchConfig::for_num_elems(outer_size as u32),
                    (
                        out.data_f32_mut()?,
                        input.data_f32()?,
                        gamma.data_f32()?,
                        beta.data_f32()?,
                        outer_size,
                        axis_size,
                        eps,
                    ),
                )
                .map_err(|e| {
                    CudaError::Kernel(format!("layer_norm_kernel launch failed: {}", e))
                })?;

            Ok(out)
        }
        super::tensor::DType::BFloat16 => {
            // WS-3.5 Y(2) sub-2c: BF16 IO with f32 accumulator. scale +
            // bias remain f32 — see the kernel-source comment block for
            // why (no FP16 LayerNorm to mirror; principled choice for a
            // wider-exponent dtype's affine params is f32).
            assert_eq!(
                gamma.dtype(),
                super::tensor::DType::Float32,
                "BF16 LayerNorm: gamma must be f32 (affine params stay in f32)"
            );
            assert_eq!(
                beta.dtype(),
                super::tensor::DType::Float32,
                "BF16 LayerNorm: beta must be f32 (affine params stay in f32)"
            );
            let mut out = pool.get_tensor_bf16(ctx, shape.to_vec())?;

            // SAFETY: layer_norm_bf16_kernel signature is
            // `(__nv_bfloat16* out, const __nv_bfloat16* inp,
            //   const float* gamma, const float* beta,
            //   size_t outer_size, size_t axis_size, float eps)`.
            let kernel = unsafe {
                kernels.module().typed_kernel::<(
                    &mut DeviceSlice<'_, half::bf16>,
                    &DeviceSlice<'_, half::bf16>,
                    &DeviceSlice<'_, f32>,
                    &DeviceSlice<'_, f32>,
                    usize,
                    usize,
                    f32,
                )>("layer_norm_bf16_kernel")
            }
            .map_err(|e| {
                CudaError::Kernel(format!("layer_norm_bf16_kernel lookup failed: {}", e))
            })?;

            kernel
                .launch(
                    ctx.garboard_stream(),
                    &LaunchConfig::for_num_elems(outer_size as u32),
                    (
                        out.data_bf16_mut()?,
                        input.data_bf16()?,
                        gamma.data_f32()?,
                        beta.data_f32()?,
                        outer_size,
                        axis_size,
                        eps,
                    ),
                )
                .map_err(|e| {
                    CudaError::Kernel(format!("layer_norm_bf16_kernel launch failed: {}", e))
                })?;

            Ok(out)
        }
        dt => Err(CudaError::Kernel(format!(
            "LayerNorm: unsupported dtype {}",
            dt.name()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::memory_pool::GpuMemoryPool;

    fn setup() -> (IconnxCudaContext, ReductionKernelCache, GpuMemoryPool) {
        let ctx = IconnxCudaContext::new().expect("Failed to create CUDA context");
        let kernels = ReductionKernelCache::new(&ctx).expect("Failed to compile kernels");
        let pool = GpuMemoryPool::new();
        (ctx, kernels, pool)
    }

    #[test]
    #[ignore] // Requires CUDA GPU
    fn test_kernel_compilation() {
        let (ctx, _kernels, _pool) = setup();
        ctx.sync().unwrap();
        println!("All reduction kernels compiled successfully!");
    }

    #[test]
    #[ignore] // Requires CUDA GPU
    fn test_reduce_sum_all() {
        let (ctx, kernels, mut pool) = setup();

        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let input = GpuTensor::from_host_f32(&ctx, &data, vec![5]).unwrap();

        let result = gpu_reduce_sum_all(&ctx, &kernels, &mut pool, &input).unwrap();
        assert!(
            (result - 15.0).abs() < 1e-5,
            "Expected 15.0, got {}",
            result
        );
    }

    #[test]
    #[ignore] // Requires CUDA GPU
    fn test_reduce_sum_axis() {
        let (ctx, kernels, mut pool) = setup();

        // 2x3 matrix, reduce along last axis (columns)
        // [[1, 2, 3], [4, 5, 6]] -> [6, 15]
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = GpuTensor::from_host_f32(&ctx, &data, vec![2, 3]).unwrap();

        let output = gpu_reduce_sum_axis(&ctx, &kernels, &mut pool, &input).unwrap();
        let result = output.to_host_f32(&ctx).unwrap();

        assert_eq!(output.shape(), &[2]);
        assert!((result[0] - 6.0).abs() < 1e-5);
        assert!((result[1] - 15.0).abs() < 1e-5);
    }

    #[test]
    #[ignore] // Requires CUDA GPU
    fn test_reduce_mean_axis() {
        let (ctx, kernels, mut pool) = setup();

        // 2x3 matrix: mean along last axis
        // [[1, 2, 3], [4, 5, 6]] -> [2, 5]
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = GpuTensor::from_host_f32(&ctx, &data, vec![2, 3]).unwrap();

        let output = gpu_reduce_mean_axis(&ctx, &kernels, &mut pool, &input).unwrap();
        let result = output.to_host_f32(&ctx).unwrap();

        assert_eq!(output.shape(), &[2]);
        assert!((result[0] - 2.0).abs() < 1e-5);
        assert!((result[1] - 5.0).abs() < 1e-5);
    }

    #[test]
    #[ignore] // Requires CUDA GPU
    fn test_softmax_1d() {
        let (ctx, kernels, mut pool) = setup();

        // 1D softmax: [1, 2, 3]
        let data = vec![1.0f32, 2.0, 3.0];
        let input = GpuTensor::from_host_f32(&ctx, &data, vec![3]).unwrap();

        let output = gpu_softmax(&ctx, &kernels, &mut pool, &input).unwrap();
        let result = output.to_host_f32(&ctx).unwrap();

        // Expected: [e^1/Z, e^2/Z, e^3/Z] where Z = e^1 + e^2 + e^3
        let z: f32 = 1.0_f32.exp() + 2.0_f32.exp() + 3.0_f32.exp();
        assert!((result[0] - (1.0_f32.exp() / z)).abs() < 1e-5);
        assert!((result[1] - (2.0_f32.exp() / z)).abs() < 1e-5);
        assert!((result[2] - (3.0_f32.exp() / z)).abs() < 1e-5);

        // Sum must equal 1.0
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    #[ignore] // Requires CUDA GPU
    fn test_softmax_2d() {
        let (ctx, kernels, mut pool) = setup();

        // 2x3 matrix: softmax along last axis
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = GpuTensor::from_host_f32(&ctx, &data, vec![2, 3]).unwrap();

        let output = gpu_softmax(&ctx, &kernels, &mut pool, &input).unwrap();
        let result = output.to_host_f32(&ctx).unwrap();

        // Each row should sum to 1.0
        let sum_row_0: f32 = result[0..3].iter().sum();
        let sum_row_1: f32 = result[3..6].iter().sum();
        assert!((sum_row_0 - 1.0).abs() < 1e-5);
        assert!((sum_row_1 - 1.0).abs() < 1e-5);
    }

    #[test]
    #[ignore] // Requires CUDA GPU
    fn test_softmax_numerical_stability() {
        let (ctx, kernels, mut pool) = setup();

        // Large values that would overflow without numerical stability
        let data = vec![1000.0f32, 1001.0, 1002.0];
        let input = GpuTensor::from_host_f32(&ctx, &data, vec![3]).unwrap();

        let output = gpu_softmax(&ctx, &kernels, &mut pool, &input).unwrap();
        let result = output.to_host_f32(&ctx).unwrap();

        // Must produce valid probabilities summing to 1
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(result.iter().all(|&x| (0.0..=1.0).contains(&x)));
        // Values should be monotonically increasing (last largest)
        assert!(result[0] < result[1]);
        assert!(result[1] < result[2]);
    }

    #[test]
    #[ignore] // Requires CUDA GPU
    fn test_large_reduction() {
        let (ctx, kernels, mut pool) = setup();

        // Test with a larger tensor. Summing 0..n in f32 with the naive
        // left-fold (what Vec::sum does) bleeds precision as the
        // accumulator grows beyond ~2^24. Use the true closed-form sum
        // n*(n-1)/2 via f64 as the oracle so the test is comparing to
        // the mathematical truth, not an already-imprecise CPU sum.
        let n = 10000;
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let input = GpuTensor::from_host_f32(&ctx, &data, vec![n]).unwrap();

        let result = gpu_reduce_sum_all(&ctx, &kernels, &mut pool, &input).unwrap();
        let expected: f64 = (n as f64) * (n as f64 - 1.0) / 2.0;

        // Tolerance 1e-5 relative is comfortable; GPU atomicAdd order
        // yields a result within a handful of ULPs of the true sum.
        let rel_err = ((result as f64 - expected) / expected).abs();
        assert!(
            rel_err < 1e-5,
            "Expected {}, got {} (rel err {:.2e})",
            expected,
            result,
            rel_err,
        );
    }

    #[test]
    #[ignore] // Requires CUDA GPU
    fn test_layer_norm() {
        let (ctx, kernels, mut pool) = setup();

        // Input: [2, 4] - 2 rows of 4 features each
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let input = GpuTensor::from_host_f32(&ctx, &data, vec![2, 4]).unwrap();

        // Gamma = 1.0, Beta = 0.0 -> just standard normalization
        let gamma = GpuTensor::from_host_f32(&ctx, &[1.0; 4], vec![4]).unwrap();
        let beta = GpuTensor::from_host_f32(&ctx, &[0.0; 4], vec![4]).unwrap();

        let eps = 1e-5;
        let output = gpu_layer_norm(&ctx, &kernels, &mut pool, &input, &gamma, &beta, eps).unwrap();
        let result = output.to_host_f32(&ctx).unwrap();

        assert_eq!(output.shape(), &[2, 4]);

        // For row 0: [1,2,3,4], mean=2.5, var=1.25, std≈1.118
        // Normalized: ((x-2.5) / 1.118)
        // Expected: [-1.342, -0.447, 0.447, 1.342]
        assert!((result[0] - (-1.342)).abs() < 0.01);
        assert!((result[3] - 1.342).abs() < 0.01);

        // Each row should have mean ≈ 0 and variance ≈ 1
        let mean_row_0: f32 = result[0..4].iter().sum::<f32>() / 4.0;
        let mean_row_1: f32 = result[4..8].iter().sum::<f32>() / 4.0;
        assert!(mean_row_0.abs() < 1e-5);
        assert!(mean_row_1.abs() < 1e-5);
    }

    #[test]
    #[ignore] // Requires CUDA GPU
    fn test_layer_norm_with_affine() {
        let (ctx, kernels, mut pool) = setup();

        // Input: [1, 4] - 1 row of 4 features
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let input = GpuTensor::from_host_f32(&ctx, &data, vec![1, 4]).unwrap();

        // Gamma = [2, 2, 2, 2], Beta = [1, 1, 1, 1]
        let gamma = GpuTensor::from_host_f32(&ctx, &[2.0; 4], vec![4]).unwrap();
        let beta = GpuTensor::from_host_f32(&ctx, &[1.0; 4], vec![4]).unwrap();

        let eps = 1e-5;
        let output = gpu_layer_norm(&ctx, &kernels, &mut pool, &input, &gamma, &beta, eps).unwrap();
        let result = output.to_host_f32(&ctx).unwrap();

        // Mean of [1,2,3,4] = 2.5, std ≈ 1.118
        // Normalized: [-1.342, -0.447, 0.447, 1.342]
        // With gamma=2: [-2.684, -0.894, 0.894, 2.684]
        // With beta=1: [-1.684, 0.106, 1.894, 3.684]
        assert!((result[0] - (-1.684)).abs() < 0.01);
        assert!((result[3] - 3.684).abs() < 0.01);
    }

    // =========================================================================
    // WS-3 M3.4 sub-3: FP16 reduction contract tests.
    //
    // Tolerance is set generously vs. f32 because (a) FP16 inputs already lose
    // ~3 decimal digits, (b) the kernel rounds intermediates back through
    // FP16 storage between passes for the softmax case. The accumulator
    // stays in f32 so long axes don't overflow.
    // =========================================================================

    fn h(values: &[f32]) -> Vec<half::f16> {
        values.iter().copied().map(half::f16::from_f32).collect()
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_reduce_mean_axis_float16() {
        let (ctx, kernels, mut pool) = setup();

        // [[1, 2, 3], [4, 5, 6]] -> mean along last axis -> [2.0, 5.0].
        let input = GpuTensor::from_host_f16(&ctx, &h(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), vec![2, 3])
            .unwrap();

        let output = gpu_reduce_mean_axis(&ctx, &kernels, &mut pool, &input).unwrap();
        assert_eq!(output.dtype(), super::super::tensor::DType::Float16);
        let result = output.to_host_f16(&ctx).unwrap();
        assert_eq!(result.len(), 2);
        assert!((result[0].to_f32() - 2.0).abs() < 1e-3);
        assert!((result[1].to_f32() - 5.0).abs() < 1e-3);
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_reduce_mean_axis_float16_long_axis_no_overflow() {
        // 1500-wide axis matches Whisper attention's K dimension. Pure-FP16
        // accumulation would overflow once values * length exceed 65504;
        // f32 accumulator keeps the sum in float until the final divide,
        // so the mean stays in f16 range.
        let (ctx, kernels, mut pool) = setup();

        let axis = 1500;
        let mut data = Vec::with_capacity(axis);
        for _ in 0..axis {
            data.push(half::f16::from_f32(50.0));
        }
        // Sum would be 75000 in f32 — well past f16 max (65504). Mean is 50.
        let input = GpuTensor::from_host_f16(&ctx, &data, vec![1, axis]).unwrap();

        let output = gpu_reduce_mean_axis(&ctx, &kernels, &mut pool, &input).unwrap();
        let result = output.to_host_f16(&ctx).unwrap();
        assert_eq!(result.len(), 1);
        assert!(
            (result[0].to_f32() - 50.0).abs() < 0.5,
            "got {} (expected ~50.0); pure-FP16 accumulator would have overflowed",
            result[0].to_f32()
        );
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_softmax_float16() {
        let (ctx, kernels, mut pool) = setup();

        // 2 rows, 3 cols. Softmax along last axis. Expected per-row:
        //   row 0 = softmax([1,2,3])
        //   row 1 = softmax([1,1,1]) = [1/3, 1/3, 1/3].
        let input = GpuTensor::from_host_f16(&ctx, &h(&[1.0, 2.0, 3.0, 1.0, 1.0, 1.0]), vec![2, 3])
            .unwrap();

        let output = gpu_softmax(&ctx, &kernels, &mut pool, &input).unwrap();
        assert_eq!(output.dtype(), super::super::tensor::DType::Float16);
        let result = output.to_host_f16(&ctx).unwrap();
        assert_eq!(result.len(), 6);

        // Row 0 hand-computed:
        //   exp([-2,-1,0]) = [0.1353, 0.3679, 1.0], sum ≈ 1.503,
        //   softmax ≈ [0.0900, 0.2447, 0.6652].
        let row0_expected = [0.0900_f32, 0.2447, 0.6652];
        for (i, &exp) in row0_expected.iter().enumerate() {
            let got = result[i].to_f32();
            assert!(
                (got - exp).abs() < 5e-3,
                "row0[{}]: got {} vs {}",
                i,
                got,
                exp
            );
        }

        // Row 1 — uniform input -> uniform softmax.
        for (i, slot) in result.iter().enumerate().skip(3) {
            let got = slot.to_f32();
            assert!(
                (got - 1.0 / 3.0).abs() < 5e-3,
                "row1[{}]: got {}, expected 1/3",
                i - 3,
                got
            );
            // Each row should sum to 1.
        }
        let row1_sum: f32 = result[3..].iter().map(|h| h.to_f32()).sum();
        assert!((row1_sum - 1.0).abs() < 1e-2, "row1 sum = {}", row1_sum);
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_softmax_float16_numerical_stability() {
        // Large input magnitudes exercise the max-subtraction step.
        // softmax([100, 100, 100]) = [1/3, 1/3, 1/3]. Without subtracting
        // the max, exp(100) overflows.
        let (ctx, kernels, mut pool) = setup();

        let input =
            GpuTensor::from_host_f16(&ctx, &h(&[100.0, 100.0, 100.0]), vec![1, 3]).unwrap();
        let output = gpu_softmax(&ctx, &kernels, &mut pool, &input).unwrap();
        let result = output.to_host_f16(&ctx).unwrap();

        for slot in &result {
            let got = slot.to_f32();
            assert!(
                (got - 1.0 / 3.0).abs() < 5e-3,
                "got {} (expected 1/3)",
                got
            );
        }
    }
}
