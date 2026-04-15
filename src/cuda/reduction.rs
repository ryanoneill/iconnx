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
];

/// Reduction CUDA kernel source code.
const REDUCTION_KERNELS: &str = r#"
// Define infinity constant for NVRTC (can't include math_constants.h)
#define CUDART_INF_F __int_as_float(0x7f800000)

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
        "reduce_mean_kernel",
        out.data_f32_mut()?,
        input.data_f32()?,
        outer_size,
        axis_size,
    )?;

    Ok(out)
}

/// GPU Softmax along last axis (numerically stable).
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
    .map_err(|e| CudaError::Kernel(format!("layer_norm_kernel lookup failed: {}", e)))?;

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
        .map_err(|e| CudaError::Kernel(format!("layer_norm_kernel launch failed: {}", e)))?;

    Ok(out)
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
}
