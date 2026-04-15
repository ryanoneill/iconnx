use crate::cuda::bridge::GbKernelArg;
use super::context::{CudaError, IconnxCudaContext};
use super::memory_pool::GpuMemoryPool;
use super::tensor::GpuTensor;
/// GPU Reduction Kernels using NVRTC
///
/// Provides GPU-accelerated reduction operations (sum, mean, softmax, layernorm).
/// These operations reduce data along specified axes.
use cudarc::driver::{CudaFunction, CudaModule, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use std::collections::HashMap;
use std::sync::Arc;

/// Manages compiled reduction kernels with caching
pub struct ReductionKernelCache {
    #[allow(dead_code)]
    module: Arc<CudaModule>,
    functions: HashMap<&'static str, CudaFunction>,
}

impl ReductionKernelCache {
    /// Compile all reduction kernels and create the cache
    pub fn new(ctx: &IconnxCudaContext) -> Result<Self, CudaError> {
        let ptx = compile_ptx(REDUCTION_KERNELS)
            .map_err(|e| CudaError::Kernel(format!("NVRTC compilation failed: {:?}", e)))?;

        let module = ctx
            .context()
            .load_module(ptx)
            .map_err(|e| CudaError::Kernel(format!("Module load failed: {}", e)))?;

        let mut functions = HashMap::new();

        for name in KERNEL_NAMES {
            let func = module.load_function(name).map_err(|e| {
                CudaError::Kernel(format!("Failed to load kernel '{}': {}", name, e))
            })?;
            functions.insert(*name, func);
        }

        Ok(Self { module, functions })
    }

    pub fn get(&self, name: &'static str) -> Option<&CudaFunction> {
        self.functions.get(name)
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

/// Reduction CUDA kernel source code
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
// Reduction Operations
// =============================================================================

/// GPU ReduceSum: Sum all elements to a single value
pub fn gpu_reduce_sum_all(
    ctx: &IconnxCudaContext,
    kernels: &ReductionKernelCache,
    pool: &mut GpuMemoryPool,
    input: &GpuTensor,
) -> Result<f32, CudaError> {
    let n = input.len();
    let mut out = pool.get_tensor_f32(ctx, vec![1])?;

    let func = kernels
        .get("reduce_sum_kernel")
        .ok_or_else(|| CudaError::Kernel("reduce_sum_kernel not found".into()))?;

    let block_size = 256u32;
    let grid_size = (n as u32).div_ceil(block_size);
    let shared_mem = block_size * std::mem::size_of::<f32>() as u32;

    let config = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: shared_mem,
    };

    unsafe {
        ctx.stream()
            .launch_builder(func)
            .arg(&GbKernelArg::new_mut(out.data_f32_mut()?))
            .arg(&GbKernelArg::new(input.data_f32()?))
            .arg(&n)
            .launch(config)
            .map_err(|e| CudaError::Kernel(e.to_string()))?;
    }

    let result = out.to_host_f32(ctx)?;
    Ok(result[0])
}

/// GPU ReduceSum along last axis: input[..., N] -> output[...]
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

    // Handle scalar case
    if outer_size == 0 {
        return pool.get_tensor_f32(ctx, vec![1]);
    }

    let out_shape: Vec<usize> = if shape.len() > 1 {
        shape[..shape.len() - 1].to_vec()
    } else {
        vec![1]
    };
    let mut out = pool.get_tensor_f32(ctx, out_shape)?;

    let func = kernels
        .get("reduce_sum_axis_kernel")
        .ok_or_else(|| CudaError::Kernel("reduce_sum_axis_kernel not found".into()))?;

    let config = LaunchConfig::for_num_elems(outer_size as u32);

    unsafe {
        ctx.stream()
            .launch_builder(func)
            .arg(&GbKernelArg::new_mut(out.data_f32_mut()?))
            .arg(&GbKernelArg::new(input.data_f32()?))
            .arg(&outer_size)
            .arg(&axis_size)
            .launch(config)
            .map_err(|e| CudaError::Kernel(e.to_string()))?;
    }

    Ok(out)
}

/// GPU ReduceMean along last axis: input[..., N] -> output[...]
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

    let func = kernels
        .get("reduce_mean_kernel")
        .ok_or_else(|| CudaError::Kernel("reduce_mean_kernel not found".into()))?;

    let config = LaunchConfig::for_num_elems(outer_size as u32);

    unsafe {
        ctx.stream()
            .launch_builder(func)
            .arg(&GbKernelArg::new_mut(out.data_f32_mut()?))
            .arg(&GbKernelArg::new(input.data_f32()?))
            .arg(&outer_size)
            .arg(&axis_size)
            .launch(config)
            .map_err(|e| CudaError::Kernel(e.to_string()))?;
    }

    Ok(out)
}

/// GPU Softmax along last axis (numerically stable)
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

    let func = kernels
        .get("softmax_kernel")
        .ok_or_else(|| CudaError::Kernel("softmax_kernel not found".into()))?;

    let config = LaunchConfig::for_num_elems(outer_size as u32);

    unsafe {
        ctx.stream()
            .launch_builder(func)
            .arg(&GbKernelArg::new_mut(out.data_f32_mut()?))
            .arg(&GbKernelArg::new(input.data_f32()?))
            .arg(&outer_size)
            .arg(&axis_size)
            .launch(config)
            .map_err(|e| CudaError::Kernel(e.to_string()))?;
    }

    Ok(out)
}

/// GPU LayerNormalization along last axis
/// y = (x - mean) / sqrt(var + eps) * gamma + beta
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

    let func = kernels
        .get("layer_norm_kernel")
        .ok_or_else(|| CudaError::Kernel("layer_norm_kernel not found".into()))?;

    let config = LaunchConfig::for_num_elems(outer_size as u32);

    unsafe {
        ctx.stream()
            .launch_builder(func)
            .arg(&GbKernelArg::new_mut(out.data_f32_mut()?))
            .arg(&GbKernelArg::new(input.data_f32()?))
            .arg(&GbKernelArg::new(gamma.data_f32()?))
            .arg(&GbKernelArg::new(beta.data_f32()?))
            .arg(&outer_size)
            .arg(&axis_size)
            .arg(&eps)
            .launch(config)
            .map_err(|e| CudaError::Kernel(e.to_string()))?;
    }

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

        // 2x3 matrix, mean along last axis
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

        let data = vec![1.0f32, 2.0, 3.0];
        let input = GpuTensor::from_host_f32(&ctx, &data, vec![3]).unwrap();

        let output = gpu_softmax(&ctx, &kernels, &mut pool, &input).unwrap();
        let result = output.to_host_f32(&ctx).unwrap();

        // softmax([1,2,3]) = [0.0900, 0.2447, 0.6652]
        let sum: f32 = result.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Softmax should sum to 1, got {}",
            sum
        );

        // Verify relative ordering
        assert!(result[0] < result[1]);
        assert!(result[1] < result[2]);
    }

    #[test]
    #[ignore] // Requires CUDA GPU
    fn test_softmax_2d() {
        let (ctx, kernels, mut pool) = setup();

        // 2x3 matrix, softmax along last axis
        let data = vec![1.0f32, 2.0, 3.0, 1.0, 1.0, 1.0];
        let input = GpuTensor::from_host_f32(&ctx, &data, vec![2, 3]).unwrap();

        let output = gpu_softmax(&ctx, &kernels, &mut pool, &input).unwrap();
        let result = output.to_host_f32(&ctx).unwrap();

        // Each row should sum to 1
        let row1_sum: f32 = result[0..3].iter().sum();
        let row2_sum: f32 = result[3..6].iter().sum();
        assert!((row1_sum - 1.0).abs() < 1e-5, "Row 1 should sum to 1");
        assert!((row2_sum - 1.0).abs() < 1e-5, "Row 2 should sum to 1");

        // Second row is uniform, should be ~0.333 each
        assert!((result[3] - 0.3333).abs() < 0.01);
        assert!((result[4] - 0.3333).abs() < 0.01);
        assert!((result[5] - 0.3333).abs() < 0.01);
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

        // Should still work and sum to 1
        let sum: f32 = result.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Softmax should sum to 1 even with large inputs"
        );

        // Values should be finite
        for &v in &result {
            assert!(v.is_finite(), "Softmax produced non-finite value");
        }
    }

    #[test]
    #[ignore] // Requires CUDA GPU
    fn test_layer_norm() {
        let (ctx, kernels, mut pool) = setup();

        // Simple 1x4 input
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let input = GpuTensor::from_host_f32(&ctx, &data, vec![1, 4]).unwrap();

        // gamma = 1, beta = 0 (identity transform on normalized values)
        let gamma = GpuTensor::from_host_f32(&ctx, &[1.0f32, 1.0, 1.0, 1.0], vec![4]).unwrap();
        let beta = GpuTensor::from_host_f32(&ctx, &[0.0f32, 0.0, 0.0, 0.0], vec![4]).unwrap();

        let output = gpu_layer_norm(&ctx, &kernels, &mut pool, &input, &gamma, &beta, 1e-5).unwrap();
        let result = output.to_host_f32(&ctx).unwrap();

        // Mean of normalized should be ~0
        let mean: f32 = result.iter().sum::<f32>() / result.len() as f32;
        assert!(
            mean.abs() < 1e-5,
            "LayerNorm mean should be ~0, got {}",
            mean
        );

        // Variance should be ~1
        let var: f32 =
            result.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / result.len() as f32;
        assert!(
            (var - 1.0).abs() < 0.1,
            "LayerNorm variance should be ~1, got {}",
            var
        );
    }

    #[test]
    #[ignore] // Requires CUDA GPU
    fn test_layer_norm_with_affine() {
        let (ctx, kernels, mut pool) = setup();

        // 2x4 input
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let input = GpuTensor::from_host_f32(&ctx, &data, vec![2, 4]).unwrap();

        // gamma = 2, beta = 1
        let gamma = GpuTensor::from_host_f32(&ctx, &[2.0f32, 2.0, 2.0, 2.0], vec![4]).unwrap();
        let beta = GpuTensor::from_host_f32(&ctx, &[1.0f32, 1.0, 1.0, 1.0], vec![4]).unwrap();

        let output = gpu_layer_norm(&ctx, &kernels, &mut pool, &input, &gamma, &beta, 1e-5).unwrap();
        let result = output.to_host_f32(&ctx).unwrap();

        // After affine transform, mean should be ~1 (beta), not 0
        let mean: f32 = result.iter().sum::<f32>() / result.len() as f32;
        assert!(
            (mean - 1.0).abs() < 0.1,
            "LayerNorm with affine mean should be ~1"
        );
    }

    #[test]
    #[ignore] // Requires CUDA GPU
    fn test_large_reduction() {
        let (ctx, kernels, mut pool) = setup();

        // Test with larger tensor
        let n = 10000;
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let input = GpuTensor::from_host_f32(&ctx, &data, vec![n]).unwrap();

        let result = gpu_reduce_sum_all(&ctx, &kernels, &mut pool, &input).unwrap();

        // Sum of 0..n = n*(n-1)/2
        let expected = (n * (n - 1) / 2) as f32;
        assert!(
            (result - expected).abs() / expected < 1e-4,
            "Expected {}, got {}",
            expected,
            result
        );
    }
}
