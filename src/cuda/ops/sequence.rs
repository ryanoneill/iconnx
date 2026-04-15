//! Sequence and math operations for GPU tensors

use crate::cuda::bridge::GbKernelArg;
use super::cache::OpsKernelCache;
use crate::cuda::context::{CudaError, IconnxCudaContext};
use crate::cuda::tensor::GpuTensor;
use cudarc::driver::{LaunchConfig, PushKernelArg};

/// GPU Atan: out = atan(inp)
pub fn gpu_atan(
    ctx: &IconnxCudaContext,
    cache: &OpsKernelCache,
    input: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    let n = input.len();

    let mut output = GpuTensor::zeros_f32(ctx, input.shape().to_vec())?;

    let kernel = cache
        .get("atan_kernel")
        .ok_or_else(|| CudaError::Kernel("atan_kernel not found".into()))?;

    let config = LaunchConfig::for_num_elems(n as u32);

    unsafe {
        ctx.stream()
            .launch_builder(kernel)
            .arg(&GbKernelArg::new_mut(output.data_f32_mut()?))
            .arg(&GbKernelArg::new(input.data_f32()?))
            .arg(&n)
            .launch(config)
            .map_err(|e| CudaError::Kernel(format!("atan launch failed: {}", e)))?;
    }

    Ok(output)
}

/// GPU Range: Generate sequence [start, start+delta, ...]
pub fn gpu_range(
    ctx: &IconnxCudaContext,
    cache: &OpsKernelCache,
    start: f32,
    delta: f32,
    n: usize,
) -> Result<GpuTensor, CudaError> {
    let mut output = GpuTensor::zeros_f32(ctx, vec![n])?;

    let kernel = cache
        .get("range_kernel")
        .ok_or_else(|| CudaError::Kernel("range_kernel not found".into()))?;

    let config = LaunchConfig::for_num_elems(n as u32);

    unsafe {
        ctx.stream()
            .launch_builder(kernel)
            .arg(&GbKernelArg::new_mut(output.data_f32_mut()?))
            .arg(&start)
            .arg(&delta)
            .arg(&n)
            .launch(config)
            .map_err(|e| CudaError::Kernel(format!("range launch failed: {}", e)))?;
    }

    Ok(output)
}

/// GPU Range Int64: Generate sequence [start, start+delta, ...]
pub fn gpu_range_i64(
    ctx: &IconnxCudaContext,
    cache: &OpsKernelCache,
    start: i64,
    delta: i64,
    n: usize,
) -> Result<GpuTensor, CudaError> {
    let mut output = GpuTensor::zeros_i64(ctx, vec![n])?;

    let kernel = cache
        .get("range_i64_kernel")
        .ok_or_else(|| CudaError::Kernel("range_i64_kernel not found".into()))?;

    let config = LaunchConfig::for_num_elems(n as u32);

    unsafe {
        ctx.stream()
            .launch_builder(kernel)
            .arg(&GbKernelArg::new_mut(output.data_i64_mut()?))
            .arg(&start)
            .arg(&delta)
            .arg(&n)
            .launch(config)
            .map_err(|e| CudaError::Kernel(format!("range_i64 launch failed: {}", e)))?;
    }

    Ok(output)
}

/// GPU CumSum: Cumulative sum along specified axis
/// Supports Float32, Int64, and Int32 tensors
///
/// # Arguments
/// * `axis` - The axis along which to compute cumulative sum (can be negative)
pub fn gpu_cumsum(
    ctx: &IconnxCudaContext,
    cache: &OpsKernelCache,
    input: &GpuTensor,
    axis: i64,
    exclusive: bool,
    reverse: bool,
) -> Result<GpuTensor, CudaError> {
    let shape = input.shape();
    let dtype = input.dtype();
    let ndim = shape.len();

    // Handle negative axis
    let axis_idx = if axis < 0 {
        (ndim as i64 + axis) as usize
    } else {
        axis as usize
    };

    if axis_idx >= ndim {
        return Err(CudaError::Kernel(format!(
            "CumSum axis {} out of bounds for tensor with {} dimensions",
            axis, ndim
        )));
    }

    // Compute dimensions for strided access
    // For shape [A, B, C] with axis=1: axis_len=B, inner_stride=C, outer_size=A
    let axis_len = shape[axis_idx];
    let inner_stride: usize = shape[axis_idx + 1..].iter().product::<usize>().max(1);
    let outer_size: usize = shape[..axis_idx].iter().product::<usize>().max(1);
    let num_lines = outer_size * inner_stride;

    // Launch one thread per cumsum line
    let config = LaunchConfig::for_num_elems(num_lines as u32);
    let exclusive_i = if exclusive { 1i32 } else { 0i32 };
    let reverse_i = if reverse { 1i32 } else { 0i32 };

    match dtype {
        crate::cuda::tensor::DType::Float32 => {
            let mut output = GpuTensor::zeros_f32(ctx, shape.to_vec())?;
            let kernel = cache
                .get("cumsum_kernel")
                .ok_or_else(|| CudaError::Kernel("cumsum_kernel not found".into()))?;

            unsafe {
                ctx.stream()
                    .launch_builder(kernel)
                    .arg(&GbKernelArg::new_mut(output.data_f32_mut()?))
                    .arg(&GbKernelArg::new(input.data_f32()?))
                    .arg(&num_lines)
                    .arg(&axis_len)
                    .arg(&inner_stride)
                    .arg(&exclusive_i)
                    .arg(&reverse_i)
                    .launch(config)
                    .map_err(|e| CudaError::Kernel(format!("cumsum launch failed: {}", e)))?;
            }
            Ok(output)
        }
        crate::cuda::tensor::DType::Int64 => {
            let mut output = GpuTensor::zeros_i64(ctx, shape.to_vec())?;
            let kernel = cache
                .get("cumsum_i64_kernel")
                .ok_or_else(|| CudaError::Kernel("cumsum_i64_kernel not found".into()))?;

            unsafe {
                ctx.stream()
                    .launch_builder(kernel)
                    .arg(&GbKernelArg::new_mut(output.data_i64_mut()?))
                    .arg(&GbKernelArg::new(input.data_i64()?))
                    .arg(&num_lines)
                    .arg(&axis_len)
                    .arg(&inner_stride)
                    .arg(&exclusive_i)
                    .arg(&reverse_i)
                    .launch(config)
                    .map_err(|e| CudaError::Kernel(format!("cumsum_i64 launch failed: {}", e)))?;
            }
            Ok(output)
        }
        crate::cuda::tensor::DType::Int32 => {
            let mut output = GpuTensor::zeros_i32(ctx, shape.to_vec())?;
            let kernel = cache
                .get("cumsum_i32_kernel")
                .ok_or_else(|| CudaError::Kernel("cumsum_i32_kernel not found".into()))?;

            unsafe {
                ctx.stream()
                    .launch_builder(kernel)
                    .arg(&GbKernelArg::new_mut(output.data_i32_mut()?))
                    .arg(&GbKernelArg::new(input.data_i32()?))
                    .arg(&num_lines)
                    .arg(&axis_len)
                    .arg(&inner_stride)
                    .arg(&exclusive_i)
                    .arg(&reverse_i)
                    .launch(config)
                    .map_err(|e| CudaError::Kernel(format!("cumsum_i32 launch failed: {}", e)))?;
            }
            Ok(output)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (IconnxCudaContext, OpsKernelCache) {
        let ctx = IconnxCudaContext::new().expect("Failed to create CUDA context");
        let cache = OpsKernelCache::new(&ctx).expect("Failed to compile kernels");
        (ctx, cache)
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_atan() {
        let (ctx, cache) = setup();

        let input_data = vec![0.0f32, 1.0, -1.0, 0.5];
        let input = GpuTensor::from_host_f32(&ctx, &input_data, vec![4]).unwrap();

        let output = gpu_atan(&ctx, &cache, &input).unwrap();
        let result = output.to_host_f32(&ctx).unwrap();

        assert!((result[0] - 0.0).abs() < 1e-5);
        assert!((result[1] - std::f32::consts::FRAC_PI_4).abs() < 1e-5);
        assert!((result[2] - (-std::f32::consts::FRAC_PI_4)).abs() < 1e-5);
        assert!((result[3] - 0.4636476).abs() < 1e-5);

        println!("Atan test passed!");
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_range() {
        let (ctx, cache) = setup();

        let output = gpu_range(&ctx, &cache, 0.0, 1.0, 5).unwrap();
        let result = output.to_host_f32(&ctx).unwrap();

        assert_eq!(result, vec![0.0, 1.0, 2.0, 3.0, 4.0]);

        let output2 = gpu_range(&ctx, &cache, 2.0, 0.5, 4).unwrap();
        let result2 = output2.to_host_f32(&ctx).unwrap();

        assert_eq!(result2, vec![2.0, 2.5, 3.0, 3.5]);

        println!("Range test passed!");
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_cumsum() {
        let (ctx, cache) = setup();

        let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let input = GpuTensor::from_host_f32(&ctx, &input_data, vec![5]).unwrap();

        // Inclusive cumsum along axis 0: [1, 3, 6, 10, 15]
        let output = gpu_cumsum(&ctx, &cache, &input, 0, false, false).unwrap();
        let result = output.to_host_f32(&ctx).unwrap();
        assert_eq!(result, vec![1.0, 3.0, 6.0, 10.0, 15.0]);

        // Exclusive cumsum along axis 0: [0, 1, 3, 6, 10]
        let output_ex = gpu_cumsum(&ctx, &cache, &input, 0, true, false).unwrap();
        let result_ex = output_ex.to_host_f32(&ctx).unwrap();
        assert_eq!(result_ex, vec![0.0, 1.0, 3.0, 6.0, 10.0]);

        // Reverse inclusive along axis 0: [15, 14, 12, 9, 5]
        let output_rev = gpu_cumsum(&ctx, &cache, &input, 0, false, true).unwrap();
        let result_rev = output_rev.to_host_f32(&ctx).unwrap();
        assert_eq!(result_rev, vec![15.0, 14.0, 12.0, 9.0, 5.0]);

        println!("CumSum test passed!");
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_cumsum_axis() {
        let (ctx, cache) = setup();

        // 2D tensor [2, 3]
        let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = GpuTensor::from_host_f32(&ctx, &input_data, vec![2, 3]).unwrap();

        // Cumsum along axis 0: [[1,2,3], [5,7,9]]
        let output_axis0 = gpu_cumsum(&ctx, &cache, &input, 0, false, false).unwrap();
        let result0 = output_axis0.to_host_f32(&ctx).unwrap();
        assert_eq!(result0, vec![1.0, 2.0, 3.0, 5.0, 7.0, 9.0]);

        // Cumsum along axis 1: [[1,3,6], [4,9,15]]
        let output_axis1 = gpu_cumsum(&ctx, &cache, &input, 1, false, false).unwrap();
        let result1 = output_axis1.to_host_f32(&ctx).unwrap();
        assert_eq!(result1, vec![1.0, 3.0, 6.0, 4.0, 9.0, 15.0]);

        // Cumsum along axis -1 (same as axis 1)
        let output_neg = gpu_cumsum(&ctx, &cache, &input, -1, false, false).unwrap();
        let result_neg = output_neg.to_host_f32(&ctx).unwrap();
        assert_eq!(result_neg, vec![1.0, 3.0, 6.0, 4.0, 9.0, 15.0]);

        println!("CumSum axis test passed!");
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_cumsum_3d() {
        let (ctx, cache) = setup();

        // 3D tensor [1, 3, 2] - matches the problematic case in the model
        // Data: [[[1,2], [3,4], [5,6]]]
        let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = GpuTensor::from_host_f32(&ctx, &input_data, vec![1, 3, 2]).unwrap();

        // Cumsum along axis 1 (the middle axis with size 3)
        // Expected: [[[1,2], [4,6], [9,12]]]
        let output = gpu_cumsum(&ctx, &cache, &input, 1, false, false).unwrap();
        let result = output.to_host_f32(&ctx).unwrap();
        assert_eq!(result, vec![1.0, 2.0, 4.0, 6.0, 9.0, 12.0]);

        println!("CumSum 3D test passed!");
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_cumsum_i32() {
        let (ctx, cache) = setup();

        let input_data = vec![1i32, 2, 3, 4, 5];
        let input = GpuTensor::from_host_i32(&ctx, &input_data, vec![5]).unwrap();

        // Inclusive cumsum along axis 0: [1, 3, 6, 10, 15]
        let output = gpu_cumsum(&ctx, &cache, &input, 0, false, false).unwrap();
        let result = output.to_host_i32(&ctx).unwrap();
        assert_eq!(result, vec![1, 3, 6, 10, 15]);

        println!("CumSum Int32 test passed!");
    }
}
