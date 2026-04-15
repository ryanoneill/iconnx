//! Type casting operations for GPU tensors

use crate::cuda::bridge::GbKernelArg;
use super::cache::OpsKernelCache;
use crate::cuda::context::{CudaError, IconnxCudaContext};
use crate::cuda::tensor::{DType, GpuTensor};
use cudarc::driver::{LaunchConfig, PushKernelArg};

/// GPU Cast: Convert tensor from one dtype to another
///
/// Supports conversions between Float32, Int64, and Int32.
pub fn gpu_cast(
    ctx: &IconnxCudaContext,
    cache: &OpsKernelCache,
    input: &GpuTensor,
    target_dtype: DType,
) -> Result<GpuTensor, CudaError> {
    let src_dtype = input.dtype();

    // If same type, just clone the tensor (no-op cast)
    // This avoids expensive D2H -> H2D round-trip
    if src_dtype == target_dtype {
        return Ok(input.clone());
    }

    let n = input.len();
    let shape = input.shape().to_vec();
    let config = LaunchConfig::for_num_elems(n as u32);

    match (src_dtype, target_dtype) {
        // Float32 → Int64
        (DType::Float32, DType::Int64) => {
            let kernel = cache
                .get("cast_f32_to_i64_kernel")
                .ok_or_else(|| CudaError::Kernel("cast_f32_to_i64_kernel not found".into()))?;
            let mut output = GpuTensor::zeros_i64(ctx, shape)?;

            unsafe {
                ctx.stream()
                    .launch_builder(kernel)
                    .arg(&GbKernelArg::new_mut(output.data_i64_mut()?))
                    .arg(&GbKernelArg::new(input.data_f32()?))
                    .arg(&n)
                    .launch(config)
                    .map_err(|e| {
                        CudaError::Kernel(format!("cast_f32_to_i64 launch failed: {}", e))
                    })?;
            }
            Ok(output)
        }

        // Float32 → Int32
        (DType::Float32, DType::Int32) => {
            let kernel = cache
                .get("cast_f32_to_i32_kernel")
                .ok_or_else(|| CudaError::Kernel("cast_f32_to_i32_kernel not found".into()))?;
            let mut output = GpuTensor::zeros_i32(ctx, shape)?;

            unsafe {
                ctx.stream()
                    .launch_builder(kernel)
                    .arg(&GbKernelArg::new_mut(output.data_i32_mut()?))
                    .arg(&GbKernelArg::new(input.data_f32()?))
                    .arg(&n)
                    .launch(config)
                    .map_err(|e| {
                        CudaError::Kernel(format!("cast_f32_to_i32 launch failed: {}", e))
                    })?;
            }
            Ok(output)
        }

        // Int64 → Float32
        (DType::Int64, DType::Float32) => {
            let kernel = cache
                .get("cast_i64_to_f32_kernel")
                .ok_or_else(|| CudaError::Kernel("cast_i64_to_f32_kernel not found".into()))?;
            let mut output = GpuTensor::zeros_f32(ctx, shape)?;

            unsafe {
                ctx.stream()
                    .launch_builder(kernel)
                    .arg(&GbKernelArg::new_mut(output.data_f32_mut()?))
                    .arg(&GbKernelArg::new(input.data_i64()?))
                    .arg(&n)
                    .launch(config)
                    .map_err(|e| {
                        CudaError::Kernel(format!("cast_i64_to_f32 launch failed: {}", e))
                    })?;
            }
            Ok(output)
        }

        // Int64 → Int32
        (DType::Int64, DType::Int32) => {
            let kernel = cache
                .get("cast_i64_to_i32_kernel")
                .ok_or_else(|| CudaError::Kernel("cast_i64_to_i32_kernel not found".into()))?;
            let mut output = GpuTensor::zeros_i32(ctx, shape)?;

            unsafe {
                ctx.stream()
                    .launch_builder(kernel)
                    .arg(&GbKernelArg::new_mut(output.data_i32_mut()?))
                    .arg(&GbKernelArg::new(input.data_i64()?))
                    .arg(&n)
                    .launch(config)
                    .map_err(|e| {
                        CudaError::Kernel(format!("cast_i64_to_i32 launch failed: {}", e))
                    })?;
            }
            Ok(output)
        }

        // Int32 → Float32
        (DType::Int32, DType::Float32) => {
            let kernel = cache
                .get("cast_i32_to_f32_kernel")
                .ok_or_else(|| CudaError::Kernel("cast_i32_to_f32_kernel not found".into()))?;
            let mut output = GpuTensor::zeros_f32(ctx, shape)?;

            unsafe {
                ctx.stream()
                    .launch_builder(kernel)
                    .arg(&GbKernelArg::new_mut(output.data_f32_mut()?))
                    .arg(&GbKernelArg::new(input.data_i32()?))
                    .arg(&n)
                    .launch(config)
                    .map_err(|e| {
                        CudaError::Kernel(format!("cast_i32_to_f32 launch failed: {}", e))
                    })?;
            }
            Ok(output)
        }

        // Int32 → Int64
        (DType::Int32, DType::Int64) => {
            let kernel = cache
                .get("cast_i32_to_i64_kernel")
                .ok_or_else(|| CudaError::Kernel("cast_i32_to_i64_kernel not found".into()))?;
            let mut output = GpuTensor::zeros_i64(ctx, shape)?;

            unsafe {
                ctx.stream()
                    .launch_builder(kernel)
                    .arg(&GbKernelArg::new_mut(output.data_i64_mut()?))
                    .arg(&GbKernelArg::new(input.data_i32()?))
                    .arg(&n)
                    .launch(config)
                    .map_err(|e| {
                        CudaError::Kernel(format!("cast_i32_to_i64 launch failed: {}", e))
                    })?;
            }
            Ok(output)
        }

        // Same-type cases already handled above by early return
        (DType::Float32, DType::Float32)
        | (DType::Int64, DType::Int64)
        | (DType::Int32, DType::Int32) => unreachable!("Same-type cast handled above"),
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
    fn test_cast_f32_to_i64() {
        let (ctx, cache) = setup();

        let data = vec![1.5f32, 2.4, -3.6, 4.0, 0.0];
        let input = GpuTensor::from_host_f32(&ctx, &data, vec![5]).unwrap();

        let output = gpu_cast(&ctx, &cache, &input, DType::Int64).unwrap();
        assert_eq!(output.dtype(), DType::Int64);

        let result = output.to_host_i64(&ctx).unwrap();
        // With rounding: 1.5→2, 2.4→2, -3.6→-4, 4.0→4, 0.0→0
        assert_eq!(result, vec![2i64, 2, -4, 4, 0]);

        println!("Cast f32 to i64 test passed!");
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_cast_i64_to_f32() {
        let (ctx, cache) = setup();

        let data = vec![1i64, 2, -3, 4, 0];
        let input = GpuTensor::from_host_i64(&ctx, &data, vec![5]).unwrap();

        let output = gpu_cast(&ctx, &cache, &input, DType::Float32).unwrap();
        assert_eq!(output.dtype(), DType::Float32);

        let result = output.to_host_f32(&ctx).unwrap();
        assert_eq!(result, vec![1.0f32, 2.0, -3.0, 4.0, 0.0]);

        println!("Cast i64 to f32 test passed!");
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_cast_f32_to_i32() {
        let (ctx, cache) = setup();

        let data = vec![1.5f32, -2.4, 3.6];
        let input = GpuTensor::from_host_f32(&ctx, &data, vec![3]).unwrap();

        let output = gpu_cast(&ctx, &cache, &input, DType::Int32).unwrap();
        assert_eq!(output.dtype(), DType::Int32);

        let result = output.to_host_i32(&ctx).unwrap();
        // With rounding: 1.5→2, -2.4→-2, 3.6→4
        assert_eq!(result, vec![2i32, -2, 4]);

        println!("Cast f32 to i32 test passed!");
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_cast_i32_to_f32() {
        let (ctx, cache) = setup();

        let data = vec![1i32, -2, 3];
        let input = GpuTensor::from_host_i32(&ctx, &data, vec![3]).unwrap();

        let output = gpu_cast(&ctx, &cache, &input, DType::Float32).unwrap();
        assert_eq!(output.dtype(), DType::Float32);

        let result = output.to_host_f32(&ctx).unwrap();
        assert_eq!(result, vec![1.0f32, -2.0, 3.0]);

        println!("Cast i32 to f32 test passed!");
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_cast_same_type() {
        let (ctx, cache) = setup();

        let data = vec![1.0f32, 2.0, 3.0];
        let input = GpuTensor::from_host_f32(&ctx, &data, vec![3]).unwrap();

        // Cast to same type should work
        let output = gpu_cast(&ctx, &cache, &input, DType::Float32).unwrap();
        assert_eq!(output.dtype(), DType::Float32);

        let result = output.to_host_f32(&ctx).unwrap();
        assert_eq!(result, data);

        println!("Cast same type test passed!");
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_cast_chain() {
        let (ctx, cache) = setup();

        // f32 → i64 → i32 → f32 should work
        let data = vec![1.0f32, 2.0, 3.0];
        let f32_input = GpuTensor::from_host_f32(&ctx, &data, vec![3]).unwrap();

        let i64_tensor = gpu_cast(&ctx, &cache, &f32_input, DType::Int64).unwrap();
        assert_eq!(i64_tensor.dtype(), DType::Int64);

        let i32_tensor = gpu_cast(&ctx, &cache, &i64_tensor, DType::Int32).unwrap();
        assert_eq!(i32_tensor.dtype(), DType::Int32);

        let f32_output = gpu_cast(&ctx, &cache, &i32_tensor, DType::Float32).unwrap();
        assert_eq!(f32_output.dtype(), DType::Float32);

        let result = f32_output.to_host_f32(&ctx).unwrap();
        assert_eq!(result, data);

        println!("Cast chain test passed!");
    }
}
