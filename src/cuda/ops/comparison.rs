//! Comparison and logical operations for GPU tensors

use crate::cuda::bridge::GbKernelArg;
use super::cache::OpsKernelCache;
use crate::cuda::context::{CudaError, IconnxCudaContext};
use crate::cuda::tensor::GpuTensor;
use cudarc::driver::{LaunchConfig, PushKernelArg};

/// GPU Equal: out = (a == b) ? 1.0 : 0.0
pub fn gpu_equal(
    ctx: &IconnxCudaContext,
    cache: &OpsKernelCache,
    a: &GpuTensor,
    b: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    binary_comparison(ctx, cache, a, b, "equal_kernel")
}

/// GPU Less: out = (a < b) ? 1.0 : 0.0
pub fn gpu_less(
    ctx: &IconnxCudaContext,
    cache: &OpsKernelCache,
    a: &GpuTensor,
    b: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    binary_comparison(ctx, cache, a, b, "less_kernel")
}

/// GPU Greater: out = (a > b) ? 1.0 : 0.0
pub fn gpu_greater(
    ctx: &IconnxCudaContext,
    cache: &OpsKernelCache,
    a: &GpuTensor,
    b: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    binary_comparison(ctx, cache, a, b, "greater_kernel")
}

/// GPU GreaterOrEqual: out = (a >= b) ? 1.0 : 0.0
pub fn gpu_greater_or_equal(
    ctx: &IconnxCudaContext,
    cache: &OpsKernelCache,
    a: &GpuTensor,
    b: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    binary_comparison(ctx, cache, a, b, "greater_or_equal_kernel")
}

/// GPU LessOrEqual: out = (a <= b) ? 1.0 : 0.0
pub fn gpu_less_or_equal(
    ctx: &IconnxCudaContext,
    cache: &OpsKernelCache,
    a: &GpuTensor,
    b: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    binary_comparison(ctx, cache, a, b, "less_or_equal_kernel")
}

/// GPU NotEqual: out = (a != b) ? 1.0 : 0.0
pub fn gpu_not_equal(
    ctx: &IconnxCudaContext,
    cache: &OpsKernelCache,
    a: &GpuTensor,
    b: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    binary_comparison(ctx, cache, a, b, "not_equal_kernel")
}

/// GPU And: out = (a && b) where non-zero is true
pub fn gpu_and(
    ctx: &IconnxCudaContext,
    cache: &OpsKernelCache,
    a: &GpuTensor,
    b: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    binary_comparison(ctx, cache, a, b, "and_kernel")
}

/// GPU Or: out = (a || b) where non-zero is true
pub fn gpu_or(
    ctx: &IconnxCudaContext,
    cache: &OpsKernelCache,
    a: &GpuTensor,
    b: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    binary_comparison(ctx, cache, a, b, "or_kernel")
}

/// GPU Not: out = !a where non-zero is true
pub fn gpu_not(
    ctx: &IconnxCudaContext,
    cache: &OpsKernelCache,
    a: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    let n = a.len();
    let mut output = GpuTensor::zeros_f32(ctx, a.shape().to_vec())?;

    let kernel = cache
        .get("not_kernel")
        .ok_or_else(|| CudaError::Kernel("not_kernel not found".into()))?;

    let config = LaunchConfig::for_num_elems(n as u32);

    unsafe {
        ctx.stream()
            .launch_builder(kernel)
            .arg(&GbKernelArg::new_mut(output.data_f32_mut()?))
            .arg(&GbKernelArg::new(a.data_f32()?))
            .arg(&n)
            .launch(config)
            .map_err(|e| CudaError::Kernel(format!("not launch failed: {}", e)))?;
    }

    Ok(output)
}

/// Helper for binary comparison operations
/// Supports Float32 and Int64 input tensors, always outputs Float32 (0.0 or 1.0)
fn binary_comparison(
    ctx: &IconnxCudaContext,
    cache: &OpsKernelCache,
    a: &GpuTensor,
    b: &GpuTensor,
    kernel_name: &'static str,
) -> Result<GpuTensor, CudaError> {
    let n = a.len();
    assert_eq!(b.len(), n, "b length mismatch");

    let dtype = a.dtype();
    if b.dtype() != dtype {
        return Err(CudaError::Kernel(format!(
            "Comparison dtype mismatch: {} vs {}",
            dtype.name(),
            b.dtype().name()
        )));
    }

    let mut output = GpuTensor::zeros_f32(ctx, a.shape().to_vec())?;
    let config = LaunchConfig::for_num_elems(n as u32);

    match dtype {
        crate::cuda::tensor::DType::Float32 => {
            let kernel = cache
                .get(kernel_name)
                .ok_or_else(|| CudaError::Kernel(format!("{} not found", kernel_name)))?;

            unsafe {
                ctx.stream()
                    .launch_builder(kernel)
                    .arg(&GbKernelArg::new_mut(output.data_f32_mut()?))
                    .arg(&GbKernelArg::new(a.data_f32()?))
                    .arg(&GbKernelArg::new(b.data_f32()?))
                    .arg(&n)
                    .launch(config)
                    .map_err(|e| {
                        CudaError::Kernel(format!("{} launch failed: {}", kernel_name, e))
                    })?;
            }
        }
        crate::cuda::tensor::DType::Int64 => {
            // Use the i64 version of the kernel
            let i64_kernel_name = format!("{}_i64", kernel_name.trim_end_matches("_kernel"));
            let kernel_name_str: &str = Box::leak(i64_kernel_name.into_boxed_str());
            let kernel = cache
                .get(kernel_name_str)
                .ok_or_else(|| CudaError::Kernel(format!("{} not found", kernel_name_str)))?;

            unsafe {
                ctx.stream()
                    .launch_builder(kernel)
                    .arg(&GbKernelArg::new_mut(output.data_f32_mut()?))
                    .arg(&GbKernelArg::new(a.data_i64()?))
                    .arg(&GbKernelArg::new(b.data_i64()?))
                    .arg(&n)
                    .launch(config)
                    .map_err(|e| {
                        CudaError::Kernel(format!("{} launch failed: {}", kernel_name_str, e))
                    })?;
            }
        }
        crate::cuda::tensor::DType::Int32 => {
            // Use the i32 version of the kernel
            let i32_kernel_name = format!("{}_i32", kernel_name.trim_end_matches("_kernel"));
            let kernel_name_str: &str = Box::leak(i32_kernel_name.into_boxed_str());
            let kernel = cache
                .get(kernel_name_str)
                .ok_or_else(|| CudaError::Kernel(format!("{} not found", kernel_name_str)))?;

            unsafe {
                ctx.stream()
                    .launch_builder(kernel)
                    .arg(&GbKernelArg::new_mut(output.data_f32_mut()?))
                    .arg(&GbKernelArg::new(a.data_i32()?))
                    .arg(&GbKernelArg::new(b.data_i32()?))
                    .arg(&n)
                    .launch(config)
                    .map_err(|e| {
                        CudaError::Kernel(format!("{} launch failed: {}", kernel_name_str, e))
                    })?;
            }
        }
    }

    Ok(output)
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
    fn test_comparison_ops() {
        let (ctx, cache) = setup();

        let a_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let a = GpuTensor::from_host_f32(&ctx, &a_data, vec![5]).unwrap();

        let b_data = vec![1.0f32, 3.0, 2.0, 4.0, 6.0];
        let b = GpuTensor::from_host_f32(&ctx, &b_data, vec![5]).unwrap();

        // Equal: [1, 0, 0, 1, 0]
        let eq_result = gpu_equal(&ctx, &cache, &a, &b)
            .unwrap()
            .to_host_f32(&ctx)
            .unwrap();
        assert_eq!(eq_result, vec![1.0, 0.0, 0.0, 1.0, 0.0]);

        // Less: [0, 1, 0, 0, 1]
        let lt_result = gpu_less(&ctx, &cache, &a, &b)
            .unwrap()
            .to_host_f32(&ctx)
            .unwrap();
        assert_eq!(lt_result, vec![0.0, 1.0, 0.0, 0.0, 1.0]);

        // Greater: [0, 0, 1, 0, 0]
        let gt_result = gpu_greater(&ctx, &cache, &a, &b)
            .unwrap()
            .to_host_f32(&ctx)
            .unwrap();
        assert_eq!(gt_result, vec![0.0, 0.0, 1.0, 0.0, 0.0]);

        println!("Comparison ops test passed!");
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_extended_comparison_ops() {
        let (ctx, cache) = setup();

        let a_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let a = GpuTensor::from_host_f32(&ctx, &a_data, vec![5]).unwrap();

        let b_data = vec![1.0f32, 3.0, 2.0, 4.0, 6.0];
        let b = GpuTensor::from_host_f32(&ctx, &b_data, vec![5]).unwrap();

        // GreaterOrEqual: [1, 0, 1, 1, 0]
        let ge_result = gpu_greater_or_equal(&ctx, &cache, &a, &b)
            .unwrap()
            .to_host_f32(&ctx)
            .unwrap();
        assert_eq!(ge_result, vec![1.0, 0.0, 1.0, 1.0, 0.0]);

        // LessOrEqual: [1, 1, 0, 1, 1]
        let le_result = gpu_less_or_equal(&ctx, &cache, &a, &b)
            .unwrap()
            .to_host_f32(&ctx)
            .unwrap();
        assert_eq!(le_result, vec![1.0, 1.0, 0.0, 1.0, 1.0]);

        // NotEqual: [0, 1, 1, 0, 1]
        let ne_result = gpu_not_equal(&ctx, &cache, &a, &b)
            .unwrap()
            .to_host_f32(&ctx)
            .unwrap();
        assert_eq!(ne_result, vec![0.0, 1.0, 1.0, 0.0, 1.0]);

        println!("Extended comparison ops test passed!");
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_logical_ops() {
        let (ctx, cache) = setup();

        let a_data = vec![1.0f32, 0.0, 1.0, 0.0];
        let a = GpuTensor::from_host_f32(&ctx, &a_data, vec![4]).unwrap();

        let b_data = vec![1.0f32, 1.0, 0.0, 0.0];
        let b = GpuTensor::from_host_f32(&ctx, &b_data, vec![4]).unwrap();

        // And: [1, 0, 0, 0]
        let and_result = gpu_and(&ctx, &cache, &a, &b)
            .unwrap()
            .to_host_f32(&ctx)
            .unwrap();
        assert_eq!(and_result, vec![1.0, 0.0, 0.0, 0.0]);

        // Or: [1, 1, 1, 0]
        let or_result = gpu_or(&ctx, &cache, &a, &b)
            .unwrap()
            .to_host_f32(&ctx)
            .unwrap();
        assert_eq!(or_result, vec![1.0, 1.0, 1.0, 0.0]);

        // Not a: [0, 1, 0, 1]
        let not_result = gpu_not(&ctx, &cache, &a)
            .unwrap()
            .to_host_f32(&ctx)
            .unwrap();
        assert_eq!(not_result, vec![0.0, 1.0, 0.0, 1.0]);

        println!("Logical ops test passed!");
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_comparison_ops_int32() {
        let (ctx, cache) = setup();

        // Test Int32 comparison
        let a_data = vec![1i32, 2, 3, 4, 5];
        let a = GpuTensor::from_host_i32(&ctx, &a_data, vec![5]).unwrap();

        let b_data = vec![1i32, 3, 2, 4, 6];
        let b = GpuTensor::from_host_i32(&ctx, &b_data, vec![5]).unwrap();

        // Equal: [1, 0, 0, 1, 0]
        let eq_result = gpu_equal(&ctx, &cache, &a, &b)
            .unwrap()
            .to_host_f32(&ctx)
            .unwrap();
        assert_eq!(eq_result, vec![1.0, 0.0, 0.0, 1.0, 0.0]);

        // Less: [0, 1, 0, 0, 1]
        let lt_result = gpu_less(&ctx, &cache, &a, &b)
            .unwrap()
            .to_host_f32(&ctx)
            .unwrap();
        assert_eq!(lt_result, vec![0.0, 1.0, 0.0, 0.0, 1.0]);

        // Greater: [0, 0, 1, 0, 0]
        let gt_result = gpu_greater(&ctx, &cache, &a, &b)
            .unwrap()
            .to_host_f32(&ctx)
            .unwrap();
        assert_eq!(gt_result, vec![0.0, 0.0, 1.0, 0.0, 0.0]);

        println!("Int32 comparison ops test passed!");
    }
}
