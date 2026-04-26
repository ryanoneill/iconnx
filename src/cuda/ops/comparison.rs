//! Comparison and logical operations for GPU tensors (garboard TypedKernel).

use super::cache::OpsKernelCache;
use crate::cuda::context::{CudaError, IconnxCudaContext};
use crate::cuda::tensor::GpuTensor;
use garboard::{DeviceSlice, LaunchConfig};

/// GPU Equal: out = (a == b) ? 1.0 : 0.0.
pub fn gpu_equal(
    ctx: &IconnxCudaContext,
    cache: &OpsKernelCache,
    a: &GpuTensor,
    b: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    binary_comparison(ctx, cache, a, b, "equal_kernel")
}

/// GPU Less: out = (a < b) ? 1.0 : 0.0.
pub fn gpu_less(
    ctx: &IconnxCudaContext,
    cache: &OpsKernelCache,
    a: &GpuTensor,
    b: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    binary_comparison(ctx, cache, a, b, "less_kernel")
}

/// GPU Greater: out = (a > b) ? 1.0 : 0.0.
pub fn gpu_greater(
    ctx: &IconnxCudaContext,
    cache: &OpsKernelCache,
    a: &GpuTensor,
    b: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    binary_comparison(ctx, cache, a, b, "greater_kernel")
}

/// GPU GreaterOrEqual: out = (a >= b) ? 1.0 : 0.0.
pub fn gpu_greater_or_equal(
    ctx: &IconnxCudaContext,
    cache: &OpsKernelCache,
    a: &GpuTensor,
    b: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    binary_comparison(ctx, cache, a, b, "greater_or_equal_kernel")
}

/// GPU LessOrEqual: out = (a <= b) ? 1.0 : 0.0.
pub fn gpu_less_or_equal(
    ctx: &IconnxCudaContext,
    cache: &OpsKernelCache,
    a: &GpuTensor,
    b: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    binary_comparison(ctx, cache, a, b, "less_or_equal_kernel")
}

/// GPU NotEqual: out = (a != b) ? 1.0 : 0.0.
pub fn gpu_not_equal(
    ctx: &IconnxCudaContext,
    cache: &OpsKernelCache,
    a: &GpuTensor,
    b: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    binary_comparison(ctx, cache, a, b, "not_equal_kernel")
}

/// GPU And: out = (a && b) where non-zero is true.
pub fn gpu_and(
    ctx: &IconnxCudaContext,
    cache: &OpsKernelCache,
    a: &GpuTensor,
    b: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    binary_comparison(ctx, cache, a, b, "and_kernel")
}

/// GPU Or: out = (a || b) where non-zero is true.
pub fn gpu_or(
    ctx: &IconnxCudaContext,
    cache: &OpsKernelCache,
    a: &GpuTensor,
    b: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    binary_comparison(ctx, cache, a, b, "or_kernel")
}

/// GPU Not: out = !a where non-zero is true.
pub fn gpu_not(
    ctx: &IconnxCudaContext,
    cache: &OpsKernelCache,
    a: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    let n = a.len();
    let mut output = GpuTensor::zeros_f32(ctx, a.shape().to_vec())?;

    // SAFETY: not_kernel signature is `(float* out, const float* a, size_t n)`.
    let kernel = unsafe {
        cache.module().typed_kernel::<(
            &mut DeviceSlice<'_, f32>,
            &DeviceSlice<'_, f32>,
            usize,
        )>("not_kernel")
    }
    .map_err(|e| CudaError::Kernel(format!("not_kernel lookup failed: {}", e)))?;

    kernel
        .launch(
            ctx.garboard_stream(),
            &LaunchConfig::for_num_elems(n as u32),
            (output.data_f32_mut()?, a.data_f32()?, n),
        )
        .map_err(|e| CudaError::Kernel(format!("not launch failed: {}", e)))?;

    Ok(output)
}

/// Helper for binary comparison operations. Supports Float32, Int64, and
/// Int32 input tensors; always outputs Float32 (0.0 or 1.0).
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
            // SAFETY: f32 comparison kernels have signature
            // `(float* out, const float* a, const float* b, size_t n)`.
            let kernel = unsafe {
                cache.module().typed_kernel::<(
                    &mut DeviceSlice<'_, f32>,
                    &DeviceSlice<'_, f32>,
                    &DeviceSlice<'_, f32>,
                    usize,
                )>(kernel_name)
            }
            .map_err(|e| {
                CudaError::Kernel(format!("{} lookup failed: {}", kernel_name, e))
            })?;

            kernel
                .launch(
                    ctx.garboard_stream(),
                    &config,
                    (output.data_f32_mut()?, a.data_f32()?, b.data_f32()?, n),
                )
                .map_err(|e| {
                    CudaError::Kernel(format!("{} launch failed: {}", kernel_name, e))
                })?;
        }
        crate::cuda::tensor::DType::Int64 => {
            // Int64 kernels drop the trailing `_kernel` and append `_i64`
            // (e.g., `equal_kernel` → `equal_i64`).
            let i64_kernel_name = format!(
                "{}_i64",
                kernel_name.trim_end_matches("_kernel")
            );
            // SAFETY: i64 comparison kernels have signature
            // `(float* out, const long long* a, const long long* b, size_t n)`.
            let kernel = unsafe {
                cache.module().typed_kernel::<(
                    &mut DeviceSlice<'_, f32>,
                    &DeviceSlice<'_, i64>,
                    &DeviceSlice<'_, i64>,
                    usize,
                )>(&i64_kernel_name)
            }
            .map_err(|e| {
                CudaError::Kernel(format!("{} lookup failed: {}", i64_kernel_name, e))
            })?;

            kernel
                .launch(
                    ctx.garboard_stream(),
                    &config,
                    (output.data_f32_mut()?, a.data_i64()?, b.data_i64()?, n),
                )
                .map_err(|e| {
                    CudaError::Kernel(format!("{} launch failed: {}", i64_kernel_name, e))
                })?;
        }
        crate::cuda::tensor::DType::Int32 => {
            // Int32 kernels drop the trailing `_kernel` and append `_i32`.
            let i32_kernel_name = format!(
                "{}_i32",
                kernel_name.trim_end_matches("_kernel")
            );
            // SAFETY: i32 comparison kernels have signature
            // `(float* out, const int* a, const int* b, size_t n)`.
            let kernel = unsafe {
                cache.module().typed_kernel::<(
                    &mut DeviceSlice<'_, f32>,
                    &DeviceSlice<'_, i32>,
                    &DeviceSlice<'_, i32>,
                    usize,
                )>(&i32_kernel_name)
            }
            .map_err(|e| {
                CudaError::Kernel(format!("{} lookup failed: {}", i32_kernel_name, e))
            })?;

            kernel
                .launch(
                    ctx.garboard_stream(),
                    &config,
                    (output.data_f32_mut()?, a.data_i32()?, b.data_i32()?, n),
                )
                .map_err(|e| {
                    CudaError::Kernel(format!("{} launch failed: {}", i32_kernel_name, e))
                })?;
        }
        crate::cuda::tensor::DType::Int8 | crate::cuda::tensor::DType::UInt8 => {
            return Err(CudaError::Kernel(format!(
                "Comparison does not support {} (WS-4 — quantization graph should not route INT8/UINT8 through comparison ops)",
                dtype.name()
            )));
        }
        crate::cuda::tensor::DType::Float16 | crate::cuda::tensor::DType::Bool => {
            return Err(CudaError::Kernel(format!(
                "Comparison does not support {} (WS-3 M3.5 — Float16/Bool dispatch arm pending; M3.5 also migrates comparison output from f32 1.0/0.0 to native GpuTensor::Bool)",
                dtype.name()
            )));
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
        let a = GpuTensor::from_host_f32(&ctx, &[1.0f32, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
        let b = GpuTensor::from_host_f32(&ctx, &[1.0f32, 3.0, 2.0, 4.0, 6.0], vec![5]).unwrap();
        assert_eq!(
            gpu_equal(&ctx, &cache, &a, &b).unwrap().to_host_f32(&ctx).unwrap(),
            vec![1.0, 0.0, 0.0, 1.0, 0.0]
        );
        assert_eq!(
            gpu_less(&ctx, &cache, &a, &b).unwrap().to_host_f32(&ctx).unwrap(),
            vec![0.0, 1.0, 0.0, 0.0, 1.0]
        );
        assert_eq!(
            gpu_greater(&ctx, &cache, &a, &b).unwrap().to_host_f32(&ctx).unwrap(),
            vec![0.0, 0.0, 1.0, 0.0, 0.0]
        );
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_extended_comparison_ops() {
        let (ctx, cache) = setup();
        let a = GpuTensor::from_host_f32(&ctx, &[1.0f32, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
        let b = GpuTensor::from_host_f32(&ctx, &[1.0f32, 3.0, 2.0, 4.0, 6.0], vec![5]).unwrap();
        assert_eq!(
            gpu_greater_or_equal(&ctx, &cache, &a, &b).unwrap().to_host_f32(&ctx).unwrap(),
            vec![1.0, 0.0, 1.0, 1.0, 0.0]
        );
        assert_eq!(
            gpu_less_or_equal(&ctx, &cache, &a, &b).unwrap().to_host_f32(&ctx).unwrap(),
            vec![1.0, 1.0, 0.0, 1.0, 1.0]
        );
        assert_eq!(
            gpu_not_equal(&ctx, &cache, &a, &b).unwrap().to_host_f32(&ctx).unwrap(),
            vec![0.0, 1.0, 1.0, 0.0, 1.0]
        );
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_logical_ops() {
        let (ctx, cache) = setup();
        let a = GpuTensor::from_host_f32(&ctx, &[1.0f32, 0.0, 1.0, 0.0], vec![4]).unwrap();
        let b = GpuTensor::from_host_f32(&ctx, &[1.0f32, 1.0, 0.0, 0.0], vec![4]).unwrap();
        assert_eq!(
            gpu_and(&ctx, &cache, &a, &b).unwrap().to_host_f32(&ctx).unwrap(),
            vec![1.0, 0.0, 0.0, 0.0]
        );
        assert_eq!(
            gpu_or(&ctx, &cache, &a, &b).unwrap().to_host_f32(&ctx).unwrap(),
            vec![1.0, 1.0, 1.0, 0.0]
        );
        assert_eq!(
            gpu_not(&ctx, &cache, &a).unwrap().to_host_f32(&ctx).unwrap(),
            vec![0.0, 1.0, 0.0, 1.0]
        );
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_comparison_ops_int32() {
        let (ctx, cache) = setup();
        let a = GpuTensor::from_host_i32(&ctx, &[1i32, 2, 3, 4, 5], vec![5]).unwrap();
        let b = GpuTensor::from_host_i32(&ctx, &[1i32, 3, 2, 4, 6], vec![5]).unwrap();
        assert_eq!(
            gpu_equal(&ctx, &cache, &a, &b).unwrap().to_host_f32(&ctx).unwrap(),
            vec![1.0, 0.0, 0.0, 1.0, 0.0]
        );
        assert_eq!(
            gpu_less(&ctx, &cache, &a, &b).unwrap().to_host_f32(&ctx).unwrap(),
            vec![0.0, 1.0, 0.0, 0.0, 1.0]
        );
        assert_eq!(
            gpu_greater(&ctx, &cache, &a, &b).unwrap().to_host_f32(&ctx).unwrap(),
            vec![0.0, 0.0, 1.0, 0.0, 0.0]
        );
    }
}
