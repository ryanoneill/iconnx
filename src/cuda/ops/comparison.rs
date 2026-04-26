//! Comparison and logical operations for GPU tensors (garboard TypedKernel).
//!
//! WS-3 M3.5 sub-B: comparison and logical ops produce native
//! [`GpuTensor::Bool`] (a `DeviceSlice<u8>` of `0`/`1` bytes) rather than
//! `Float32` `0.0`/`1.0`. Logical ops also consume Bool inputs end-to-end.
//! The kernels' C-side output type is `unsigned char*`, matching `u8`.

use super::cache::OpsKernelCache;
use crate::cuda::context::{CudaError, IconnxCudaContext};
use crate::cuda::tensor::GpuTensor;
use garboard::{DeviceSlice, LaunchConfig};

/// GPU Equal: out = (a == b) ? true : false.
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

/// GPU And: out = (a && b). Both inputs and output are native Bool.
pub fn gpu_and(
    ctx: &IconnxCudaContext,
    cache: &OpsKernelCache,
    a: &GpuTensor,
    b: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    binary_logical(ctx, cache, a, b, "and_kernel")
}

/// GPU Or: out = (a || b). Both inputs and output are native Bool.
pub fn gpu_or(
    ctx: &IconnxCudaContext,
    cache: &OpsKernelCache,
    a: &GpuTensor,
    b: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    binary_logical(ctx, cache, a, b, "or_kernel")
}

/// GPU Not: out = !a, both as native Bool (`GpuTensor::Bool`).
pub fn gpu_not(
    ctx: &IconnxCudaContext,
    cache: &OpsKernelCache,
    a: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    if a.dtype() != crate::cuda::tensor::DType::Bool {
        return Err(CudaError::Kernel(format!(
            "Not: input must be Bool, got {} (WS-3 M3.5 — comparison/logical chain operates on native Bool)",
            a.dtype().name()
        )));
    }
    let n = a.len();
    let mut output = GpuTensor::zeros_bool(ctx, a.shape().to_vec())?;

    // SAFETY: not_kernel signature is `(unsigned char* out, const unsigned char* a, size_t n)`.
    let kernel = unsafe {
        cache.module().typed_kernel::<(
            &mut DeviceSlice<'_, u8>,
            &DeviceSlice<'_, u8>,
            usize,
        )>("not_kernel")
    }
    .map_err(|e| CudaError::Kernel(format!("not_kernel lookup failed: {}", e)))?;

    kernel
        .launch(
            ctx.garboard_stream(),
            &LaunchConfig::for_num_elems(n as u32),
            (output.data_bool_mut()?, a.data_bool()?, n),
        )
        .map_err(|e| CudaError::Kernel(format!("not launch failed: {}", e)))?;

    Ok(output)
}

/// Binary logical (And/Or) shared launch helper: both operands and the
/// output are Bool. The C-side kernel signature is
/// `(unsigned char* out, const unsigned char* a, const unsigned char* b, size_t n)`.
fn binary_logical(
    ctx: &IconnxCudaContext,
    cache: &OpsKernelCache,
    a: &GpuTensor,
    b: &GpuTensor,
    kernel_name: &'static str,
) -> Result<GpuTensor, CudaError> {
    let n = a.len();
    assert_eq!(b.len(), n, "logical: b length mismatch");
    if a.dtype() != crate::cuda::tensor::DType::Bool || b.dtype() != crate::cuda::tensor::DType::Bool {
        return Err(CudaError::Kernel(format!(
            "{}: both inputs must be Bool, got {} / {} (WS-3 M3.5)",
            kernel_name,
            a.dtype().name(),
            b.dtype().name()
        )));
    }

    let mut output = GpuTensor::zeros_bool(ctx, a.shape().to_vec())?;
    let config = LaunchConfig::for_num_elems(n as u32);

    let kernel = unsafe {
        cache.module().typed_kernel::<(
            &mut DeviceSlice<'_, u8>,
            &DeviceSlice<'_, u8>,
            &DeviceSlice<'_, u8>,
            usize,
        )>(kernel_name)
    }
    .map_err(|e| CudaError::Kernel(format!("{} lookup failed: {}", kernel_name, e)))?;

    kernel
        .launch(
            ctx.garboard_stream(),
            &config,
            (
                output.data_bool_mut()?,
                a.data_bool()?,
                b.data_bool()?,
                n,
            ),
        )
        .map_err(|e| CudaError::Kernel(format!("{} launch failed: {}", kernel_name, e)))?;

    Ok(output)
}

/// Helper for binary comparison operations. Supports Float32, Int64, and
/// Int32 input tensors; always outputs `GpuTensor::Bool` (`u8` 1/0 bytes).
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

    let mut output = GpuTensor::zeros_bool(ctx, a.shape().to_vec())?;
    let config = LaunchConfig::for_num_elems(n as u32);

    match dtype {
        crate::cuda::tensor::DType::Float32 => {
            // SAFETY: f32 comparison kernels have signature
            // `(unsigned char* out, const float* a, const float* b, size_t n)`.
            let kernel = unsafe {
                cache.module().typed_kernel::<(
                    &mut DeviceSlice<'_, u8>,
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
                    (output.data_bool_mut()?, a.data_f32()?, b.data_f32()?, n),
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
            // `(unsigned char* out, const long long* a, const long long* b, size_t n)`.
            let kernel = unsafe {
                cache.module().typed_kernel::<(
                    &mut DeviceSlice<'_, u8>,
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
                    (output.data_bool_mut()?, a.data_i64()?, b.data_i64()?, n),
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
            // `(unsigned char* out, const int* a, const int* b, size_t n)`.
            let kernel = unsafe {
                cache.module().typed_kernel::<(
                    &mut DeviceSlice<'_, u8>,
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
                    (output.data_bool_mut()?, a.data_i32()?, b.data_i32()?, n),
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
                "Comparison does not support {} (Whisper-FP16 doesn't compare FP16 directly; Bool inputs route through And/Or/Not, not the binary comparison helper)",
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

    fn bool_to_vec(t: &GpuTensor, ctx: &IconnxCudaContext) -> Vec<bool> {
        t.to_host_bool(ctx).unwrap()
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_comparison_ops() {
        let (ctx, cache) = setup();
        let a = GpuTensor::from_host_f32(&ctx, &[1.0f32, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
        let b = GpuTensor::from_host_f32(&ctx, &[1.0f32, 3.0, 2.0, 4.0, 6.0], vec![5]).unwrap();

        let eq = gpu_equal(&ctx, &cache, &a, &b).unwrap();
        assert_eq!(eq.dtype(), crate::cuda::tensor::DType::Bool);
        assert_eq!(bool_to_vec(&eq, &ctx), vec![true, false, false, true, false]);

        let lt = gpu_less(&ctx, &cache, &a, &b).unwrap();
        assert_eq!(lt.dtype(), crate::cuda::tensor::DType::Bool);
        assert_eq!(bool_to_vec(&lt, &ctx), vec![false, true, false, false, true]);

        let gt = gpu_greater(&ctx, &cache, &a, &b).unwrap();
        assert_eq!(bool_to_vec(&gt, &ctx), vec![false, false, true, false, false]);
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_extended_comparison_ops() {
        let (ctx, cache) = setup();
        let a = GpuTensor::from_host_f32(&ctx, &[1.0f32, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
        let b = GpuTensor::from_host_f32(&ctx, &[1.0f32, 3.0, 2.0, 4.0, 6.0], vec![5]).unwrap();
        assert_eq!(
            bool_to_vec(&gpu_greater_or_equal(&ctx, &cache, &a, &b).unwrap(), &ctx),
            vec![true, false, true, true, false]
        );
        assert_eq!(
            bool_to_vec(&gpu_less_or_equal(&ctx, &cache, &a, &b).unwrap(), &ctx),
            vec![true, true, false, true, true]
        );
        assert_eq!(
            bool_to_vec(&gpu_not_equal(&ctx, &cache, &a, &b).unwrap(), &ctx),
            vec![false, true, true, false, true]
        );
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_logical_ops() {
        let (ctx, cache) = setup();
        let a = GpuTensor::from_host_bool(&ctx, &[true, false, true, false], vec![4]).unwrap();
        let b = GpuTensor::from_host_bool(&ctx, &[true, true, false, false], vec![4]).unwrap();
        assert_eq!(
            bool_to_vec(&gpu_and(&ctx, &cache, &a, &b).unwrap(), &ctx),
            vec![true, false, false, false]
        );
        assert_eq!(
            bool_to_vec(&gpu_or(&ctx, &cache, &a, &b).unwrap(), &ctx),
            vec![true, true, true, false]
        );
        assert_eq!(
            bool_to_vec(&gpu_not(&ctx, &cache, &a).unwrap(), &ctx),
            vec![false, true, false, true]
        );
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_comparison_ops_int32() {
        let (ctx, cache) = setup();
        let a = GpuTensor::from_host_i32(&ctx, &[1i32, 2, 3, 4, 5], vec![5]).unwrap();
        let b = GpuTensor::from_host_i32(&ctx, &[1i32, 3, 2, 4, 6], vec![5]).unwrap();
        assert_eq!(
            bool_to_vec(&gpu_equal(&ctx, &cache, &a, &b).unwrap(), &ctx),
            vec![true, false, false, true, false]
        );
        assert_eq!(
            bool_to_vec(&gpu_less(&ctx, &cache, &a, &b).unwrap(), &ctx),
            vec![false, true, false, false, true]
        );
        assert_eq!(
            bool_to_vec(&gpu_greater(&ctx, &cache, &a, &b).unwrap(), &ctx),
            vec![false, false, true, false, false]
        );
    }
}
