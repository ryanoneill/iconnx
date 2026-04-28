//! Sequence and math operations for GPU tensors (garboard TypedKernel).

use super::cache::OpsKernelCache;
use crate::cuda::context::{CudaError, IconnxCudaContext};
use crate::cuda::tensor::GpuTensor;
use garboard::{DeviceSlice, LaunchConfig};

/// GPU Atan: out = atan(inp).
pub fn gpu_atan(
    ctx: &IconnxCudaContext,
    cache: &OpsKernelCache,
    input: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    let n = input.len();
    let mut output = GpuTensor::zeros_f32(ctx, input.shape().to_vec())?;

    // SAFETY: atan_kernel signature is `(float* out, const float* in, size_t n)`.
    let kernel = unsafe {
        cache.module().typed_kernel::<(
            &mut DeviceSlice<'_, f32>,
            &DeviceSlice<'_, f32>,
            usize,
        )>("atan_kernel")
    }
    .map_err(|e| CudaError::Kernel(format!("atan_kernel lookup failed: {}", e)))?;

    kernel
        .launch(
            ctx.garboard_stream(),
            &LaunchConfig::for_num_elems(n as u32),
            (output.data_f32_mut()?, input.data_f32()?, n),
        )
        .map_err(|e| CudaError::Kernel(format!("atan launch failed: {}", e)))?;

    Ok(output)
}

/// GPU Range: generate sequence `[start, start+delta, ...]` of length `n`.
pub fn gpu_range(
    ctx: &IconnxCudaContext,
    cache: &OpsKernelCache,
    start: f32,
    delta: f32,
    n: usize,
) -> Result<GpuTensor, CudaError> {
    let mut output = GpuTensor::zeros_f32(ctx, vec![n])?;

    // SAFETY: range_kernel signature is
    // `(float* out, float start, float delta, size_t n)`.
    let kernel = unsafe {
        cache.module().typed_kernel::<(
            &mut DeviceSlice<'_, f32>,
            f32,
            f32,
            usize,
        )>("range_kernel")
    }
    .map_err(|e| CudaError::Kernel(format!("range_kernel lookup failed: {}", e)))?;

    kernel
        .launch(
            ctx.garboard_stream(),
            &LaunchConfig::for_num_elems(n as u32),
            (output.data_f32_mut()?, start, delta, n),
        )
        .map_err(|e| CudaError::Kernel(format!("range launch failed: {}", e)))?;

    Ok(output)
}

/// GPU Range Int64.
pub fn gpu_range_i64(
    ctx: &IconnxCudaContext,
    cache: &OpsKernelCache,
    start: i64,
    delta: i64,
    n: usize,
) -> Result<GpuTensor, CudaError> {
    let mut output = GpuTensor::zeros_i64(ctx, vec![n])?;

    // SAFETY: range_i64_kernel signature is
    // `(long long* out, long long start, long long delta, size_t n)`.
    let kernel = unsafe {
        cache.module().typed_kernel::<(
            &mut DeviceSlice<'_, i64>,
            i64,
            i64,
            usize,
        )>("range_i64_kernel")
    }
    .map_err(|e| CudaError::Kernel(format!("range_i64_kernel lookup failed: {}", e)))?;

    kernel
        .launch(
            ctx.garboard_stream(),
            &LaunchConfig::for_num_elems(n as u32),
            (output.data_i64_mut()?, start, delta, n),
        )
        .map_err(|e| CudaError::Kernel(format!("range_i64 launch failed: {}", e)))?;

    Ok(output)
}

/// GPU CumSum: cumulative sum along a specified axis. Supports Float32,
/// Int64, and Int32 tensors.
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

    // Handle negative axis.
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

    // Compute dimensions for strided access.
    let axis_len = shape[axis_idx];
    let inner_stride: usize = shape[axis_idx + 1..].iter().product::<usize>().max(1);
    let outer_size: usize = shape[..axis_idx].iter().product::<usize>().max(1);
    let num_lines = outer_size * inner_stride;

    let config = LaunchConfig::for_num_elems(num_lines as u32);
    let exclusive_i = if exclusive { 1i32 } else { 0i32 };
    let reverse_i = if reverse { 1i32 } else { 0i32 };

    match dtype {
        crate::cuda::tensor::DType::Float32 => {
            let mut output = GpuTensor::zeros_f32(ctx, shape.to_vec())?;
            // SAFETY: cumsum_kernel signature is
            // `(float* out, const float* in, size_t num_lines, size_t axis_len,
            //   size_t inner_stride, int exclusive, int reverse)`.
            let kernel = unsafe {
                cache.module().typed_kernel::<(
                    &mut DeviceSlice<'_, f32>,
                    &DeviceSlice<'_, f32>,
                    usize, usize, usize,
                    i32, i32,
                )>("cumsum_kernel")
            }
            .map_err(|e| {
                CudaError::Kernel(format!("cumsum_kernel lookup failed: {}", e))
            })?;

            kernel
                .launch(
                    ctx.garboard_stream(),
                    &config,
                    (
                        output.data_f32_mut()?,
                        input.data_f32()?,
                        num_lines,
                        axis_len,
                        inner_stride,
                        exclusive_i,
                        reverse_i,
                    ),
                )
                .map_err(|e| CudaError::Kernel(format!("cumsum launch failed: {}", e)))?;
            Ok(output)
        }
        crate::cuda::tensor::DType::Int64 => {
            let mut output = GpuTensor::zeros_i64(ctx, shape.to_vec())?;
            // SAFETY: cumsum_i64_kernel — i64 counterpart.
            let kernel = unsafe {
                cache.module().typed_kernel::<(
                    &mut DeviceSlice<'_, i64>,
                    &DeviceSlice<'_, i64>,
                    usize, usize, usize,
                    i32, i32,
                )>("cumsum_i64_kernel")
            }
            .map_err(|e| {
                CudaError::Kernel(format!("cumsum_i64_kernel lookup failed: {}", e))
            })?;

            kernel
                .launch(
                    ctx.garboard_stream(),
                    &config,
                    (
                        output.data_i64_mut()?,
                        input.data_i64()?,
                        num_lines,
                        axis_len,
                        inner_stride,
                        exclusive_i,
                        reverse_i,
                    ),
                )
                .map_err(|e| {
                    CudaError::Kernel(format!("cumsum_i64 launch failed: {}", e))
                })?;
            Ok(output)
        }
        crate::cuda::tensor::DType::Int32 => {
            let mut output = GpuTensor::zeros_i32(ctx, shape.to_vec())?;
            // SAFETY: cumsum_i32_kernel — i32 counterpart.
            let kernel = unsafe {
                cache.module().typed_kernel::<(
                    &mut DeviceSlice<'_, i32>,
                    &DeviceSlice<'_, i32>,
                    usize, usize, usize,
                    i32, i32,
                )>("cumsum_i32_kernel")
            }
            .map_err(|e| {
                CudaError::Kernel(format!("cumsum_i32_kernel lookup failed: {}", e))
            })?;

            kernel
                .launch(
                    ctx.garboard_stream(),
                    &config,
                    (
                        output.data_i32_mut()?,
                        input.data_i32()?,
                        num_lines,
                        axis_len,
                        inner_stride,
                        exclusive_i,
                        reverse_i,
                    ),
                )
                .map_err(|e| {
                    CudaError::Kernel(format!("cumsum_i32 launch failed: {}", e))
                })?;
            Ok(output)
        }
        crate::cuda::tensor::DType::Int8 | crate::cuda::tensor::DType::UInt8 => {
            Err(CudaError::Kernel(format!(
                "CumSum does not support {} (WS-4 — quantization graph should not route INT8/UINT8 through CumSum)",
                dtype.name()
            )))
        }
        crate::cuda::tensor::DType::Float16 | crate::cuda::tensor::DType::Bool => {
            Err(CudaError::Kernel(format!(
                "CumSum does not support {} (WS-3 M3.4 — Float16/Bool dispatch arm pending)",
                dtype.name()
            )))
        }
        crate::cuda::tensor::DType::BFloat16 => {
            // WS-3.5: BF16 is FP16-touched-surfaces only (spec §1 R3).
            // CumSum has no FP16 kernel today (the FP16/Bool arm above
            // returns the same kind of unsupported error), so BF16
            // inherits the same scope. A roster-driven BF16 CumSum
            // kernel would land alongside an FP16 CumSum kernel.
            Err(CudaError::UnsupportedDtype {
                op: "CumSum",
                dtype: "bfloat16",
                reason: "no FP16 CumSum kernel exists today; BF16 inherits the same scope per WS-3.5 spec §1 R3"
                    .to_string(),
            })
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
        let input = GpuTensor::from_host_f32(&ctx, &[0.0f32, 1.0, -1.0, 0.5], vec![4]).unwrap();
        let output = gpu_atan(&ctx, &cache, &input).unwrap();
        let result = output.to_host_f32(&ctx).unwrap();
        assert!((result[0] - 0.0).abs() < 1e-5);
        assert!((result[1] - std::f32::consts::FRAC_PI_4).abs() < 1e-5);
        assert!((result[2] - (-std::f32::consts::FRAC_PI_4)).abs() < 1e-5);
        assert!((result[3] - 0.4636476).abs() < 1e-5);
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_range() {
        let (ctx, cache) = setup();
        let output = gpu_range(&ctx, &cache, 0.0, 1.0, 5).unwrap();
        assert_eq!(output.to_host_f32(&ctx).unwrap(), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        let output2 = gpu_range(&ctx, &cache, 2.0, 0.5, 4).unwrap();
        assert_eq!(output2.to_host_f32(&ctx).unwrap(), vec![2.0, 2.5, 3.0, 3.5]);
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_cumsum() {
        let (ctx, cache) = setup();
        let input = GpuTensor::from_host_f32(&ctx, &[1.0f32, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
        assert_eq!(
            gpu_cumsum(&ctx, &cache, &input, 0, false, false).unwrap().to_host_f32(&ctx).unwrap(),
            vec![1.0, 3.0, 6.0, 10.0, 15.0]
        );
        assert_eq!(
            gpu_cumsum(&ctx, &cache, &input, 0, true, false).unwrap().to_host_f32(&ctx).unwrap(),
            vec![0.0, 1.0, 3.0, 6.0, 10.0]
        );
        assert_eq!(
            gpu_cumsum(&ctx, &cache, &input, 0, false, true).unwrap().to_host_f32(&ctx).unwrap(),
            vec![15.0, 14.0, 12.0, 9.0, 5.0]
        );
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_cumsum_axis() {
        let (ctx, cache) = setup();
        let input = GpuTensor::from_host_f32(&ctx, &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        assert_eq!(
            gpu_cumsum(&ctx, &cache, &input, 0, false, false).unwrap().to_host_f32(&ctx).unwrap(),
            vec![1.0, 2.0, 3.0, 5.0, 7.0, 9.0]
        );
        assert_eq!(
            gpu_cumsum(&ctx, &cache, &input, 1, false, false).unwrap().to_host_f32(&ctx).unwrap(),
            vec![1.0, 3.0, 6.0, 4.0, 9.0, 15.0]
        );
        assert_eq!(
            gpu_cumsum(&ctx, &cache, &input, -1, false, false).unwrap().to_host_f32(&ctx).unwrap(),
            vec![1.0, 3.0, 6.0, 4.0, 9.0, 15.0]
        );
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_cumsum_3d() {
        let (ctx, cache) = setup();
        let input = GpuTensor::from_host_f32(&ctx, &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]).unwrap();
        assert_eq!(
            gpu_cumsum(&ctx, &cache, &input, 1, false, false).unwrap().to_host_f32(&ctx).unwrap(),
            vec![1.0, 2.0, 4.0, 6.0, 9.0, 12.0]
        );
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_cumsum_i32() {
        let (ctx, cache) = setup();
        let input = GpuTensor::from_host_i32(&ctx, &[1i32, 2, 3, 4, 5], vec![5]).unwrap();
        assert_eq!(
            gpu_cumsum(&ctx, &cache, &input, 0, false, false).unwrap().to_host_i32(&ctx).unwrap(),
            vec![1, 3, 6, 10, 15]
        );
    }
}
