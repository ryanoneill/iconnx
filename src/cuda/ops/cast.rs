//! Type casting operations for GPU tensors (garboard TypedKernel).

use super::cache::OpsKernelCache;
use crate::cuda::context::{CudaError, IconnxCudaContext};
use crate::cuda::tensor::{DType, GpuTensor};
use garboard::{DeviceSlice, LaunchConfig};

/// Launch a generic `(out, in, n)` cast kernel with typed slice
/// parameters `Out` and `In`.
///
/// # Safety
/// The kernel named `name` must have the C signature
/// `(Out* out, const In* in, size_t n)`. The caller promises this by
/// supplying type parameters that match the Rust side of the cast.
unsafe fn launch_cast_kernel<Out, In>(
    ctx: &IconnxCudaContext,
    cache: &OpsKernelCache,
    name: &'static str,
    out: &mut DeviceSlice<'static, Out>,
    inp: &DeviceSlice<'static, In>,
    n: usize,
) -> Result<(), CudaError>
where
    Out: bytemuck::Pod,
    In: bytemuck::Pod,
{
    // SAFETY: see function contract.
    let kernel = unsafe {
        cache.module().typed_kernel::<(
            &mut DeviceSlice<'_, Out>,
            &DeviceSlice<'_, In>,
            usize,
        )>(name)
    }
    .map_err(|e| CudaError::Kernel(format!("{} lookup failed: {}", name, e)))?;

    kernel
        .launch(
            ctx.garboard_stream(),
            &LaunchConfig::for_num_elems(n as u32),
            (out, inp, n),
        )
        .map_err(|e| CudaError::Kernel(format!("{} launch failed: {}", name, e)))
}

/// GPU Cast: convert tensor from one dtype to another.
///
/// Supports conversions between Float32, Int64, and Int32. INT8/UINT8 are
/// not handled here — those cross the float/int boundary via
/// `DequantizeLinear` (M4.4) and `DynamicQuantizeLinear` (M4.5); plain
/// Cast on those dtypes returns `CudaError::Kernel`.
pub fn gpu_cast(
    ctx: &IconnxCudaContext,
    cache: &OpsKernelCache,
    input: &GpuTensor,
    target_dtype: DType,
) -> Result<GpuTensor, CudaError> {
    let src_dtype = input.dtype();

    // Same type: return a cheap Arc clone (no D2H + H2D round trip).
    if src_dtype == target_dtype {
        return Ok(input.clone());
    }

    let n = input.len();
    let shape = input.shape().to_vec();

    match (src_dtype, target_dtype) {
        (DType::Float32, DType::Int64) => {
            let mut output = GpuTensor::zeros_i64(ctx, shape)?;
            // SAFETY: cast_f32_to_i64_kernel has C signature
            // `(long long* out, const float* in, size_t n)`.
            unsafe {
                launch_cast_kernel::<i64, f32>(
                    ctx,
                    cache,
                    "cast_f32_to_i64_kernel",
                    output.data_i64_mut()?,
                    input.data_f32()?,
                    n,
                )?;
            }
            Ok(output)
        }

        (DType::Float32, DType::Int32) => {
            let mut output = GpuTensor::zeros_i32(ctx, shape)?;
            // SAFETY: `(int* out, const float* in, size_t n)`.
            unsafe {
                launch_cast_kernel::<i32, f32>(
                    ctx,
                    cache,
                    "cast_f32_to_i32_kernel",
                    output.data_i32_mut()?,
                    input.data_f32()?,
                    n,
                )?;
            }
            Ok(output)
        }

        (DType::Int64, DType::Float32) => {
            let mut output = GpuTensor::zeros_f32(ctx, shape)?;
            // SAFETY: `(float* out, const long long* in, size_t n)`.
            unsafe {
                launch_cast_kernel::<f32, i64>(
                    ctx,
                    cache,
                    "cast_i64_to_f32_kernel",
                    output.data_f32_mut()?,
                    input.data_i64()?,
                    n,
                )?;
            }
            Ok(output)
        }

        (DType::Int64, DType::Int32) => {
            let mut output = GpuTensor::zeros_i32(ctx, shape)?;
            // SAFETY: `(int* out, const long long* in, size_t n)`.
            unsafe {
                launch_cast_kernel::<i32, i64>(
                    ctx,
                    cache,
                    "cast_i64_to_i32_kernel",
                    output.data_i32_mut()?,
                    input.data_i64()?,
                    n,
                )?;
            }
            Ok(output)
        }

        (DType::Int32, DType::Float32) => {
            let mut output = GpuTensor::zeros_f32(ctx, shape)?;
            // SAFETY: `(float* out, const int* in, size_t n)`.
            unsafe {
                launch_cast_kernel::<f32, i32>(
                    ctx,
                    cache,
                    "cast_i32_to_f32_kernel",
                    output.data_f32_mut()?,
                    input.data_i32()?,
                    n,
                )?;
            }
            Ok(output)
        }

        (DType::Int32, DType::Int64) => {
            let mut output = GpuTensor::zeros_i64(ctx, shape)?;
            // SAFETY: `(long long* out, const int* in, size_t n)`.
            unsafe {
                launch_cast_kernel::<i64, i32>(
                    ctx,
                    cache,
                    "cast_i32_to_i64_kernel",
                    output.data_i64_mut()?,
                    input.data_i32()?,
                    n,
                )?;
            }
            Ok(output)
        }

        // Same-type cases already handled above by early return.
        (DType::Float32, DType::Float32)
        | (DType::Int64, DType::Int64)
        | (DType::Int32, DType::Int32)
        | (DType::Int8, DType::Int8)
        | (DType::UInt8, DType::UInt8)
        | (DType::Float16, DType::Float16)
        | (DType::Bool, DType::Bool) => unreachable!("Same-type cast handled above"),

        // WS-3 M3.5: FP16 boundary casts. Float16 ↔ Float32 is the
        // primary boundary for Whisper's mixed-precision attention;
        // Float16 ↔ Int32 / Int64 covers shape-arithmetic results being
        // mixed back into FP16 paths.
        (DType::Float16, DType::Float32) => {
            let mut output = GpuTensor::zeros_f32(ctx, shape)?;
            // SAFETY: cast_f16_to_f32_kernel: `(float* out, const __half* in, size_t n)`.
            unsafe {
                launch_cast_kernel::<f32, half::f16>(
                    ctx,
                    cache,
                    "cast_f16_to_f32_kernel",
                    output.data_f32_mut()?,
                    input.data_f16()?,
                    n,
                )?;
            }
            Ok(output)
        }
        (DType::Float32, DType::Float16) => {
            let mut output = GpuTensor::zeros_f16(ctx, shape)?;
            // SAFETY: cast_f32_to_f16_kernel: `(__half* out, const float* in, size_t n)`.
            unsafe {
                launch_cast_kernel::<half::f16, f32>(
                    ctx,
                    cache,
                    "cast_f32_to_f16_kernel",
                    output.data_f16_mut()?,
                    input.data_f32()?,
                    n,
                )?;
            }
            Ok(output)
        }
        (DType::Float16, DType::Int32) => {
            let mut output = GpuTensor::zeros_i32(ctx, shape)?;
            // SAFETY: cast_f16_to_i32_kernel: `(int* out, const __half* in, size_t n)`.
            unsafe {
                launch_cast_kernel::<i32, half::f16>(
                    ctx,
                    cache,
                    "cast_f16_to_i32_kernel",
                    output.data_i32_mut()?,
                    input.data_f16()?,
                    n,
                )?;
            }
            Ok(output)
        }
        (DType::Int32, DType::Float16) => {
            let mut output = GpuTensor::zeros_f16(ctx, shape)?;
            // SAFETY: cast_i32_to_f16_kernel: `(__half* out, const int* in, size_t n)`.
            unsafe {
                launch_cast_kernel::<half::f16, i32>(
                    ctx,
                    cache,
                    "cast_i32_to_f16_kernel",
                    output.data_f16_mut()?,
                    input.data_i32()?,
                    n,
                )?;
            }
            Ok(output)
        }
        (DType::Float16, DType::Int64) => {
            let mut output = GpuTensor::zeros_i64(ctx, shape)?;
            // SAFETY: cast_f16_to_i64_kernel: `(long long* out, const __half* in, size_t n)`.
            unsafe {
                launch_cast_kernel::<i64, half::f16>(
                    ctx,
                    cache,
                    "cast_f16_to_i64_kernel",
                    output.data_i64_mut()?,
                    input.data_f16()?,
                    n,
                )?;
            }
            Ok(output)
        }
        (DType::Int64, DType::Float16) => {
            let mut output = GpuTensor::zeros_f16(ctx, shape)?;
            // SAFETY: cast_i64_to_f16_kernel: `(__half* out, const long long* in, size_t n)`.
            unsafe {
                launch_cast_kernel::<half::f16, i64>(
                    ctx,
                    cache,
                    "cast_i64_to_f16_kernel",
                    output.data_f16_mut()?,
                    input.data_i64()?,
                    n,
                )?;
            }
            Ok(output)
        }

        // INT8/UINT8 casts: WS-4 quantization graph dequantizes via
        // DequantizeLinear (M4.5) and quantizes via DynamicQuantizeLinear
        // (M4.6). Plain Cast is not on that path.
        (DType::Int8, _) | (DType::UInt8, _) | (_, DType::Int8) | (_, DType::UInt8) => {
            Err(CudaError::Kernel(format!(
                "Cast does not support {} -> {} (WS-4 — INT8/UINT8 should use DequantizeLinear / DynamicQuantizeLinear, not Cast)",
                src_dtype.name(),
                target_dtype.name()
            )))
        }

        // WS-3 M3.5 sub-B: ONNX Cast(to=BOOL) produces native
        // GpuTensor::Bool. Reverse direction (Bool→numeric) emits 0/1.
        // Float16 ↔ Bool stays unsupported — no active graph routes
        // FP16 through a Bool cast and the round-trip would lose
        // information (every non-zero f16 → 1, then 1 → 1.0_f16).
        (DType::Float32, DType::Bool) => {
            let mut output = GpuTensor::zeros_bool(ctx, shape)?;
            // SAFETY: cast_f32_to_bool_kernel: `(unsigned char* out, const float* in, size_t n)`.
            unsafe {
                launch_cast_kernel::<u8, f32>(
                    ctx,
                    cache,
                    "cast_f32_to_bool_kernel",
                    output.data_bool_mut()?,
                    input.data_f32()?,
                    n,
                )?;
            }
            Ok(output)
        }
        (DType::Int32, DType::Bool) => {
            let mut output = GpuTensor::zeros_bool(ctx, shape)?;
            // SAFETY: cast_i32_to_bool_kernel: `(unsigned char* out, const int* in, size_t n)`.
            unsafe {
                launch_cast_kernel::<u8, i32>(
                    ctx,
                    cache,
                    "cast_i32_to_bool_kernel",
                    output.data_bool_mut()?,
                    input.data_i32()?,
                    n,
                )?;
            }
            Ok(output)
        }
        (DType::Int64, DType::Bool) => {
            let mut output = GpuTensor::zeros_bool(ctx, shape)?;
            // SAFETY: cast_i64_to_bool_kernel: `(unsigned char* out, const long long* in, size_t n)`.
            unsafe {
                launch_cast_kernel::<u8, i64>(
                    ctx,
                    cache,
                    "cast_i64_to_bool_kernel",
                    output.data_bool_mut()?,
                    input.data_i64()?,
                    n,
                )?;
            }
            Ok(output)
        }
        (DType::Bool, DType::Float32) => {
            let mut output = GpuTensor::zeros_f32(ctx, shape)?;
            // SAFETY: cast_bool_to_f32_kernel: `(float* out, const unsigned char* in, size_t n)`.
            unsafe {
                launch_cast_kernel::<f32, u8>(
                    ctx,
                    cache,
                    "cast_bool_to_f32_kernel",
                    output.data_f32_mut()?,
                    input.data_bool()?,
                    n,
                )?;
            }
            Ok(output)
        }
        (DType::Bool, DType::Int32) => {
            let mut output = GpuTensor::zeros_i32(ctx, shape)?;
            // SAFETY: cast_bool_to_i32_kernel: `(int* out, const unsigned char* in, size_t n)`.
            unsafe {
                launch_cast_kernel::<i32, u8>(
                    ctx,
                    cache,
                    "cast_bool_to_i32_kernel",
                    output.data_i32_mut()?,
                    input.data_bool()?,
                    n,
                )?;
            }
            Ok(output)
        }
        (DType::Bool, DType::Int64) => {
            let mut output = GpuTensor::zeros_i64(ctx, shape)?;
            // SAFETY: cast_bool_to_i64_kernel: `(long long* out, const unsigned char* in, size_t n)`.
            unsafe {
                launch_cast_kernel::<i64, u8>(
                    ctx,
                    cache,
                    "cast_bool_to_i64_kernel",
                    output.data_i64_mut()?,
                    input.data_bool()?,
                    n,
                )?;
            }
            Ok(output)
        }

        // Float16 ↔ Bool: no active graph routes FP16 through a Bool
        // cast; round-trip would lose information (every non-zero f16 → 1,
        // then 1 → 1.0_f16). Stay loud rather than silently truncating.
        (DType::Float16, DType::Bool) | (DType::Bool, DType::Float16) => {
            Err(CudaError::Kernel(format!(
                "Cast does not support {} -> {} (no active model graph requires this; would lose information)",
                src_dtype.name(),
                target_dtype.name()
            )))
        }

        // WS-3.5 Y(2) sub-2d will replace this catchall arm with 10
        // explicit dtype-pair arms (FP32↔BF16, FP16↔BF16, Int32↔BF16,
        // Int64↔BF16) plus an explicit-unsupported (Bool, BFloat16) arm.
        // Same-type `(DType::BFloat16, DType::BFloat16)` is already
        // handled by the early-return at the top of `gpu_cast`.
        (DType::BFloat16, _) | (_, DType::BFloat16) => Err(CudaError::UnsupportedDtype {
            op: "Cast",
            dtype: "bfloat16",
            reason: format!(
                "BF16 Cast pair {} -> {} will land in Y(2) sub-2d (10 explicit dtype-pair arms + (Bool, BFloat16) unsupported)",
                src_dtype.name(),
                target_dtype.name()
            ),
        }),
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
        assert_eq!(result, vec![2i32, -2, 4]);
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
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_cast_same_type() {
        let (ctx, cache) = setup();
        let data = vec![1.0f32, 2.0, 3.0];
        let input = GpuTensor::from_host_f32(&ctx, &data, vec![3]).unwrap();
        let output = gpu_cast(&ctx, &cache, &input, DType::Float32).unwrap();
        assert_eq!(output.dtype(), DType::Float32);
        let result = output.to_host_f32(&ctx).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_cast_chain() {
        let (ctx, cache) = setup();
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
    }
}
