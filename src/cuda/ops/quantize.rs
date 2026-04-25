//! WS-4 quantization GPU ops.
//!
//! Today: `DequantizeLinear` (M4.4). Future siblings will land alongside
//! (`DynamicQuantizeLinear` — M4.5 — and downstream paths driven by
//! `MatMulInteger` — M4.6).

use super::cache::OpsKernelCache;
use crate::cuda::context::{CudaError, IconnxCudaContext};
use crate::cuda::memory_pool::GpuMemoryPool;
use crate::cuda::tensor::{DType, GpuTensor};
use garboard::{DeviceSlice, LaunchConfig};

/// GPU `DequantizeLinear`: `out = (x - zp) * scale`, output FP32.
///
/// `scale` must be FP32. `zp`, when supplied, must share dtype with `x`.
/// Per-tensor (scalar `scale`) is signalled by `axis_size = 1` and
/// `inner = n` so the per-element axis lookup collapses to index 0.
///
/// `zp = None` is materialized as a zero buffer of length `axis_size` in
/// `x`'s dtype before launch; the kernel never observes a null device
/// pointer. This trades one tiny allocation per call for a uniform kernel
/// signature across the present/absent cases — substantially simpler than
/// either threading optionality through the kernel ABI or maintaining
/// duplicate "no-zp" kernel variants.
pub fn gpu_dequantize_linear(
    ctx: &IconnxCudaContext,
    cache: &OpsKernelCache,
    pool: &mut GpuMemoryPool,
    x: &GpuTensor,
    scale: &GpuTensor,
    zp: Option<&GpuTensor>,
    axis: usize,
) -> Result<GpuTensor, CudaError> {
    let shape: Vec<usize> = x.shape().to_vec();
    let n = x.len();

    // Detect per-tensor (scalar scale or single-element 1-D) vs per-axis.
    let scale_shape = scale.shape();
    let scale_len = scale.len();
    let per_tensor = scale_shape.is_empty() || (scale_shape.len() == 1 && scale_len == 1);

    // Reduce per-element axis lookup to index 0 in the per-tensor case by
    // setting axis_size = 1 and inner = n. Otherwise compute the strides
    // from the input shape.
    let (axis_size, inner): (usize, usize) = if per_tensor {
        (1usize, n.max(1))
    } else {
        let ndim = shape.len();
        if axis >= ndim {
            return Err(CudaError::Kernel(format!(
                "DequantizeLinear: axis {} out of range for ndim {}",
                axis, ndim
            )));
        }
        let axis_size = shape[axis];
        if scale_len != axis_size {
            return Err(CudaError::Kernel(format!(
                "DequantizeLinear: per-axis scale length {} doesn't match \
                 input shape[{}] = {}",
                scale_len, axis, axis_size
            )));
        }
        let inner: usize = shape[axis + 1..].iter().product::<usize>().max(1);
        (axis_size, inner)
    };

    // `scale` must be FP32 by spec.
    if scale.dtype() != DType::Float32 {
        return Err(CudaError::Kernel(format!(
            "DequantizeLinear: scale must be Float32, got {}",
            scale.dtype().name()
        )));
    }

    // `zp` must share dtype with `x` when supplied.
    if let Some(zp_t) = zp {
        if zp_t.dtype() != x.dtype() {
            return Err(CudaError::Kernel(format!(
                "DequantizeLinear: zero_point dtype {} must match input \
                 dtype {}",
                zp_t.dtype().name(),
                x.dtype().name()
            )));
        }
        // Per-axis: zp shape must match scale shape (or be scalar).
        let zp_shape = zp_t.shape();
        let zp_per_tensor =
            zp_shape.is_empty() || (zp_shape.len() == 1 && zp_t.len() == 1);
        if !per_tensor && !zp_per_tensor && zp_t.len() != axis_size {
            return Err(CudaError::Kernel(format!(
                "DequantizeLinear: per-axis zero_point length {} doesn't \
                 match axis_size {}",
                zp_t.len(),
                axis_size
            )));
        }
    }

    let mut output = pool.get_tensor_f32(ctx, shape)?;
    let config = LaunchConfig::for_num_elems(n as u32);

    match x.dtype() {
        DType::Int8 => {
            // Synthesize a zeros zp on demand so the kernel signature is
            // uniform. Length = axis_size; per-tensor falls through with
            // axis_size = 1.
            let zp_owned: Option<GpuTensor> = match zp {
                Some(t) => {
                    // Broadcast a scalar zp to length `axis_size` if needed.
                    if !per_tensor && t.len() == 1 {
                        let host = t.to_host_i8(ctx)?;
                        let buf = vec![host[0]; axis_size];
                        Some(GpuTensor::from_host_i8(ctx, &buf, vec![axis_size])?)
                    } else {
                        None
                    }
                }
                None => Some(GpuTensor::zeros_i8(ctx, vec![axis_size])?),
            };
            let zp_ref: &GpuTensor = zp_owned.as_ref().unwrap_or_else(|| zp.unwrap());
            // SAFETY: signature matches `dequantize_linear_i8_kernel`.
            let kernel = unsafe {
                cache.module().typed_kernel::<(
                    &mut DeviceSlice<'_, f32>,
                    &DeviceSlice<'_, i8>,
                    &DeviceSlice<'_, f32>,
                    &DeviceSlice<'_, i8>,
                    usize,
                    usize,
                    usize,
                )>("dequantize_linear_i8_kernel")
            }
            .map_err(|e| {
                CudaError::Kernel(format!("dequantize_linear_i8_kernel lookup: {}", e))
            })?;
            kernel
                .launch(
                    ctx.garboard_stream(),
                    &config,
                    (
                        output.data_f32_mut()?,
                        x.data_i8()?,
                        scale.data_f32()?,
                        zp_ref.data_i8()?,
                        n,
                        axis_size,
                        inner,
                    ),
                )
                .map_err(|e| {
                    CudaError::Kernel(format!("dequantize_linear_i8_kernel launch: {}", e))
                })?;
            Ok(output)
        }
        DType::UInt8 => {
            let zp_owned: Option<GpuTensor> = match zp {
                Some(t) => {
                    if !per_tensor && t.len() == 1 {
                        let host = t.to_host_u8(ctx)?;
                        let buf = vec![host[0]; axis_size];
                        Some(GpuTensor::from_host_u8(ctx, &buf, vec![axis_size])?)
                    } else {
                        None
                    }
                }
                None => Some(GpuTensor::zeros_u8(ctx, vec![axis_size])?),
            };
            let zp_ref: &GpuTensor = zp_owned.as_ref().unwrap_or_else(|| zp.unwrap());
            // SAFETY: signature matches `dequantize_linear_u8_kernel`.
            let kernel = unsafe {
                cache.module().typed_kernel::<(
                    &mut DeviceSlice<'_, f32>,
                    &DeviceSlice<'_, u8>,
                    &DeviceSlice<'_, f32>,
                    &DeviceSlice<'_, u8>,
                    usize,
                    usize,
                    usize,
                )>("dequantize_linear_u8_kernel")
            }
            .map_err(|e| {
                CudaError::Kernel(format!("dequantize_linear_u8_kernel lookup: {}", e))
            })?;
            kernel
                .launch(
                    ctx.garboard_stream(),
                    &config,
                    (
                        output.data_f32_mut()?,
                        x.data_u8()?,
                        scale.data_f32()?,
                        zp_ref.data_u8()?,
                        n,
                        axis_size,
                        inner,
                    ),
                )
                .map_err(|e| {
                    CudaError::Kernel(format!("dequantize_linear_u8_kernel launch: {}", e))
                })?;
            Ok(output)
        }
        DType::Int32 => {
            let zp_owned: Option<GpuTensor> = match zp {
                Some(t) => {
                    if !per_tensor && t.len() == 1 {
                        let host = ctx.dtoh(t.data_i32()?)?;
                        // dtoh may return a bucket-sized vec; keep first elem.
                        let buf = vec![host[0]; axis_size];
                        Some(GpuTensor::from_host_i32(ctx, &buf, vec![axis_size])?)
                    } else {
                        None
                    }
                }
                None => {
                    let zeros = ctx.alloc_zeros::<i32>(axis_size)?;
                    Some(GpuTensor::new_i32(zeros, vec![axis_size]))
                }
            };
            let zp_ref: &GpuTensor = zp_owned.as_ref().unwrap_or_else(|| zp.unwrap());
            // SAFETY: signature matches `dequantize_linear_i32_kernel`.
            let kernel = unsafe {
                cache.module().typed_kernel::<(
                    &mut DeviceSlice<'_, f32>,
                    &DeviceSlice<'_, i32>,
                    &DeviceSlice<'_, f32>,
                    &DeviceSlice<'_, i32>,
                    usize,
                    usize,
                    usize,
                )>("dequantize_linear_i32_kernel")
            }
            .map_err(|e| {
                CudaError::Kernel(format!("dequantize_linear_i32_kernel lookup: {}", e))
            })?;
            kernel
                .launch(
                    ctx.garboard_stream(),
                    &config,
                    (
                        output.data_f32_mut()?,
                        x.data_i32()?,
                        scale.data_f32()?,
                        zp_ref.data_i32()?,
                        n,
                        axis_size,
                        inner,
                    ),
                )
                .map_err(|e| {
                    CudaError::Kernel(format!("dequantize_linear_i32_kernel launch: {}", e))
                })?;
            Ok(output)
        }
        other => Err(CudaError::Kernel(format!(
            "DequantizeLinear: input dtype {} unsupported (must be Int8, \
             UInt8, or Int32)",
            other.name()
        ))),
    }
}
