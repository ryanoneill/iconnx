//! Host-side wrappers for fused GPU kernels
//!
//! This file holds the `gpu_fused_*` host wrapper functions. Each wrapper:
//! - Validates tensor shapes (handling any supported broadcast patterns).
//! - Allocates an output tensor from the `GpuMemoryPool`.
//! - Looks up the compiled kernel in `FusedKernelCache` and launches it.
//!
//! The shared `elementwise_config` and `compute_broadcast_stride` helpers
//! live in the parent `mod.rs`. All wrappers are re-exported at
//! `crate::cuda::kernels::fused::*` via `pub use wrappers::*` in `mod.rs`.

use super::{compute_broadcast_stride, elementwise_config, FusedKernelCache};
use crate::cuda::context::{CudaError, IconnxCudaContext};
use crate::cuda::memory_pool::GpuMemoryPool;
use crate::cuda::tensor::GpuTensor;
use cudarc::driver::PushKernelArg;

/// Returns true iff `w` is a valid per-channel broadcast of `x`:
/// all leading dims are 1, and `w.shape().last()` equals the last axis of `x`.
///
/// This is the only broadcast contract supported by
/// `gpu_fused_mul_sin_pow_mul_add`: `w` must have shape `[C]`, `[1, C]`,
/// `[1, 1, C]`, and so on, where `C` matches `x`'s last axis size.
fn is_per_channel_broadcast(x: &GpuTensor, w: &GpuTensor) -> bool {
    let x_shape = x.shape();
    let w_shape = w.shape();
    let channels = match x_shape.last() {
        Some(&c) if c > 0 => c,
        _ => return false,
    };
    if w_shape.last() != Some(&channels) {
        return false;
    }
    // All dims except the last must be 1.
    w_shape
        .iter()
        .take(w_shape.len().saturating_sub(1))
        .all(|&d| d == 1)
}

// ============================================================================
// Fused operation implementations
// ============================================================================

/// GPU fused div_rsqrt: out = numerator / sqrt(variance + eps)
/// Replaces Add -> Sqrt -> Div pattern
/// Supports broadcasting when variance has trailing 1 dimensions
pub fn gpu_fused_div_rsqrt(
    ctx: &IconnxCudaContext,
    cache: &FusedKernelCache,
    pool: &mut GpuMemoryPool,
    numerator: &GpuTensor,
    variance: &GpuTensor,
    eps: f32,
) -> Result<GpuTensor, CudaError> {
    // Check if shapes match or if broadcasting is possible
    let num_shape = numerator.shape();
    let var_shape = variance.shape();

    // Simple case: shapes match
    if num_shape == var_shape {
        let n = numerator.len();
        let mut out = pool.get_tensor_f32(ctx, numerator.shape().to_vec())?;

        let func = cache
            .get("fused_div_rsqrt_scalar_eps_kernel")
            .ok_or_else(|| {
                CudaError::Kernel("fused_div_rsqrt_scalar_eps_kernel not found".into())
            })?;

        unsafe {
            ctx.stream()
                .launch_builder(func)
                .arg(out.data_f32_mut()?)
                .arg(numerator.data_f32()?)
                .arg(variance.data_f32()?)
                .arg(&eps)
                .arg(&n)
                .launch(elementwise_config(n))
                .map_err(|e| CudaError::Kernel(format!("fused_div_rsqrt launch failed: {}", e)))?;
        }

        return Ok(out);
    }

    // Broadcasting case: check if variance can be broadcast to numerator shape
    // Supported pattern: variance has trailing 1 dimensions that broadcast to numerator
    // e.g., numerator [1, 512, 5], variance [1, 512, 1] -> broadcast last dim
    if let Some(variance_stride) = compute_broadcast_stride(num_shape, var_shape) {
        let n = numerator.len();
        let mut out = pool.get_tensor_f32(ctx, numerator.shape().to_vec())?;

        let func = cache
            .get("fused_div_rsqrt_broadcast_kernel")
            .ok_or_else(|| {
                CudaError::Kernel("fused_div_rsqrt_broadcast_kernel not found".into())
            })?;

        unsafe {
            ctx.stream()
                .launch_builder(func)
                .arg(out.data_f32_mut()?)
                .arg(numerator.data_f32()?)
                .arg(variance.data_f32()?)
                .arg(&eps)
                .arg(&n)
                .arg(&variance_stride)
                .launch(elementwise_config(n))
                .map_err(|e| {
                    CudaError::Kernel(format!("fused_div_rsqrt_broadcast launch failed: {}", e))
                })?;
        }

        return Ok(out);
    }

    // Unsupported broadcast pattern
    Err(CudaError::Kernel(format!(
        "fused_div_rsqrt: unsupported broadcast {:?} vs {:?}",
        num_shape, var_shape
    )))
}

/// GPU fused add_mul_add: out = (x + a) * b + c
/// Replaces Add -> Mul -> Add pattern
pub fn gpu_fused_add_mul_add(
    ctx: &IconnxCudaContext,
    cache: &FusedKernelCache,
    pool: &mut GpuMemoryPool,
    x: &GpuTensor,
    a: &GpuTensor,
    b: &GpuTensor,
    c: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    let n = x.len();

    // Check if a is scalar (broadcast case)
    if a.len() == 1 {
        // Use scalar version - need to read scalar value from GPU
        // For now, use tensor version with broadcasting handled at caller
    }

    // All shapes must match for the simple case
    if x.shape() != a.shape() || x.shape() != b.shape() || x.shape() != c.shape() {
        return Err(CudaError::Kernel(format!(
            "fused_add_mul_add: shape mismatch x={:?} a={:?} b={:?} c={:?}",
            x.shape(),
            a.shape(),
            b.shape(),
            c.shape()
        )));
    }

    let mut out = pool.get_tensor_f32(ctx, x.shape().to_vec())?;

    let func = cache
        .get("fused_add_mul_add_kernel")
        .ok_or_else(|| CudaError::Kernel("fused_add_mul_add_kernel not found".into()))?;

    unsafe {
        ctx.stream()
            .launch_builder(func)
            .arg(out.data_f32_mut()?)
            .arg(x.data_f32()?)
            .arg(a.data_f32()?)
            .arg(b.data_f32()?)
            .arg(c.data_f32()?)
            .arg(&n)
            .launch(elementwise_config(n))
            .map_err(|e| CudaError::Kernel(format!("fused_add_mul_add launch failed: {}", e)))?;
    }

    Ok(out)
}

/// GPU fused GELU: out = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
/// Replaces full GELU activation pattern (8 ops -> 1)
pub fn gpu_fused_gelu(
    ctx: &IconnxCudaContext,
    cache: &FusedKernelCache,
    pool: &mut GpuMemoryPool,
    x: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    let n = x.len();
    let mut out = pool.get_tensor_f32(ctx, x.shape().to_vec())?;

    let func = cache
        .get("fused_gelu_kernel")
        .ok_or_else(|| CudaError::Kernel("fused_gelu_kernel not found".into()))?;

    unsafe {
        ctx.stream()
            .launch_builder(func)
            .arg(out.data_f32_mut()?)
            .arg(x.data_f32()?)
            .arg(&n)
            .launch(elementwise_config(n))
            .map_err(|e| CudaError::Kernel(format!("fused_gelu launch failed: {}", e)))?;
    }

    Ok(out)
}

/// GPU fused mul_add: out = a * b + c
/// Replaces Mul -> Add pattern (reduces kernel launches)
pub fn gpu_fused_mul_add(
    ctx: &IconnxCudaContext,
    cache: &FusedKernelCache,
    pool: &mut GpuMemoryPool,
    a: &GpuTensor,
    b: &GpuTensor,
    c: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    let n = a.len();

    // Shapes must match for elementwise operation
    if a.shape() != b.shape() || a.shape() != c.shape() {
        return Err(CudaError::Kernel(format!(
            "fused_mul_add: shape mismatch a={:?} b={:?} c={:?}",
            a.shape(),
            b.shape(),
            c.shape()
        )));
    }

    let mut out = pool.get_tensor_f32(ctx, a.shape().to_vec())?;

    let func = cache
        .get("fused_mul_add_kernel")
        .ok_or_else(|| CudaError::Kernel("fused_mul_add_kernel not found".into()))?;

    unsafe {
        ctx.stream()
            .launch_builder(func)
            .arg(out.data_f32_mut()?)
            .arg(a.data_f32()?)
            .arg(b.data_f32()?)
            .arg(c.data_f32()?)
            .arg(&n)
            .launch(elementwise_config(n))
            .map_err(|e| CudaError::Kernel(format!("fused_mul_add launch failed: {}", e)))?;
    }

    Ok(out)
}

/// GPU fused add_mul: out = (a + b) * c
/// Replaces Add -> Mul pattern (reduces kernel launches)
pub fn gpu_fused_add_mul(
    ctx: &IconnxCudaContext,
    cache: &FusedKernelCache,
    pool: &mut GpuMemoryPool,
    a: &GpuTensor,
    b: &GpuTensor,
    c: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    let n = a.len();

    // Shapes must match for elementwise operation
    if a.shape() != b.shape() || a.shape() != c.shape() {
        return Err(CudaError::Kernel(format!(
            "fused_add_mul: shape mismatch a={:?} b={:?} c={:?}",
            a.shape(),
            b.shape(),
            c.shape()
        )));
    }

    let mut out = pool.get_tensor_f32(ctx, a.shape().to_vec())?;

    let func = cache
        .get("fused_add_mul_kernel")
        .ok_or_else(|| CudaError::Kernel("fused_add_mul_kernel not found".into()))?;

    unsafe {
        ctx.stream()
            .launch_builder(func)
            .arg(out.data_f32_mut()?)
            .arg(a.data_f32()?)
            .arg(b.data_f32()?)
            .arg(c.data_f32()?)
            .arg(&n)
            .launch(elementwise_config(n))
            .map_err(|e| CudaError::Kernel(format!("fused_add_mul launch failed: {}", e)))?;
    }

    Ok(out)
}

/// GPU fused sub_mul: out = (a - b) * c
/// Replaces Sub -> Mul pattern (reduces kernel launches)
pub fn gpu_fused_sub_mul(
    ctx: &IconnxCudaContext,
    cache: &FusedKernelCache,
    pool: &mut GpuMemoryPool,
    a: &GpuTensor,
    b: &GpuTensor,
    c: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    let n = a.len();

    // Shapes must match for elementwise operation
    if a.shape() != b.shape() || a.shape() != c.shape() {
        return Err(CudaError::Kernel(format!(
            "fused_sub_mul: shape mismatch a={:?} b={:?} c={:?}",
            a.shape(),
            b.shape(),
            c.shape()
        )));
    }

    let mut out = pool.get_tensor_f32(ctx, a.shape().to_vec())?;

    let func = cache
        .get("fused_sub_mul_kernel")
        .ok_or_else(|| CudaError::Kernel("fused_sub_mul_kernel not found".into()))?;

    unsafe {
        ctx.stream()
            .launch_builder(func)
            .arg(out.data_f32_mut()?)
            .arg(a.data_f32()?)
            .arg(b.data_f32()?)
            .arg(c.data_f32()?)
            .arg(&n)
            .launch(elementwise_config(n))
            .map_err(|e| CudaError::Kernel(format!("fused_sub_mul launch failed: {}", e)))?;
    }

    Ok(out)
}

/// GPU fused div_mul: out = (a / b) * c
/// Replaces Div -> Mul pattern (reduces kernel launches)
pub fn gpu_fused_div_mul(
    ctx: &IconnxCudaContext,
    cache: &FusedKernelCache,
    pool: &mut GpuMemoryPool,
    a: &GpuTensor,
    b: &GpuTensor,
    c: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    let n = a.len();

    // Shapes must match for elementwise operation
    if a.shape() != b.shape() || a.shape() != c.shape() {
        return Err(CudaError::Kernel(format!(
            "fused_div_mul: shape mismatch a={:?} b={:?} c={:?}",
            a.shape(),
            b.shape(),
            c.shape()
        )));
    }

    let mut out = pool.get_tensor_f32(ctx, a.shape().to_vec())?;

    let func = cache
        .get("fused_div_mul_kernel")
        .ok_or_else(|| CudaError::Kernel("fused_div_mul_kernel not found".into()))?;

    unsafe {
        ctx.stream()
            .launch_builder(func)
            .arg(out.data_f32_mut()?)
            .arg(a.data_f32()?)
            .arg(b.data_f32()?)
            .arg(c.data_f32()?)
            .arg(&n)
            .launch(elementwise_config(n))
            .map_err(|e| CudaError::Kernel(format!("fused_div_mul launch failed: {}", e)))?;
    }

    Ok(out)
}

/// GPU fused mul_sin_pow_mul_add: y = sin(x * w0)^p * w1 + b
///
/// Replaces the 5-op chain Mul -> Sin -> Pow -> Mul -> Add used in vocoder
/// periodic activations (48 occurrences in Kokoro).
///
/// All tensors must have the same shape. A broadcast variant is added in
/// a separate task.
///
/// `p` is passed as f32. Integer exponents (detected via `p.fract() == 0.0`)
/// use a squaring loop that handles negative bases correctly; non-integer
/// exponents use `powf` (and require non-negative bases).
#[allow(clippy::too_many_arguments)]
pub fn gpu_fused_mul_sin_pow_mul_add(
    ctx: &IconnxCudaContext,
    cache: &FusedKernelCache,
    pool: &mut GpuMemoryPool,
    x: &GpuTensor,
    w0: &GpuTensor,
    w1: &GpuTensor,
    b: &GpuTensor,
    p: f32,
) -> Result<GpuTensor, CudaError> {
    // Non-integer exponents would need `powf`, which in CUDA returns NaN for
    // negative bases. We cannot prove at runtime that every `sin(x * w0)`
    // intermediate will be non-negative, so we conservatively refuse
    // fractional exponents and force the caller to use the unfused chain.
    if p.fract() != 0.0 {
        return Err(CudaError::Kernel(format!(
            "gpu_fused_mul_sin_pow_mul_add: non-integer exponent {p} not supported \
             (would require proving sin(x*w0) >= 0 for all lanes)"
        )));
    }

    // Same-shape fast path: all four tensors share one shape.
    let shape = x.shape();
    if w0.shape() == shape && w1.shape() == shape && b.shape() == shape {
        let n = x.len();
        let mut out = pool.get_tensor_f32(ctx, shape.to_vec())?;

        let func = cache
            .get("fused_mul_sin_pow_mul_add_kernel")
            .ok_or_else(|| {
                CudaError::Kernel("fused_mul_sin_pow_mul_add_kernel not found".into())
            })?;

        let p_is_int: i32 = if p.fract() == 0.0 { 1 } else { 0 };

        unsafe {
            ctx.stream()
                .launch_builder(func)
                .arg(out.data_f32_mut()?)
                .arg(x.data_f32()?)
                .arg(w0.data_f32()?)
                .arg(w1.data_f32()?)
                .arg(b.data_f32()?)
                .arg(&p)
                .arg(&p_is_int)
                .arg(&n)
                .launch(elementwise_config(n))
                .map_err(|e| {
                    CudaError::Kernel(format!(
                        "fused_mul_sin_pow_mul_add launch failed: {}",
                        e
                    ))
                })?;
        }

        return Ok(out);
    }

    // Broadcast case: w0, w1, b all have per-channel broadcast shape
    // (leading dims all 1, last dim matching x's last axis). We only accept
    // shapes like [C], [1, C], [1, 1, C], etc., so the broadcast stride is
    // exactly `channels` (the last-axis length of x).
    let w_per_channel = is_per_channel_broadcast(x, w0)
        && is_per_channel_broadcast(x, w1)
        && is_per_channel_broadcast(x, b)
        && w0.shape() == w1.shape()
        && w1.shape() == b.shape();

    if w_per_channel {
        let channels = *x
            .shape()
            .last()
            .expect("is_per_channel_broadcast ensured non-empty last axis");
        let broadcast_stride = channels;
        let n = x.len();
        let mut out = pool.get_tensor_f32(ctx, shape.to_vec())?;

        let func = cache
            .get("fused_mul_sin_pow_mul_add_broadcast_kernel")
            .ok_or_else(|| {
                CudaError::Kernel(
                    "fused_mul_sin_pow_mul_add_broadcast_kernel not found".into(),
                )
            })?;

        let p_is_int: i32 = if p.fract() == 0.0 { 1 } else { 0 };

        unsafe {
            ctx.stream()
                .launch_builder(func)
                .arg(out.data_f32_mut()?)
                .arg(x.data_f32()?)
                .arg(w0.data_f32()?)
                .arg(w1.data_f32()?)
                .arg(b.data_f32()?)
                .arg(&p)
                .arg(&p_is_int)
                .arg(&n)
                .arg(&broadcast_stride)
                .launch(elementwise_config(n))
                .map_err(|e| {
                    CudaError::Kernel(format!(
                        "fused_mul_sin_pow_mul_add_broadcast launch failed: {}",
                        e
                    ))
                })?;
        }

        return Ok(out);
    }

    Err(CudaError::Kernel(format!(
        "gpu_fused_mul_sin_pow_mul_add: unsupported shape combination \
         (x={:?}, w0={:?}, w1={:?}, b={:?}) -- need same-shape fast path or \
         per-channel broadcast (w shapes like [1,...,1,C] matching x's last axis)",
        x.shape(),
        w0.shape(),
        w1.shape(),
        b.shape()
    )))
}
