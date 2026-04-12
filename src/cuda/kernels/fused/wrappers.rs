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

/// Returns `Some(k_dim)` iff:
/// - `x.shape() == w1.shape()`
/// - `w0.shape() == b.shape()`
/// - `x`'s shape is `w0`'s shape with the LAST axis replaced by 1 (i.e.,
///   `x` and `w1` broadcast trailing-dim over `w0` and `b`).
/// - The last dim of `w0` (`k_dim`) is >= 1.
///
/// This is the shape pattern Kokoro's vocoder uses for per-channel scale:
/// x: [..., C, 1], w0: [..., C, K], w1: [..., C, 1], b: [..., C, K].
fn is_per_channel_scale_broadcast(
    x: &GpuTensor,
    w0: &GpuTensor,
    w1: &GpuTensor,
    b: &GpuTensor,
) -> Option<usize> {
    let x_shape = x.shape();
    let w0_shape = w0.shape();
    let w1_shape = w1.shape();
    let b_shape = b.shape();

    if x_shape != w1_shape {
        return None;
    }
    if w0_shape != b_shape {
        return None;
    }
    if x_shape.len() != w0_shape.len() {
        return None;
    }
    let ndim = w0_shape.len();
    if ndim == 0 {
        return None;
    }
    // All leading dims must match
    for i in 0..ndim - 1 {
        if x_shape[i] != w0_shape[i] {
            return None;
        }
    }
    // x's last dim must be 1, w0's must be >= 1
    if *x_shape.last().unwrap() != 1 {
        return None;
    }
    let k_dim = *w0_shape.last().unwrap();
    if k_dim == 0 {
        return None;
    }
    Some(k_dim)
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

/// Compute broadcast parameters for c in the AddMulAdd fusion.
///
/// Returns `Some((c_inner_stride, c_size))` if c is broadcast-compatible
/// with x (same ndim, each c dim is 1 or matches x, at least one dim is 1,
/// and the non-broadcast dims form a single contiguous block).
///
/// Returns `None` if shapes are incompatible or the broadcast pattern
/// is non-contiguous (e.g., c = [C, 1, C]).
pub(super) fn compute_c_broadcast_params(
    x_shape: &[usize],
    c_shape: &[usize],
) -> Option<(usize, usize)> {
    if x_shape.len() != c_shape.len() {
        return None;
    }

    // Classify each dimension as broadcast (c=1, x>1) or matching (c==x)
    // Also reject incompatible (c!=1 && c!=x)
    let mut has_broadcast = false;
    for (x_dim, c_dim) in x_shape.iter().zip(c_shape.iter()) {
        if *c_dim == *x_dim {
            // matching
        } else if *c_dim == 1 {
            has_broadcast = true;
        } else {
            return None; // incompatible
        }
    }

    if !has_broadcast {
        // All dims match -- this is the same-shape case, not broadcast
        return None;
    }

    // Find the contiguous non-broadcast block.
    // Non-broadcast dimension indices (where c_dim == x_dim AND x_dim > 1)
    // Dims where both are 1 can be treated as either broadcast or matching.
    let non_bcast_indices: Vec<usize> = x_shape
        .iter()
        .zip(c_shape.iter())
        .enumerate()
        .filter(|(_i, (x_d, c_d))| **c_d == **x_d && **x_d > 1)
        .map(|(i, _)| i)
        .collect();

    if non_bcast_indices.is_empty() {
        // c is all 1s -- degenerate scalar broadcast.
        // c_size = 1, c_inner_stride = 1 works (c_idx = (i/1)%1 = 0 always).
        return Some((1, 1));
    }

    // Check contiguity: indices must be consecutive
    for window in non_bcast_indices.windows(2) {
        if window[1] != window[0] + 1 {
            return None; // non-contiguous non-broadcast block
        }
    }

    let block_start = non_bcast_indices[0];
    let block_end = *non_bcast_indices.last().unwrap() + 1; // exclusive

    // c_size = product of x dims in [block_start..block_end)
    let c_size: usize = x_shape[block_start..block_end].iter().product();

    // c_inner_stride = product of x dims in [block_end..ndim)
    let c_inner_stride: usize = x_shape[block_end..].iter().product();

    Some((c_inner_stride, c_size))
}

/// GPU fused add_mul_add: out = (x + a) * b + c
/// Replaces Add -> Mul -> Add pattern
///
/// Three-way dispatch:
/// 1. All four shapes identical -> existing same-shape kernel
/// 2. x, a, b same shape + c broadcasts -> broadcast-c kernel
/// 3. Incompatible shapes -> FusionShapeMismatch sentinel for fallback
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

    // Path 1: All four shapes identical -> existing same-shape kernel
    if x.shape() == a.shape() && x.shape() == b.shape() && x.shape() == c.shape() {
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
                .map_err(|e| {
                    CudaError::Kernel(format!("fused_add_mul_add launch failed: {}", e))
                })?;
        }

        return Ok(out);
    }

    // Path 2: x, a, b same shape + c broadcasts -> broadcast-c kernel
    if x.shape() == a.shape() && x.shape() == b.shape() {
        if let Some((c_inner_stride, c_size)) =
            compute_c_broadcast_params(x.shape(), c.shape())
        {
            let mut out = pool.get_tensor_f32(ctx, x.shape().to_vec())?;

            let func = cache.get("fused_add_mul_add_bcast_c_kernel").ok_or_else(|| {
                CudaError::Kernel("fused_add_mul_add_bcast_c_kernel not found".into())
            })?;

            unsafe {
                ctx.stream()
                    .launch_builder(func)
                    .arg(out.data_f32_mut()?)
                    .arg(x.data_f32()?)
                    .arg(a.data_f32()?)
                    .arg(b.data_f32()?)
                    .arg(c.data_f32()?)
                    .arg(&n)
                    .arg(&c_inner_stride)
                    .arg(&c_size)
                    .launch(elementwise_config(n))
                    .map_err(|e| {
                        CudaError::Kernel(format!(
                            "fused_add_mul_add_bcast_c launch failed: {}",
                            e
                        ))
                    })?;
            }

            return Ok(out);
        }
    }

    // Path 3: Per-channel affine — x=[1,C,1], b=[1,C,T], c=[1,C,1], a empty/scalar
    // This is Kokoro's AdaIN pattern: scale * normalized_input + bias
    if b.shape().len() == 3 && b.shape()[0] == 1 {
        let c_dim = b.shape()[1];
        let t_dim = b.shape()[2];
        let is_per_channel_x = x.shape() == [1, c_dim, 1];
        let is_per_channel_c = c.shape() == [1, c_dim, 1];
        let is_a_trivial = a.len() <= 1; // empty or scalar

        if t_dim > 1 && is_per_channel_x && is_per_channel_c && is_a_trivial {
            let n = b.len();
            let mut out = pool.get_tensor_f32(ctx, b.shape().to_vec())?;

            let func = cache
                .get("fused_channel_affine_kernel")
                .ok_or_else(|| {
                    CudaError::Kernel("fused_channel_affine_kernel not found".into())
                })?;

            unsafe {
                ctx.stream()
                    .launch_builder(func)
                    .arg(out.data_f32_mut()?)
                    .arg(b.data_f32()?)    // input (full shape)
                    .arg(x.data_f32()?)    // scale (per-channel)
                    .arg(c.data_f32()?)    // bias (per-channel)
                    .arg(&n)
                    .arg(&t_dim)
                    .launch(elementwise_config(n))
                    .map_err(|e| {
                        CudaError::Kernel(format!(
                            "fused_channel_affine launch failed: {}",
                            e
                        ))
                    })?;
            }

            return Ok(out);
        }
    }

    // Path 4: shapes incompatible -> sentinel error for fallback
    Err(CudaError::FusionShapeMismatch(format!(
        "fused_add_mul_add: incompatible shapes x={:?} a={:?} b={:?} c={:?}",
        x.shape(),
        a.shape(),
        b.shape(),
        c.shape()
    )))
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
/// Supports three shape layouts:
/// 1. **Same-shape fast path:** `x`, `w0`, `w1`, and `b` all share one shape.
/// 2. **Per-channel broadcast:** `w0`, `w1`, `b` have shape `[C]`, `[1, C]`,
///    `[1, 1, C]`, etc., where `C` matches the last axis of `x`.
/// 3. **Per-channel scale broadcast:** `x` and `w1` have shape `[..., C, 1]`
///    (per-channel scalars), while `w0` and `b` have full shape `[..., C, K]`.
///    Every `K` output elements share the same `x[c]` and `w1[c]` values.
///    This is the shape pattern Kokoro's vocoder periodic activation uses.
///
/// `p` is passed as f32 but must be integer-valued (`p.fract() == 0.0`).
/// The kernel uses a squaring loop that handles negative bases correctly;
/// fractional exponents would require proving `sin(x * w0) >= 0` for all
/// lanes, so they are rejected up front and the caller is expected to fall
/// back to the unfused chain.
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

        unsafe {
            ctx.stream()
                .launch_builder(func)
                .arg(out.data_f32_mut()?)
                .arg(x.data_f32()?)
                .arg(w0.data_f32()?)
                .arg(w1.data_f32()?)
                .arg(b.data_f32()?)
                .arg(&p)
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

        unsafe {
            ctx.stream()
                .launch_builder(func)
                .arg(out.data_f32_mut()?)
                .arg(x.data_f32()?)
                .arg(w0.data_f32()?)
                .arg(w1.data_f32()?)
                .arg(b.data_f32()?)
                .arg(&p)
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

    // Per-channel-scale broadcast branch: x and w1 broadcast over the
    // trailing dim of w0 and b. This is Kokoro's vocoder periodic
    // activation shape.
    if let Some(k_dim) = is_per_channel_scale_broadcast(x, w0, w1, b) {
        let n = w0.len();
        let mut out = pool.get_tensor_f32(ctx, w0.shape().to_vec())?;

        let func = cache
            .get("fused_mul_sin_pow_mul_add_per_channel_scale_kernel")
            .ok_or_else(|| {
                CudaError::Kernel(
                    "fused_mul_sin_pow_mul_add_per_channel_scale_kernel not found".into(),
                )
            })?;

        unsafe {
            ctx.stream()
                .launch_builder(func)
                .arg(out.data_f32_mut()?)
                .arg(x.data_f32()?)
                .arg(w0.data_f32()?)
                .arg(w1.data_f32()?)
                .arg(b.data_f32()?)
                .arg(&p)
                .arg(&n)
                .arg(&k_dim)
                .launch(elementwise_config(n))
                .map_err(|e| {
                    CudaError::Kernel(format!(
                        "fused_mul_sin_pow_mul_add_per_channel_scale launch failed: {}",
                        e
                    ))
                })?;
        }

        return Ok(out);
    }

    Err(CudaError::Kernel(format!(
        "gpu_fused_mul_sin_pow_mul_add: unsupported shape combination \
         (x={:?}, w0={:?}, w1={:?}, b={:?}) -- need same-shape fast path, \
         per-channel broadcast (w shapes like [1,...,1,C] matching x's last axis), \
         or per-channel-scale broadcast (x and w1 shape [..., C, 1] over w0 and b \
         shape [..., C, K])",
        x.shape(),
        w0.shape(),
        w1.shape(),
        b.shape()
    )))
}
