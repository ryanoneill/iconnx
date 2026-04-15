//! Host-side wrappers for fused GPU kernels.
//!
//! This file holds the `gpu_fused_*` host wrapper functions. Each wrapper:
//! - Validates tensor shapes (handling any supported broadcast patterns).
//! - Allocates an output tensor from the `GpuMemoryPool`.
//! - Resolves a typed kernel from the `FusedKernelCache` module and
//!   launches it on garboard's stream.
//!
//! The shared `elementwise_config` and `compute_broadcast_stride` helpers
//! live in the parent `mod.rs`. All wrappers are re-exported at
//! `crate::cuda::kernels::fused::*` via `pub use wrappers::*` in `mod.rs`.

use super::{compute_broadcast_stride, elementwise_config, FusedKernelCache};
use crate::cuda::context::{CudaError, IconnxCudaContext};
use crate::cuda::memory_pool::GpuMemoryPool;
use crate::cuda::tensor::GpuTensor;
use garboard::DeviceSlice;

/// Returns true iff `w` is a valid per-channel broadcast of `x`:
/// all leading dims are 1, and `w.shape().last()` equals the last axis of `x`.
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
    w_shape
        .iter()
        .take(w_shape.len().saturating_sub(1))
        .all(|&d| d == 1)
}

/// Returns `Some(k_dim)` iff the tensors form the per-channel-scale
/// broadcast pattern used by Kokoro's vocoder: `x` and `w1` have shape
/// `[..., C, 1]`; `w0` and `b` have shape `[..., C, K]`.
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
    for i in 0..ndim - 1 {
        if x_shape[i] != w0_shape[i] {
            return None;
        }
    }
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

/// GPU fused div_rsqrt: out = numerator / sqrt(variance + eps).
/// Replaces Add -> Sqrt -> Div pattern. Supports broadcasting when
/// variance has trailing 1 dimensions.
pub fn gpu_fused_div_rsqrt(
    ctx: &IconnxCudaContext,
    cache: &FusedKernelCache,
    pool: &mut GpuMemoryPool,
    numerator: &GpuTensor,
    variance: &GpuTensor,
    eps: f32,
) -> Result<GpuTensor, CudaError> {
    let num_shape = numerator.shape();
    let var_shape = variance.shape();

    // Simple case: shapes match.
    if num_shape == var_shape {
        let n = numerator.len();
        let mut out = pool.get_tensor_f32(ctx, numerator.shape().to_vec())?;

        // SAFETY: fused_div_rsqrt_scalar_eps_kernel signature is
        // `(float* out, const float* numerator, const float* variance,
        //   float eps, size_t n)`.
        let kernel = unsafe {
            cache.module().typed_kernel::<(
                &mut DeviceSlice<'_, f32>,
                &DeviceSlice<'_, f32>,
                &DeviceSlice<'_, f32>,
                f32,
                usize,
            )>("fused_div_rsqrt_scalar_eps_kernel")
        }
        .map_err(|e| {
            CudaError::Kernel(format!(
                "fused_div_rsqrt_scalar_eps_kernel lookup failed: {}",
                e
            ))
        })?;

        kernel
            .launch(
                ctx.garboard_stream(),
                &elementwise_config(n),
                (
                    out.data_f32_mut()?,
                    numerator.data_f32()?,
                    variance.data_f32()?,
                    eps,
                    n,
                ),
            )
            .map_err(|e| CudaError::Kernel(format!("fused_div_rsqrt launch failed: {}", e)))?;

        return Ok(out);
    }

    // Broadcasting case: variance has trailing 1 dimensions.
    if let Some(variance_stride) = compute_broadcast_stride(num_shape, var_shape) {
        let n = numerator.len();
        let mut out = pool.get_tensor_f32(ctx, numerator.shape().to_vec())?;

        // SAFETY: fused_div_rsqrt_broadcast_kernel signature is
        // `(float* out, const float* numerator, const float* variance,
        //   float eps, size_t n, size_t variance_stride)`.
        let kernel = unsafe {
            cache.module().typed_kernel::<(
                &mut DeviceSlice<'_, f32>,
                &DeviceSlice<'_, f32>,
                &DeviceSlice<'_, f32>,
                f32,
                usize,
                usize,
            )>("fused_div_rsqrt_broadcast_kernel")
        }
        .map_err(|e| {
            CudaError::Kernel(format!(
                "fused_div_rsqrt_broadcast_kernel lookup failed: {}",
                e
            ))
        })?;

        kernel
            .launch(
                ctx.garboard_stream(),
                &elementwise_config(n),
                (
                    out.data_f32_mut()?,
                    numerator.data_f32()?,
                    variance.data_f32()?,
                    eps,
                    n,
                    variance_stride,
                ),
            )
            .map_err(|e| {
                CudaError::Kernel(format!("fused_div_rsqrt_broadcast launch failed: {}", e))
            })?;

        return Ok(out);
    }

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
pub(super) fn compute_c_broadcast_params(
    x_shape: &[usize],
    c_shape: &[usize],
) -> Option<(usize, usize)> {
    if x_shape.len() != c_shape.len() {
        return None;
    }

    let mut has_broadcast = false;
    for (x_dim, c_dim) in x_shape.iter().zip(c_shape.iter()) {
        if *c_dim == *x_dim {
            // matching
        } else if *c_dim == 1 {
            has_broadcast = true;
        } else {
            return None;
        }
    }

    if !has_broadcast {
        return None;
    }

    let non_bcast_indices: Vec<usize> = x_shape
        .iter()
        .zip(c_shape.iter())
        .enumerate()
        .filter(|(_i, (x_d, c_d))| **c_d == **x_d && **x_d > 1)
        .map(|(i, _)| i)
        .collect();

    if non_bcast_indices.is_empty() {
        return Some((1, 1));
    }

    for window in non_bcast_indices.windows(2) {
        if window[1] != window[0] + 1 {
            return None;
        }
    }

    let block_start = non_bcast_indices[0];
    let block_end = *non_bcast_indices.last().unwrap() + 1;

    let c_size: usize = x_shape[block_start..block_end].iter().product();
    let c_inner_stride: usize = x_shape[block_end..].iter().product();

    Some((c_inner_stride, c_size))
}

/// Launch a simple "binary-op with output" fused kernel:
/// `(out, a, b, c, n)`. Covers mul_add, add_mul, sub_mul, div_mul.
#[allow(clippy::too_many_arguments)]
fn launch_ternary_f32(
    ctx: &IconnxCudaContext,
    cache: &FusedKernelCache,
    name: &'static str,
    out: &mut DeviceSlice<'static, f32>,
    a: &DeviceSlice<'static, f32>,
    b: &DeviceSlice<'static, f32>,
    c: &DeviceSlice<'static, f32>,
    n: usize,
) -> Result<(), CudaError> {
    // SAFETY: kernel signature is
    // `(float* out, const float* a, const float* b, const float* c, size_t n)`.
    let kernel = unsafe {
        cache.module().typed_kernel::<(
            &mut DeviceSlice<'_, f32>,
            &DeviceSlice<'_, f32>,
            &DeviceSlice<'_, f32>,
            &DeviceSlice<'_, f32>,
            usize,
        )>(name)
    }
    .map_err(|e| CudaError::Kernel(format!("{} lookup failed: {}", name, e)))?;

    kernel
        .launch(
            ctx.garboard_stream(),
            &elementwise_config(n),
            (out, a, b, c, n),
        )
        .map_err(|e| CudaError::Kernel(format!("{} launch failed: {}", name, e)))
}

/// GPU fused add_mul_add: out = (x + a) * b + c.
/// Replaces Add -> Mul -> Add pattern.
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

    // Path 1: All four shapes identical.
    if x.shape() == a.shape() && x.shape() == b.shape() && x.shape() == c.shape() {
        let mut out = pool.get_tensor_f32(ctx, x.shape().to_vec())?;

        // SAFETY: fused_add_mul_add_kernel signature is
        // `(float* out, const float* x, const float* a, const float* b,
        //   const float* c, size_t n)`.
        let kernel = unsafe {
            cache.module().typed_kernel::<(
                &mut DeviceSlice<'_, f32>,
                &DeviceSlice<'_, f32>,
                &DeviceSlice<'_, f32>,
                &DeviceSlice<'_, f32>,
                &DeviceSlice<'_, f32>,
                usize,
            )>("fused_add_mul_add_kernel")
        }
        .map_err(|e| {
            CudaError::Kernel(format!("fused_add_mul_add_kernel lookup failed: {}", e))
        })?;

        kernel
            .launch(
                ctx.garboard_stream(),
                &elementwise_config(n),
                (
                    out.data_f32_mut()?,
                    x.data_f32()?,
                    a.data_f32()?,
                    b.data_f32()?,
                    c.data_f32()?,
                    n,
                ),
            )
            .map_err(|e| CudaError::Kernel(format!("fused_add_mul_add launch failed: {}", e)))?;

        return Ok(out);
    }

    // Path 2: x, a, b same shape + c broadcasts.
    if x.shape() == a.shape() && x.shape() == b.shape() {
        if let Some((c_inner_stride, c_size)) =
            compute_c_broadcast_params(x.shape(), c.shape())
        {
            let mut out = pool.get_tensor_f32(ctx, x.shape().to_vec())?;

            // SAFETY: fused_add_mul_add_bcast_c_kernel signature is
            // `(float* out, const float* x, const float* a, const float* b,
            //   const float* c, size_t n, size_t c_inner_stride, size_t c_size)`.
            let kernel = unsafe {
                cache.module().typed_kernel::<(
                    &mut DeviceSlice<'_, f32>,
                    &DeviceSlice<'_, f32>,
                    &DeviceSlice<'_, f32>,
                    &DeviceSlice<'_, f32>,
                    &DeviceSlice<'_, f32>,
                    usize,
                    usize,
                    usize,
                )>("fused_add_mul_add_bcast_c_kernel")
            }
            .map_err(|e| {
                CudaError::Kernel(format!(
                    "fused_add_mul_add_bcast_c_kernel lookup failed: {}",
                    e
                ))
            })?;

            kernel
                .launch(
                    ctx.garboard_stream(),
                    &elementwise_config(n),
                    (
                        out.data_f32_mut()?,
                        x.data_f32()?,
                        a.data_f32()?,
                        b.data_f32()?,
                        c.data_f32()?,
                        n,
                        c_inner_stride,
                        c_size,
                    ),
                )
                .map_err(|e| {
                    CudaError::Kernel(format!(
                        "fused_add_mul_add_bcast_c launch failed: {}",
                        e
                    ))
                })?;

            return Ok(out);
        }
    }

    // Path 3: Per-channel affine — x=[1,C,1], b=[1,C,T], c=[1,C,1], a
    // empty/scalar. This is Kokoro's AdaIN pattern.
    if b.shape().len() == 3 && b.shape()[0] == 1 {
        let c_dim = b.shape()[1];
        let t_dim = b.shape()[2];
        let is_per_channel_x = x.shape() == [1, c_dim, 1];
        let is_per_channel_c = c.shape() == [1, c_dim, 1];
        let is_a_trivial = a.len() <= 1;

        if t_dim > 1 && is_per_channel_x && is_per_channel_c && is_a_trivial {
            let n = b.len();
            let mut out = pool.get_tensor_f32(ctx, b.shape().to_vec())?;

            // SAFETY: fused_channel_affine_kernel signature is
            // `(float* out, const float* input, const float* scale,
            //   const float* bias, size_t n, size_t T)`.
            let kernel = unsafe {
                cache.module().typed_kernel::<(
                    &mut DeviceSlice<'_, f32>,
                    &DeviceSlice<'_, f32>,
                    &DeviceSlice<'_, f32>,
                    &DeviceSlice<'_, f32>,
                    usize,
                    usize,
                )>("fused_channel_affine_kernel")
            }
            .map_err(|e| {
                CudaError::Kernel(format!(
                    "fused_channel_affine_kernel lookup failed: {}",
                    e
                ))
            })?;

            kernel
                .launch(
                    ctx.garboard_stream(),
                    &elementwise_config(n),
                    (
                        out.data_f32_mut()?,
                        b.data_f32()?, // input (full shape)
                        x.data_f32()?, // scale (per-channel)
                        c.data_f32()?, // bias (per-channel)
                        n,
                        t_dim,
                    ),
                )
                .map_err(|e| {
                    CudaError::Kernel(format!("fused_channel_affine launch failed: {}", e))
                })?;

            return Ok(out);
        }
    }

    // Path 4: sentinel for fallback.
    Err(CudaError::FusionShapeMismatch(format!(
        "fused_add_mul_add: incompatible shapes x={:?} a={:?} b={:?} c={:?}",
        x.shape(),
        a.shape(),
        b.shape(),
        c.shape()
    )))
}

/// GPU fused GELU: `out = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
pub fn gpu_fused_gelu(
    ctx: &IconnxCudaContext,
    cache: &FusedKernelCache,
    pool: &mut GpuMemoryPool,
    x: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    let n = x.len();
    let mut out = pool.get_tensor_f32(ctx, x.shape().to_vec())?;

    // SAFETY: fused_gelu_kernel signature is
    // `(float* out, const float* x, size_t n)`.
    let kernel = unsafe {
        cache.module().typed_kernel::<(
            &mut DeviceSlice<'_, f32>,
            &DeviceSlice<'_, f32>,
            usize,
        )>("fused_gelu_kernel")
    }
    .map_err(|e| CudaError::Kernel(format!("fused_gelu_kernel lookup failed: {}", e)))?;

    kernel
        .launch(
            ctx.garboard_stream(),
            &elementwise_config(n),
            (out.data_f32_mut()?, x.data_f32()?, n),
        )
        .map_err(|e| CudaError::Kernel(format!("fused_gelu launch failed: {}", e)))?;

    Ok(out)
}

/// GPU fused mul_add: out = a * b + c.
pub fn gpu_fused_mul_add(
    ctx: &IconnxCudaContext,
    cache: &FusedKernelCache,
    pool: &mut GpuMemoryPool,
    a: &GpuTensor,
    b: &GpuTensor,
    c: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    let n = a.len();

    if a.shape() != b.shape() || a.shape() != c.shape() {
        return Err(CudaError::Kernel(format!(
            "fused_mul_add: shape mismatch a={:?} b={:?} c={:?}",
            a.shape(),
            b.shape(),
            c.shape()
        )));
    }

    let mut out = pool.get_tensor_f32(ctx, a.shape().to_vec())?;
    launch_ternary_f32(
        ctx,
        cache,
        "fused_mul_add_kernel",
        out.data_f32_mut()?,
        a.data_f32()?,
        b.data_f32()?,
        c.data_f32()?,
        n,
    )?;
    Ok(out)
}

/// GPU fused add_mul: out = (a + b) * c.
pub fn gpu_fused_add_mul(
    ctx: &IconnxCudaContext,
    cache: &FusedKernelCache,
    pool: &mut GpuMemoryPool,
    a: &GpuTensor,
    b: &GpuTensor,
    c: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    let n = a.len();

    if a.shape() != b.shape() || a.shape() != c.shape() {
        return Err(CudaError::Kernel(format!(
            "fused_add_mul: shape mismatch a={:?} b={:?} c={:?}",
            a.shape(),
            b.shape(),
            c.shape()
        )));
    }

    let mut out = pool.get_tensor_f32(ctx, a.shape().to_vec())?;
    launch_ternary_f32(
        ctx,
        cache,
        "fused_add_mul_kernel",
        out.data_f32_mut()?,
        a.data_f32()?,
        b.data_f32()?,
        c.data_f32()?,
        n,
    )?;
    Ok(out)
}

/// GPU fused sub_mul: out = (a - b) * c.
pub fn gpu_fused_sub_mul(
    ctx: &IconnxCudaContext,
    cache: &FusedKernelCache,
    pool: &mut GpuMemoryPool,
    a: &GpuTensor,
    b: &GpuTensor,
    c: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    let n = a.len();

    if a.shape() != b.shape() || a.shape() != c.shape() {
        return Err(CudaError::Kernel(format!(
            "fused_sub_mul: shape mismatch a={:?} b={:?} c={:?}",
            a.shape(),
            b.shape(),
            c.shape()
        )));
    }

    let mut out = pool.get_tensor_f32(ctx, a.shape().to_vec())?;
    launch_ternary_f32(
        ctx,
        cache,
        "fused_sub_mul_kernel",
        out.data_f32_mut()?,
        a.data_f32()?,
        b.data_f32()?,
        c.data_f32()?,
        n,
    )?;
    Ok(out)
}

/// GPU fused div_mul: out = (a / b) * c.
pub fn gpu_fused_div_mul(
    ctx: &IconnxCudaContext,
    cache: &FusedKernelCache,
    pool: &mut GpuMemoryPool,
    a: &GpuTensor,
    b: &GpuTensor,
    c: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    let n = a.len();

    if a.shape() != b.shape() || a.shape() != c.shape() {
        return Err(CudaError::Kernel(format!(
            "fused_div_mul: shape mismatch a={:?} b={:?} c={:?}",
            a.shape(),
            b.shape(),
            c.shape()
        )));
    }

    let mut out = pool.get_tensor_f32(ctx, a.shape().to_vec())?;
    launch_ternary_f32(
        ctx,
        cache,
        "fused_div_mul_kernel",
        out.data_f32_mut()?,
        a.data_f32()?,
        b.data_f32()?,
        c.data_f32()?,
        n,
    )?;
    Ok(out)
}

/// GPU fused mul_sin_pow_mul_add: y = sin(x * w0)^p * w1 + b.
/// Vocoder periodic activation (48 occurrences in Kokoro).
/// Supports three shape layouts — see the comment inside.
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
    // Non-integer exponents would need `powf`, which in CUDA returns NaN
    // for negative bases. We can't prove at runtime that every
    // `sin(x * w0)` intermediate is non-negative, so refuse and let the
    // caller fall back to the unfused chain.
    if p.fract() != 0.0 {
        return Err(CudaError::Kernel(format!(
            "gpu_fused_mul_sin_pow_mul_add: non-integer exponent {p} not supported \
             (would require proving sin(x*w0) >= 0 for all lanes)"
        )));
    }

    // Same-shape fast path.
    let shape = x.shape();
    if w0.shape() == shape && w1.shape() == shape && b.shape() == shape {
        let n = x.len();
        let mut out = pool.get_tensor_f32(ctx, shape.to_vec())?;

        // SAFETY: fused_mul_sin_pow_mul_add_kernel signature is
        // `(float* out, const float* x, const float* w0, const float* w1,
        //   const float* b, float p, size_t n)`.
        let kernel = unsafe {
            cache.module().typed_kernel::<(
                &mut DeviceSlice<'_, f32>,
                &DeviceSlice<'_, f32>,
                &DeviceSlice<'_, f32>,
                &DeviceSlice<'_, f32>,
                &DeviceSlice<'_, f32>,
                f32,
                usize,
            )>("fused_mul_sin_pow_mul_add_kernel")
        }
        .map_err(|e| {
            CudaError::Kernel(format!(
                "fused_mul_sin_pow_mul_add_kernel lookup failed: {}",
                e
            ))
        })?;

        kernel
            .launch(
                ctx.garboard_stream(),
                &elementwise_config(n),
                (
                    out.data_f32_mut()?,
                    x.data_f32()?,
                    w0.data_f32()?,
                    w1.data_f32()?,
                    b.data_f32()?,
                    p,
                    n,
                ),
            )
            .map_err(|e| {
                CudaError::Kernel(format!(
                    "fused_mul_sin_pow_mul_add launch failed: {}",
                    e
                ))
            })?;

        return Ok(out);
    }

    // Per-channel broadcast: w0, w1, b shape like [1,...,1,C] matching x's
    // last axis.
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

        // SAFETY: fused_mul_sin_pow_mul_add_broadcast_kernel signature is
        // `(float* out, const float* x, const float* w0, const float* w1,
        //   const float* b, float p, size_t n, size_t broadcast_stride)`.
        let kernel = unsafe {
            cache.module().typed_kernel::<(
                &mut DeviceSlice<'_, f32>,
                &DeviceSlice<'_, f32>,
                &DeviceSlice<'_, f32>,
                &DeviceSlice<'_, f32>,
                &DeviceSlice<'_, f32>,
                f32,
                usize,
                usize,
            )>("fused_mul_sin_pow_mul_add_broadcast_kernel")
        }
        .map_err(|e| {
            CudaError::Kernel(format!(
                "fused_mul_sin_pow_mul_add_broadcast_kernel lookup failed: {}",
                e
            ))
        })?;

        kernel
            .launch(
                ctx.garboard_stream(),
                &elementwise_config(n),
                (
                    out.data_f32_mut()?,
                    x.data_f32()?,
                    w0.data_f32()?,
                    w1.data_f32()?,
                    b.data_f32()?,
                    p,
                    n,
                    broadcast_stride,
                ),
            )
            .map_err(|e| {
                CudaError::Kernel(format!(
                    "fused_mul_sin_pow_mul_add_broadcast launch failed: {}",
                    e
                ))
            })?;

        return Ok(out);
    }

    // Per-channel-scale broadcast branch.
    if let Some(k_dim) = is_per_channel_scale_broadcast(x, w0, w1, b) {
        let n = w0.len();
        let mut out = pool.get_tensor_f32(ctx, w0.shape().to_vec())?;

        // SAFETY: fused_mul_sin_pow_mul_add_per_channel_scale_kernel
        // signature is
        // `(float* out, const float* x, const float* w0, const float* w1,
        //   const float* b, float p, size_t n, size_t k_dim)`.
        let kernel = unsafe {
            cache.module().typed_kernel::<(
                &mut DeviceSlice<'_, f32>,
                &DeviceSlice<'_, f32>,
                &DeviceSlice<'_, f32>,
                &DeviceSlice<'_, f32>,
                &DeviceSlice<'_, f32>,
                f32,
                usize,
                usize,
            )>("fused_mul_sin_pow_mul_add_per_channel_scale_kernel")
        }
        .map_err(|e| {
            CudaError::Kernel(format!(
                "fused_mul_sin_pow_mul_add_per_channel_scale_kernel lookup failed: {}",
                e
            ))
        })?;

        kernel
            .launch(
                ctx.garboard_stream(),
                &elementwise_config(n),
                (
                    out.data_f32_mut()?,
                    x.data_f32()?,
                    w0.data_f32()?,
                    w1.data_f32()?,
                    b.data_f32()?,
                    p,
                    n,
                    k_dim,
                ),
            )
            .map_err(|e| {
                CudaError::Kernel(format!(
                    "fused_mul_sin_pow_mul_add_per_channel_scale launch failed: {}",
                    e
                ))
            })?;

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
