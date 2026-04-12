//! Fused GPU kernels for common operator patterns
//!
//! These kernels combine multiple operations into single kernel launches to:
//! - Reduce kernel launch overhead
//! - Eliminate intermediate memory allocations
//! - Reduce memory bandwidth by keeping data in registers
//!
//! Patterns implemented:
//! - add_sqrt_div: y / sqrt(x + eps) - normalization denominator (65 occurrences)
//! - add_mul_add: (x + a) * b + c - affine transform (73 occurrences)
//! - gelu: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) - GELU activation (12 occurrences)
//!
//! This module is split across two files:
//! - `mod.rs` (this file): the `FusedKernelCache` struct, the CUDA source
//!   raw string, the kernel name list, and shared helpers used by the host
//!   wrappers (`elementwise_config`, `compute_broadcast_stride`).
//! - `wrappers.rs`: the `gpu_fused_*` host wrapper functions, re-exported
//!   through `pub use wrappers::*` so external callers see a flat namespace.

use crate::cuda::context::{CudaError, IconnxCudaContext};
use cudarc::driver::{CudaFunction, CudaModule, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use std::collections::HashMap;
use std::sync::Arc;

pub(super) mod wrappers;
pub use wrappers::*;

/// Cache for fused kernels
pub struct FusedKernelCache {
    #[allow(dead_code)]
    module: Arc<CudaModule>,
    functions: HashMap<&'static str, CudaFunction>,
}

impl FusedKernelCache {
    /// Compile all fused kernels and create the cache
    pub fn new(ctx: &IconnxCudaContext) -> Result<Self, CudaError> {
        let ptx = compile_ptx(FUSED_KERNELS)
            .map_err(|e| CudaError::Kernel(format!("NVRTC fused kernel compilation failed: {:?}", e)))?;

        let module = ctx
            .context()
            .load_module(ptx)
            .map_err(|e| CudaError::Kernel(format!("Fused module load failed: {}", e)))?;

        let mut functions = HashMap::new();

        for name in FUSED_KERNEL_NAMES {
            let func = module.load_function(name).map_err(|e| {
                CudaError::Kernel(format!("Failed to load fused kernel '{}': {}", name, e))
            })?;
            functions.insert(*name, func);
        }

        Ok(Self { module, functions })
    }

    /// Get a compiled kernel function by name
    pub fn get(&self, name: &'static str) -> Option<&CudaFunction> {
        self.functions.get(name)
    }
}

const FUSED_KERNEL_NAMES: &[&str] = &[
    // Normalization pattern: y / sqrt(variance + eps)
    "fused_div_rsqrt_kernel",
    "fused_div_rsqrt_scalar_eps_kernel",
    "fused_div_rsqrt_broadcast_kernel", // For broadcasting variance to numerator shape
    // Affine transform: (x + a) * b + c
    "fused_add_mul_add_kernel",
    "fused_add_mul_add_scalar_kernel",
    "fused_add_mul_add_bcast_c_kernel",
    // Full GELU activation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    "fused_gelu_kernel",
    // Mul-Add: y = a * b + c (fused multiply-add)
    "fused_mul_add_kernel",
    // Add-Mul: y = (a + b) * c (fused add-multiply)
    "fused_add_mul_kernel",
    // Sub-Mul: y = (a - b) * c (fused sub-multiply)
    "fused_sub_mul_kernel",
    // Div-Mul: y = (a / b) * c (fused div-multiply)
    "fused_div_mul_kernel",
    // Mul-Sin-Pow-Mul-Add: y = sin(x * w0)^p * w1 + b
    // Vocoder periodic activation (48 occurrences in Kokoro)
    "fused_mul_sin_pow_mul_add_kernel",
    "fused_mul_sin_pow_mul_add_broadcast_kernel",
    "fused_mul_sin_pow_mul_add_per_channel_scale_kernel",
];

/// CUDA source for fused kernels
const FUSED_KERNELS: &str = r#"
// ============================================================================
// Fused div_rsqrt: out = numerator / sqrt(variance + eps)
// Used in normalization (65 occurrences in Kokoro)
// Replaces: Add -> Sqrt -> Div
// ============================================================================

// Version with eps as tensor (for broadcasting)
extern "C" __global__ void fused_div_rsqrt_kernel(
    float* out,
    const float* numerator,
    const float* variance,
    const float* eps,
    size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = numerator[i] * rsqrtf(variance[i] + eps[i]);
    }
}

// Version with eps as scalar (more common case)
extern "C" __global__ void fused_div_rsqrt_scalar_eps_kernel(
    float* out,
    const float* numerator,
    const float* variance,
    float eps,
    size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = numerator[i] * rsqrtf(variance[i] + eps);
    }
}

// Version with broadcasting: variance has shape [..., 1] broadcast to [..., P]
// variance_stride is the number of elements before repeating (i.e., P)
extern "C" __global__ void fused_div_rsqrt_broadcast_kernel(
    float* out,
    const float* numerator,
    const float* variance,
    float eps,
    size_t n,
    size_t variance_stride
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        size_t var_idx = i / variance_stride;
        out[i] = numerator[i] * rsqrtf(variance[var_idx] + eps);
    }
}

// ============================================================================
// Fused add_mul_add: out = (x + a) * b + c
// Affine transform pattern (73 occurrences in Kokoro)
// Replaces: Add -> Mul -> Add
// ============================================================================

// All tensors version (elementwise with matching shapes)
extern "C" __global__ void fused_add_mul_add_kernel(
    float* out,
    const float* x,
    const float* a,
    const float* b,
    const float* c,
    size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = (x[i] + a[i]) * b[i] + c[i];
    }
}

// Common case: a is scalar (broadcast), b and c are tensors
extern "C" __global__ void fused_add_mul_add_scalar_kernel(
    float* out,
    const float* x,
    float a,
    const float* b,
    const float* c,
    size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = (x[i] + a) * b[i] + c[i];
    }
}

// Broadcast-c version: x, a, b have identical shape, c broadcasts via
// a contiguous non-broadcast block. c_inner_stride is the product of
// x dims after the non-broadcast block; c_size is the product of dims
// in the non-broadcast block.
// AdaIN: c=[1,C,1], x=[1,C,T] -> c_inner_stride=T, c_size=C
// LSTM:  c=[1,1,C], x=[1,T,C] -> c_inner_stride=1, c_size=C
extern "C" __global__ void fused_add_mul_add_bcast_c_kernel(
    float* out,
    const float* x,
    const float* a,
    const float* b,
    const float* c,
    size_t n,
    size_t c_inner_stride,
    size_t c_size
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        size_t c_idx = (i / c_inner_stride) % c_size;
        out[i] = (x[i] + a[i]) * b[i] + c[c_idx];
    }
}

// ============================================================================
// Fused GELU: out = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// Full GELU activation function (12 occurrences in Kokoro)
// Replaces: Pow -> Mul -> Add -> Mul -> Tanh -> Add -> Mul -> Mul (8 ops)
// ============================================================================

extern "C" __global__ void fused_gelu_kernel(
    float* out,
    const float* x,
    size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float xi = x[i];
        // Constants for GELU approximation
        const float SQRT_2_OVER_PI = 0.7978845608f;  // sqrt(2/pi)
        const float COEFF = 0.044715f;

        // Compute: x + 0.044715 * x^3
        float x3 = xi * xi * xi;
        float inner = xi + COEFF * x3;

        // Compute: tanh(sqrt(2/pi) * inner)
        float tanh_arg = SQRT_2_OVER_PI * inner;
        float tanh_val = tanhf(tanh_arg);

        // Final: x * 0.5 * (1 + tanh_val)
        out[i] = xi * 0.5f * (1.0f + tanh_val);
    }
}

// ============================================================================
// Fused mul_add: out = a * b + c
// Simple multiply-add pattern (619 Mul + Add combinations in Kokoro)
// Replaces: Mul -> Add
// ============================================================================

extern "C" __global__ void fused_mul_add_kernel(
    float* out,
    const float* a,
    const float* b,
    const float* c,
    size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] * b[i] + c[i];
    }
}

// ============================================================================
// Fused add_mul: out = (a + b) * c
// Simple add-multiply pattern (complement to mul_add)
// Replaces: Add -> Mul
// ============================================================================

extern "C" __global__ void fused_add_mul_kernel(
    float* out,
    const float* a,
    const float* b,
    const float* c,
    size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = (a[i] + b[i]) * c[i];
    }
}

// ============================================================================
// Fused sub_mul: out = (a - b) * c
// Simple sub-multiply pattern
// Replaces: Sub -> Mul
// ============================================================================

extern "C" __global__ void fused_sub_mul_kernel(
    float* out,
    const float* a,
    const float* b,
    const float* c,
    size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = (a[i] - b[i]) * c[i];
    }
}

// ============================================================================
// Fused div_mul: out = (a / b) * c
// Simple div-multiply pattern
// Replaces: Div -> Mul
// ============================================================================

extern "C" __global__ void fused_div_mul_kernel(
    float* out,
    const float* a,
    const float* b,
    const float* c,
    size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = (a[i] / b[i]) * c[i];
    }
}

// ============================================================================
// Fused mul_sin_pow_mul_add: y = sin(x * w0)^p * w1 + b
// Vocoder periodic activation (48 occurrences in Kokoro)
// Replaces: Mul -> Sin -> Pow -> Mul -> Add
// ============================================================================

// The host wrapper refuses fractional exponents, so only the integer
// squaring loop is reachable here; `p` is always an integer value stored
// in f32.
extern "C" __global__ void fused_mul_sin_pow_mul_add_kernel(
    float* out,
    const float* x,
    const float* w0,
    const float* w1,
    const float* b,
    float p,
    size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float s = sinf(x[i] * w0[i]);
        // Integer exponent path: handles negative bases correctly.
        int ip = (int) p;
        float pow_val = 1.0f;
        float base = s;
        int exp = ip < 0 ? -ip : ip;
        while (exp > 0) {
            if (exp & 1) pow_val *= base;
            base *= base;
            exp >>= 1;
        }
        if (ip < 0) pow_val = 1.0f / pow_val;
        out[i] = pow_val * w1[i] + b[i];
    }
}

// Broadcast variant: w0, w1, and b may broadcast along the leading dimension
// (they have shape [1, C] while x has shape [N, C]).
// broadcast_stride is the number of elements per broadcast row (e.g., C when
// broadcasting [1, C] over [N, C]).
// Same integer-only exponent contract as the same-shape kernel above:
// the host wrapper guarantees `p` is an integer value stored in f32.
extern "C" __global__ void fused_mul_sin_pow_mul_add_broadcast_kernel(
    float* out,
    const float* x,
    const float* w0,
    const float* w1,
    const float* b,
    float p,
    size_t n,
    size_t broadcast_stride
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        size_t bi = i % broadcast_stride;
        float s = sinf(x[i] * w0[bi]);
        int ip = (int) p;
        float pow_val = 1.0f;
        float base = s;
        int exp = ip < 0 ? -ip : ip;
        while (exp > 0) {
            if (exp & 1) pow_val *= base;
            base *= base;
            exp >>= 1;
        }
        if (ip < 0) pow_val = 1.0f / pow_val;
        out[i] = pow_val * w1[bi] + b[bi];
    }
}

// Per-channel-scale broadcast variant:
//   x  is per-channel scalar, shape [..., C, 1]  (small_total = C)
//   w0 is full shape,         shape [..., C, K]  (full_total  = C*K)
//   w1 is per-channel scalar, shape [..., C, 1]
//   b  is full shape,         shape [..., C, K]
//
// Each output element [i] (in flat row-major layout of the full shape)
// uses x[i / k_dim] and w1[i / k_dim] (broadcast over the trailing
// dim), and w0[i], b[i] at full index.
//
// `k_dim` is the size of the trailing dimension (the one being broadcast
// over). `n` is the total output element count (= small_total * k_dim).
extern "C" __global__ void fused_mul_sin_pow_mul_add_per_channel_scale_kernel(
    float* out,
    const float* x,
    const float* w0,
    const float* w1,
    const float* b,
    float p,
    size_t n,
    size_t k_dim
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        size_t chan_idx = i / k_dim;
        float s = sinf(x[chan_idx] * w0[i]);
        // Integer exponentiation by squaring. p is guaranteed to be an
        // integer by the host guard before this kernel is invoked.
        int ip = (int) p;
        float pow_val = 1.0f;
        float base = s;
        int exp = ip < 0 ? -ip : ip;
        while (exp > 0) {
            if (exp & 1) pow_val *= base;
            base *= base;
            exp >>= 1;
        }
        if (ip < 0) pow_val = 1.0f / pow_val;
        out[i] = pow_val * w1[chan_idx] + b[i];
    }
}
"#;

/// Compute optimal launch configuration for elementwise operations
pub(super) fn elementwise_config(n: usize) -> LaunchConfig {
    LaunchConfig::for_num_elems(n as u32)
}

/// Compute the broadcast stride for last-dimension broadcasting.
/// Returns Some(stride) if variance can be broadcast to numerator by repeating
/// each variance element `stride` times along the last dimension(s).
/// Returns None if broadcasting is not possible with this simple pattern.
///
/// The simple stride formula (i / stride) only works when broadcast dimensions
/// are contiguous at the trailing end of the shape.
pub(super) fn compute_broadcast_stride(num_shape: &[usize], var_shape: &[usize]) -> Option<usize> {
    // Must have same number of dimensions
    if num_shape.len() != var_shape.len() {
        return None;
    }

    // Iterate in reverse (from last to first dimension)
    // Broadcast dims (v=1, n>1) must be contiguous at the end
    let mut stride = 1usize;
    let mut found_broadcast = false;
    let mut seen_matching_gt_1 = false;

    for (n, v) in num_shape.iter().zip(var_shape.iter()).rev() {
        if *n == *v {
            // Matching dimension
            if *n > 1 {
                seen_matching_gt_1 = true;
            }
        } else if *v == 1 {
            // Broadcast dimension: variance has 1, numerator has > 1
            if seen_matching_gt_1 {
                // There's a matching dim at higher index than this broadcast
                // e.g., [1, 512, 5] vs [1, 1, 5] - can't use simple stride formula
                return None;
            }
            stride *= n;
            found_broadcast = true;
        } else {
            // Incompatible dimensions (n != v and v != 1)
            return None;
        }
    }

    if found_broadcast {
        Some(stride)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::memory_pool::GpuMemoryPool;
    use crate::cuda::tensor::GpuTensor;

    fn setup() -> (IconnxCudaContext, FusedKernelCache, GpuMemoryPool) {
        let ctx = IconnxCudaContext::new().expect("Failed to create CUDA context");
        let cache = FusedKernelCache::new(&ctx).expect("Failed to compile fused kernels");
        let pool = GpuMemoryPool::new();
        (ctx, cache, pool)
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_fused_div_rsqrt() {
        let (ctx, cache, mut pool) = setup();

        // Test: numerator / sqrt(variance + eps)
        // numerator = [4.0, 9.0, 16.0]
        // variance = [3.0, 8.0, 15.0]
        // eps = 1.0
        // sqrt(variance + eps) = [2.0, 3.0, 4.0]
        // result = [2.0, 3.0, 4.0]
        let numerator = GpuTensor::from_host_f32(&ctx, &[4.0, 9.0, 16.0], vec![3]).unwrap();
        let variance = GpuTensor::from_host_f32(&ctx, &[3.0, 8.0, 15.0], vec![3]).unwrap();

        let result = gpu_fused_div_rsqrt(&ctx, &cache, &mut pool, &numerator, &variance, 1.0).unwrap();
        let host_result = result.to_host_f32(&ctx).unwrap();

        assert!((host_result[0] - 2.0).abs() < 1e-5);
        assert!((host_result[1] - 3.0).abs() < 1e-5);
        assert!((host_result[2] - 4.0).abs() < 1e-5);
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_fused_add_mul_add() {
        let (ctx, cache, mut pool) = setup();

        // Test: (x + a) * b + c
        // x = [1.0, 2.0, 3.0]
        // a = [0.5, 0.5, 0.5]
        // b = [2.0, 2.0, 2.0]
        // c = [1.0, 1.0, 1.0]
        // (x + a) = [1.5, 2.5, 3.5]
        // * b = [3.0, 5.0, 7.0]
        // + c = [4.0, 6.0, 8.0]
        let x = GpuTensor::from_host_f32(&ctx, &[1.0, 2.0, 3.0], vec![3]).unwrap();
        let a = GpuTensor::from_host_f32(&ctx, &[0.5, 0.5, 0.5], vec![3]).unwrap();
        let b = GpuTensor::from_host_f32(&ctx, &[2.0, 2.0, 2.0], vec![3]).unwrap();
        let c = GpuTensor::from_host_f32(&ctx, &[1.0, 1.0, 1.0], vec![3]).unwrap();

        let result = gpu_fused_add_mul_add(&ctx, &cache, &mut pool, &x, &a, &b, &c).unwrap();
        let host_result = result.to_host_f32(&ctx).unwrap();

        assert!((host_result[0] - 4.0).abs() < 1e-5);
        assert!((host_result[1] - 6.0).abs() < 1e-5);
        assert!((host_result[2] - 8.0).abs() < 1e-5);
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_fused_gelu() {
        let (ctx, cache, mut pool) = setup();

        // Test: GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        // Reference values computed via:
        // import torch; torch.nn.functional.gelu(torch.tensor([0.0, 1.0, -1.0, 2.0]))
        // tensor([0.0000, 0.8413, -0.1587, 1.9545])
        let x = GpuTensor::from_host_f32(&ctx, &[0.0, 1.0, -1.0, 2.0], vec![4]).unwrap();

        let result = gpu_fused_gelu(&ctx, &cache, &mut pool, &x).unwrap();
        let host_result = result.to_host_f32(&ctx).unwrap();

        // GELU(0) = 0
        assert!(host_result[0].abs() < 1e-5, "GELU(0) = {}", host_result[0]);
        // GELU(1) ≈ 0.8413
        assert!(
            (host_result[1] - 0.8413).abs() < 1e-3,
            "GELU(1) = {}",
            host_result[1]
        );
        // GELU(-1) ≈ -0.1587
        assert!(
            (host_result[2] - (-0.1587)).abs() < 1e-3,
            "GELU(-1) = {}",
            host_result[2]
        );
        // GELU(2) ≈ 1.9545
        assert!(
            (host_result[3] - 1.9545).abs() < 1e-3,
            "GELU(2) = {}",
            host_result[3]
        );
    }

    #[test]
    fn test_compute_broadcast_stride() {
        // [1, 512, 5] vs [1, 512, 1] -> stride = 5
        assert_eq!(
            compute_broadcast_stride(&[1, 512, 5], &[1, 512, 1]),
            Some(5)
        );

        // [1, 512, 10] vs [1, 512, 1] -> stride = 10
        assert_eq!(
            compute_broadcast_stride(&[1, 512, 10], &[1, 512, 1]),
            Some(10)
        );

        // Same shape -> None (no broadcast needed, use regular kernel)
        assert_eq!(compute_broadcast_stride(&[1, 512, 5], &[1, 512, 5]), None);

        // Different rank -> None
        assert_eq!(compute_broadcast_stride(&[1, 512, 5], &[512, 1]), None);

        // Incompatible dimensions -> None
        assert_eq!(compute_broadcast_stride(&[1, 512, 5], &[1, 256, 1]), None);

        // Middle broadcast not supported -> None
        assert_eq!(compute_broadcast_stride(&[1, 512, 5], &[1, 1, 5]), None);

        // Multiple trailing broadcast dimensions
        // [2, 4, 8] vs [2, 1, 1] -> stride = 4 * 8 = 32
        assert_eq!(compute_broadcast_stride(&[2, 4, 8], &[2, 1, 1]), Some(32));
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_fused_div_rsqrt_broadcast() {
        let (ctx, cache, mut pool) = setup();

        // Test: numerator / sqrt(variance + eps) with broadcasting
        // numerator shape: [1, 2, 3] (6 elements)
        // variance shape: [1, 2, 1] (2 elements, broadcast last dim)
        // eps = 0.0
        //
        // numerator = [[1, 2, 3], [4, 5, 6]]  (reshaped as [1,2,3])
        // variance = [[4], [9]]  (reshaped as [1,2,1])
        // sqrt(variance) = [[2], [3]]
        // result = [[0.5, 1.0, 1.5], [1.33, 1.67, 2.0]]
        let numerator = GpuTensor::from_host_f32(
            &ctx,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![1, 2, 3],
        )
        .unwrap();
        let variance = GpuTensor::from_host_f32(&ctx, &[4.0, 9.0], vec![1, 2, 1]).unwrap();

        let result = gpu_fused_div_rsqrt(&ctx, &cache, &mut pool, &numerator, &variance, 0.0).unwrap();
        let host_result = result.to_host_f32(&ctx).unwrap();

        // First row: [1, 2, 3] / sqrt(4) = [0.5, 1.0, 1.5]
        assert!((host_result[0] - 0.5).abs() < 1e-5);
        assert!((host_result[1] - 1.0).abs() < 1e-5);
        assert!((host_result[2] - 1.5).abs() < 1e-5);

        // Second row: [4, 5, 6] / sqrt(9) = [1.33, 1.67, 2.0]
        assert!((host_result[3] - 4.0 / 3.0).abs() < 1e-5);
        assert!((host_result[4] - 5.0 / 3.0).abs() < 1e-5);
        assert!((host_result[5] - 2.0).abs() < 1e-5);
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_fused_mul_add() {
        let (ctx, cache, mut pool) = setup();

        // Test: a * b + c
        // a = [1.0, 2.0, 3.0]
        // b = [2.0, 3.0, 4.0]
        // c = [0.5, 1.0, 1.5]
        // result = [2.5, 7.0, 13.5]
        let a = GpuTensor::from_host_f32(&ctx, &[1.0, 2.0, 3.0], vec![3]).unwrap();
        let b = GpuTensor::from_host_f32(&ctx, &[2.0, 3.0, 4.0], vec![3]).unwrap();
        let c = GpuTensor::from_host_f32(&ctx, &[0.5, 1.0, 1.5], vec![3]).unwrap();

        let result = gpu_fused_mul_add(&ctx, &cache, &mut pool, &a, &b, &c).unwrap();
        let host_result = result.to_host_f32(&ctx).unwrap();

        // 1.0 * 2.0 + 0.5 = 2.5
        assert!((host_result[0] - 2.5).abs() < 1e-5);
        // 2.0 * 3.0 + 1.0 = 7.0
        assert!((host_result[1] - 7.0).abs() < 1e-5);
        // 3.0 * 4.0 + 1.5 = 13.5
        assert!((host_result[2] - 13.5).abs() < 1e-5);
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_fused_add_mul() {
        let (ctx, cache, mut pool) = setup();

        // Test: (a + b) * c
        // a = [1.0, 2.0, 3.0]
        // b = [0.5, 1.0, 1.5]
        // c = [2.0, 3.0, 4.0]
        // result = [3.0, 9.0, 18.0]
        let a = GpuTensor::from_host_f32(&ctx, &[1.0, 2.0, 3.0], vec![3]).unwrap();
        let b = GpuTensor::from_host_f32(&ctx, &[0.5, 1.0, 1.5], vec![3]).unwrap();
        let c = GpuTensor::from_host_f32(&ctx, &[2.0, 3.0, 4.0], vec![3]).unwrap();

        let result = gpu_fused_add_mul(&ctx, &cache, &mut pool, &a, &b, &c).unwrap();
        let host_result = result.to_host_f32(&ctx).unwrap();

        // (1.0 + 0.5) * 2.0 = 3.0
        assert!((host_result[0] - 3.0).abs() < 1e-5);
        // (2.0 + 1.0) * 3.0 = 9.0
        assert!((host_result[1] - 9.0).abs() < 1e-5);
        // (3.0 + 1.5) * 4.0 = 18.0
        assert!((host_result[2] - 18.0).abs() < 1e-5);
    }
}
