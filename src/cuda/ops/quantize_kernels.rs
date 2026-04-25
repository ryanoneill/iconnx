//! WS-4 quantization NVRTC kernels — `DequantizeLinear` (M4.4),
//! `DynamicQuantizeLinear` (M4.5), `MatMulInteger` (M4.6).
//!
//! Lives in its own file to avoid piling onto `src/cuda/ops/kernels.rs`,
//! which is already past the project's 1000-line guideline. Both source
//! strings are concatenated at runtime by `OpsKernelCache::new` and
//! compiled into a single garboard `Module`, mirroring the
//! `src/cuda/conv/kernels.rs` precedent.

/// Kernel names compiled from this module's source. Validated at cache
/// creation alongside `super::kernels::KERNEL_NAMES`.
pub const QUANTIZE_KERNEL_NAMES: &[&str] = &[
    // M4.4 DequantizeLinear: y = (x - zp) * scale.
    "dequantize_linear_i8_kernel",
    "dequantize_linear_u8_kernel",
    "dequantize_linear_i32_kernel",
];

/// CUDA kernel source for the WS-4 quantization op family.
pub const QUANTIZE_KERNELS: &str = r#"
// ============================================================================
// WS-4 M4.4 DequantizeLinear: y = (x - zp) * scale.
//
// Per-element axis coordinate is recovered from the flat index `i`:
//   axis_coord = (i / inner) % axis_size
// where `inner = product(shape[axis+1..])` and `axis_size = shape[axis]`.
//
// Per-tensor (scalar scale) is encoded by callers as `axis_size = 1` and
// `inner = n` so the modulo collapses to 0 — a single scale/zp covers the
// whole tensor without a separate kernel.
//
// `zp` is never null: callers synthesize a zero buffer when the ONNX node
// omits the optional input. This keeps the typed_kernel signature uniform
// across the (zp present / zp absent) cases without nullable device pointers.
// ============================================================================
extern "C" __global__ void dequantize_linear_i8_kernel(
    float* out, const signed char* x, const float* scale, const signed char* zp,
    size_t n, size_t axis_size, size_t inner
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    size_t a = (i / inner) % axis_size;
    out[i] = (float)((int)x[i] - (int)zp[a]) * scale[a];
}

extern "C" __global__ void dequantize_linear_u8_kernel(
    float* out, const unsigned char* x, const float* scale, const unsigned char* zp,
    size_t n, size_t axis_size, size_t inner
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    size_t a = (i / inner) % axis_size;
    out[i] = (float)((int)x[i] - (int)zp[a]) * scale[a];
}

extern "C" __global__ void dequantize_linear_i32_kernel(
    float* out, const int* x, const float* scale, const int* zp,
    size_t n, size_t axis_size, size_t inner
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    size_t a = (i / inner) % axis_size;
    // Promote to long long before subtracting to avoid INT_MIN-INT_MIN UB.
    out[i] = (float)((long long)x[i] - (long long)zp[a]) * scale[a];
}
"#;
