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
    // M4.6 MatMulInteger: 4 sign-combination variants.
    "matmul_integer_s8s8_kernel",
    "matmul_integer_u8s8_kernel",
    "matmul_integer_s8u8_kernel",
    "matmul_integer_u8u8_kernel",
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

// ============================================================================
// WS-4 M4.6 MatMulInteger: integer GEMM with optional zero-point subtraction.
//
// Four sign-combination variants — s8s8, u8s8, s8u8, u8u8 — to cover every
// (a, b) dtype pairing the ONNX spec admits. DistilBERT-INT8 specifically
// uses the u8s8 variant (UINT8 activations × INT8 weights).
//
// Naive one-thread-per-output GEMM. Each thread computes a single
// `out[m, n]` cell over the full inner dim. Shared-memory tiling is a
// perf-workstream follow-up; correctness is the gate for M4.6.
//
// Per-tensor scalar zps are passed as 1-element device buffers; the kernel
// reads `zp[0]`. zero-pointers for absent zp are NOT supported by the
// typed_kernel ABI — the wrapper synthesizes 1-element zero buffers when
// the ONNX node omits zp (matches the M4.4 DequantizeLinear convention).
//
// Shape layout: `a` is row-major `[M, K]`, `b` is row-major `[K, N]`,
// `out` is row-major `[M, N]`. Output is always Int32 — no scale is
// applied here; that's a separate downstream op (Mul or fused
// DequantizeLinear).
//
// Worst-case accumulator: K * 127 * 127 ≈ K * 16k. K=3072 (DistilBERT FFN)
// → ~50M, well within INT32_MAX (2.1e9). Input range bounds verified by
// the matmul_integer_k_3072_overflow_safety_distilbert_ffn test on the
// CPU forward (the GPU oracle).
//
// All four kernels share an identical signature shape:
//   (int* out, const T_a* a, const T_b* b, const T_a* a_zp,
//    const T_b* b_zp, size_t m, size_t k_dim, size_t n)
// so the wrapper's typed_kernel tuple structure changes only the operand
// element types. Promotion to `int` happens explicitly per-element so
// signed/unsigned extension is unambiguous.
// ============================================================================
extern "C" __global__ void matmul_integer_s8s8_kernel(
    int* out, const signed char* a, const signed char* b,
    const signed char* a_zp, const signed char* b_zp,
    size_t m, size_t k_dim, size_t n
) {
    size_t mn = blockIdx.x * blockDim.x + threadIdx.x;
    if (mn >= m * n) return;
    size_t row = mn / n;
    size_t col = mn % n;
    int azp = (int)a_zp[0];
    int bzp = (int)b_zp[0];
    int acc = 0;
    for (size_t kk = 0; kk < k_dim; ++kk) {
        int av = (int)a[row * k_dim + kk] - azp;
        int bv = (int)b[kk * n + col] - bzp;
        acc += av * bv;
    }
    out[mn] = acc;
}

extern "C" __global__ void matmul_integer_u8s8_kernel(
    int* out, const unsigned char* a, const signed char* b,
    const unsigned char* a_zp, const signed char* b_zp,
    size_t m, size_t k_dim, size_t n
) {
    size_t mn = blockIdx.x * blockDim.x + threadIdx.x;
    if (mn >= m * n) return;
    size_t row = mn / n;
    size_t col = mn % n;
    int azp = (int)a_zp[0];
    int bzp = (int)b_zp[0];
    int acc = 0;
    for (size_t kk = 0; kk < k_dim; ++kk) {
        int av = (int)a[row * k_dim + kk] - azp;
        int bv = (int)b[kk * n + col] - bzp;
        acc += av * bv;
    }
    out[mn] = acc;
}

extern "C" __global__ void matmul_integer_s8u8_kernel(
    int* out, const signed char* a, const unsigned char* b,
    const signed char* a_zp, const unsigned char* b_zp,
    size_t m, size_t k_dim, size_t n
) {
    size_t mn = blockIdx.x * blockDim.x + threadIdx.x;
    if (mn >= m * n) return;
    size_t row = mn / n;
    size_t col = mn % n;
    int azp = (int)a_zp[0];
    int bzp = (int)b_zp[0];
    int acc = 0;
    for (size_t kk = 0; kk < k_dim; ++kk) {
        int av = (int)a[row * k_dim + kk] - azp;
        int bv = (int)b[kk * n + col] - bzp;
        acc += av * bv;
    }
    out[mn] = acc;
}

extern "C" __global__ void matmul_integer_u8u8_kernel(
    int* out, const unsigned char* a, const unsigned char* b,
    const unsigned char* a_zp, const unsigned char* b_zp,
    size_t m, size_t k_dim, size_t n
) {
    size_t mn = blockIdx.x * blockDim.x + threadIdx.x;
    if (mn >= m * n) return;
    size_t row = mn / n;
    size_t col = mn % n;
    int azp = (int)a_zp[0];
    int bzp = (int)b_zp[0];
    int acc = 0;
    for (size_t kk = 0; kk < k_dim; ++kk) {
        int av = (int)a[row * k_dim + kk] - azp;
        int bv = (int)b[kk * n + col] - bzp;
        acc += av * bv;
    }
    out[mn] = acc;
}
"#;
