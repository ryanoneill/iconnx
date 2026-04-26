//! WS-3 M3.4 sub-2 — FP16 elementwise NVRTC kernels.
//!
//! These mirror the f32 elementwise kernels in `mod.rs` but operate on
//! `__half` operands. Unlike the memcpy-class FP16 kernels in
//! `cuda/ops/kernels_fp16.rs`, these read FP values and perform
//! arithmetic, so the C-side parameter type is `__half*` (from
//! `<cuda_fp16.h>`), which is layout-compatible with `half::f16` —
//! 2-byte aligned, 2 bytes wide, single `unsigned short` field. Garboard's
//! `TypedKernel<&DeviceSlice<'_, half::f16>>` paired with `__half*` works
//! because the C ABI sees both as 8-byte device pointers at launch time.
//!
//! ## Op-level precision strategy
//!
//! - **Add / Sub / Mul / Div**: pure FP16 via `__hadd` / `__hsub` /
//!   `__hmul` / `__hdiv`. Single binary op, no accumulation — round-half-
//!   to-even semantics match CPU reference.
//! - **Sqrt**: pure FP16 via `hsqrt`. Square-root is monotone over the
//!   non-negative range, no intermediate overflow risk.
//! - **Pow / Erf**: round-trip through f32 via `__half2float` / `powf` /
//!   `erff` / `__float2half`. CUDA has no `__half`-typed pow or erf; the
//!   f32 round-trip is the standard ONNX runtime approach (mirrors the
//!   CUDNN_DATA_HALF reduction pattern with f32 accumulator). Worst-case
//!   precision loss is bounded by the f16 → f32 → f16 round-trip ULP.
//!
//! Lives in its own file rather than appending to `mod.rs` (already at
//! ~1191 lines, past the project's 1000-line guideline). Source is
//! concatenated into the elementwise garboard `Module` by `KernelCache::new`.

/// Kernel names compiled from this module's source. Validated at cache
/// creation alongside `super::KERNEL_NAMES`.
pub const KERNEL_NAMES_FP16: &[&str] = &[
    "add_f16_kernel",
    "sub_f16_kernel",
    "mul_f16_kernel",
    "div_f16_kernel",
    "sqrt_f16_kernel",
    "pow_f16_kernel",
    "erf_f16_kernel",
];

/// CUDA kernel source for the WS-3 FP16 elementwise op family.
///
/// `<cuda_fp16.h>` is supplied by NVRTC's default include path on every
/// CUDA toolkit ≥ 9.0; no extra include directory needed.
pub const ELEMENTWISE_KERNELS_FP16: &str = r#"
#include <cuda_fp16.h>

// =============================================================================
// FP16 binary elementwise: out[i] = a[i] op b[i]
// =============================================================================

extern "C" __global__ void add_f16_kernel(__half* out, const __half* a, const __half* b, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __hadd(a[i], b[i]);
    }
}

extern "C" __global__ void sub_f16_kernel(__half* out, const __half* a, const __half* b, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __hsub(a[i], b[i]);
    }
}

extern "C" __global__ void mul_f16_kernel(__half* out, const __half* a, const __half* b, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __hmul(a[i], b[i]);
    }
}

// FP16 Div with the same epsilon-protection pattern as div_kernel: prevents
// 0/0 = NaN by clamping the denominator's magnitude. Computed in f32 then
// rounded back so the `fabsf < eps` predicate stays meaningful below f16's
// 6e-5 minimum normal.
extern "C" __global__ void div_f16_kernel(__half* out, const __half* a, const __half* b, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float eps = 1e-8f;
        float denom = __half2float(b[i]);
        if (fabsf(denom) < eps) {
            denom = (denom >= 0.0f) ? eps : -eps;
        }
        out[i] = __float2half(__half2float(a[i]) / denom);
    }
}

// =============================================================================
// FP16 unary math: out[i] = f(x[i])
// =============================================================================

// Pure FP16 sqrt with non-negative guard; matches the f32 sqrt_kernel.
extern "C" __global__ void sqrt_f16_kernel(__half* out, const __half* x, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        __half v = x[i];
        // __hge: half >=. Substitute zero for negative inputs.
        out[i] = __hge(v, __float2half(0.0f)) ? hsqrt(v) : __float2half(0.0f);
    }
}

// Pow rounds through f32 — no `__half`-typed `pow` intrinsic exists.
extern "C" __global__ void pow_f16_kernel(__half* out, const __half* base, const __half* exp_vals, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __float2half(powf(__half2float(base[i]), __half2float(exp_vals[i])));
    }
}

// Erf rounds through f32 — same reasoning as pow.
extern "C" __global__ void erf_f16_kernel(__half* out, const __half* x, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __float2half(erff(__half2float(x[i])));
    }
}
"#;
