//! WS-3.5 Y(2) sub-2b — BF16 elementwise NVRTC kernels.
//!
//! These mirror the f32 elementwise kernels in `mod.rs` but operate on
//! `__nv_bfloat16` operands. Unlike the memcpy-class BF16 kernels in
//! `cuda/ops/kernels_bf16.rs`, these read FP values and perform
//! arithmetic, so the C-side parameter type is `__nv_bfloat16*` (from
//! `<cuda_bf16.h>`), which is layout-compatible with `half::bf16` —
//! 2-byte aligned, 2 bytes wide, single `unsigned short` field.
//! Garboard's `TypedKernel<&DeviceSlice<'_, half::bf16>>` paired with
//! `__nv_bfloat16*` works because the C ABI sees both as 8-byte device
//! pointers at launch time.
//!
//! ## Op-level precision strategy
//!
//! - **Add / Sub / Mul / Div**: pure BF16 via `__hadd` / `__hsub` /
//!   `__hmul` / `__hdiv` on `__nv_bfloat16` (sm_80+ via `<cuda_bf16.h>`).
//!   Single binary op, no accumulation — round-half-to-even semantics
//!   match CPU reference.
//! - **Sqrt / Pow / Erf**: round-trip through f32 via
//!   `__bfloat162float` / `sqrtf` / `powf` / `erff` /
//!   `__float2bfloat16_rn`. CUDA has no `__nv_bfloat16`-typed pow, erf,
//!   or sqrt intrinsic; the f32 round-trip is the standard ONNX runtime
//!   approach. Worst-case precision loss is bounded by the bf16 → f32 →
//!   bf16 round-trip ULP.
//!
//! Lives in its own file rather than appending to `mod.rs` (already
//! past the project's 1000-line guideline). Source is concatenated into
//! the elementwise garboard `Module` by `KernelCache::new`. Sub-2e will
//! append broadcast-scalar Add and Sin/Cos to the same source string.

/// Kernel names compiled from this module's source. Validated at cache
/// creation alongside `super::KERNEL_NAMES` and
/// `super::kernels_fp16::KERNEL_NAMES_FP16`.
pub const KERNEL_NAMES_BF16: &[&str] = &[
    "add_bf16_kernel",
    "sub_bf16_kernel",
    "mul_bf16_kernel",
    "div_bf16_kernel",
    "sqrt_bf16_kernel",
    "pow_bf16_kernel",
    "erf_bf16_kernel",
];

/// CUDA kernel source for the WS-3.5 BF16 elementwise op family.
///
/// `<cuda_bf16.h>` is supplied by NVRTC's default include path on every
/// CUDA toolkit ≥ 11.0; no extra include directory needed. The
/// `__hadd`/`__hsub`/`__hmul`/`__hdiv` intrinsics on `__nv_bfloat16`
/// require sm_80+ (Ampere); the hardware-capability gate at
/// `GpuTensor::zeros_bf16` / `from_host_bf16` enforces this at
/// construction so dispatch sites can assume the runtime device is
/// sm_80+.
pub const KERNELS_BF16: &str = r#"
#include <cuda_bf16.h>

// =============================================================================
// BF16 binary elementwise: out[i] = a[i] op b[i]
// =============================================================================

extern "C" __global__ void add_bf16_kernel(__nv_bfloat16* out, const __nv_bfloat16* a, const __nv_bfloat16* b, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __hadd(a[i], b[i]);
    }
}

extern "C" __global__ void sub_bf16_kernel(__nv_bfloat16* out, const __nv_bfloat16* a, const __nv_bfloat16* b, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __hsub(a[i], b[i]);
    }
}

extern "C" __global__ void mul_bf16_kernel(__nv_bfloat16* out, const __nv_bfloat16* a, const __nv_bfloat16* b, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __hmul(a[i], b[i]);
    }
}

// BF16 Div with the same epsilon-protection pattern as div_kernel: prevents
// 0/0 = NaN by clamping the denominator's magnitude. Computed in f32 then
// rounded back so the `fabsf < eps` predicate stays meaningful below
// bf16's 1.18e-38 minimum normal (BF16 has FP32-range exponent so the
// epsilon is well within the representable range, but BF16 mantissa
// resolution is coarser than FP16).
extern "C" __global__ void div_bf16_kernel(__nv_bfloat16* out, const __nv_bfloat16* a, const __nv_bfloat16* b, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float eps = 1e-8f;
        float denom = __bfloat162float(b[i]);
        if (fabsf(denom) < eps) {
            denom = (denom >= 0.0f) ? eps : -eps;
        }
        out[i] = __float2bfloat16_rn(__bfloat162float(a[i]) / denom);
    }
}

// =============================================================================
// BF16 unary math: out[i] = f(x[i])
// =============================================================================

// BF16 sqrt rounds through f32 — `hsqrt` exists for __half but not
// __nv_bfloat16, so f32 is the standard route (and avoids the precision
// trap of pure-bf16 reciprocal-sqrt approximations).
extern "C" __global__ void sqrt_bf16_kernel(__nv_bfloat16* out, const __nv_bfloat16* x, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = __bfloat162float(x[i]);
        // Guard against sqrt(negative) — match the f32 sqrt_kernel's policy.
        out[i] = __float2bfloat16_rn(v >= 0.0f ? sqrtf(v) : 0.0f);
    }
}

// Pow rounds through f32 — no `__nv_bfloat16`-typed `pow` intrinsic exists.
extern "C" __global__ void pow_bf16_kernel(__nv_bfloat16* out, const __nv_bfloat16* base, const __nv_bfloat16* exp_vals, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __float2bfloat16_rn(powf(__bfloat162float(base[i]), __bfloat162float(exp_vals[i])));
    }
}

// Erf rounds through f32 — same reasoning as pow.
extern "C" __global__ void erf_bf16_kernel(__nv_bfloat16* out, const __nv_bfloat16* x, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __float2bfloat16_rn(erff(__bfloat162float(x[i])));
    }
}
"#;
