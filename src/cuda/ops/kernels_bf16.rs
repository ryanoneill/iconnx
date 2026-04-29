//! WS-3.5 BF16 NVRTC kernels — BFloat16 mirrors of the memcpy-class
//! layout ops (transpose, concat, gather, expand, copy) used by the BF16
//! dispatch breadth wave (Y(2) sub-2a) plus the Cast dtype-pair kernels
//! used by Y(2) sub-2d.
//!
//! Memcpy-class kernels are pure index gymnastics — no arithmetic on
//! the element values — so they only need byte-level identity with
//! `half::bf16`. The C-side parameter type is `unsigned short*` because
//! `half::bf16` is `#[repr(transparent)] struct(u16)`, sharing size and
//! alignment with `unsigned short`. This avoids pulling in
//! `<cuda_bf16.h>` for symbols that never reference the BF16 numeric
//! value.
//!
//! Cast kernels DO need `<cuda_bf16.h>` because they read FP values via
//! `__bfloat162float` / `__float2bfloat16_rn`. The FP16↔BF16 pair routes
//! via f32 intermediate per spec §2 — ONNX doesn't define a direct
//! half-half cast and iconnx avoids precision-coupling between the two
//! half formats.
//!
//! Lives in its own file (parallel to `kernels_fp16.rs`) per the
//! project's 1000-line file guideline; source is concatenated into the
//! single garboard `Module` by `OpsKernelCache::new`.

/// Kernel names compiled from this module's source. Validated at cache
/// creation alongside `super::kernels::KERNEL_NAMES`.
pub const KERNEL_NAMES_BF16: &[&str] = &[
    // Y(2) sub-2a (memcpy-class BF16 layout ops):
    "transpose_general_bf16_kernel",
    "concat_bf16_kernel",
    "gather_bf16_kernel",
    "expand_bf16_kernel",
    "copy_bf16_kernel",
    // Y(2) sub-2d (Cast — BF16 dtype-pair arms):
    "cast_bf16_to_f32_kernel",
    "cast_f32_to_bf16_kernel",
    "cast_bf16_to_f16_kernel",
    "cast_f16_to_bf16_kernel",
    "cast_bf16_to_i32_kernel",
    "cast_i32_to_bf16_kernel",
    "cast_bf16_to_i64_kernel",
    "cast_i64_to_bf16_kernel",
];

/// CUDA kernel source for the WS-3.5 BF16 op family. See module docs
/// for why `unsigned short*` is the correct C type for memcpy-only
/// kernels. `<cuda_bf16.h>` is needed for the Cast kernels which read
/// FP values via `__bfloat162float` / `__float2bfloat16_rn`.
pub const OPS_KERNELS_BF16: &str = r#"
#include <cuda_bf16.h>

// ============================================================================
// WS-3.5 Y(2) sub-2a Transpose (BFloat16). Mirrors `transpose_general_kernel`
// in `kernels.rs` byte-for-byte with `unsigned short*` data pointers.
// ============================================================================
extern "C" __global__ void transpose_general_bf16_kernel(
    unsigned short* out, const unsigned short* inp,
    const size_t* in_shape, const size_t* in_strides,
    const size_t* out_strides, const size_t* perm,
    size_t ndim, size_t total_elements
) {
    size_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= total_elements) return;
    size_t out_coords[6];
    size_t remaining = out_idx;
    for (size_t d = 0; d < ndim; d++) {
        out_coords[d] = remaining / out_strides[d];
        remaining = remaining % out_strides[d];
    }
    size_t in_idx = 0;
    for (size_t d = 0; d < ndim; d++) {
        in_idx += out_coords[d] * in_strides[perm[d]];
    }
    out[out_idx] = inp[in_idx];
}

// ============================================================================
// WS-3.5 Y(2) sub-2a Concat (BFloat16). Mirrors `concat_kernel` byte-for-byte.
// ============================================================================
extern "C" __global__ void concat_bf16_kernel(
    unsigned short* out, const unsigned short* inp,
    size_t inp_offset, size_t inp_axis_size,
    size_t outer_size, size_t inner_size,
    size_t out_axis_size, size_t total_inp_elements
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_inp_elements) return;
    size_t outer = idx / (inp_axis_size * inner_size);
    size_t rem = idx % (inp_axis_size * inner_size);
    size_t axis_pos = rem / inner_size;
    size_t inner = rem % inner_size;
    size_t out_axis_pos = inp_offset + axis_pos;
    size_t out_idx = outer * (out_axis_size * inner_size) + out_axis_pos * inner_size + inner;
    out[out_idx] = inp[idx];
}

// ============================================================================
// WS-3.5 Y(2) sub-2a Gather (BFloat16). Mirrors `gather_kernel` byte-for-byte.
// Negative indices are wrapped via the standard Python/ONNX convention;
// out-of-bounds clamps to [0, dim0_size-1] to match the f32 / i64 / u8 arms.
// ============================================================================
extern "C" __global__ void gather_bf16_kernel(
    unsigned short* out, const unsigned short* inp, const long long* indices,
    size_t num_indices, size_t slice_size, size_t dim0_size
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_indices * slice_size) return;
    size_t which_index = idx / slice_size;
    size_t offset_in_slice = idx % slice_size;
    long long raw_index = indices[which_index];
    long long adjusted = (raw_index < 0) ? (raw_index + (long long)dim0_size) : raw_index;
    size_t actual_index;
    if (adjusted < 0) {
        actual_index = 0;
    } else if ((size_t)adjusted >= dim0_size) {
        actual_index = dim0_size - 1;
    } else {
        actual_index = (size_t)adjusted;
    }
    size_t in_idx = actual_index * slice_size + offset_in_slice;
    out[idx] = inp[in_idx];
}

// ============================================================================
// WS-3.5 Y(2) sub-2a Expand (BFloat16). Mirrors `expand_kernel` byte-for-byte.
// ============================================================================
extern "C" __global__ void expand_bf16_kernel(
    unsigned short* out, const unsigned short* inp,
    const size_t* out_shape, const size_t* out_strides,
    const size_t* inp_shape, const size_t* inp_strides,
    size_t ndim, size_t total_elements
) {
    size_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= total_elements) return;
    size_t coords[8];
    size_t remaining = out_idx;
    for (size_t d = 0; d < ndim; d++) {
        coords[d] = remaining / out_strides[d];
        remaining = remaining % out_strides[d];
    }
    size_t inp_idx = 0;
    for (size_t d = 0; d < ndim; d++) {
        size_t coord = (inp_shape[d] == 1) ? 0 : coords[d];
        inp_idx += coord * inp_strides[d];
    }
    out[out_idx] = inp[inp_idx];
}

// ============================================================================
// WS-3.5 Y(2) sub-2a Copy (BFloat16). Mirrors `copy_kernel` (f32 variant)
// byte-for-byte. WS-1 M1.4's gpu_copy is per-dtype dispatch with explicit
// kernels; BF16 is the new arm.
// ============================================================================
extern "C" __global__ void copy_bf16_kernel(
    unsigned short* out, const unsigned short* inp, size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = inp[i];
    }
}

// ============================================================================
// WS-3.5 Y(2) sub-2d Cast — BF16 dtype-pair kernels.
//
// `__float2bfloat16_rn` is round-to-nearest-even, matching IEEE bfloat16's
// default rounding mode and `half::bf16::from_f32` on the CPU side. The
// integer ↔ BF16 pairs go through f32 (`(float)x` then
// `__float2bfloat16_rn`, or `__bfloat162float` then `(int)x` /
// `(long long)x`) — there's no `__nv_bfloat16`-typed integer conversion
// intrinsic, and going via f32 keeps the same truncate-toward-zero
// semantics ONNX Cast specifies for integer targets.
//
// Half-half casts (FP16↔BF16) go through f32 intermediate per spec §2 —
// ONNX doesn't define a direct half-half cast and iconnx avoids
// precision-coupling between the two half formats.
// ============================================================================

extern "C" __global__ void cast_bf16_to_f32_kernel(
    float* out, const __nv_bfloat16* inp, size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __bfloat162float(inp[i]);
    }
}

extern "C" __global__ void cast_f32_to_bf16_kernel(
    __nv_bfloat16* out, const float* inp, size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __float2bfloat16_rn(inp[i]);
    }
}

extern "C" __global__ void cast_bf16_to_f16_kernel(
    __half* out, const __nv_bfloat16* inp, size_t n
) {
    // Half-half via f32 intermediate (no direct intrinsic; precision-isolation).
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __float2half_rn(__bfloat162float(inp[i]));
    }
}

extern "C" __global__ void cast_f16_to_bf16_kernel(
    __nv_bfloat16* out, const __half* inp, size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __float2bfloat16_rn(__half2float(inp[i]));
    }
}

extern "C" __global__ void cast_bf16_to_i32_kernel(
    int* out, const __nv_bfloat16* inp, size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = (int)__bfloat162float(inp[i]);
    }
}

extern "C" __global__ void cast_i32_to_bf16_kernel(
    __nv_bfloat16* out, const int* inp, size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __float2bfloat16_rn((float)inp[i]);
    }
}

extern "C" __global__ void cast_bf16_to_i64_kernel(
    long long* out, const __nv_bfloat16* inp, size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = (long long)__bfloat162float(inp[i]);
    }
}

extern "C" __global__ void cast_i64_to_bf16_kernel(
    __nv_bfloat16* out, const long long* inp, size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __float2bfloat16_rn((float)inp[i]);
    }
}
"#;
