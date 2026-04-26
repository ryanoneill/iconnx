//! WS-3 FP16 NVRTC kernels — Float16 mirrors of the memcpy-class layout
//! ops (transpose, concat, gather) used by the FP16 op-dispatch breadth
//! sub-commit (M3.4 sub-1).
//!
//! These kernels are pure index gymnastics — no arithmetic on the element
//! values — so they only need byte-level identity with `half::f16`. The
//! C-side parameter type is `unsigned short*` because `half::f16` is
//! `#[repr(transparent)] struct(u16)`, sharing size and alignment with
//! `unsigned short`. This avoids pulling in `<cuda_fp16.h>` for symbols
//! that never reference the FP16 numeric value.
//!
//! Lives in its own file to stay under the project's 1000-line guideline
//! for `kernels.rs`. Source is concatenated into the single garboard
//! `Module` by `OpsKernelCache::new`.

/// Kernel names compiled from this module's source. Validated at cache
/// creation alongside `super::kernels::KERNEL_NAMES`.
pub const KERNEL_NAMES_FP16: &[&str] = &[
    // M3.4 sub-1 (memcpy-class FP16 layout ops, Whisper-Tiny encoder coverage):
    "transpose_general_f16_kernel",
    "concat_f16_kernel",
    "gather_f16_kernel",
    // M3.5 (Cast — FP16 boundary). FP16 ↔ Float32 is the primary boundary
    // for Whisper's mixed-precision attention; FP16 ↔ Int32 / Int64 covers
    // shape-arithmetic results being mixed back into FP16 paths.
    "cast_f16_to_f32_kernel",
    "cast_f32_to_f16_kernel",
    "cast_f16_to_i32_kernel",
    "cast_i32_to_f16_kernel",
    "cast_f16_to_i64_kernel",
    "cast_i64_to_f16_kernel",
];

/// CUDA kernel source for the WS-3 FP16 op family. See module docs for
/// why `unsigned short*` is the correct C type for memcpy-only kernels.
/// `<cuda_fp16.h>` is supplied by NVRTC's default include path (it is
/// already needed for the M3.5 Cast kernels below, which read FP values
/// via `__half2float` / `__float2half_rn`).
pub const OPS_KERNELS_FP16: &str = r#"
#include <cuda_fp16.h>

// ============================================================================
// WS-3 M3.4 sub-1 Transpose (Float16). Mirrors `transpose_general_kernel`
// in `kernels.rs` byte-for-byte with `unsigned short*` data pointers in
// place of `float*`. Up to 6 dims, matches the f32 / i64 / i32 / u8
// variants. Element values are never read as FP — this is a pure index
// shuffle — so byte-level identity with half::f16 (which is repr(transparent)
// over u16) is sufficient.
// ============================================================================
extern "C" __global__ void transpose_general_f16_kernel(
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
// WS-3 M3.4 sub-1 Concat (Float16). Mirrors `concat_kernel` in `kernels.rs`
// byte-for-byte with `unsigned short*` data pointers. Memcpy-class — no
// reduction across the axis, just slot placement.
// ============================================================================
extern "C" __global__ void concat_f16_kernel(
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
// WS-3 M3.4 sub-1 Gather (Float16). Mirrors `gather_kernel` /
// `gather_i64_kernel` in `kernels.rs` byte-for-byte with `unsigned short*`
// data pointers. Indices stay i64 (ONNX spec). Negative indices are
// wrapped via the standard Python/ONNX convention; out-of-bounds clamps
// to [0, dim0_size-1] to match the f32 / i64 / u8 arms.
// ============================================================================
extern "C" __global__ void gather_f16_kernel(
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
// WS-3 M3.5 Cast — FP16 boundary kernels.
//
// `__float2half_rn` is round-to-nearest-even, matching IEEE binary16's
// default rounding mode and `half::f16::from_f32` on the CPU side. The
// integer ↔ FP16 pairs go through f32 (`(float)x` then `__float2half_rn`,
// or `__half2float` then `(int)x` / `(long long)x`) — there's no
// `__half`-typed integer conversion intrinsic, and going via f32 keeps
// the same truncate-toward-zero semantics ONNX Cast specifies for
// integer targets.
// ============================================================================

extern "C" __global__ void cast_f16_to_f32_kernel(
    float* out, const __half* inp, size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __half2float(inp[i]);
    }
}

extern "C" __global__ void cast_f32_to_f16_kernel(
    __half* out, const float* inp, size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __float2half_rn(inp[i]);
    }
}

extern "C" __global__ void cast_f16_to_i32_kernel(
    int* out, const __half* inp, size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = (int)__half2float(inp[i]);
    }
}

extern "C" __global__ void cast_i32_to_f16_kernel(
    __half* out, const int* inp, size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __float2half_rn((float)inp[i]);
    }
}

extern "C" __global__ void cast_f16_to_i64_kernel(
    long long* out, const __half* inp, size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = (long long)__half2float(inp[i]);
    }
}

extern "C" __global__ void cast_i64_to_f16_kernel(
    __half* out, const long long* inp, size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __float2half_rn((float)inp[i]);
    }
}
"#;
