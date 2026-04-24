//! CUDA kernel source code for ops module

/// All kernel names that need to be loaded
pub const KERNEL_NAMES: &[&str] = &[
    "transpose_2d_kernel",
    "transpose_general_kernel",
    "transpose_general_i64_kernel",
    "transpose_general_i32_kernel",
    "concat_kernel",
    "concat_i64_kernel",
    "concat_i32_kernel",
    "gather_kernel",
    "slice_kernel",
    "where_kernel",
    "where_i64_kernel",
    "where_i32_kernel",
    "equal_kernel",
    "equal_i64",
    "less_kernel",
    "less_i64",
    "greater_kernel",
    "greater_i64",
    "greater_or_equal_i64",
    "less_or_equal_i64",
    "not_equal_i64",
    // Int32 comparison kernels
    "equal_i32",
    "less_i32",
    "greater_i32",
    "greater_or_equal_i32",
    "less_or_equal_i32",
    "not_equal_i32",
    // Cast kernels - proper typed versions
    "cast_f32_to_i64_kernel",
    "cast_f32_to_i32_kernel",
    "cast_i64_to_f32_kernel",
    "cast_i64_to_i32_kernel",
    "cast_i32_to_f32_kernel",
    "cast_i32_to_i64_kernel",
    "greater_or_equal_kernel",
    "less_or_equal_kernel",
    "not_equal_kernel",
    "and_kernel",
    "or_kernel",
    "not_kernel",
    "atan_kernel",
    "expand_kernel",
    "expand_i64_kernel",
    "expand_i32_kernel",
    "pad_kernel",
    "pad_3d_scalar_kernel",
    "pad_4d_scalar_kernel",
    "pad_reflect_3d_kernel",
    "pad_reflect_kernel",
    "pad_edge_kernel",
    "scatter_nd_kernel",
    "range_kernel",
    "range_i64_kernel",
    "cumsum_kernel",
    "cumsum_i64_kernel",
    "cumsum_i32_kernel",
    "copy_kernel",
    "resize_nearest_kernel",
    "resize_3d_scalar_kernel",
    "resize_4d_scalar_kernel",
    "resize_linear_3d_kernel",
    "resize_linear_4d_kernel",
    "slice_nd_kernel",
    "slice_nd_i64_kernel",
    "slice_nd_i32_kernel",
    "slice_3d_scalar_kernel",
    "slice_3d_i64_scalar_kernel",
    "slice_4d_scalar_kernel",
    "slice_4d_i64_scalar_kernel",
    "gather_i64_kernel",
    "fill_kernel",
    "fill_i64_kernel",
];

/// CUDA kernel source code
pub const OPS_KERNELS: &str = r#"
// Simple 2D transpose: [M, N] -> [N, M]
extern "C" __global__ void transpose_2d_kernel(
    float* out, const float* inp, size_t rows, size_t cols
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    size_t row = idx / cols;
    size_t col = idx % cols;
    out[col * rows + row] = inp[idx];
}

// General N-D transpose with permutation (up to 6 dims)
extern "C" __global__ void transpose_general_kernel(
    float* out, const float* inp,
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

// N-dimensional transpose kernel (Int64)
extern "C" __global__ void transpose_general_i64_kernel(
    long long* out, const long long* inp,
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

// N-dimensional transpose kernel (Int32)
//
// Functionally identical to transpose_general_kernel — both move 4-byte
// scalars by index — but exists as its own symbol so the Rust type system
// (garboard's TypedKernel) can enforce that an i32 slice is paired with
// an i32-typed kernel. The cost is one extra NVRTC symbol in the module.
extern "C" __global__ void transpose_general_i32_kernel(
    int* out, const int* inp,
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

// Concatenate tensors along an axis (Float32)
extern "C" __global__ void concat_kernel(
    float* out, const float* inp,
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

// Concatenate tensors along an axis (Int64)
extern "C" __global__ void concat_i64_kernel(
    long long* out, const long long* inp,
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

// Concatenate tensors along an axis (Int32)
extern "C" __global__ void concat_i32_kernel(
    int* out, const int* inp,
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

// Gather: out[i] = inp[indices[i]] along axis 0 (Float32)
// Supports negative indices (e.g., -1 = last element)
// Clamps out-of-bounds indices to valid range [0, dim0_size-1]
extern "C" __global__ void gather_kernel(
    float* out, const float* inp, const long long* indices,
    size_t num_indices, size_t slice_size, size_t dim0_size
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_indices * slice_size) return;
    size_t which_index = idx / slice_size;
    size_t offset_in_slice = idx % slice_size;
    long long raw_index = indices[which_index];
    // Handle negative indices (Python/ONNX convention)
    long long adjusted = (raw_index < 0) ? (raw_index + (long long)dim0_size) : raw_index;
    // Clamp to valid range [0, dim0_size-1]
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

// Gather: out[i] = inp[indices[i]] along axis 0 (Int64)
// Supports negative indices (e.g., -1 = last element)
// Clamps out-of-bounds indices to valid range [0, dim0_size-1]
extern "C" __global__ void gather_i64_kernel(
    long long* out, const long long* inp, const long long* indices,
    size_t num_indices, size_t slice_size, size_t dim0_size
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_indices * slice_size) return;
    size_t which_index = idx / slice_size;
    size_t offset_in_slice = idx % slice_size;
    long long raw_index = indices[which_index];
    // Handle negative indices (Python/ONNX convention)
    long long adjusted = (raw_index < 0) ? (raw_index + (long long)dim0_size) : raw_index;
    // Clamp to valid range [0, dim0_size-1]
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

// Slice: Extract contiguous slice
extern "C" __global__ void slice_kernel(
    float* out, const float* inp, size_t start_offset, size_t total_elements
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    out[idx] = inp[start_offset + idx];
}

// Where: out = condition ? x : y (Float32)
extern "C" __global__ void where_kernel(
    float* out, const float* condition, const float* x, const float* y, size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = (condition[idx] != 0.0f) ? x[idx] : y[idx];
}

// Where Int64: out = condition ? x : y (condition is float, x/y are int64)
extern "C" __global__ void where_i64_kernel(
    long long* out, const float* condition, const long long* x, const long long* y, size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = (condition[idx] != 0.0f) ? x[idx] : y[idx];
}

// Where Int32: out = condition ? x : y (condition is float, x/y are int32)
extern "C" __global__ void where_i32_kernel(
    int* out, const float* condition, const int* x, const int* y, size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = (condition[idx] != 0.0f) ? x[idx] : y[idx];
}

// Equal: out = (a == b) ? 1.0 : 0.0
extern "C" __global__ void equal_kernel(float* out, const float* a, const float* b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = (a[idx] == b[idx]) ? 1.0f : 0.0f;
}

// Equal Int64: out = (a == b) ? 1.0 : 0.0
extern "C" __global__ void equal_i64(float* out, const long long* a, const long long* b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = (a[idx] == b[idx]) ? 1.0f : 0.0f;
}

// Less: out = (a < b) ? 1.0 : 0.0
extern "C" __global__ void less_kernel(float* out, const float* a, const float* b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = (a[idx] < b[idx]) ? 1.0f : 0.0f;
}

// Less Int64: out = (a < b) ? 1.0 : 0.0
extern "C" __global__ void less_i64(float* out, const long long* a, const long long* b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = (a[idx] < b[idx]) ? 1.0f : 0.0f;
}

// Greater: out = (a > b) ? 1.0 : 0.0
extern "C" __global__ void greater_kernel(float* out, const float* a, const float* b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = (a[idx] > b[idx]) ? 1.0f : 0.0f;
}

// Greater Int64: out = (a > b) ? 1.0 : 0.0
extern "C" __global__ void greater_i64(float* out, const long long* a, const long long* b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = (a[idx] > b[idx]) ? 1.0f : 0.0f;
}

// GreaterOrEqual Int64: out = (a >= b) ? 1.0 : 0.0
extern "C" __global__ void greater_or_equal_i64(float* out, const long long* a, const long long* b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = (a[idx] >= b[idx]) ? 1.0f : 0.0f;
}

// LessOrEqual Int64: out = (a <= b) ? 1.0 : 0.0
extern "C" __global__ void less_or_equal_i64(float* out, const long long* a, const long long* b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = (a[idx] <= b[idx]) ? 1.0f : 0.0f;
}

// NotEqual Int64: out = (a != b) ? 1.0 : 0.0
extern "C" __global__ void not_equal_i64(float* out, const long long* a, const long long* b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = (a[idx] != b[idx]) ? 1.0f : 0.0f;
}

// Equal Int32: out = (a == b) ? 1.0 : 0.0
extern "C" __global__ void equal_i32(float* out, const int* a, const int* b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = (a[idx] == b[idx]) ? 1.0f : 0.0f;
}

// Less Int32: out = (a < b) ? 1.0 : 0.0
extern "C" __global__ void less_i32(float* out, const int* a, const int* b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = (a[idx] < b[idx]) ? 1.0f : 0.0f;
}

// Greater Int32: out = (a > b) ? 1.0 : 0.0
extern "C" __global__ void greater_i32(float* out, const int* a, const int* b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = (a[idx] > b[idx]) ? 1.0f : 0.0f;
}

// GreaterOrEqual Int32: out = (a >= b) ? 1.0 : 0.0
extern "C" __global__ void greater_or_equal_i32(float* out, const int* a, const int* b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = (a[idx] >= b[idx]) ? 1.0f : 0.0f;
}

// LessOrEqual Int32: out = (a <= b) ? 1.0 : 0.0
extern "C" __global__ void less_or_equal_i32(float* out, const int* a, const int* b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = (a[idx] <= b[idx]) ? 1.0f : 0.0f;
}

// NotEqual Int32: out = (a != b) ? 1.0 : 0.0
extern "C" __global__ void not_equal_i32(float* out, const int* a, const int* b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = (a[idx] != b[idx]) ? 1.0f : 0.0f;
}

// Cast f32 to i64 - round to nearest integer
extern "C" __global__ void cast_f32_to_i64_kernel(long long* out, const float* inp, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = (long long)(inp[idx] + (inp[idx] >= 0 ? 0.5f : -0.5f));
}

// Cast f32 to i32 - round to nearest integer
extern "C" __global__ void cast_f32_to_i32_kernel(int* out, const float* inp, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = (int)(inp[idx] + (inp[idx] >= 0 ? 0.5f : -0.5f));
}

// Cast i64 to f32
extern "C" __global__ void cast_i64_to_f32_kernel(float* out, const long long* inp, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = (float)inp[idx];
}

// Cast i64 to i32 (truncates)
extern "C" __global__ void cast_i64_to_i32_kernel(int* out, const long long* inp, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = (int)inp[idx];
}

// Cast i32 to f32
extern "C" __global__ void cast_i32_to_f32_kernel(float* out, const int* inp, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = (float)inp[idx];
}

// Cast i32 to i64 (zero-extend)
extern "C" __global__ void cast_i32_to_i64_kernel(long long* out, const int* inp, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = (long long)inp[idx];
}

// GreaterOrEqual: out = (a >= b) ? 1.0 : 0.0
extern "C" __global__ void greater_or_equal_kernel(float* out, const float* a, const float* b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = (a[idx] >= b[idx]) ? 1.0f : 0.0f;
}

// LessOrEqual: out = (a <= b) ? 1.0 : 0.0
extern "C" __global__ void less_or_equal_kernel(float* out, const float* a, const float* b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = (a[idx] <= b[idx]) ? 1.0f : 0.0f;
}

// NotEqual: out = (a != b) ? 1.0 : 0.0
extern "C" __global__ void not_equal_kernel(float* out, const float* a, const float* b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = (a[idx] != b[idx]) ? 1.0f : 0.0f;
}

// And: out = (a != 0 && b != 0) ? 1.0 : 0.0
extern "C" __global__ void and_kernel(float* out, const float* a, const float* b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = (a[idx] != 0.0f && b[idx] != 0.0f) ? 1.0f : 0.0f;
}

// Or: out = (a != 0 || b != 0) ? 1.0 : 0.0
extern "C" __global__ void or_kernel(float* out, const float* a, const float* b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = (a[idx] != 0.0f || b[idx] != 0.0f) ? 1.0f : 0.0f;
}

// Not: out = (a == 0) ? 1.0 : 0.0
extern "C" __global__ void not_kernel(float* out, const float* a, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = (a[idx] == 0.0f) ? 1.0f : 0.0f;
}

// Atan: out = atan(inp)
extern "C" __global__ void atan_kernel(float* out, const float* inp, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = atanf(inp[idx]);
}

// Expand: Broadcast input to output shape (up to 8 dims)
extern "C" __global__ void expand_kernel(
    float* out, const float* inp,
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

// Expand Int64: Broadcast input to output shape (up to 8 dims)
extern "C" __global__ void expand_i64_kernel(
    long long* out, const long long* inp,
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

// Expand Int32: Broadcast input to output shape (up to 8 dims)
extern "C" __global__ void expand_i32_kernel(
    int* out, const int* inp,
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

// Pad: Pad tensor with constant value (up to 8 dims)
extern "C" __global__ void pad_kernel(
    float* out, const float* inp,
    const size_t* inp_shape, const size_t* out_shape,
    const size_t* inp_strides, const size_t* out_strides,
    const size_t* pads_begin, size_t ndim,
    float pad_value, size_t total_out_elements
) {
    size_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= total_out_elements) return;
    size_t out_coords[8];
    size_t remaining = out_idx;
    for (size_t d = 0; d < ndim; d++) {
        out_coords[d] = remaining / out_strides[d];
        remaining = remaining % out_strides[d];
    }
    bool in_bounds = true;
    size_t inp_idx = 0;
    for (size_t d = 0; d < ndim; d++) {
        size_t inp_coord = out_coords[d] - pads_begin[d];
        if (out_coords[d] < pads_begin[d] || inp_coord >= inp_shape[d]) {
            in_bounds = false;
            break;
        }
        inp_idx += inp_coord * inp_strides[d];
    }
    out[out_idx] = in_bounds ? inp[inp_idx] : pad_value;
}

// Optimized 3D Pad with packed parameters (avoids 19-arg tuple limit).
// params layout: [inp_d0..2, out_d0..2, inp_s0..2, out_s0..2, pad_b0..2] = 15 values.
extern "C" __global__ void pad_3d_scalar_kernel(
    float* out, const float* inp,
    const size_t* params,
    float pad_value, size_t total_out_elements
) {
    size_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= total_out_elements) return;

    // Unpack params
    size_t inp_d0 = params[0], inp_d1 = params[1], inp_d2 = params[2];
    // out_d0..2 at params[3..6] — unused directly but define output layout.
    size_t inp_s0 = params[6], inp_s1 = params[7], inp_s2 = params[8];
    size_t out_s0 = params[9], out_s1 = params[10], out_s2 = params[11];
    size_t pad_b0 = params[12], pad_b1 = params[13], pad_b2 = params[14];

    // Decompose output index into coordinates
    size_t c0 = out_idx / out_s0;
    size_t rem = out_idx % out_s0;
    size_t c1 = rem / out_s1;
    size_t c2 = rem % out_s1;

    // Check bounds and compute input coordinate
    bool in_bounds = true;
    size_t inp_idx = 0;

    if (c0 < pad_b0 || c0 - pad_b0 >= inp_d0) {
        in_bounds = false;
    } else {
        inp_idx += (c0 - pad_b0) * inp_s0;
    }

    if (in_bounds) {
        if (c1 < pad_b1 || c1 - pad_b1 >= inp_d1) {
            in_bounds = false;
        } else {
            inp_idx += (c1 - pad_b1) * inp_s1;
        }
    }

    if (in_bounds) {
        if (c2 < pad_b2 || c2 - pad_b2 >= inp_d2) {
            in_bounds = false;
        } else {
            inp_idx += (c2 - pad_b2) * inp_s2;
        }
    }

    out[out_idx] = in_bounds ? inp[inp_idx] : pad_value;
}

// Optimized 4D Pad with packed parameters (avoids 24-arg tuple limit).
// params layout: [inp_d0..3, out_d0..3, inp_s0..3, out_s0..3, pad_b0..3] = 20 values.
extern "C" __global__ void pad_4d_scalar_kernel(
    float* out, const float* inp,
    const size_t* params,
    float pad_value, size_t total_out_elements
) {
    size_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= total_out_elements) return;

    // Unpack params
    size_t inp_d0 = params[0], inp_d1 = params[1], inp_d2 = params[2], inp_d3 = params[3];
    // out_d0..3 at params[4..8] — unused directly.
    size_t inp_s0 = params[8], inp_s1 = params[9], inp_s2 = params[10], inp_s3 = params[11];
    size_t out_s0 = params[12], out_s1 = params[13], out_s2 = params[14], out_s3 = params[15];
    size_t pad_b0 = params[16], pad_b1 = params[17], pad_b2 = params[18], pad_b3 = params[19];

    // Decompose output index into coordinates
    size_t c0 = out_idx / out_s0;
    size_t rem = out_idx % out_s0;
    size_t c1 = rem / out_s1;
    rem = rem % out_s1;
    size_t c2 = rem / out_s2;
    size_t c3 = rem % out_s2;

    // Check bounds and compute input coordinate
    bool in_bounds = true;
    size_t inp_idx = 0;

    if (c0 < pad_b0 || c0 - pad_b0 >= inp_d0) {
        in_bounds = false;
    } else {
        inp_idx += (c0 - pad_b0) * inp_s0;
    }

    if (in_bounds) {
        if (c1 < pad_b1 || c1 - pad_b1 >= inp_d1) {
            in_bounds = false;
        } else {
            inp_idx += (c1 - pad_b1) * inp_s1;
        }
    }

    if (in_bounds) {
        if (c2 < pad_b2 || c2 - pad_b2 >= inp_d2) {
            in_bounds = false;
        } else {
            inp_idx += (c2 - pad_b2) * inp_s2;
        }
    }

    if (in_bounds) {
        if (c3 < pad_b3 || c3 - pad_b3 >= inp_d3) {
            in_bounds = false;
        } else {
            inp_idx += (c3 - pad_b3) * inp_s3;
        }
    }

    out[out_idx] = in_bounds ? inp[inp_idx] : pad_value;
}

// Helper device function for reflect index calculation
// Given output coordinate c, pad_begin p, and input dimension d,
// returns the input coordinate using reflection
__device__ inline size_t reflect_index(size_t c, size_t pad_begin, size_t inp_dim) {
    // Convert to signed for easier math
    long long ic = (long long)c - (long long)pad_begin;

    if (ic < 0) {
        // Reflect from beginning: -1 -> 1, -2 -> 2, etc.
        ic = -ic;
    }
    if (ic >= (long long)inp_dim) {
        // Reflect from end: if inp_dim=5, index 5->3, 6->2, etc.
        ic = 2 * (long long)inp_dim - ic - 2;
    }

    // Handle multiple reflections (for very large padding)
    // This handles cases where padding > input dimension
    while (ic < 0 || ic >= (long long)inp_dim) {
        if (ic < 0) {
            ic = -ic;
        }
        if (ic >= (long long)inp_dim) {
            ic = 2 * (long long)inp_dim - ic - 2;
        }
    }

    return (size_t)ic;
}

// Optimized 3D Pad with reflect mode
// Optimized 3D Pad reflect with packed parameters (avoids 18-arg tuple limit).
// params layout: [inp_d0..2, out_d0..2, inp_s0..2, out_s0..2, pad_b0..2] = 15 values.
extern "C" __global__ void pad_reflect_3d_kernel(
    float* out, const float* inp,
    const size_t* params,
    size_t total_out_elements
) {
    size_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= total_out_elements) return;

    // Unpack params
    size_t inp_d0 = params[0], inp_d1 = params[1], inp_d2 = params[2];
    // out_d0..2 at params[3..6] — unused directly.
    size_t inp_s0 = params[6], inp_s1 = params[7], inp_s2 = params[8];
    size_t out_s0 = params[9], out_s1 = params[10], out_s2 = params[11];
    size_t pad_b0 = params[12], pad_b1 = params[13], pad_b2 = params[14];

    // Decompose output index into coordinates
    size_t c0 = out_idx / out_s0;
    size_t rem = out_idx % out_s0;
    size_t c1 = rem / out_s1;
    size_t c2 = rem % out_s1;

    // Compute reflected input coordinates
    size_t i0 = (pad_b0 > 0 || c0 >= pad_b0 + inp_d0) ? reflect_index(c0, pad_b0, inp_d0) : (c0 - pad_b0);
    size_t i1 = (pad_b1 > 0 || c1 >= pad_b1 + inp_d1) ? reflect_index(c1, pad_b1, inp_d1) : (c1 - pad_b1);
    size_t i2 = (pad_b2 > 0 || c2 >= pad_b2 + inp_d2) ? reflect_index(c2, pad_b2, inp_d2) : (c2 - pad_b2);

    size_t inp_idx = i0 * inp_s0 + i1 * inp_s1 + i2 * inp_s2;
    out[out_idx] = inp[inp_idx];
}

// Generic Pad with reflect mode (up to 8 dims, uses global memory for shapes)
extern "C" __global__ void pad_reflect_kernel(
    float* out, const float* inp,
    const size_t* inp_shape, const size_t* out_shape,
    const size_t* inp_strides, const size_t* out_strides,
    const size_t* pads_begin, size_t ndim,
    size_t total_out_elements
) {
    size_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= total_out_elements) return;

    // Decompose output index into coordinates and compute input index
    size_t inp_idx = 0;
    size_t remaining = out_idx;

    for (size_t d = 0; d < ndim; d++) {
        size_t coord = remaining / out_strides[d];
        remaining = remaining % out_strides[d];

        // Compute reflected input coordinate
        size_t inp_coord = reflect_index(coord, pads_begin[d], inp_shape[d]);
        inp_idx += inp_coord * inp_strides[d];
    }

    out[out_idx] = inp[inp_idx];
}

// Generic Pad with edge mode (replicate edges, up to 8 dims)
extern "C" __global__ void pad_edge_kernel(
    float* out, const float* inp,
    const size_t* inp_shape, const size_t* out_shape,
    const size_t* inp_strides, const size_t* out_strides,
    const size_t* pads_begin, size_t ndim,
    size_t total_out_elements
) {
    size_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= total_out_elements) return;

    // Decompose output index into coordinates and compute input index
    size_t inp_idx = 0;
    size_t remaining = out_idx;

    for (size_t d = 0; d < ndim; d++) {
        size_t coord = remaining / out_strides[d];
        remaining = remaining % out_strides[d];

        // Clamp coordinate to valid input range (edge replication)
        long long ic = (long long)coord - (long long)pads_begin[d];
        if (ic < 0) ic = 0;
        if (ic >= (long long)inp_shape[d]) ic = (long long)inp_shape[d] - 1;

        inp_idx += (size_t)ic * inp_strides[d];
    }

    out[out_idx] = inp[inp_idx];
}

// ScatterND: Scatter updates into output at given indices
extern "C" __global__ void scatter_nd_kernel(
    float* out, const float* updates, const long long* indices,
    const size_t* shape, size_t num_updates, size_t slice_size,
    size_t index_depth, size_t ndim
) {
    size_t update_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (update_idx >= num_updates * slice_size) return;
    size_t which_update = update_idx / slice_size;
    size_t offset_in_slice = update_idx % slice_size;
    size_t out_linear = 0;
    for (size_t d = 0; d < index_depth; d++) {
        size_t s = 1;
        for (size_t dd = d + 1; dd < ndim; dd++) {
            s *= shape[dd];
        }
        out_linear += indices[which_update * index_depth + d] * s;
    }
    out_linear += offset_in_slice;
    out[out_linear] = updates[update_idx];
}

// Range: Generate sequence [start, start+delta, ...]
extern "C" __global__ void range_kernel(float* out, float start, float delta, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = start + delta * (float)idx;
}

// Range Int64: Generate a sequence of integers
extern "C" __global__ void range_i64_kernel(long long* out, long long start, long long delta, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = start + delta * (long long)idx;
}

// CumSum: Cumulative sum along arbitrary axis (one thread per cumsum line)
// For shape [A, B, C] with axis=1: num_lines=A*C, axis_len=B, inner_stride=C
extern "C" __global__ void cumsum_kernel(
    float* out, const float* inp,
    size_t num_lines, size_t axis_len, size_t inner_stride,
    int exclusive, int reverse
) {
    size_t line = blockIdx.x * blockDim.x + threadIdx.x;
    if (line >= num_lines) return;

    // Compute starting offset: line = outer * inner_stride + inner
    size_t outer = line / inner_stride;
    size_t inner = line % inner_stride;
    size_t base = outer * axis_len * inner_stride + inner;

    if (reverse) {
        float sum = 0.0f;
        for (size_t ax = axis_len; ax > 0; ax--) {
            size_t idx = base + (ax - 1) * inner_stride;
            if (exclusive) {
                out[idx] = sum;
                sum += inp[idx];
            } else {
                sum += inp[idx];
                out[idx] = sum;
            }
        }
    } else {
        float sum = 0.0f;
        for (size_t ax = 0; ax < axis_len; ax++) {
            size_t idx = base + ax * inner_stride;
            if (exclusive) {
                out[idx] = sum;
                sum += inp[idx];
            } else {
                sum += inp[idx];
                out[idx] = sum;
            }
        }
    }
}

// CumSum Int64: Cumulative sum along arbitrary axis (one thread per cumsum line)
extern "C" __global__ void cumsum_i64_kernel(
    long long* out, const long long* inp,
    size_t num_lines, size_t axis_len, size_t inner_stride,
    int exclusive, int reverse
) {
    size_t line = blockIdx.x * blockDim.x + threadIdx.x;
    if (line >= num_lines) return;

    size_t outer = line / inner_stride;
    size_t inner = line % inner_stride;
    size_t base = outer * axis_len * inner_stride + inner;

    if (reverse) {
        long long sum = 0;
        for (size_t ax = axis_len; ax > 0; ax--) {
            size_t idx = base + (ax - 1) * inner_stride;
            if (exclusive) {
                out[idx] = sum;
                sum += inp[idx];
            } else {
                sum += inp[idx];
                out[idx] = sum;
            }
        }
    } else {
        long long sum = 0;
        for (size_t ax = 0; ax < axis_len; ax++) {
            size_t idx = base + ax * inner_stride;
            if (exclusive) {
                out[idx] = sum;
                sum += inp[idx];
            } else {
                sum += inp[idx];
                out[idx] = sum;
            }
        }
    }
}

// CumSum Int32: Cumulative sum along arbitrary axis (one thread per cumsum line)
extern "C" __global__ void cumsum_i32_kernel(
    int* out, const int* inp,
    size_t num_lines, size_t axis_len, size_t inner_stride,
    int exclusive, int reverse
) {
    size_t line = blockIdx.x * blockDim.x + threadIdx.x;
    if (line >= num_lines) return;

    size_t outer = line / inner_stride;
    size_t inner = line % inner_stride;
    size_t base = outer * axis_len * inner_stride + inner;

    if (reverse) {
        int sum = 0;
        for (size_t ax = axis_len; ax > 0; ax--) {
            size_t idx = base + (ax - 1) * inner_stride;
            if (exclusive) {
                out[idx] = sum;
                sum += inp[idx];
            } else {
                sum += inp[idx];
                out[idx] = sum;
            }
        }
    } else {
        int sum = 0;
        for (size_t ax = 0; ax < axis_len; ax++) {
            size_t idx = base + ax * inner_stride;
            if (exclusive) {
                out[idx] = sum;
                sum += inp[idx];
            } else {
                sum += inp[idx];
                out[idx] = sum;
            }
        }
    }
}

// Copy: Simple memory copy
extern "C" __global__ void copy_kernel(float* out, const float* inp, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = inp[idx];
}

// Resize: Nearest-neighbor interpolation (up to 8 dims)
// coord_mode: 0 = asymmetric, 1 = half_pixel
// nearest_mode: 0 = floor, 1 = round_prefer_floor
extern "C" __global__ void resize_nearest_kernel(
    float* out, const float* inp,
    const size_t* inp_shape, const size_t* out_shape,
    const size_t* inp_strides, const size_t* out_strides,
    const float* scales, size_t ndim, size_t total_out_elements,
    int coord_mode, int nearest_mode
) {
    size_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= total_out_elements) return;

    // Convert flat output index to multi-dimensional coordinates
    size_t out_coords[8];
    size_t remaining = out_idx;
    for (size_t d = 0; d < ndim; d++) {
        out_coords[d] = remaining / out_strides[d];
        remaining = remaining % out_strides[d];
    }

    // Map output coordinates to input coordinates (nearest neighbor)
    size_t inp_idx = 0;
    for (size_t d = 0; d < ndim; d++) {
        // Transform coordinate based on coord_mode
        float out_f = (float)out_coords[d];
        float inp_coord_f;
        if (coord_mode == 1) {
            // half_pixel
            inp_coord_f = (out_f + 0.5f) / scales[d] - 0.5f;
        } else {
            // asymmetric
            inp_coord_f = out_f / scales[d];
        }

        // Apply nearest mode
        float result;
        if (nearest_mode == 1) {
            // round_prefer_floor
            float frac = inp_coord_f - floorf(inp_coord_f);
            if (frac == 0.5f) {
                result = floorf(inp_coord_f);
            } else {
                result = roundf(inp_coord_f);
            }
        } else {
            // floor
            result = floorf(inp_coord_f);
        }

        // Clamp to valid range
        size_t inp_coord;
        if (result < 0.0f) {
            inp_coord = 0;
        } else {
            inp_coord = (size_t)result;
            if (inp_coord >= inp_shape[d]) {
                inp_coord = inp_shape[d] - 1;
            }
        }
        inp_idx += inp_coord * inp_strides[d];
    }

    out[out_idx] = inp[inp_idx];
}

// Helper: Transform output coordinate to input coordinate
// coord_mode: 0 = asymmetric, 1 = half_pixel
// Returns float coordinate before rounding
__device__ float resize_transform_coord(size_t out_coord, float scale, int coord_mode) {
    float out_f = (float)out_coord;
    if (coord_mode == 1) {
        // half_pixel: (out_coord + 0.5) / scale - 0.5
        return (out_f + 0.5f) / scale - 0.5f;
    } else {
        // asymmetric: out_coord / scale
        return out_f / scale;
    }
}

// Helper: Apply nearest mode rounding
// nearest_mode: 0 = floor, 1 = round_prefer_floor
__device__ size_t resize_apply_nearest(float coord, size_t max_idx, int nearest_mode) {
    float result;
    if (nearest_mode == 1) {
        // round_prefer_floor: round to nearest, ties go to floor
        float frac = coord - floorf(coord);
        if (frac == 0.5f) {
            result = floorf(coord);
        } else {
            result = roundf(coord);
        }
    } else {
        // floor
        result = floorf(coord);
    }
    // Clamp to valid range
    if (result < 0.0f) return 0;
    size_t idx = (size_t)result;
    if (idx >= max_idx) return max_idx - 1;
    return idx;
}

// Optimized 3D Resize with scalar parameters (no H2D transfers)
// coord_mode: 0 = asymmetric, 1 = half_pixel
// nearest_mode: 0 = floor, 1 = round_prefer_floor
// Optimized 3D Resize nearest with packed params (avoids 20-arg limit).
// params layout: [inp_d0..2, out_d0..2, inp_s0..2, out_s0..2] = 12 values.
extern "C" __global__ void resize_3d_scalar_kernel(
    float* out, const float* inp,
    const size_t* params,
    float scale0, float scale1, float scale2,
    size_t total_out_elements,
    int coord_mode, int nearest_mode
) {
    size_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= total_out_elements) return;

    size_t inp_d0 = params[0], inp_d1 = params[1], inp_d2 = params[2];
    size_t inp_s0 = params[6], inp_s1 = params[7], inp_s2 = params[8];
    size_t out_s0 = params[9], out_s1 = params[10], out_s2 = params[11];

    // Decompose output index into coordinates
    size_t c0 = out_idx / out_s0;
    size_t rem = out_idx % out_s0;
    size_t c1 = rem / out_s1;
    size_t c2 = rem % out_s1;

    // Transform and round coordinates
    float f0 = resize_transform_coord(c0, scale0, coord_mode);
    float f1 = resize_transform_coord(c1, scale1, coord_mode);
    float f2 = resize_transform_coord(c2, scale2, coord_mode);

    size_t i0 = resize_apply_nearest(f0, inp_d0, nearest_mode);
    size_t i1 = resize_apply_nearest(f1, inp_d1, nearest_mode);
    size_t i2 = resize_apply_nearest(f2, inp_d2, nearest_mode);

    size_t inp_idx = i0 * inp_s0 + i1 * inp_s1 + i2 * inp_s2;
    out[out_idx] = inp[inp_idx];
}

// Optimized 4D Resize nearest with packed params (avoids 25-arg limit).
// params layout: [inp_d0..3, out_d0..3, inp_s0..3, out_s0..3] = 16 values.
extern "C" __global__ void resize_4d_scalar_kernel(
    float* out, const float* inp,
    const size_t* params,
    float scale0, float scale1, float scale2, float scale3,
    size_t total_out_elements,
    int coord_mode, int nearest_mode
) {
    size_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= total_out_elements) return;

    size_t inp_d0 = params[0], inp_d1 = params[1], inp_d2 = params[2], inp_d3 = params[3];
    size_t inp_s0 = params[8], inp_s1 = params[9], inp_s2 = params[10], inp_s3 = params[11];
    size_t out_s0 = params[12], out_s1 = params[13], out_s2 = params[14], out_s3 = params[15];

    // Decompose output index into coordinates
    size_t c0 = out_idx / out_s0;
    size_t rem = out_idx % out_s0;
    size_t c1 = rem / out_s1;
    rem = rem % out_s1;
    size_t c2 = rem / out_s2;
    size_t c3 = rem % out_s2;

    // Transform and round coordinates
    float f0 = resize_transform_coord(c0, scale0, coord_mode);
    float f1 = resize_transform_coord(c1, scale1, coord_mode);
    float f2 = resize_transform_coord(c2, scale2, coord_mode);
    float f3 = resize_transform_coord(c3, scale3, coord_mode);

    size_t i0 = resize_apply_nearest(f0, inp_d0, nearest_mode);
    size_t i1 = resize_apply_nearest(f1, inp_d1, nearest_mode);
    size_t i2 = resize_apply_nearest(f2, inp_d2, nearest_mode);
    size_t i3 = resize_apply_nearest(f3, inp_d3, nearest_mode);

    size_t inp_idx = i0 * inp_s0 + i1 * inp_s1 + i2 * inp_s2 + i3 * inp_s3;
    out[out_idx] = inp[inp_idx];
}

// 3D Linear interpolation (trilinear) for Resize
// coord_mode: 0 = asymmetric, 1 = half_pixel
extern "C" __global__ void resize_linear_3d_kernel(
    float* out, const float* inp,
    size_t inp_d0, size_t inp_d1, size_t inp_d2,
    size_t out_d0, size_t out_d1, size_t out_d2,
    size_t inp_s0, size_t inp_s1, size_t inp_s2,
    float scale0, float scale1, float scale2,
    size_t total_out_elements,
    int coord_mode
) {
    size_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= total_out_elements) return;

    // Decompose output index into coordinates
    size_t out_s0 = out_d1 * out_d2;
    size_t out_s1 = out_d2;
    size_t c0 = out_idx / out_s0;
    size_t rem = out_idx % out_s0;
    size_t c1 = rem / out_s1;
    size_t c2 = rem % out_s1;

    // Transform coordinates to input space
    float f0 = resize_transform_coord(c0, scale0, coord_mode);
    float f1 = resize_transform_coord(c1, scale1, coord_mode);
    float f2 = resize_transform_coord(c2, scale2, coord_mode);

    // Clamp to valid range
    f0 = fmaxf(0.0f, fminf(f0, (float)(inp_d0 - 1)));
    f1 = fmaxf(0.0f, fminf(f1, (float)(inp_d1 - 1)));
    f2 = fmaxf(0.0f, fminf(f2, (float)(inp_d2 - 1)));

    // Get floor indices
    size_t i0_lo = (size_t)floorf(f0);
    size_t i1_lo = (size_t)floorf(f1);
    size_t i2_lo = (size_t)floorf(f2);

    // Get ceil indices (clamped)
    size_t i0_hi = (i0_lo + 1 < inp_d0) ? i0_lo + 1 : inp_d0 - 1;
    size_t i1_hi = (i1_lo + 1 < inp_d1) ? i1_lo + 1 : inp_d1 - 1;
    size_t i2_hi = (i2_lo + 1 < inp_d2) ? i2_lo + 1 : inp_d2 - 1;

    // Compute interpolation weights
    float w0 = f0 - (float)i0_lo;
    float w1 = f1 - (float)i1_lo;
    float w2 = f2 - (float)i2_lo;

    // Load all 8 corner values for trilinear interpolation
    float v000 = inp[i0_lo * inp_s0 + i1_lo * inp_s1 + i2_lo * inp_s2];
    float v001 = inp[i0_lo * inp_s0 + i1_lo * inp_s1 + i2_hi * inp_s2];
    float v010 = inp[i0_lo * inp_s0 + i1_hi * inp_s1 + i2_lo * inp_s2];
    float v011 = inp[i0_lo * inp_s0 + i1_hi * inp_s1 + i2_hi * inp_s2];
    float v100 = inp[i0_hi * inp_s0 + i1_lo * inp_s1 + i2_lo * inp_s2];
    float v101 = inp[i0_hi * inp_s0 + i1_lo * inp_s1 + i2_hi * inp_s2];
    float v110 = inp[i0_hi * inp_s0 + i1_hi * inp_s1 + i2_lo * inp_s2];
    float v111 = inp[i0_hi * inp_s0 + i1_hi * inp_s1 + i2_hi * inp_s2];

    // Trilinear blend: interpolate along dim2, then dim1, then dim0
    float v00 = v000 * (1.0f - w2) + v001 * w2;
    float v01 = v010 * (1.0f - w2) + v011 * w2;
    float v10 = v100 * (1.0f - w2) + v101 * w2;
    float v11 = v110 * (1.0f - w2) + v111 * w2;

    float v0 = v00 * (1.0f - w1) + v01 * w1;
    float v1 = v10 * (1.0f - w1) + v11 * w1;

    out[out_idx] = v0 * (1.0f - w0) + v1 * w0;
}

// 4D Linear interpolation for Resize
// coord_mode: 0 = asymmetric, 1 = half_pixel
// 4D Linear interpolation with packed params (avoids 20-arg limit).
// params layout: [inp_d0..3, out_d0..3, inp_s0..3] = 12 values.
extern "C" __global__ void resize_linear_4d_kernel(
    float* out, const float* inp,
    const size_t* params,
    float scale0, float scale1, float scale2, float scale3,
    size_t total_out_elements,
    int coord_mode
) {
    size_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= total_out_elements) return;

    size_t inp_d0 = params[0], inp_d1 = params[1], inp_d2 = params[2], inp_d3 = params[3];
    size_t out_d0 = params[4], out_d1 = params[5], out_d2 = params[6], out_d3 = params[7];
    size_t inp_s0 = params[8], inp_s1 = params[9], inp_s2 = params[10], inp_s3 = params[11];

    // Decompose output index into coordinates
    size_t out_s0 = out_d1 * out_d2 * out_d3;
    size_t out_s1 = out_d2 * out_d3;
    size_t out_s2 = out_d3;
    size_t c0 = out_idx / out_s0;
    size_t rem = out_idx % out_s0;
    size_t c1 = rem / out_s1;
    rem = rem % out_s1;
    size_t c2 = rem / out_s2;
    size_t c3 = rem % out_s2;

    // Transform coordinates to input space
    float f0 = resize_transform_coord(c0, scale0, coord_mode);
    float f1 = resize_transform_coord(c1, scale1, coord_mode);
    float f2 = resize_transform_coord(c2, scale2, coord_mode);
    float f3 = resize_transform_coord(c3, scale3, coord_mode);

    // Clamp to valid range
    f0 = fmaxf(0.0f, fminf(f0, (float)(inp_d0 - 1)));
    f1 = fmaxf(0.0f, fminf(f1, (float)(inp_d1 - 1)));
    f2 = fmaxf(0.0f, fminf(f2, (float)(inp_d2 - 1)));
    f3 = fmaxf(0.0f, fminf(f3, (float)(inp_d3 - 1)));

    // Get floor indices
    size_t i0_lo = (size_t)floorf(f0);
    size_t i1_lo = (size_t)floorf(f1);
    size_t i2_lo = (size_t)floorf(f2);
    size_t i3_lo = (size_t)floorf(f3);

    // Get ceil indices (clamped)
    size_t i0_hi = (i0_lo + 1 < inp_d0) ? i0_lo + 1 : inp_d0 - 1;
    size_t i1_hi = (i1_lo + 1 < inp_d1) ? i1_lo + 1 : inp_d1 - 1;
    size_t i2_hi = (i2_lo + 1 < inp_d2) ? i2_lo + 1 : inp_d2 - 1;
    size_t i3_hi = (i3_lo + 1 < inp_d3) ? i3_lo + 1 : inp_d3 - 1;

    // Compute interpolation weights
    float w0 = f0 - (float)i0_lo;
    float w1 = f1 - (float)i1_lo;
    float w2 = f2 - (float)i2_lo;
    float w3 = f3 - (float)i3_lo;

    // For 4D, we need 16 corners. For efficiency, we interpolate progressively.
    // First interpolate along dim3, then dim2, then dim1, then dim0

    // Base offsets for low/high in each dimension
    size_t base_lo0 = i0_lo * inp_s0;
    size_t base_hi0 = i0_hi * inp_s0;
    size_t base_lo1 = i1_lo * inp_s1;
    size_t base_hi1 = i1_hi * inp_s1;
    size_t base_lo2 = i2_lo * inp_s2;
    size_t base_hi2 = i2_hi * inp_s2;
    size_t base_lo3 = i3_lo * inp_s3;
    size_t base_hi3 = i3_hi * inp_s3;

    // Interpolate along dim3 for all 8 combinations of dim0,dim1,dim2
    float v000 = inp[base_lo0 + base_lo1 + base_lo2 + base_lo3] * (1.0f - w3)
               + inp[base_lo0 + base_lo1 + base_lo2 + base_hi3] * w3;
    float v001 = inp[base_lo0 + base_lo1 + base_hi2 + base_lo3] * (1.0f - w3)
               + inp[base_lo0 + base_lo1 + base_hi2 + base_hi3] * w3;
    float v010 = inp[base_lo0 + base_hi1 + base_lo2 + base_lo3] * (1.0f - w3)
               + inp[base_lo0 + base_hi1 + base_lo2 + base_hi3] * w3;
    float v011 = inp[base_lo0 + base_hi1 + base_hi2 + base_lo3] * (1.0f - w3)
               + inp[base_lo0 + base_hi1 + base_hi2 + base_hi3] * w3;
    float v100 = inp[base_hi0 + base_lo1 + base_lo2 + base_lo3] * (1.0f - w3)
               + inp[base_hi0 + base_lo1 + base_lo2 + base_hi3] * w3;
    float v101 = inp[base_hi0 + base_lo1 + base_hi2 + base_lo3] * (1.0f - w3)
               + inp[base_hi0 + base_lo1 + base_hi2 + base_hi3] * w3;
    float v110 = inp[base_hi0 + base_hi1 + base_lo2 + base_lo3] * (1.0f - w3)
               + inp[base_hi0 + base_hi1 + base_lo2 + base_hi3] * w3;
    float v111 = inp[base_hi0 + base_hi1 + base_hi2 + base_lo3] * (1.0f - w3)
               + inp[base_hi0 + base_hi1 + base_hi2 + base_hi3] * w3;

    // Interpolate along dim2
    float v00 = v000 * (1.0f - w2) + v001 * w2;
    float v01 = v010 * (1.0f - w2) + v011 * w2;
    float v10 = v100 * (1.0f - w2) + v101 * w2;
    float v11 = v110 * (1.0f - w2) + v111 * w2;

    // Interpolate along dim1
    float v0 = v00 * (1.0f - w1) + v01 * w1;
    float v1 = v10 * (1.0f - w1) + v11 * w1;

    // Interpolate along dim0
    out[out_idx] = v0 * (1.0f - w0) + v1 * w0;
}

// N-dimensional Slice: Extract a slice from input tensor
// For each output coordinate, compute corresponding input coordinate using:
//   input_coord[d] = starts[d] + output_coord[d] * steps[d]
// Supports arbitrary axes, negative steps (reverse), up to 8 dimensions
extern "C" __global__ void slice_nd_kernel(
    float* out, const float* inp,
    const size_t* inp_strides, const size_t* out_strides,
    const long long* starts, const long long* steps,
    size_t ndim, size_t total_out_elements
) {
    size_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= total_out_elements) return;

    // Convert flat output index to multi-dimensional coordinates
    size_t out_coords[8];
    size_t remaining = out_idx;
    for (size_t d = 0; d < ndim; d++) {
        out_coords[d] = remaining / out_strides[d];
        remaining = remaining % out_strides[d];
    }

    // Map output coordinates to input coordinates using slice parameters
    size_t inp_idx = 0;
    for (size_t d = 0; d < ndim; d++) {
        // input_coord = starts[d] + output_coord * steps[d]
        long long inp_coord = starts[d] + (long long)out_coords[d] * steps[d];
        inp_idx += (size_t)inp_coord * inp_strides[d];
    }

    out[out_idx] = inp[inp_idx];
}

// N-dimensional Slice for Int64: Extract a slice from input tensor
extern "C" __global__ void slice_nd_i64_kernel(
    long long* out, const long long* inp,
    const size_t* inp_strides, const size_t* out_strides,
    const long long* starts, const long long* steps,
    size_t ndim, size_t total_out_elements
) {
    size_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= total_out_elements) return;

    // Convert flat output index to multi-dimensional coordinates
    size_t out_coords[8];
    size_t remaining = out_idx;
    for (size_t d = 0; d < ndim; d++) {
        out_coords[d] = remaining / out_strides[d];
        remaining = remaining % out_strides[d];
    }

    // Map output coordinates to input coordinates using slice parameters
    size_t inp_idx = 0;
    for (size_t d = 0; d < ndim; d++) {
        long long inp_coord = starts[d] + (long long)out_coords[d] * steps[d];
        inp_idx += (size_t)inp_coord * inp_strides[d];
    }

    out[out_idx] = inp[inp_idx];
}

// N-dimensional Slice for Int32: Extract a slice from input tensor
extern "C" __global__ void slice_nd_i32_kernel(
    int* out, const int* inp,
    const size_t* inp_strides, const size_t* out_strides,
    const long long* starts, const long long* steps,
    size_t ndim, size_t total_out_elements
) {
    size_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= total_out_elements) return;

    // Convert flat output index to multi-dimensional coordinates
    size_t out_coords[8];
    size_t remaining = out_idx;
    for (size_t d = 0; d < ndim; d++) {
        out_coords[d] = remaining / out_strides[d];
        remaining = remaining % out_strides[d];
    }

    // Map output coordinates to input coordinates using slice parameters
    size_t inp_idx = 0;
    for (size_t d = 0; d < ndim; d++) {
        long long inp_coord = starts[d] + (long long)out_coords[d] * steps[d];
        inp_idx += (size_t)inp_coord * inp_strides[d];
    }

    out[out_idx] = inp[inp_idx];
}

// Optimized 3D Slice with scalar parameters (no H2D transfers)
// Parameters passed directly as kernel arguments instead of GPU arrays
extern "C" __global__ void slice_3d_scalar_kernel(
    float* out, const float* inp,
    size_t inp_stride0, size_t inp_stride1, size_t inp_stride2,
    size_t out_stride0, size_t out_stride1, size_t out_stride2,
    long long start0, long long start1, long long start2,
    long long step0, long long step1, long long step2,
    size_t total_out_elements
) {
    size_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= total_out_elements) return;

    // Decompose flat index to 3D coordinates
    size_t c0 = out_idx / out_stride0;
    size_t rem = out_idx % out_stride0;
    size_t c1 = rem / out_stride1;
    size_t c2 = rem % out_stride1;

    // Map to input coordinates
    size_t inp_idx = (size_t)(start0 + (long long)c0 * step0) * inp_stride0
                   + (size_t)(start1 + (long long)c1 * step1) * inp_stride1
                   + (size_t)(start2 + (long long)c2 * step2) * inp_stride2;

    out[out_idx] = inp[inp_idx];
}

// Optimized 3D Slice for Int64 with scalar parameters
extern "C" __global__ void slice_3d_i64_scalar_kernel(
    long long* out, const long long* inp,
    size_t inp_stride0, size_t inp_stride1, size_t inp_stride2,
    size_t out_stride0, size_t out_stride1, size_t out_stride2,
    long long start0, long long start1, long long start2,
    long long step0, long long step1, long long step2,
    size_t total_out_elements
) {
    size_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= total_out_elements) return;

    size_t c0 = out_idx / out_stride0;
    size_t rem = out_idx % out_stride0;
    size_t c1 = rem / out_stride1;
    size_t c2 = rem % out_stride1;

    size_t inp_idx = (size_t)(start0 + (long long)c0 * step0) * inp_stride0
                   + (size_t)(start1 + (long long)c1 * step1) * inp_stride1
                   + (size_t)(start2 + (long long)c2 * step2) * inp_stride2;

    out[out_idx] = inp[inp_idx];
}

// Optimized 4D Slice with scalar parameters (no H2D transfers)
// Optimized 4D Slice with packed params (avoids 19-arg limit).
// strides layout: [inp_s0..3, out_s0..3] = 8 size_t values.
// slice_params layout: [start0..3, step0..3] = 8 long long values.
extern "C" __global__ void slice_4d_scalar_kernel(
    float* out, const float* inp,
    const size_t* strides, const long long* slice_params,
    size_t total_out_elements
) {
    size_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= total_out_elements) return;

    size_t inp_stride0 = strides[0], inp_stride1 = strides[1];
    size_t inp_stride2 = strides[2], inp_stride3 = strides[3];
    size_t out_stride0 = strides[4], out_stride1 = strides[5];
    size_t out_stride2 = strides[6];
    long long start0 = slice_params[0], start1 = slice_params[1];
    long long start2 = slice_params[2], start3 = slice_params[3];
    long long step0 = slice_params[4], step1 = slice_params[5];
    long long step2 = slice_params[6], step3 = slice_params[7];

    size_t c0 = out_idx / out_stride0;
    size_t rem = out_idx % out_stride0;
    size_t c1 = rem / out_stride1;
    rem = rem % out_stride1;
    size_t c2 = rem / out_stride2;
    size_t c3 = rem % out_stride2;

    size_t inp_idx = (size_t)(start0 + (long long)c0 * step0) * inp_stride0
                   + (size_t)(start1 + (long long)c1 * step1) * inp_stride1
                   + (size_t)(start2 + (long long)c2 * step2) * inp_stride2
                   + (size_t)(start3 + (long long)c3 * step3) * inp_stride3;

    out[out_idx] = inp[inp_idx];
}

// Optimized 4D Slice for Int64 with packed params (same layout as f32 variant).
extern "C" __global__ void slice_4d_i64_scalar_kernel(
    long long* out, const long long* inp,
    const size_t* strides, const long long* slice_params,
    size_t total_out_elements
) {
    size_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= total_out_elements) return;

    size_t inp_stride0 = strides[0], inp_stride1 = strides[1];
    size_t inp_stride2 = strides[2], inp_stride3 = strides[3];
    size_t out_stride0 = strides[4], out_stride1 = strides[5];
    size_t out_stride2 = strides[6];
    long long start0 = slice_params[0], start1 = slice_params[1];
    long long start2 = slice_params[2], start3 = slice_params[3];
    long long step0 = slice_params[4], step1 = slice_params[5];
    long long step2 = slice_params[6], step3 = slice_params[7];

    size_t c0 = out_idx / out_stride0;
    size_t rem = out_idx % out_stride0;
    size_t c1 = rem / out_stride1;
    rem = rem % out_stride1;
    size_t c2 = rem / out_stride2;
    size_t c3 = rem % out_stride2;

    size_t inp_idx = (size_t)(start0 + (long long)c0 * step0) * inp_stride0
                   + (size_t)(start1 + (long long)c1 * step1) * inp_stride1
                   + (size_t)(start2 + (long long)c2 * step2) * inp_stride2
                   + (size_t)(start3 + (long long)c3 * step3) * inp_stride3;

    out[out_idx] = inp[inp_idx];
}

// Fill kernel - fills array with constant value
extern "C" __global__ void fill_kernel(
    float* out, float value, size_t total_elements
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    out[idx] = value;
}

// Int64 variant of fill_kernel. Needed for ONNX ConstantOfShape when the
// `value` attribute carries an Int64 scalar (common after a Shape op
// whose output is Int64 by spec). Using the f32 fill_kernel with a
// reinterpret would be unsound; this kernel writes typed i64 values.
extern "C" __global__ void fill_i64_kernel(
    long long* out, long long value, size_t total_elements
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    out[idx] = value;
}
"#;
