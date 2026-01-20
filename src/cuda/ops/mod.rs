//! Memory layout and utility operations for GPU tensors
//!
//! This module provides GPU-accelerated implementations of:
//! - Layout operations: transpose, gather, slice, copy
//! - Comparison operations: equal, less, greater, etc.
//! - Logical operations: and, or, not
//! - Sequence operations: range, cumsum
//! - Math operations: atan
//! - Type casting operations: cast between dtypes

mod cache;
mod cast;
mod comparison;
mod kernels;
mod layout;
mod sequence;

pub use cache::OpsKernelCache;

// Cast operations
pub use cast::gpu_cast;

// Layout operations
#[allow(unused_imports)]
pub use layout::{
    gpu_concat, gpu_constant_of_shape, gpu_constant_of_shape_direct, gpu_copy, gpu_expand,
    gpu_gather, gpu_gather_from_gpu, gpu_nonzero, gpu_pad, gpu_resize, gpu_scatter_nd, gpu_shape,
    gpu_slice_contiguous, gpu_slice_nd, gpu_transpose_2d, gpu_transpose_nd, gpu_where,
    ResizeCoordMode, ResizeMode, ResizeNearestMode,
};

// Comparison operations
pub use comparison::{
    gpu_equal, gpu_greater, gpu_greater_or_equal, gpu_less, gpu_less_or_equal, gpu_not_equal,
};

// Logical operations
pub use comparison::{gpu_and, gpu_not, gpu_or};

// Sequence and math operations
pub use sequence::{gpu_atan, gpu_cumsum, gpu_range, gpu_range_i64};
