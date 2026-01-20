//! GPU Convolution operations using im2col + cuBLAS
//!
//! Implements Conv2D, Conv1D, and their transposed variants using the im2col
//! transformation, which converts convolution into matrix multiplication.
//!
//! ## Forward Convolution Algorithm:
//! 1. im2col: Transform input patches into columns
//! 2. GEMM: kernel @ col_matrix (using cuBLAS)
//! 3. Add bias (if provided)
//!
//! ## Transposed Convolution Algorithm:
//! 1. GEMM: kernel^T @ input
//! 2. col2im: Scatter column values back to spatial positions
//! 3. Add bias (if provided)

mod cache;
mod forward;
mod kernels;
mod params;
mod transpose;

pub use cache::ConvKernelCache;
pub use params::{Conv1dParams, Conv2dParams};

// Forward convolution
pub use forward::{add_bias, gpu_conv1d, gpu_conv2d, gpu_im2col, gpu_im2col_1d};

// Transposed convolution
pub use transpose::{gpu_col2im, gpu_col2im_1d, gpu_conv_transpose_1d, gpu_conv_transpose_2d};
