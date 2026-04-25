//! Operator implementations for Iconnx
//!
//! Each operator will be test-driven:
//! 1. Write test comparing to ort
//! 2. Implement operator
//! 3. Validate accuracy
//!
//! Note: These are CPU reference implementations. GPU versions are used in production.
//! Keeping these for testing and validation against ORT.

pub mod add;
pub mod and;
pub mod atan;
pub mod cast;
pub mod clip;
pub mod concat;
pub mod constant;
pub mod constant_of_shape;
pub mod conv;
pub mod conv_transpose;
pub mod cos;
pub mod cumsum;
pub mod div;
pub mod equal;
pub mod erf;
pub mod exp;
pub mod expand;
pub mod fftw_stft;
pub mod flatten;
pub mod floor;
pub mod gather;
pub mod gemm;
pub mod global_average_pool;
pub mod greater;
pub mod greater_or_equal;
pub mod layer_normalization;
pub mod leaky_relu;
pub mod less;
pub mod lstm;
pub mod matmul;
pub mod mul;
pub mod nonzero;
pub mod pad;
pub mod pow;
pub mod range;
pub mod reduce_mean;
pub mod reduce_sum;
pub mod relu;
pub mod reshape;
pub mod resize;
pub mod round;
pub mod scatter_nd;
pub mod shape;
pub mod sigmoid;
pub mod sin;
pub mod slice;
pub mod softmax;
pub mod split;
pub mod sqrt;
pub mod squeeze;
pub mod stft;
pub mod sub;
pub mod tanh;
pub mod transpose;
pub mod unsqueeze;
pub mod where_op;
