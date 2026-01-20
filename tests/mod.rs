//! iconnx inference framework tests
//!
//! Comprehensive test suite for validating ONNX operator implementations
//! against ONNX Runtime as ground truth.

mod conv_dilation_test;
mod debug_where_expand;
mod end_to_end_test;
mod gather_axis_attribute_test;
mod graph_executor_test;
mod kokoro_debug_test;
mod kokoro_inference_test;
mod kokoro_progressive_test;
mod kokoro_validation_test;
mod onnx_parser_test;
mod operators;
mod squeeze_multidtype_test;
mod targeted_range_test;
mod tensor_test;
mod unsqueeze_axes_test;
mod weight_extraction_test;

#[cfg(feature = "debug-inference")]
mod validation_test;
