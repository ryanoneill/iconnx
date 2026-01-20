/// Constant operator for Iconnx
///
/// Returns a constant tensor (stored as attribute in ONNX)
/// ONNX spec: https://onnx.ai/onnx/operators/onnx__Constant.html
use crate::tensor::Tensor;

/// Constant operator - produces constant tensor
pub struct Constant;

impl Constant {
    /// Forward pass: return constant value
    ///
    /// # Arguments
    /// * `inputs` - Should be empty (constant defined in graph)
    /// * `attributes` - Contains the constant value
    ///
    /// # Returns
    /// Constant tensor
    ///
    /// Note: In ONNX, constant value is stored as attribute.
    /// GraphExecutor handles Constant nodes specially, extracting value from attributes.
    pub fn forward(
        _inputs: &[Tensor],
        _attributes: &crate::attributes::NodeAttributes,
    ) -> Tensor {
        // Constant values are loaded as initializers, not executed
        // This is a placeholder for graph execution compatibility
        panic!("Constant operator should not be executed - value should be in initializers");
    }
}
