/// Mul operator for Iconnx
///
/// Implements element-wise multiplication with broadcasting support
/// ONNX spec: https://onnx.ai/onnx/operators/onnx__Mul.html
use crate::tensor::Tensor;

/// Mul operator - performs element-wise multiplication
pub struct Mul;

impl Mul {
    /// Forward pass: element-wise multiplication of inputs
    ///
    /// Supports:
    /// - Multiple inputs (multiply all of them)
    /// - Broadcasting (automatically handled by ndarray)
    /// - Any number of dimensions
    ///
    /// # Arguments
    /// * `inputs` - Slice of input tensors to multiply
    ///
    /// # Returns
    /// Tensor containing element-wise product
    ///
    /// # Panics
    /// Panics if inputs is empty or broadcasting fails
    pub fn forward(
        inputs: &[Tensor],
        _attributes: &crate::attributes::NodeAttributes,
    ) -> Tensor {
        assert!(!inputs.is_empty(), "Mul requires at least one input");

        if inputs.len() == 1 {
            // Single input: return clone (identity)
            return inputs[0].clone();
        }

        #[cfg(debug_assertions)]
        {}

        // Use dispatch_multi for dtype preservation
        Tensor::dispatch_multi(
            inputs,
            |arrays| {
                let mut result = arrays[0].clone();
                for arr in &arrays[1..] {
                    result = &result * *arr;
                }
                result
            },
            |arrays| {
                let mut result = arrays[0].clone();
                for arr in &arrays[1..] {
                    result = &result * *arr;
                }
                result
            },
        )
    }
}
