/// Sqrt operator for Iconnx
///
/// Element-wise square root
/// ONNX spec: https://onnx.ai/onnx/operators/onnx__Sqrt.html
use crate::tensor::Tensor;

/// Sqrt operator - element-wise square root
pub struct Sqrt;

impl Sqrt {
    /// Forward pass: element-wise sqrt
    ///
    /// # Arguments
    /// * `inputs` - Single input tensor
    ///
    /// # Returns
    /// Tensor with sqrt applied element-wise
    pub fn forward(
        inputs: &[Tensor],
        _attributes: &crate::attributes::NodeAttributes,
    ) -> Tensor {
        assert_eq!(inputs.len(), 1, "Sqrt requires exactly 1 input");

        // Use map_preserving_dtype to handle multiple numeric types
        // For sqrt, we need to convert int64 to f64, apply sqrt, then convert back
        inputs[0].map_preserving_dtype(
            |arr| arr.mapv(|x| x.sqrt()),                 // f32 -> f32
            |arr| arr.mapv(|x| (x as f64).sqrt() as i64), // i64 -> i64
        )
    }
}
