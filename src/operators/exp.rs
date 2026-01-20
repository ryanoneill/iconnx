/// Exp operator for Iconnx
///
/// Element-wise natural exponential: e^x
/// ONNX spec: https://onnx.ai/onnx/operators/onnx__Exp.html
use crate::tensor::Tensor;

/// Exp operator - element-wise exponential
pub struct Exp;

impl Exp {
    /// Forward pass: element-wise e^x
    ///
    /// # Arguments
    /// * `inputs` - Single input tensor
    ///
    /// # Returns
    /// Tensor with exp applied element-wise
    pub fn forward(
        inputs: &[Tensor],
        _attributes: &crate::attributes::NodeAttributes,
    ) -> Tensor {
        assert_eq!(inputs.len(), 1, "Exp requires exactly 1 input");

        inputs[0].map_preserving_dtype(
            |arr| arr.mapv(|x| x.exp()),
            |arr| arr.mapv(|x| (x as f64).exp() as i64),
        )
    }
}
