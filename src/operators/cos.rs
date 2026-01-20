/// Cos operator for Iconnx
///
/// Element-wise cosine
/// ONNX spec: https://onnx.ai/onnx/operators/onnx__Cos.html
use crate::tensor::Tensor;

/// Cos operator - element-wise cosine
pub struct Cos;

impl Cos {
    /// Forward pass: element-wise cos
    ///
    /// # Arguments
    /// * `inputs` - Single input tensor
    ///
    /// # Returns
    /// Tensor with cos applied element-wise
    pub fn forward(
        inputs: &[Tensor],
        _attributes: &crate::attributes::NodeAttributes,
    ) -> Tensor {
        assert_eq!(inputs.len(), 1, "Cos requires exactly 1 input");

        let data = inputs[0].to_array();
        let result = data.mapv(|x| x.cos());

        Tensor::from_array(result)
    }
}
