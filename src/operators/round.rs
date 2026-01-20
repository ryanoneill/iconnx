/// Round operator for Iconnx
///
/// Element-wise round
/// ONNX spec: https://onnx.ai/onnx/operators/onnx__Round.html
use crate::tensor::Tensor;

/// Round operator - element-wise round
pub struct Round;

impl Round {
    /// Forward pass: element-wise round
    ///
    /// # Arguments
    /// * `inputs` - Single input tensor
    ///
    /// # Returns
    /// Tensor with round applied element-wise
    pub fn forward(
        inputs: &[Tensor],
        _attributes: &crate::attributes::NodeAttributes,
    ) -> Tensor {
        assert_eq!(inputs.len(), 1, "Round requires exactly 1 input");

        let data = inputs[0].to_array();
        let result = data.mapv(|x| x.round());

        Tensor::from_array(result)
    }
}
