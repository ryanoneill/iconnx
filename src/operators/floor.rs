/// Floor operator for Iconnx
///
/// Element-wise floor
/// ONNX spec: https://onnx.ai/onnx/operators/onnx__Floor.html
use crate::tensor::Tensor;

/// Floor operator - element-wise floor
pub struct Floor;

impl Floor {
    /// Forward pass: element-wise floor
    ///
    /// # Arguments
    /// * `inputs` - Single input tensor
    ///
    /// # Returns
    /// Tensor with floor applied element-wise
    pub fn forward(
        inputs: &[Tensor],
        _attributes: &crate::attributes::NodeAttributes,
    ) -> Tensor {
        assert_eq!(inputs.len(), 1, "Floor requires exactly 1 input");

        let data = inputs[0].to_array();
        let result = data.mapv(|x| x.floor());

        Tensor::from_array(result)
    }
}
