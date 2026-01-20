/// Sin operator for Iconnx
///
/// Element-wise sine
/// ONNX spec: https://onnx.ai/onnx/operators/onnx__Sin.html
use crate::tensor::Tensor;

/// Sin operator - element-wise sine
pub struct Sin;

impl Sin {
    /// Forward pass: element-wise sin
    ///
    /// # Arguments
    /// * `inputs` - Single input tensor
    ///
    /// # Returns
    /// Tensor with sin applied element-wise
    pub fn forward(
        inputs: &[Tensor],
        _attributes: &crate::attributes::NodeAttributes,
    ) -> Tensor {
        assert_eq!(inputs.len(), 1, "Sin requires exactly 1 input");

        let data = inputs[0].to_array();
        let result = data.mapv(|x| x.sin());

        Tensor::from_array(result)
    }
}
