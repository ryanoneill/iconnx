/// Sigmoid activation operator for Iconnx
///
/// Sigmoid(x) = 1 / (1 + exp(-x))
/// ONNX spec: https://onnx.ai/onnx/operators/onnx__Sigmoid.html
use crate::tensor::Tensor;

/// Sigmoid operator - logistic activation function
pub struct Sigmoid;

impl Sigmoid {
    /// Forward pass: element-wise sigmoid
    ///
    /// # Arguments
    /// * `inputs` - Single input tensor
    ///
    /// # Returns
    /// Tensor with sigmoid applied element-wise
    pub fn forward(
        inputs: &[Tensor],
        _attributes: &crate::attributes::NodeAttributes,
    ) -> Tensor {
        assert_eq!(inputs.len(), 1, "Sigmoid requires exactly 1 input");

        let data = inputs[0].to_array();
        let result = data.mapv(|x| 1.0 / (1.0 + (-x).exp()));

        Tensor::from_array(result)
    }
}
