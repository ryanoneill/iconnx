/// Tanh activation operator for Iconnx
///
/// Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
/// ONNX spec: https://onnx.ai/onnx/operators/onnx__Tanh.html
use crate::tensor::Tensor;

/// Tanh operator - hyperbolic tangent activation
pub struct Tanh;

impl Tanh {
    /// Forward pass: element-wise tanh
    ///
    /// # Arguments
    /// * `inputs` - Single input tensor
    ///
    /// # Returns
    /// Tensor with tanh applied element-wise
    pub fn forward(
        inputs: &[Tensor],
        _attributes: &crate::attributes::NodeAttributes,
    ) -> Tensor {
        assert_eq!(inputs.len(), 1, "Tanh requires exactly 1 input");

        let data = inputs[0].to_array();
        let result = data.mapv(|x| x.tanh());

        Tensor::from_array(result)
    }
}
