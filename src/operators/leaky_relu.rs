/// LeakyRelu operator for Iconnx
///
/// Leaky ReLU activation: x if x > 0 else alpha * x
/// ONNX spec: https://onnx.ai/onnx/operators/onnx__LeakyRelu.html
use crate::tensor::Tensor;

/// LeakyRelu operator - leaky rectified linear unit
pub struct LeakyRelu;

impl LeakyRelu {
    /// Forward pass: element-wise leaky ReLU
    ///
    /// # Arguments
    /// * `inputs` - [data]
    ///   - data: input tensor
    /// * `attributes` - ONNX attributes:
    ///   - alpha: slope for negative values (default: 0.01)
    ///
    /// # Returns
    /// Tensor with LeakyRelu applied element-wise
    pub fn forward(
        inputs: &[Tensor],
        attributes: &crate::attributes::NodeAttributes,
    ) -> Tensor {
        assert_eq!(inputs.len(), 1, "LeakyRelu requires exactly 1 input [data]");

        // Get alpha from attributes (default 0.01 per ONNX spec)
        let alpha = attributes.get_float("alpha").unwrap_or(0.01);

        // Apply to Float32 data
        let data = inputs[0].to_float32();
        let data_arr = data.to_array();

        let result = data_arr.mapv(|x| if x > 0.0 { x } else { alpha * x });

        Tensor::from_array(result)
    }
}
