/// Equal operator for Iconnx
///
/// Element-wise x == y comparison
/// ONNX spec: https://onnx.ai/onnx/operators/onnx__Equal.html
use crate::tensor::Tensor;

/// Equal operator - element-wise comparison
pub struct Equal;

impl Equal {
    /// Forward pass: element-wise x == y
    ///
    /// # Arguments
    /// * `inputs` - [x, y] tensors
    ///
    /// # Returns
    /// Tensor with 1.0 where x == y, 0.0 otherwise
    pub fn forward(
        inputs: &[Tensor],
        _attributes: &crate::attributes::NodeAttributes,
    ) -> Tensor {
        assert_eq!(inputs.len(), 2, "Equal requires exactly 2 inputs");

        // Handle both Int64 and Float32 comparisons
        if inputs[0].is_int64() && inputs[1].is_int64() {
            let x = inputs[0].to_array_i64();
            let y = inputs[1].to_array_i64();

            let result = x
                .iter()
                .zip(y.iter())
                .map(|(a, b)| if a == b { 1.0 } else { 0.0 })
                .collect::<Vec<_>>();

            Tensor::from_vec(result, x.shape().to_vec())
        } else {
            // Convert to Float32 and compare
            let x = inputs[0].to_float32();
            let y = inputs[1].to_float32();

            let x_arr = x.to_array();
            let y_arr = y.to_array();

            let result = x_arr
                .iter()
                .zip(y_arr.iter())
                .map(|(a, b)| if a == b { 1.0 } else { 0.0 })
                .collect::<Vec<_>>();

            #[cfg(debug_assertions)]
            {
                if result.len() <= 5 {}
            }

            Tensor::from_vec(result, x_arr.shape().to_vec())
        }
    }
}
