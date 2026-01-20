/// And operator for Iconnx
///
/// Logical AND (element-wise)
/// ONNX spec: https://onnx.ai/onnx/operators/onnx__And.html
use crate::tensor::Tensor;

/// And operator - element-wise logical AND
pub struct And;

impl And {
    /// Forward pass: element-wise AND
    ///
    /// # Arguments
    /// * `inputs` - [a, b]
    ///   - Both tensors with 1.0 = true, 0.0 = false
    ///
    /// # Returns
    /// Tensor with logical AND results (1.0 or 0.0)
    pub fn forward(
        inputs: &[Tensor],
        _attributes: &crate::attributes::NodeAttributes,
    ) -> Tensor {
        assert_eq!(inputs.len(), 2, "And requires exactly 2 inputs");

        let a = inputs[0].to_array();
        let b = inputs[1].to_array();

        // Element-wise AND operation
        let a_slice = a.as_slice().unwrap();
        let b_slice = b.as_slice().unwrap();

        let result_vec: Vec<f32> = a_slice
            .iter()
            .zip(b_slice.iter())
            .map(|(&a_val, &b_val)| {
                if a_val != 0.0 && b_val != 0.0 {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();

        Tensor::from_vec(result_vec, a.shape().to_vec())
    }
}
