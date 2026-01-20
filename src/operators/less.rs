/// Less operator for Iconnx
///
/// Element-wise x < y comparison
/// ONNX spec: https://onnx.ai/onnx/operators/onnx__Less.html
use crate::tensor::Tensor;

/// Less operator - element-wise comparison
pub struct Less;

impl Less {
    /// Forward pass: element-wise x < y
    ///
    /// # Arguments
    /// * `inputs` - [x, y] tensors
    ///
    /// # Returns
    /// Tensor with 1.0 where x < y, 0.0 otherwise
    pub fn forward(
        inputs: &[Tensor],
        _attributes: &crate::attributes::NodeAttributes,
    ) -> Tensor {
        assert_eq!(inputs.len(), 2, "Less requires exactly 2 inputs");

        // Convert to Float32 for comparison
        let x = inputs[0].to_float32();
        let y = inputs[1].to_float32();

        // Compute broadcast shape
        let max_rank = x.ndim().max(y.ndim());
        let mut broadcast_shape = vec![1; max_rank];

        for (i, &dim) in x.shape().iter().rev().enumerate() {
            let idx = broadcast_shape.len() - 1 - i;
            broadcast_shape[idx] = broadcast_shape[idx].max(dim);
        }
        for (i, &dim) in y.shape().iter().rev().enumerate() {
            let idx = broadcast_shape.len() - 1 - i;
            broadcast_shape[idx] = broadcast_shape[idx].max(dim);
        }

        // Broadcast both inputs
        let x_bc = x
            .to_array()
            .broadcast(broadcast_shape.as_slice())
            .unwrap()
            .to_owned();
        let y_bc = y
            .to_array()
            .broadcast(broadcast_shape.as_slice())
            .unwrap()
            .to_owned();

        // Element-wise comparison
        let result = x_bc
            .iter()
            .zip(y_bc.iter())
            .map(|(a, b)| if a < b { 1.0 } else { 0.0 })
            .collect::<Vec<_>>();

        Tensor::from_vec(result, broadcast_shape)
    }
}
