/// Add operator for Iconnx
///
/// Implements element-wise addition with broadcasting support
/// ONNX spec: https://onnx.ai/onnx/operators/onnx__Add.html
use crate::tensor::Tensor;
use ndarray::ArrayD;

/// Add operator - performs element-wise addition
pub struct Add;

impl Add {
    /// Forward pass: element-wise addition of inputs
    ///
    /// Supports:
    /// - Multiple inputs (sum all of them)
    /// - Broadcasting (automatically handled by ndarray)
    /// - Any number of dimensions
    ///
    /// # Arguments
    /// * `inputs` - Slice of input tensors to add
    ///
    /// # Returns
    /// Tensor containing element-wise sum
    ///
    /// # Panics
    /// Panics if inputs is empty or broadcasting fails
    pub fn forward(
        inputs: &[Tensor],
        _attributes: &crate::attributes::NodeAttributes,
    ) -> Tensor {
        assert!(!inputs.is_empty(), "Add requires at least one input");

        if inputs.len() == 1 {
            // Single input: return clone (identity)
            return inputs[0].clone();
        }

        // Convert all to Float32
        let as_f32: Vec<Tensor> = inputs.iter().map(|t| t.to_float32()).collect();

        // Determine broadcast shape (max of all dimensions)
        let max_rank = as_f32.iter().map(|t| t.ndim()).max().unwrap_or(0);
        let mut broadcast_shape = vec![1; max_rank];

        for tensor in &as_f32 {
            for (i, &dim) in tensor.shape().iter().rev().enumerate() {
                let idx = broadcast_shape.len() - 1 - i;
                broadcast_shape[idx] = broadcast_shape[idx].max(dim);
            }
        }

        #[cfg(debug_assertions)]
        {
            // Only log problematic cases (non-scalar, non-matching shapes)
            if as_f32.len() == 2 && !as_f32[1].shape().is_empty() {
                for _t in as_f32.iter() {}
            }
        }

        // Broadcast all inputs to common shape and sum
        let broadcasted: Vec<ArrayD<f32>> = as_f32
            .iter()
            .enumerate()
            .map(|(i, t)| {
                t.to_array()
                    .broadcast(broadcast_shape.as_slice())
                    .unwrap_or_else(|| {
                        panic!(
                            "Failed to broadcast input[{}] from {:?} to {:?}",
                            i,
                            t.shape(),
                            broadcast_shape
                        )
                    })
                    .to_owned()
            })
            .collect();

        // Sum all broadcasted arrays
        let mut result = broadcasted[0].clone();
        for arr in &broadcasted[1..] {
            result += arr;
        }

        Tensor::from_array(result)
    }
}
