//! NonZero operator for Iconnx
//!
//! Find indices of non-zero elements
//! ONNX spec: https://onnx.ai/onnx/operators/onnx__NonZero.html

#![allow(clippy::needless_range_loop)] // Explicit indexing is clearer for index operations

use crate::tensor::Tensor;

/// NonZero operator - find non-zero element indices
pub struct NonZero;

impl NonZero {
    /// Forward pass: find non-zero indices
    ///
    /// # Arguments
    /// * `inputs` - Single input tensor
    ///
    /// # Returns
    /// 2D tensor where each column is the index of a non-zero element
    /// Shape: (ndim, num_nonzero)
    pub fn forward(
        inputs: &[Tensor],
        _attributes: &crate::attributes::NodeAttributes,
    ) -> Tensor {
        assert_eq!(inputs.len(), 1, "NonZero requires exactly 1 input");

        let data = inputs[0].to_array();
        let shape = data.shape();
        let ndim = shape.len();

        // Find all non-zero elements
        let mut indices: Vec<Vec<usize>> = vec![Vec::new(); ndim];

        // Iterate over all elements
        for (idx, &value) in data.iter().zip(data.iter()) {
            if value != 0.0 {
                // Convert flat index to multi-dimensional indices
                let flat_idx = data
                    .iter()
                    .position(|&x| x == value && x == *idx)
                    .unwrap_or(0);
                let mut temp_idx = flat_idx;

                // Calculate multi-dim indices from flat index
                for dim in (0..ndim).rev() {
                    let stride: usize = shape[dim + 1..].iter().product::<usize>().max(1);
                    let idx_dim = temp_idx / stride;
                    indices[dim].push(idx_dim);
                    temp_idx %= stride;
                }
            }
        }

        // Better approach: iterate with proper multi-dim indexing
        let mut all_indices: Vec<Vec<usize>> = vec![Vec::new(); ndim];

        // Calculate strides
        let strides: Vec<usize> = (0..ndim)
            .map(|i| shape[i + 1..].iter().product::<usize>().max(1))
            .collect();

        for (flat_idx, &value) in data.iter().enumerate() {
            if value != 0.0 {
                let mut temp_idx = flat_idx;
                for dim in 0..ndim {
                    all_indices[dim].push(temp_idx / strides[dim]);
                    temp_idx %= strides[dim];
                }
            }
        }

        let num_nonzero = all_indices[0].len();

        // Build output: shape (ndim, num_nonzero)
        let mut output_data = Vec::new();
        for dim in 0..ndim {
            for &idx in &all_indices[dim] {
                output_data.push(idx as f32);
            }
        }

        if num_nonzero == 0 {
            // Special case: no non-zero elements
            Tensor::from_vec(vec![], vec![ndim, 0])
        } else {
            Tensor::from_vec(output_data, vec![ndim, num_nonzero])
        }
    }
}
