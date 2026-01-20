//! ScatterND operator for Iconnx
//!
//! Implements scatter updates into tensor using indices
//! ONNX spec: https://onnx.ai/onnx/operators/onnx__ScatterND.html

#![allow(clippy::needless_range_loop)] // Explicit indexing is clearer for scatter operations

use crate::tensor::Tensor;

/// ScatterND operator - scatter updates into data tensor using indices
pub struct ScatterND;

impl ScatterND {
    /// Forward pass: scatter updates into data tensor
    ///
    /// # Arguments
    /// * `inputs` - Exactly 3 inputs: [data, indices, updates]
    ///   - data: Base tensor to scatter into
    ///   - indices: Indices specifying where to scatter (shape: [num_updates, index_depth])
    ///   - updates: Values to scatter (shape: [num_updates, ...])
    ///
    /// # Returns
    /// Tensor with updates scattered into data
    ///
    /// # Panics
    /// Panics if shapes are incompatible
    pub fn forward(
        inputs: &[Tensor],
        _attributes: &crate::attributes::NodeAttributes,
    ) -> Tensor {
        assert_eq!(inputs.len(), 3, "ScatterND requires exactly 3 inputs");

        let data = inputs[0].to_array();
        let indices_tensor = &inputs[1];
        let updates_tensor = &inputs[2];

        #[cfg(debug_assertions)]
        {
            if indices_tensor.is_int64() && indices_tensor.len() <= 20 {}
        }

        // Clone data to create mutable output
        let mut output = data.clone();
        let output_shape = output.shape().to_vec();

        // Get indices and updates as slices
        let indices_shape = indices_tensor.shape();
        let updates_shape = updates_tensor.shape();

        assert!(!indices_shape.is_empty(), "Indices must not be empty");

        // ONNX ScatterND spec: indices shape is [q_1, ..., q_r-1, k]
        // where k (last dimension) is the index_depth
        // and [q_1, ..., q_r-1] are the batch dimensions
        let index_depth = indices_shape[indices_shape.len() - 1];
        let batch_dims = &indices_shape[..indices_shape.len() - 1];
        let num_updates: usize = batch_dims.iter().product();

        #[cfg(debug_assertions)]
        {}

        let updates_slice = updates_tensor.as_slice();

        // Calculate strides for output tensor
        let output_strides = Self::compute_strides(&output_shape);

        // Calculate elements per update
        let update_size: usize = if updates_shape.len() > 1 {
            updates_shape[1..].iter().product()
        } else {
            1
        };

        // Process each update
        // Indices are laid out in memory as [q_1, ..., q_r-1, k]
        // We iterate over the batch dimensions and extract k indices per update
        for update_idx in 0..num_updates {
            // Extract k-dimensional index for this update
            // Handle both Int64 and Float32 indices
            let mut target_indices = vec![0usize; index_depth];
            let base_pos = update_idx * index_depth;
            for i in 0..index_depth {
                let idx_pos = base_pos + i;
                target_indices[i] = if indices_tensor.is_int64() {
                    indices_tensor.as_slice_i64()[idx_pos] as usize
                } else {
                    indices_tensor.as_slice()[idx_pos] as usize
                };
            }

            // Calculate flat output position
            let output_base_idx: usize = target_indices
                .iter()
                .zip(output_strides.iter())
                .map(|(&idx, &stride)| idx * stride)
                .sum();

            // Copy update values
            let update_start = update_idx * update_size;

            if let Some(output_slice) = output.as_slice_mut() {
                for i in 0..update_size {
                    let out_idx = output_base_idx + i;
                    if out_idx >= output_slice.len() {
                        panic!("ScatterND: index {} out of bounds for output of len {} (target_indices={:?}, strides={:?})",
                            out_idx, output_slice.len(), target_indices, output_strides);
                    }
                    output_slice[out_idx] = updates_slice[update_start + i];
                }
            } else {
                // Fallback for non-contiguous arrays
                for i in 0..update_size {
                    let flat_idx = output_base_idx + i;
                    // Convert flat index back to multi-dimensional
                    let mut multi_idx = vec![0; output_shape.len()];
                    let mut remaining = flat_idx;
                    for (dim, &stride) in output_strides.iter().enumerate() {
                        multi_idx[dim] = remaining / stride;
                        remaining %= stride;
                    }

                    // Set value
                    let idx_array: Vec<_> = multi_idx.to_vec();
                    output[&idx_array[..]] = updates_slice[update_start + i];
                }
            }
        }

        Tensor::from_array(output)
    }

    /// Compute strides for a given shape
    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }
}
