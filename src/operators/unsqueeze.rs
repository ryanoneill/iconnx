/// Unsqueeze operator for Iconnx
///
/// Inserts dimensions of size 1 into tensor shape
/// ONNX spec: https://onnx.ai/onnx/operators/onnx__Unsqueeze.html
use crate::tensor::Tensor;
use ndarray::Axis;

/// Unsqueeze operator - insert singleton dimensions
pub struct Unsqueeze;

impl Unsqueeze {
    /// Forward pass: insert dimensions
    ///
    /// # Arguments
    /// * `inputs` - [data_tensor, axes_tensor]
    ///   - data_tensor: Input tensor
    ///   - axes_tensor: 1D tensor with axes to insert (f32 values)
    ///
    /// # Returns
    /// Tensor with additional dimensions of size 1
    pub fn forward(
        inputs: &[Tensor],
        _attributes: &crate::attributes::NodeAttributes,
    ) -> Tensor {
        assert_eq!(inputs.len(), 2, "Unsqueeze requires exactly 2 inputs");

        // Get axes as i64 values first (to handle negative axes)
        let axes_i64: Vec<i64> = if inputs[1].is_int64() {
            inputs[1].as_slice_i64().to_vec()
        } else {
            inputs[1].as_slice().iter().map(|&v| v as i64).collect()
        };

        // Output rank = input rank + number of axes to insert
        let output_rank = inputs[0].ndim() + axes_i64.len();

        // Normalize negative axes relative to OUTPUT rank
        // ONNX spec: axes specify positions in OUTPUT shape
        let mut axes: Vec<usize> = axes_i64
            .iter()
            .map(|&a| {
                if a < 0 {
                    (output_rank as i64 + a) as usize
                } else {
                    a as usize
                }
            })
            .collect();

        // Sort axes in ascending order for correct insertion order
        axes.sort_unstable();

        #[cfg(debug_assertions)]
        {}

        // Preserve dtype when inserting axes
        inputs[0].map_preserving_dtype(
            |data| {
                let mut result = data.clone();
                for &axis in &axes {
                    result = result.insert_axis(Axis(axis));
                }
                result
            },
            |data| {
                let mut result = data.clone();
                for &axis in &axes {
                    result = result.insert_axis(Axis(axis));
                }
                result
            },
        )
    }
}
