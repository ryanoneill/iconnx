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
    /// * `inputs` - `[data_tensor]` (opset ≤ 12, axes lives on the
    ///   `axes` attribute) **or** `[data_tensor, axes_tensor]`
    ///   (opset ≥ 13, axes is an input).
    ///   - data_tensor: Input tensor
    ///   - axes_tensor (opset ≥ 13): 1D Int64 tensor with axes to
    ///     insert.
    /// * `attributes` - For opset ≤ 12, carries the `axes` attribute
    ///   (Vec<i64>). Ignored when a second input is present.
    ///
    /// # Returns
    /// Tensor with additional dimensions of size 1
    ///
    /// # Opset compatibility
    ///
    /// Mirrors the GPU executor's dual handling
    /// (`src/cuda/executor/layout.rs::Unsqueeze`): opset 13 moved
    /// `axes` from attribute to input, but older exports (e.g.
    /// paddle2onnx opset-12 `_rapidai` rec model) still carry the
    /// attribute form. Folding any Unsqueeze in such a graph requires
    /// reading the attribute path here.
    pub fn forward(
        inputs: &[Tensor],
        attributes: &crate::attributes::NodeAttributes,
    ) -> Tensor {
        // Opset 13+: axes is an input. Opset ≤ 12: axes is an attribute.
        let axes_i64: Vec<i64> = if inputs.len() >= 2 {
            if inputs[1].is_int64() {
                inputs[1].as_slice_i64().to_vec()
            } else {
                inputs[1].as_slice().iter().map(|&v| v as i64).collect()
            }
        } else if let Some(axes_attr) = attributes.get_ints("axes") {
            axes_attr.to_vec()
        } else {
            panic!(
                "Unsqueeze requires either an axes input (opset ≥ 13) or \
                 an 'axes' attribute (opset ≤ 12); got {} input(s) and no \
                 attribute",
                inputs.len()
            );
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
