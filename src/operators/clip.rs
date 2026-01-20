/// Clip operator for Iconnx
///
/// Clamps values: min(max(x, min_val), max_val)
/// ONNX spec: https://onnx.ai/onnx/operators/onnx__Clip.html
use crate::tensor::Tensor;

/// Clip operator - clamp values to [min, max] range
pub struct Clip;

impl Clip {
    /// Forward pass: element-wise clipping
    ///
    /// # Arguments
    /// * `inputs` - [data, min, max] or [data, min] or [data]
    ///   - data: input tensor
    ///   - min: optional minimum value (scalar)
    ///   - max: optional maximum value (scalar)
    ///
    /// # Returns
    /// Tensor with values clamped to [min, max]
    pub fn forward(
        inputs: &[Tensor],
        _attributes: &crate::attributes::NodeAttributes,
    ) -> Tensor {
        assert!(
            !inputs.is_empty() && inputs.len() <= 3,
            "Clip requires 1-3 inputs [data, min?, max?]"
        );

        // Use map_preserving_dtype for Float32 and Int64
        inputs[0].map_preserving_dtype(
            |data| {
                let min_val = if inputs.len() >= 2 {
                    inputs[1].as_slice()[0]
                } else {
                    f32::NEG_INFINITY
                };
                let max_val = if inputs.len() >= 3 {
                    inputs[2].as_slice()[0]
                } else {
                    f32::INFINITY
                };
                data.mapv(|x| x.max(min_val).min(max_val))
            },
            |data| {
                let min_val = if inputs.len() >= 2 {
                    inputs[1].as_slice_i64()[0]
                } else {
                    i64::MIN
                };
                let max_val = if inputs.len() >= 3 {
                    inputs[2].as_slice_i64()[0]
                } else {
                    i64::MAX
                };
                data.mapv(|x| x.max(min_val).min(max_val))
            },
        )
    }
}
