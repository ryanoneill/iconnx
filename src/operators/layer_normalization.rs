/// LayerNormalization operator for Iconnx
///
/// Normalizes input along last axis (most common for transformers)
/// Formula: output = (x - mean) / sqrt(variance + epsilon) * scale + bias
/// ONNX spec: https://onnx.ai/onnx/operators/onnx__LayerNormalization.html
use crate::tensor::Tensor;
use ndarray::Axis;

/// LayerNormalization operator - normalize along last axis
pub struct LayerNormalization;

impl LayerNormalization {
    /// Forward pass: layer normalization
    ///
    /// # Arguments
    /// * `inputs` - [data, scale, bias]
    ///   - data: Input tensor
    ///   - scale: Scale parameters (same size as normalized axis)
    ///   - bias: Bias parameters (same size as normalized axis)
    ///
    /// # Returns
    /// Normalized tensor
    ///
    /// Note: Always normalizes along last axis (ONNX default)
    pub fn forward(
        inputs: &[Tensor],
        _attributes: &crate::attributes::NodeAttributes,
    ) -> Tensor {
        assert_eq!(inputs.len(), 3, "LayerNormalization requires 3 inputs");

        let data = inputs[0].to_array();
        let scale = inputs[1].to_array();
        let bias = inputs[2].to_array();

        let epsilon = _attributes.get_float("epsilon").unwrap_or(1e-5);

        // Get axis from attributes (default: -1 = last axis)
        let axis_attr = _attributes.get_int("axis").unwrap_or(-1);
        let axis = if axis_attr < 0 {
            (data.ndim() as i64 + axis_attr) as usize
        } else {
            axis_attr as usize
        };

        // Compute mean along last axis
        let mean = data.mean_axis(Axis(axis)).unwrap();

        // Compute variance along last axis
        // variance = mean((x - mean)^2)
        let centered = data - &mean.clone().insert_axis(Axis(axis));
        let variance = (&centered * &centered).mean_axis(Axis(axis)).unwrap();

        // Compute std with epsilon for stability
        let std = (&variance + epsilon).mapv(|v| v.sqrt());

        // Normalize: (x - mean) / std
        let normalized = &centered / &std.insert_axis(Axis(axis));

        #[cfg(debug_assertions)]
        {}

        // Apply scale and bias
        let result = &normalized
            * &scale.broadcast(normalized.raw_dim()).unwrap_or_else(|| {
                panic!(
                    "LayerNorm: Cannot broadcast scale {:?} to normalized {:?}",
                    scale.shape(),
                    normalized.shape()
                )
            })
            + &bias.broadcast(normalized.raw_dim()).unwrap_or_else(|| {
                panic!(
                    "LayerNorm: Cannot broadcast bias {:?} to normalized {:?}",
                    bias.shape(),
                    normalized.shape()
                )
            });

        Tensor::from_array(result)
    }
}
