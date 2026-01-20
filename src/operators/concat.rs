/// Concat operator for Iconnx
///
/// Concatenates tensors along specified axis
/// ONNX spec: https://onnx.ai/onnx/operators/onnx__Concat.html
use crate::tensor::Tensor;
use ndarray::{concatenate, Axis};

/// Concat operator - concatenate tensors along axis
pub struct Concat;

impl Concat {
    /// Forward pass: concatenate tensors
    ///
    /// # Arguments
    /// * `inputs` - [tensor1, tensor2, ...]
    ///   - All inputs are data tensors to concatenate
    /// * `attributes` - ONNX attributes:
    ///   - axis: Axis along which to concatenate (required)
    ///
    /// # Returns
    /// Concatenated tensor
    ///
    /// # Panics
    /// Panics if < 1 tensor or axis not specified
    pub fn forward(
        inputs: &[Tensor],
        attributes: &crate::attributes::NodeAttributes,
    ) -> Tensor {
        assert!(
            !inputs.is_empty(),
            "Concat requires at least 1 input tensor"
        );

        // Get axis from attributes (required in ONNX)
        let axis_raw = attributes
            .get_int("axis")
            .expect("Concat requires 'axis' attribute");

        // Handle negative axis (e.g., -1 = last axis)
        let ndim = inputs[0].ndim();
        let axis_value = if axis_raw < 0 {
            (ndim as i64 + axis_raw) as usize
        } else {
            axis_raw as usize
        };

        #[cfg(debug_assertions)]
        {
            if inputs.len() == 4 && inputs[0].len() == 1 {
                eprintln!("Concat (4 scalars): axis={}", axis_value);
                for (i, t) in inputs.iter().enumerate() {
                    if t.is_int64() && t.len() <= 4 {
                        eprintln!("  input[{}]: {:?} (int64)", i, t.as_slice_i64());
                    } else if t.is_float32() && t.len() <= 4 {
                        eprintln!("  input[{}]: {:?} (float32)", i, t.as_slice());
                    } else {
                        eprintln!("  input[{}]: shape={:?}, dtype={}", i, t.shape(), t.dtype());
                    }
                }
            }
        }

        // Special case: if only 1 input, just return it (identity)
        if inputs.len() == 1 {
            return inputs[0].clone();
        }

        // Use multi-dtype dispatch
        Tensor::dispatch_multi(
            inputs,
            |arrays_f32| {
                let views: Vec<_> = arrays_f32.iter().map(|a| a.view()).collect();
                concatenate(Axis(axis_value), &views)
                    .expect("Failed to concatenate Float32 tensors - check shapes match on all axes except concat axis")
            },
            |arrays_i64| {
                let views: Vec<_> = arrays_i64.iter().map(|a| a.view()).collect();
                concatenate(Axis(axis_value), &views)
                    .expect("Failed to concatenate Int64 tensors - check shapes match on all axes except concat axis")
            },
        )
    }
}
