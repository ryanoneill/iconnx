/// ReduceMean operator for Iconnx
///
/// Computes mean along specified axes
/// ONNX spec: https://onnx.ai/onnx/operators/onnx__ReduceMean.html
use crate::tensor::Tensor;
use ndarray::Axis;

/// ReduceMean operator - compute mean along axes
pub struct ReduceMean;

impl ReduceMean {
    /// Forward pass: reduce by computing mean
    ///
    /// # Arguments
    /// * `inputs` - [data_tensor] or [data_tensor, axes_tensor]
    ///
    /// # Returns
    /// Tensor with mean computed along specified axes
    pub fn forward(
        inputs: &[Tensor],
        attributes: &crate::attributes::NodeAttributes,
    ) -> Tensor {
        assert!(
            !inputs.is_empty() && inputs.len() <= 2,
            "ReduceMean requires 1 or 2 inputs"
        );

        // Get keepdims attribute (default is 1 in ONNX)
        let keepdims = attributes.get_int("keepdims").unwrap_or(1) == 1;

        // Get axes (can be int64 or float32)
        // Handle negative axes (e.g., -1 = last axis)
        let data_ndim = inputs[0].ndim();
        let axes: Vec<usize> = if inputs.len() == 2 {
            if inputs[1].is_int64() {
                inputs[1]
                    .as_slice_i64()
                    .iter()
                    .map(|&v| {
                        if v < 0 {
                            (data_ndim as i64 + v) as usize
                        } else {
                            v as usize
                        }
                    })
                    .collect()
            } else {
                inputs[1]
                    .as_slice()
                    .iter()
                    .map(|&v| {
                        let vi = v as i64;
                        if vi < 0 {
                            (data_ndim as i64 + vi) as usize
                        } else {
                            v as usize
                        }
                    })
                    .collect()
            }
        } else {
            vec![] // No axes = reduce all
        };

        // Use map_preserving_dtype to handle Float32 and Int64
        let result = inputs[0].map_preserving_dtype(
            |data| {
                if axes.is_empty() {
                    let mean = data.mean().unwrap();
                    ndarray::ArrayD::from_shape_vec(vec![], vec![mean]).unwrap()
                } else {
                    data.mean_axis(Axis(axes[0])).unwrap()
                }
            },
            |data| {
                if axes.is_empty() {
                    let mean = data.iter().sum::<i64>() / data.len() as i64;
                    ndarray::ArrayD::from_shape_vec(vec![], vec![mean]).unwrap()
                } else {
                    // For int64, compute mean manually
                    let axis_size = data.shape()[axes[0]];
                    data.sum_axis(Axis(axes[0])).mapv(|v| v / axis_size as i64)
                }
            },
        );

        // If keepdims=1, restore reduced dimensions as size 1
        if keepdims && !axes.is_empty() {
            result.map_preserving_dtype(
                |data| {
                    let mut new_shape = data.shape().to_vec();
                    new_shape.insert(axes[0], 1);
                    data.to_shape(new_shape).unwrap().to_owned().into_dyn()
                },
                |data| {
                    let mut new_shape = data.shape().to_vec();
                    new_shape.insert(axes[0], 1);
                    data.to_shape(new_shape).unwrap().to_owned().into_dyn()
                },
            )
        } else {
            result
        }
    }
}
