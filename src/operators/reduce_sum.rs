/// ReduceSum operator for Iconnx
///
/// Computes sum along specified axes
/// ONNX spec: https://onnx.ai/onnx/operators/onnx__ReduceSum.html
use crate::tensor::Tensor;
use ndarray::Axis;

/// ReduceSum operator - compute sum along axes
pub struct ReduceSum;

impl ReduceSum {
    /// Forward pass: reduce by computing sum
    ///
    /// # Arguments
    /// * `inputs` - [data_tensor] or [data_tensor, axes_tensor]
    ///
    /// # Returns
    /// Tensor with sum computed along specified axes
    pub fn forward(
        inputs: &[Tensor],
        attributes: &crate::attributes::NodeAttributes,
    ) -> Tensor {
        assert!(
            !inputs.is_empty() && inputs.len() <= 2,
            "ReduceSum requires 1 or 2 inputs"
        );

        // Get keepdims attribute (ONNX default is 1)
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
                    // Reduce all - return scalar
                    let sum = data.sum();
                    ndarray::ArrayD::from_shape_vec(vec![], vec![sum]).unwrap()
                } else {
                    // Reduce along specified axis
                    let axis = axes[0];
                    assert!(
                        axis < data.ndim(),
                        "ReduceSum: axis {} out of bounds for tensor with {} dimensions",
                        axis,
                        data.ndim()
                    );
                    data.sum_axis(Axis(axis))
                }
            },
            |data| {
                if axes.is_empty() {
                    let sum = data.sum();
                    ndarray::ArrayD::from_shape_vec(vec![], vec![sum]).unwrap()
                } else {
                    let axis = axes[0];
                    assert!(
                        axis < data.ndim(),
                        "ReduceSum: axis {} out of bounds for tensor with {} dimensions",
                        axis,
                        data.ndim()
                    );
                    data.sum_axis(Axis(axis))
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
