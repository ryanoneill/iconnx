/// Gather operator for Iconnx
///
/// Select elements along an axis using indices
/// ONNX spec: https://onnx.ai/onnx/operators/onnx__Gather.html
use crate::tensor::Tensor;
use ndarray::{Array, IxDyn};

/// Gather operator - index selection along axis
pub struct Gather;

impl Gather {
    /// Forward pass: gather elements along axis
    ///
    /// # Arguments
    /// * `inputs` - [data, indices, axis]
    ///   - data: input tensor (Float32)
    ///   - indices: indices to select (Int64 or Float32)
    ///   - axis: axis to gather along (optional, defaults to 0)
    ///
    /// # Returns
    /// Tensor with gathered elements
    pub fn forward(
        inputs: &[Tensor],
        attributes: &crate::attributes::NodeAttributes,
    ) -> Tensor {
        assert!(
            inputs.len() >= 2 && inputs.len() <= 3,
            "Gather requires 2 or 3 inputs [data, indices] or [data, indices, axis]"
        );

        // Axis can be specified as:
        // 1. Attribute "axis" (ONNX opset 13+)
        // 2. Third input (older opset)
        // Attributes take precedence for consistency with other operators
        let axis_raw = if let Some(axis_vals) = attributes.get_ints("axis") {
            // Read from attribute
            axis_vals[0]
        } else if inputs.len() == 3 {
            // Backward compatibility: read from third input
            inputs[2].as_slice()[0] as i64
        } else {
            // Default axis = 0 per ONNX spec
            0
        };

        // Handle negative axis
        let axis = if axis_raw < 0 {
            (inputs[0].ndim() as i64 + axis_raw) as usize
        } else {
            axis_raw as usize
        };

        let axis_len = inputs[0].shape()[axis];

        // Convert indices, handling negative indexing (e.g., -1 = last element)
        let idx_vec: Vec<usize> = if inputs[1].is_int64() {
            inputs[1]
                .as_slice_i64()
                .iter()
                .map(|&x| {
                    if x < 0 {
                        (axis_len as i64 + x) as usize
                    } else {
                        x as usize
                    }
                })
                .collect()
        } else {
            inputs[1]
                .as_slice()
                .iter()
                .map(|&x| {
                    let xi = x as i64;
                    if xi < 0 {
                        (axis_len as i64 + xi) as usize
                    } else {
                        x as usize
                    }
                })
                .collect()
        };

        // Get indices shape for output construction
        let indices_shape = inputs[1].shape();

        // Data can be any dtype - dispatch to typed implementation
        match &inputs[0] {
            Tensor::Float32(data) => {
                // Build output shape per ONNX spec:
                // output.shape = data.shape[:axis] + indices.shape + data.shape[axis+1:]
                let mut output_shape = Vec::new();
                output_shape.extend_from_slice(&data.shape()[..axis]);
                output_shape.extend_from_slice(indices_shape);
                output_shape.extend_from_slice(&data.shape()[axis + 1..]);

                // Gather along axis
                let output = if axis == 0 {
                    // Gather rows
                    let mut collected = Vec::new();
                    for &idx in &idx_vec {
                        let row = data.index_axis(ndarray::Axis(0), idx);
                        collected.extend(row.iter().copied());
                    }
                    Array::from_shape_vec(IxDyn(&output_shape), collected).unwrap()
                } else {
                    // General axis gather - handle non-contiguous arrays
                    let mut collected = Vec::new();
                    let outer_size: usize = data.shape()[..axis].iter().product();
                    let inner_size: usize = data.shape()[axis + 1..].iter().product();
                    let axis_size = data.shape()[axis];

                    // Ensure contiguous array for efficient slicing
                    let data_contiguous = data.as_standard_layout();

                    for outer in 0..outer_size {
                        for &idx in &idx_vec {
                            let start = outer * axis_size * inner_size + idx * inner_size;
                            let slice = data_contiguous.as_slice().unwrap();
                            collected.extend_from_slice(&slice[start..start + inner_size]);
                        }
                    }
                    Array::from_shape_vec(IxDyn(&output_shape), collected).unwrap()
                };

                // If output is [1], squeeze to scalar []
                let final_output = if output.shape() == [1] {
                    output.to_shape(vec![]).unwrap().into_owned()
                } else {
                    output
                };

                Tensor::from_array(final_output)
            }
            Tensor::Int64(data) => {
                // Int64 path - preserve dtype (important for shape tensors!)
                let mut output_shape = Vec::new();
                output_shape.extend_from_slice(&data.shape()[..axis]);
                output_shape.extend_from_slice(indices_shape);
                output_shape.extend_from_slice(&data.shape()[axis + 1..]);

                let output = if axis == 0 {
                    let mut collected = Vec::new();
                    for &idx in &idx_vec {
                        let row = data.index_axis(ndarray::Axis(0), idx);
                        collected.extend(row.iter().copied());
                    }
                    Array::from_shape_vec(IxDyn(&output_shape), collected).unwrap()
                } else {
                    let mut collected = Vec::new();
                    let outer_size: usize = data.shape()[..axis].iter().product();
                    let inner_size: usize = data.shape()[axis + 1..].iter().product();
                    let axis_size = data.shape()[axis];

                    // Ensure contiguous array for efficient slicing
                    let data_contiguous = data.as_standard_layout();

                    for outer in 0..outer_size {
                        for &idx in &idx_vec {
                            let start = outer * axis_size * inner_size + idx * inner_size;
                            let slice = data_contiguous.as_slice().unwrap();
                            collected.extend_from_slice(&slice[start..start + inner_size]);
                        }
                    }
                    Array::from_shape_vec(IxDyn(&output_shape), collected).unwrap()
                };

                let final_output = if output.shape() == [1] {
                    output.to_shape(vec![]).unwrap().into_owned()
                } else {
                    output
                };

                Tensor::from_array_i64(final_output)
            }
            _ => {
                // For other dtypes, convert to Float32
                let data_f32 = inputs[0].to_float32();
                Self::forward(&[data_f32, inputs[1].clone()], attributes)
            }
        }
    }
}
