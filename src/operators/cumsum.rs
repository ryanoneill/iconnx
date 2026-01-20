/// CumSum operator for Iconnx
///
/// Cumulative sum along axis
/// ONNX spec: https://onnx.ai/onnx/operators/onnx__CumSum.html
use crate::tensor::Tensor;

/// CumSum operator - cumulative sum along axis
pub struct CumSum;

impl CumSum {
    /// Forward pass: cumulative sum
    ///
    /// # Arguments
    /// * `inputs` - [data, axis]
    ///   - data: input tensor
    ///   - axis: axis to compute cumulative sum along
    ///
    /// # Returns
    /// Tensor with cumulative sums
    pub fn forward(
        inputs: &[Tensor],
        attributes: &crate::attributes::NodeAttributes,
    ) -> Tensor {
        assert_eq!(
            inputs.len(),
            2,
            "CumSum requires exactly 2 inputs [data, axis]"
        );

        // Get attributes
        let exclusive = attributes.get_int("exclusive").unwrap_or(0) == 1;
        let reverse = attributes.get_int("reverse").unwrap_or(0) == 1;

        // Get axis (handle negative indexing)
        let data_ndim = inputs[0].ndim();
        let axis_val = if inputs[1].is_int64() {
            inputs[1].as_slice_i64()[0]
        } else if inputs[1].is_int32() {
            inputs[1].as_slice_i32()[0] as i64
        } else {
            inputs[1].as_slice()[0] as i64
        };

        let axis_idx = if axis_val < 0 {
            (data_ndim as i64 + axis_val) as usize
        } else {
            axis_val as usize
        };

        // Use map_preserving_dtype for Int32, Int64, Float32
        match &inputs[0] {
            Tensor::Float32(data) => {
                let shape = data.shape().to_vec();
                let axis_len = shape[axis_idx];
                let stride: usize = shape[axis_idx + 1..].iter().product::<usize>().max(1);
                let outer_size: usize = shape[..axis_idx].iter().product::<usize>().max(1);

                let mut output_data = vec![0.0f32; data.len()];
                let data_contiguous = data.as_standard_layout();
                let input_slice = data_contiguous.as_slice().unwrap();

                for outer in 0..outer_size {
                    for inner in 0..stride {
                        if reverse {
                            // Reverse: compute from end backward
                            let mut cumsum = 0.0f32;
                            for ax in (0..axis_len).rev() {
                                let idx = outer * axis_len * stride + ax * stride + inner;
                                if exclusive {
                                    output_data[idx] = cumsum;
                                    cumsum += input_slice[idx];
                                } else {
                                    cumsum += input_slice[idx];
                                    output_data[idx] = cumsum;
                                }
                            }
                        } else {
                            // Forward: standard cumsum
                            let mut cumsum = 0.0f32;
                            for ax in 0..axis_len {
                                let idx = outer * axis_len * stride + ax * stride + inner;
                                if exclusive {
                                    output_data[idx] = cumsum;
                                    cumsum += input_slice[idx];
                                } else {
                                    cumsum += input_slice[idx];
                                    output_data[idx] = cumsum;
                                }
                            }
                        }
                    }
                }

                Tensor::from_vec(output_data, shape)
            }
            Tensor::Int32(data) => {
                let shape = data.shape().to_vec();
                let axis_len = shape[axis_idx];
                let stride: usize = shape[axis_idx + 1..].iter().product::<usize>().max(1);
                let outer_size: usize = shape[..axis_idx].iter().product::<usize>().max(1);

                let mut output_data = vec![0i32; data.len()];
                let data_contiguous = data.as_standard_layout();
                let input_slice = data_contiguous.as_slice().unwrap();

                for outer in 0..outer_size {
                    for inner in 0..stride {
                        if reverse {
                            let mut cumsum = 0i32;
                            for ax in (0..axis_len).rev() {
                                let idx = outer * axis_len * stride + ax * stride + inner;
                                if exclusive {
                                    output_data[idx] = cumsum;
                                    cumsum += input_slice[idx];
                                } else {
                                    cumsum += input_slice[idx];
                                    output_data[idx] = cumsum;
                                }
                            }
                        } else {
                            let mut cumsum = 0i32;
                            for ax in 0..axis_len {
                                let idx = outer * axis_len * stride + ax * stride + inner;
                                if exclusive {
                                    output_data[idx] = cumsum;
                                    cumsum += input_slice[idx];
                                } else {
                                    cumsum += input_slice[idx];
                                    output_data[idx] = cumsum;
                                }
                            }
                        }
                    }
                }

                Tensor::from_vec_i32(output_data, shape)
            }
            Tensor::Int64(data) => {
                let shape = data.shape().to_vec();
                let axis_len = shape[axis_idx];
                let stride: usize = shape[axis_idx + 1..].iter().product::<usize>().max(1);
                let outer_size: usize = shape[..axis_idx].iter().product::<usize>().max(1);

                let mut output_data = vec![0i64; data.len()];
                let data_contiguous = data.as_standard_layout();
                let input_slice = data_contiguous.as_slice().unwrap();

                for outer in 0..outer_size {
                    for inner in 0..stride {
                        if reverse {
                            let mut cumsum = 0i64;
                            for ax in (0..axis_len).rev() {
                                let idx = outer * axis_len * stride + ax * stride + inner;
                                if exclusive {
                                    output_data[idx] = cumsum;
                                    cumsum += input_slice[idx];
                                } else {
                                    cumsum += input_slice[idx];
                                    output_data[idx] = cumsum;
                                }
                            }
                        } else {
                            let mut cumsum = 0i64;
                            for ax in 0..axis_len {
                                let idx = outer * axis_len * stride + ax * stride + inner;
                                if exclusive {
                                    output_data[idx] = cumsum;
                                    cumsum += input_slice[idx];
                                } else {
                                    cumsum += input_slice[idx];
                                    output_data[idx] = cumsum;
                                }
                            }
                        }
                    }
                }

                Tensor::from_vec_i64(output_data, shape)
            }
            _ => panic!("CumSum: Unsupported dtype {}", inputs[0].dtype()),
        }
    }
}
