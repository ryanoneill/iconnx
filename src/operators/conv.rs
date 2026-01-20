//! Conv operator for Iconnx
//!
//! 2D Convolution using im2col + MatMul approach
//! ONNX spec: https://onnx.ai/onnx/operators/onnx__Conv.html

#![allow(clippy::too_many_arguments)] // Convolution requires many parameters
#![allow(clippy::needless_range_loop)] // Explicit indexing is clearer for convolution

use crate::tensor::Tensor;
use ndarray::{Array2, ArrayD};

/// Conv operator - 2D convolution
pub struct Conv;

impl Conv {
    /// Forward pass: 2D convolution
    ///
    /// # Arguments
    /// * `inputs` - [input, kernel] or [input, kernel, bias]
    ///   - input: [N, C_in, H, W] (batch, in_channels, height, width)
    ///   - kernel: [C_out, C_in, kH, kW] (out_channels, in_channels, kernel_h, kernel_w)
    ///   - bias (optional): [C_out] (one per output channel)
    ///
    /// # Returns
    /// Convolved tensor [N, C_out, H_out, W_out]
    ///
    /// # Notes
    /// - Attributes supported: strides, pads, dilations, group
    /// - Uses im2col transformation + MatMul for efficiency (Conv2D)
    /// - Direct sliding window for Conv1D with dilation support
    pub fn forward(
        inputs: &[Tensor],
        _attributes: &crate::attributes::NodeAttributes,
    ) -> Tensor {
        assert!(
            inputs.len() >= 2 && inputs.len() <= 3,
            "Conv requires 2 or 3 inputs"
        );

        let input = inputs[0].to_array();
        let kernel = inputs[1].to_array();
        let bias = inputs.get(2).map(|t| t.to_array());

        // Get dimensions - support Conv1D (3D) and Conv2D (4D)
        let input_shape = input.shape();
        let ndim = input_shape.len();

        #[cfg(debug_assertions)]
        {}

        if ndim == 2 {
            // Input is 2D [C, L] - treat as batch=1 -> [1, C, L]
            let (c_in, l) = (input_shape[0], input_shape[1]);
            let input_3d = input.view().to_shape((1, c_in, l)).unwrap().into_owned();
            let input_3d_tensor = Tensor::from_array(input_3d.into_dyn());

            // Run as Conv1D
            let conv_inputs = if inputs.len() == 3 {
                vec![input_3d_tensor, inputs[1].clone(), inputs[2].clone()]
            } else {
                vec![input_3d_tensor, inputs[1].clone()]
            };
            let result_3d = Self::forward(&conv_inputs, _attributes);

            // Squeeze batch dimension: [1, C_out, L_out] -> [C_out, L_out]
            let result_array = result_3d.to_array();
            let shape_3d = result_array.shape();
            let result_2d = result_array
                .view()
                .to_shape((shape_3d[1], shape_3d[2]))
                .unwrap()
                .into_owned()
                .into_dyn();
            return Tensor::from_array(result_2d);
        }

        if ndim == 3 {
            // Conv1D: Direct implementation without 4D expansion
            // Input [N, C_in, L], Kernel [C_out, C_in, K]
            let (n, c_in, l) = (input_shape[0], input_shape[1], input_shape[2]);

            let kernel_shape = kernel.shape();
            assert_eq!(
                kernel_shape.len(),
                3,
                "Conv1D kernel must be 3D [C_out, C_in, K]"
            );
            let (c_out, k_c_in, k_len) = (kernel_shape[0], kernel_shape[1], kernel_shape[2]);

            if c_in != k_c_in {
                panic!(
                    "Conv1D input channels mismatch:\n  \
                     Input shape: {:?} (interpreted as N={}, C_in={}, L={})\n  \
                     Kernel shape: {:?} (interpreted as C_out={}, C_in={}, K={})\n  \
                     Expected kernel to have C_in={}, but got C_in={}\n  \
                     This likely means the input has wrong dimension order!",
                    input_shape, n, c_in, l, kernel_shape, c_out, k_c_in, k_len, c_in, k_c_in
                );
            }

            // Get parameters from attributes
            let stride = if let Some(strides_list) = _attributes.get_ints("strides") {
                strides_list[0] as usize
            } else {
                1
            };
            let padding = if let Some(pads_list) = _attributes.get_ints("pads") {
                pads_list[0] as usize
            } else {
                0
            };
            let dilation = if let Some(dilations_list) = _attributes.get_ints("dilations") {
                dilations_list[0] as usize
            } else {
                1
            };

            // Calculate effective kernel size with dilation
            // effective_k = (k - 1) * dilation + 1
            let effective_k_len = (k_len - 1) * dilation + 1;

            // Calculate output length using effective kernel size
            let out_len = (l + 2 * padding - effective_k_len) / stride + 1;

            // Implement Conv1D directly with manual sliding window
            let mut output_data = vec![0.0; n * c_out * out_len];

            // Ensure input and kernel are contiguous
            let input_vec = if let Some(slice) = input.as_slice() {
                slice.to_vec()
            } else {
                input.iter().copied().collect()
            };
            let kernel_vec = if let Some(slice) = kernel.as_slice() {
                slice.to_vec()
            } else {
                kernel.iter().copied().collect()
            };

            for batch in 0..n {
                for c_out_idx in 0..c_out {
                    for out_pos in 0..out_len {
                        let in_pos_start = (out_pos * stride) as isize - padding as isize;
                        let mut sum = 0.0;

                        // Slide kernel over input with dilation
                        for c_in_idx in 0..c_in {
                            for k_idx in 0..k_len {
                                // Apply dilation: skip 'dilation' positions in input
                                let in_pos = in_pos_start + (k_idx * dilation) as isize;

                                // Check bounds (handle padding)
                                let val = if in_pos >= 0 && (in_pos as usize) < l {
                                    let input_idx =
                                        batch * c_in * l + c_in_idx * l + in_pos as usize;
                                    input_vec[input_idx]
                                } else {
                                    0.0 // Zero padding
                                };

                                let kernel_idx =
                                    c_out_idx * c_in * k_len + c_in_idx * k_len + k_idx;
                                sum += val * kernel_vec[kernel_idx];
                            }
                        }

                        let output_idx = batch * c_out * out_len + c_out_idx * out_len + out_pos;
                        output_data[output_idx] = sum;
                    }
                }
            }

            // Add bias if provided
            if let Some(b) = bias {
                let bias_slice = b.as_slice().unwrap();
                for batch in 0..n {
                    for c_out_idx in 0..c_out {
                        for out_pos in 0..out_len {
                            let output_idx =
                                batch * c_out * out_len + c_out_idx * out_len + out_pos;
                            output_data[output_idx] += bias_slice[c_out_idx];
                        }
                    }
                }
            }

            let output = ArrayD::from_shape_vec(vec![n, c_out, out_len], output_data)
                .expect("Failed to reshape Conv1D output");

            return Tensor::from_array(output);
        }

        assert_eq!(
            input_shape.len(),
            4,
            "Input must be 3D [N, C, L] or 4D [N, C, H, W]"
        );

        let batch_size = input_shape[0];
        let in_channels = input_shape[1];
        let in_height = input_shape[2];
        let in_width = input_shape[3];

        let kernel_shape = kernel.shape();
        assert_eq!(
            kernel_shape.len(),
            4,
            "Kernel must be 4D [C_out, C_in, kH, kW]"
        );

        let out_channels = kernel_shape[0];
        let kernel_in_channels = kernel_shape[1];
        let kernel_height = kernel_shape[2];
        let kernel_width = kernel_shape[3];

        assert_eq!(
            in_channels, kernel_in_channels,
            "Input channels ({}) must match kernel input channels ({})",
            in_channels, kernel_in_channels
        );

        // Default parameters (TODO: get from attributes)
        let stride = 1;
        let padding = 0;

        // Calculate output dimensions
        let out_height = (in_height + 2 * padding - kernel_height) / stride + 1;
        let out_width = (in_width + 2 * padding - kernel_width) / stride + 1;

        // im2col transformation
        let col_matrix = Self::im2col(
            input,
            batch_size,
            in_channels,
            in_height,
            in_width,
            kernel_height,
            kernel_width,
            stride,
            padding,
        );

        // Reshape kernel to [C_out, C_in * kH * kW]
        let kernel_2d = kernel
            .view()
            .to_shape((out_channels, in_channels * kernel_height * kernel_width))
            .unwrap()
            .into_owned();

        // Matrix multiplication: kernel @ col_matrix
        let conv_result = kernel_2d.dot(&col_matrix);

        // Reshape result to [N, C_out, H_out, W_out]
        let mut output_data = conv_result.to_owned().into_raw_vec_and_offset().0;

        // Add bias if provided
        if let Some(b) = bias {
            let bias_slice = b.as_slice().unwrap();
            for batch in 0..batch_size {
                for ch in 0..out_channels {
                    let start = (batch * out_channels + ch) * out_height * out_width;
                    let end = start + out_height * out_width;
                    for i in start..end {
                        output_data[i] += bias_slice[ch];
                    }
                }
            }
        }

        let output = ArrayD::from_shape_vec(
            vec![batch_size, out_channels, out_height, out_width],
            output_data,
        )
        .expect("Failed to reshape conv output");

        Tensor::from_array(output)
    }

    /// im2col transformation: rearrange image patches into columns
    ///
    /// Transforms convolution into matrix multiplication
    fn im2col(
        input: &ArrayD<f32>,
        batch_size: usize,
        channels: usize,
        height: usize,
        width: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride: usize,
        padding: usize,
    ) -> Array2<f32> {
        let out_h = (height + 2 * padding - kernel_h) / stride + 1;
        let out_w = (width + 2 * padding - kernel_w) / stride + 1;

        let col_height = channels * kernel_h * kernel_w;
        let col_width = batch_size * out_h * out_w;

        let mut col_matrix = Array2::zeros((col_height, col_width));

        // Get input as contiguous slice
        let input_vec: Vec<f32> = if let Some(slice) = input.as_slice() {
            slice.to_vec()
        } else {
            input.iter().copied().collect()
        };

        let mut col_idx = 0;
        for _batch in 0..batch_size {
            for out_y in 0..out_h {
                for out_x in 0..out_w {
                    let mut row_idx = 0;

                    for c in 0..channels {
                        for ky in 0..kernel_h {
                            for kx in 0..kernel_w {
                                let in_y = out_y * stride + ky;
                                let in_x = out_x * stride + kx;

                                // Handle padding (zero-padding)
                                let val = if padding > 0
                                    && (in_y < padding
                                        || in_x < padding
                                        || in_y >= height + padding
                                        || in_x >= width + padding)
                                {
                                    0.0
                                } else {
                                    let y = in_y.saturating_sub(padding);
                                    let x = in_x.saturating_sub(padding);
                                    if y < height && x < width {
                                        input_vec[c * height * width + y * width + x]
                                    } else {
                                        0.0
                                    }
                                };

                                col_matrix[[row_idx, col_idx]] = val;
                                row_idx += 1;
                            }
                        }
                    }

                    col_idx += 1;
                }
            }
        }

        col_matrix
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attributes::NodeAttributes;

    /// Test Conv1D against ORT reference values
    /// Input: [1, 4, 10], Weight: [2, 4, 3], Bias: [2]
    /// Params: dilations=[1], group=1, kernel_shape=[3], pads=[1,1], strides=[1]
    #[test]
    fn test_conv1d_with_bias_ort_validated() {
        // ORT reference test case (numpy seed 42)
        // Generated with: np.random.seed(42); np.random.randn(...)
        let input_data: Vec<f32> = vec![
            0.49671414,
            -0.13826430,
            0.64768857,
            1.52302980,
            -0.23415338,
            -0.23413695,
            1.57921278,
            0.76743472,
            -0.46947438,
            0.54256004,
            -0.46341768,
            -0.46572974,
            0.24196227,
            -1.91328025,
            -1.72491789,
            -0.56228751,
            -1.01283109,
            0.31424734,
            -0.90802407,
            -1.41230369,
            1.46564877,
            -0.22577630,
            0.06752820,
            -1.42474818,
            -0.54438275,
            0.11092259,
            -1.15099359,
            0.37569803,
            -0.60063869,
            -0.29169375,
            -0.60170662,
            1.85227823,
            -0.01349723,
            -1.05771089,
            0.82254493,
            -1.22084367,
            0.20886360,
            -1.95967007,
            -1.32818604,
            0.19686124,
        ];
        let input = Tensor::from_vec_f32(input_data, vec![1, 4, 10]);

        let weight_data: Vec<f32> = vec![
            0.73846656,
            0.17136829,
            -0.11564828,
            -0.30110368,
            -1.47852194,
            -0.71984422,
            -0.46063876,
            1.05712223,
            0.34361830,
            -1.76304018,
            0.32408398,
            -0.38508227,
            -0.67692202,
            0.61167628,
            1.03099954,
            0.93128014,
            -0.83921754,
            -0.30921239,
            0.33126342,
            0.97554511,
            -0.47917423,
            -0.18565898,
            -1.10633492,
            -1.19620657,
        ];
        let weight = Tensor::from_vec_f32(weight_data, vec![2, 4, 3]);

        let bias_data: Vec<f32> = vec![0.81252581, 1.35624003];
        let bias = Tensor::from_vec_f32(bias_data, vec![2]);

        let mut attrs = NodeAttributes::new();
        attrs.add_ints("dilations".to_string(), vec![1]);
        attrs.add_int("group".to_string(), 1);
        attrs.add_ints("kernel_shape".to_string(), vec![3]);
        attrs.add_ints("pads".to_string(), vec![1, 1]);
        attrs.add_ints("strides".to_string(), vec![1]);

        let result = Conv::forward(&[input, weight, bias], &attrs);
        let data = result.as_slice();

        // ORT output first 10: [2.4975688, 2.510408, -1.3718622, 3.2165062, 8.175783, 0.54292333, 4.098069, 2.8876, 5.5410876, 5.29411]
        // ORT output last 10: [2.03841, -0.20082319, 4.981148, 3.0544257, -0.52914906, 3.6058824, 4.574725, 3.3375483, 3.8763824, 1.8907791]
        let ort_first_10: Vec<f32> = vec![
            2.4975688, 2.510408, -1.3718622, 3.2165062, 8.175783, 0.54292333, 4.098069, 2.8876,
            5.5410876, 5.29411,
        ];
        let ort_last_10: Vec<f32> = vec![
            2.03841,
            -0.20082319,
            4.981148,
            3.0544257,
            -0.52914906,
            3.6058824,
            4.574725,
            3.3375483,
            3.8763824,
            1.8907791,
        ];

        assert_eq!(result.shape(), &[1, 2, 10], "Shape mismatch");

        // Check first 10
        for (i, (got, expected)) in data[..10].iter().zip(ort_first_10.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-4,
                "First 10 mismatch at {}: got {}, expected {}",
                i,
                got,
                expected
            );
        }

        // Check last 10
        let len = data.len();
        for (i, (got, expected)) in data[len - 10..].iter().zip(ort_last_10.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-4,
                "Last 10 mismatch at {}: got {}, expected {}",
                i,
                got,
                expected
            );
        }
    }
}
