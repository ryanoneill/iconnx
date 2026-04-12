//! ConvTranspose operator for Iconnx
//!
//! Transposed convolution (deconvolution) for upsampling
//! ONNX spec: https://onnx.ai/onnx/operators/onnx__ConvTranspose.html
//!
//! Implementation: Direct approach without im2col
//! For each input position, place the kernel in the output grid

#![allow(clippy::needless_range_loop)] // Explicit indexing is clearer for convolution

use crate::tensor::Tensor;
use ndarray::ArrayD;

/// ConvTranspose operator - transposed convolution
pub struct ConvTranspose;

impl ConvTranspose {
    /// Forward pass: transposed 2D convolution
    ///
    /// # Arguments
    /// * `inputs` - [input, kernel] or [input, kernel, bias]
    ///   - input: [N, C_in, H, W] (batch, in_channels, height, width)
    ///   - kernel: [C_in, C_out, kH, kW] (in_channels, out_channels, kernel_h, kernel_w)
    ///   - bias (optional): [C_out] (one per output channel)
    ///
    /// # Returns
    /// Transposed convolved tensor [N, C_out, H_out, W_out]
    ///
    /// # Notes
    /// - Default: stride=1, padding=0, output_padding=0
    /// - Output size: (H-1)*stride - 2*padding + kernel_h + output_padding
    pub fn forward(
        inputs: &[Tensor],
        attributes: &crate::attributes::NodeAttributes,
    ) -> Tensor {
        assert!(
            inputs.len() >= 2 && inputs.len() <= 3,
            "ConvTranspose requires 2 or 3 inputs"
        );

        let input = inputs[0].to_array();
        let kernel = inputs[1].to_array();
        let bias = inputs.get(2).map(|t| t.to_array());

        // Get dimensions - support both 1D (3D input) and 2D (4D input) convolutions
        let input_shape = input.shape();
        let is_1d = input_shape.len() == 3;
        let is_2d = input_shape.len() == 4;

        assert!(
            is_1d || is_2d,
            "ConvTranspose input must be 3D [N,C,L] for 1D or 4D [N,C,H,W] for 2D, got {:?}",
            input_shape
        );

        let (batch_size, in_channels, in_height, in_width) = if is_1d {
            // 1D convolution: [N, C, L] -> treat as [N, C, L, 1]
            (input_shape[0], input_shape[1], input_shape[2], 1)
        } else {
            (
                input_shape[0],
                input_shape[1],
                input_shape[2],
                input_shape[3],
            )
        };

        let kernel_shape = kernel.shape();
        let expected_kernel_dims = if is_1d { 3 } else { 4 };
        assert_eq!(
            kernel_shape.len(),
            expected_kernel_dims,
            "Kernel must be {}D for {}D convolution, got {}D",
            expected_kernel_dims,
            if is_1d { "1" } else { "2" },
            kernel_shape.len()
        );

        let (kernel_in_channels, out_channels_per_group, kernel_height, kernel_width) = if is_1d {
            // 1D kernel: [C, M/group, kW] -> treat as [C, M/group, kW, 1]
            (kernel_shape[0], kernel_shape[1], kernel_shape[2], 1)
        } else {
            (
                kernel_shape[0],
                kernel_shape[1],
                kernel_shape[2],
                kernel_shape[3],
            )
        };

        // Get attributes
        let strides = attributes.get_ints("strides").unwrap_or_default();
        let stride_h = if strides.is_empty() {
            1
        } else {
            strides[0] as usize
        };
        let stride_w = if strides.len() < 2 {
            stride_h
        } else {
            strides[1] as usize
        };

        let pads = attributes.get_ints("pads").unwrap_or_default();
        let pad_top = if pads.is_empty() { 0 } else { pads[0] as usize };
        let pad_bottom = if pads.len() < 2 {
            pad_top
        } else {
            pads[1] as usize
        };
        let pad_left = if pads.len() < 3 { 0 } else { pads[2] as usize };
        let pad_right = if pads.len() < 4 {
            pad_left
        } else {
            pads[3] as usize
        };

        let output_paddings = attributes.get_ints("output_padding").unwrap_or_default();
        let output_padding_h = if output_paddings.is_empty() {
            0
        } else {
            output_paddings[0] as usize
        };
        let output_padding_w = if output_paddings.len() < 2 {
            output_padding_h
        } else {
            output_paddings[1] as usize
        };

        let group = attributes.get_int("group").unwrap_or(1) as usize;

        // Validate dimensions and calculate total output channels
        // ONNX spec: kernel is [C, M/group, kH, kW] where C=in_channels, M=total output channels
        assert_eq!(
            kernel_in_channels, in_channels,
            "Kernel shape[0] ({}) must equal input channels ({})",
            kernel_in_channels, in_channels
        );

        // Calculate total output channels: M = kernel.shape[1] * group
        let out_channels = out_channels_per_group * group;

        assert_eq!(
            in_channels % group,
            0,
            "Input channels ({}) must be divisible by group ({})",
            in_channels,
            group
        );
        assert_eq!(
            out_channels % group,
            0,
            "Output channels ({}) must be divisible by group ({})",
            out_channels,
            group
        );

        // Calculate output dimensions
        // Formula: (H-1)*stride_h - pad_top - pad_bottom + kernel_h + output_padding_h
        let out_height =
            (in_height - 1) * stride_h + kernel_height - pad_top - pad_bottom + output_padding_h;
        let out_width = if is_1d {
            1 // For 1D conv, width is always 1
        } else {
            (in_width - 1) * stride_w + kernel_width - pad_left - pad_right + output_padding_w
        };

        // Initialize output with zeros
        let output_size = batch_size * out_channels * out_height * out_width;
        let mut output_data = vec![0.0f32; output_size];

        let input_slice = input.as_slice().unwrap();
        let kernel_slice = kernel.as_slice().unwrap();

        // Grouped convolution parameters
        let in_channels_per_group = in_channels / group;

        // For each batch
        for batch in 0..batch_size {
            // For each group
            for g in 0..group {
                // For each input channel in this group
                for in_ch_local in 0..in_channels_per_group {
                    let in_ch = g * in_channels_per_group + in_ch_local;

                    // For each output channel in this group
                    for out_ch_local in 0..out_channels_per_group {
                        let out_ch = g * out_channels_per_group + out_ch_local;

                        // For each input position
                        for in_y in 0..in_height {
                            for in_x in 0..in_width {
                                // Get input value
                                let input_idx = batch * (in_channels * in_height * in_width)
                                    + in_ch * (in_height * in_width)
                                    + in_y * in_width
                                    + in_x;
                                let input_val = input_slice[input_idx];

                                // Place kernel in output grid
                                for ky in 0..kernel_height {
                                    for kx in 0..kernel_width {
                                        // Calculate output position with stride
                                        let out_y_with_pad = in_y * stride_h + ky;
                                        let out_x_with_pad = in_x * stride_w + kx;

                                        // Apply padding offset
                                        if out_y_with_pad < pad_top || out_x_with_pad < pad_left {
                                            continue;
                                        }
                                        let out_y = out_y_with_pad - pad_top;
                                        let out_x = out_x_with_pad - pad_left;

                                        // Skip if out of bounds
                                        if out_y >= out_height || out_x >= out_width {
                                            continue;
                                        }

                                        // Get kernel value
                                        // ONNX ConvTranspose kernel shape: [C_in, M/group, kH, kW]
                                        // For depthwise (group=C_in), each input channel has its own kernel
                                        // Use global input channel index, not local
                                        let kernel_idx = in_ch
                                            * (out_channels_per_group
                                                * kernel_height
                                                * kernel_width)
                                            + out_ch_local * (kernel_height * kernel_width)
                                            + ky * kernel_width
                                            + kx;
                                        let kernel_val = kernel_slice[kernel_idx];

                                        // Accumulate in output
                                        let output_idx = batch
                                            * (out_channels * out_height * out_width)
                                            + out_ch * (out_height * out_width)
                                            + out_y * out_width
                                            + out_x;

                                        output_data[output_idx] += input_val * kernel_val;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

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

        // Create output with appropriate dimensions
        let output_shape = if is_1d {
            // 1D convolution: output is 3D [N, C_out, L_out]
            vec![batch_size, out_channels, out_height]
        } else {
            // 2D convolution: output is 4D [N, C_out, H_out, W_out]
            vec![batch_size, out_channels, out_height, out_width]
        };

        let output = ArrayD::from_shape_vec(output_shape, output_data)
            .expect("Failed to reshape conv_transpose output");

        Tensor::from_array(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attributes::NodeAttributes;

    // =====================================================
    // ORT-validated ConvTranspose tests
    // These test cases use exact values from ONNX Runtime
    // =====================================================

    /// Test simple 1D ConvTranspose (stride=2, kernel=2, no padding)
    /// ORT reference: Input [1,2] -> Output [1, 2, 2, 4]
    #[test]
    fn test_conv_transpose_1d_simple() {
        let input = Tensor::from_vec_f32(vec![1.0, 2.0], vec![1, 1, 2]);
        let weight = Tensor::from_vec_f32(vec![1.0, 2.0], vec![1, 1, 2]);

        let mut attrs = NodeAttributes::new();
        attrs.add_ints("strides".to_string(), vec![2]);
        attrs.add_ints("pads".to_string(), vec![0, 0]);
        attrs.add_ints("kernel_shape".to_string(), vec![2]);

        let result = ConvTranspose::forward(&[input, weight], &attrs);
        let data = result.as_slice();

        // ORT produces: [1, 2, 2, 4]
        let expected = [1.0, 2.0, 2.0, 4.0];
        assert_eq!(result.shape(), &[1, 1, 4], "Shape mismatch");
        assert_eq!(data.len(), expected.len(), "Length mismatch");
        for (i, (got, exp)) in data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "Mismatch at index {}: got {}, expected {}",
                i,
                got,
                exp
            );
        }
    }

    /// Test 1D depthwise ConvTranspose (group=2, stride=2, kernel=3)
    /// This matches the F0.1/pool configuration from Kokoro model
    /// ORT reference values verified
    #[test]
    fn test_conv_transpose_1d_depthwise() {
        // Input: 2 channels, 3 values each
        let input = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 2, 3]);
        // Weight: depthwise [2, 1, 3] - each channel has its own 1x3 kernel
        let weight = Tensor::from_vec_f32(vec![1.0, 2.0, 1.0, 1.0, 2.0, 1.0], vec![2, 1, 3]);

        let mut attrs = NodeAttributes::new();
        attrs.add_int("group".to_string(), 2);
        attrs.add_ints("strides".to_string(), vec![2]);
        attrs.add_ints("pads".to_string(), vec![1, 1]);
        attrs.add_ints("output_padding".to_string(), vec![1]);
        attrs.add_ints("kernel_shape".to_string(), vec![3]);

        let result = ConvTranspose::forward(&[input, weight], &attrs);
        let data = result.as_slice();

        // ORT produces: [[2, 3, 4, 5, 6, 3], [8, 9, 10, 11, 12, 6]]
        let expected = vec![
            2.0, 3.0, 4.0, 5.0, 6.0, 3.0, // channel 0
            8.0, 9.0, 10.0, 11.0, 12.0, 6.0, // channel 1
        ];
        assert_eq!(result.shape(), &[1, 2, 6], "Shape mismatch");
        assert_eq!(data.len(), expected.len(), "Length mismatch");
        for (i, (got, exp)) in data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "Mismatch at index {}: got {}, expected {}",
                i,
                got,
                exp
            );
        }
    }

    /// Test 2D ConvTranspose with bias
    /// ORT reference values verified
    #[test]
    fn test_conv_transpose_2d_with_bias() {
        // Input: 1 channel, 2x2
        let input = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
        // Weight: identity-like 2x2 kernel
        let weight = Tensor::from_vec_f32(vec![1.0, 0.0, 0.0, 1.0], vec![1, 1, 2, 2]);
        // Bias: 0.5
        let bias = Tensor::from_vec_f32(vec![0.5], vec![1]);

        let mut attrs = NodeAttributes::new();
        attrs.add_ints("strides".to_string(), vec![1, 1]);
        attrs.add_ints("pads".to_string(), vec![0, 0, 0, 0]);
        attrs.add_ints("kernel_shape".to_string(), vec![2, 2]);

        let result = ConvTranspose::forward(&[input, weight, bias], &attrs);
        let data = result.as_slice();

        // ORT produces: [1.5, 2.5, 0.5, 3.5, 5.5, 2.5, 0.5, 3.5, 4.5]
        let expected = vec![1.5, 2.5, 0.5, 3.5, 5.5, 2.5, 0.5, 3.5, 4.5];
        assert_eq!(result.shape(), &[1, 1, 3, 3], "Shape mismatch");
        assert_eq!(data.len(), expected.len(), "Length mismatch");
        for (i, (got, exp)) in data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "Mismatch at index {}: got {}, expected {}",
                i,
                got,
                exp
            );
        }
    }
}
