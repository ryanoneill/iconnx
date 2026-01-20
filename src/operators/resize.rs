/// Resize operator for Iconnx
///
/// Tensor interpolation/upsampling
/// ONNX spec: https://onnx.ai/onnx/operators/onnx__Resize.html
///
/// Supports:
/// - mode: "nearest" (default), "linear"
/// - coordinate_transformation_mode: "half_pixel", "asymmetric", etc.
use crate::tensor::Tensor;
use ndarray::Array;

/// Resize operator - tensor interpolation
pub struct Resize;

impl Resize {
    /// Forward pass: resize tensor
    ///
    /// # Arguments
    /// * `inputs` - [X, roi, scales] or [X, roi, scales, sizes]
    ///   - X: input tensor
    ///   - roi: region of interest (optional)
    ///   - scales: scale factors for each dimension
    ///   - sizes: (optional) explicit output sizes
    ///
    /// # Attributes
    /// * `mode` - "nearest" (default) or "linear"
    /// * `coordinate_transformation_mode` - "half_pixel" (default), "asymmetric", etc.
    /// * `nearest_mode` - "round_prefer_floor" (default), "floor", "ceil", "round_prefer_ceil"
    ///
    /// # Returns
    /// Resized tensor
    pub fn forward(
        inputs: &[Tensor],
        attributes: &crate::attributes::NodeAttributes,
    ) -> Tensor {
        assert!(!inputs.is_empty(), "Resize requires at least 1 input (X)");

        let x = inputs[0].to_array();

        // Get mode attribute (default to "nearest" for backward compatibility)
        let mode = attributes.get_string("mode").unwrap_or("nearest");

        // Get coordinate transformation mode (default to "half_pixel")
        let coord_mode = attributes
            .get_string("coordinate_transformation_mode")
            .unwrap_or("half_pixel");

        // Get nearest mode (how to round to nearest integer for nearest-neighbor)
        // ONNX default is "round_prefer_floor", but many models use "floor"
        let nearest_mode = attributes
            .get_string("nearest_mode")
            .unwrap_or("round_prefer_floor");

        // Handle optional inputs: roi can be omitted (empty string filtered out)
        let scales_idx = if inputs.len() == 2 { 1 } else { 2 };
        let scales = inputs[scales_idx].as_slice();

        // Calculate output shape from scales
        let input_shape = x.shape();
        let output_shape: Vec<usize> = input_shape
            .iter()
            .zip(scales.iter())
            .map(|(&dim, &scale)| (dim as f32 * scale).round() as usize)
            .collect();

        match mode {
            "linear" => Self::linear_interpolation(x, &output_shape, &scales, coord_mode),
            _ => Self::nearest_neighbor(x, &output_shape, &scales, coord_mode, nearest_mode),
        }
    }

    /// Nearest-neighbor interpolation
    ///
    /// # Arguments
    /// * `x` - input tensor
    /// * `output_shape` - target output shape
    /// * `scales` - scale factors per dimension
    /// * `coord_mode` - coordinate transformation mode (asymmetric, half_pixel, etc.)
    /// * `nearest_mode` - how to round to integer (floor, ceil, round_prefer_floor, round_prefer_ceil)
    fn nearest_neighbor(
        x: &ndarray::ArrayD<f32>,
        output_shape: &[usize],
        scales: &[f32],
        coord_mode: &str,
        nearest_mode: &str,
    ) -> Tensor {
        let input_shape = x.shape();

        // Ensure input is contiguous for slicing
        let x_contiguous = x.as_standard_layout();
        let input_data = x_contiguous.as_slice().expect("Failed to get input slice");

        // Calculate strides for input tensor
        let mut input_strides = vec![1usize; input_shape.len()];
        for i in (0..input_shape.len() - 1).rev() {
            input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
        }

        // Calculate output strides
        let mut output_strides = vec![1usize; output_shape.len()];
        for i in (0..output_shape.len() - 1).rev() {
            output_strides[i] = output_strides[i + 1] * output_shape[i + 1];
        }

        let total_elements: usize = output_shape.iter().product();
        let mut output_data = Vec::with_capacity(total_elements);

        for output_idx in 0..total_elements {
            let mut remaining = output_idx;
            let mut output_coords = vec![0usize; output_shape.len()];

            for i in 0..output_shape.len() {
                output_coords[i] = remaining / output_strides[i];
                remaining %= output_strides[i];
            }

            let input_coords: Vec<usize> = output_coords
                .iter()
                .enumerate()
                .map(|(dim_idx, &out_coord)| {
                    // Apply coordinate transformation mode
                    let input_coord_f32 = Self::transform_coordinate(
                        out_coord,
                        scales[dim_idx],
                        input_shape[dim_idx],
                        output_shape[dim_idx],
                        coord_mode,
                    );

                    // Apply nearest mode to convert float to integer index
                    let input_coord = Self::apply_nearest_mode(input_coord_f32, nearest_mode);

                    // Clamp to valid range
                    input_coord.min(input_shape[dim_idx] - 1)
                })
                .collect();

            let input_idx: usize = input_coords
                .iter()
                .zip(input_strides.iter())
                .map(|(&coord, &stride)| coord * stride)
                .sum();

            output_data.push(input_data[input_idx]);
        }

        let output_array = Array::from_shape_vec(output_shape.to_vec(), output_data)
            .expect("Failed to create output array");

        Tensor::from_array(output_array.into_dyn())
    }

    /// Transform output coordinate to input coordinate based on transformation mode
    fn transform_coordinate(
        out_coord: usize,
        scale: f32,
        input_len: usize,
        output_len: usize,
        mode: &str,
    ) -> f32 {
        let out_coord_f32 = out_coord as f32;

        match mode {
            "half_pixel" => {
                // x_original = (x_resized + 0.5) / scale - 0.5
                (out_coord_f32 + 0.5) / scale - 0.5
            }
            "asymmetric" => {
                // x_original = x_resized / scale
                out_coord_f32 / scale
            }
            "align_corners" => {
                // x_original = x_resized * (length_original - 1) / (length_resized - 1)
                if output_len == 1 {
                    0.0
                } else {
                    out_coord_f32 * (input_len - 1) as f32 / (output_len - 1) as f32
                }
            }
            "pytorch_half_pixel" => {
                if output_len > 1 {
                    (out_coord_f32 + 0.5) / scale - 0.5
                } else {
                    0.0
                }
            }
            _ => {
                // Default to asymmetric for unknown modes
                out_coord_f32 / scale
            }
        }
    }

    /// Apply nearest mode to convert float coordinate to integer index
    fn apply_nearest_mode(coord: f32, mode: &str) -> usize {
        let result = match mode {
            "floor" => coord.floor(),
            "ceil" => coord.ceil(),
            "round_prefer_ceil" => {
                // Round to nearest, ties go to ceiling
                if coord - coord.floor() == 0.5 {
                    coord.ceil()
                } else {
                    coord.round()
                }
            }
            _ => {
                // "round_prefer_floor" and default: Round to nearest, ties go to floor (ONNX default)
                if coord - coord.floor() == 0.5 {
                    coord.floor()
                } else {
                    coord.round()
                }
            }
        };
        result.max(0.0) as usize
    }

    /// Linear (multilinear) interpolation
    fn linear_interpolation(
        x: &ndarray::ArrayD<f32>,
        output_shape: &[usize],
        scales: &[f32],
        coord_mode: &str,
    ) -> Tensor {
        let input_shape = x.shape();
        let ndim = input_shape.len();

        // Ensure input is contiguous for slicing
        let x_contiguous = x.as_standard_layout();
        let input_data = x_contiguous.as_slice().expect("Failed to get input slice");

        // Calculate strides for input tensor
        let mut input_strides = vec![1usize; ndim];
        for i in (0..ndim - 1).rev() {
            input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
        }

        // Calculate output strides
        let mut output_strides = vec![1usize; ndim];
        for i in (0..ndim - 1).rev() {
            output_strides[i] = output_strides[i + 1] * output_shape[i + 1];
        }

        let total_elements: usize = output_shape.iter().product();
        let mut output_data = Vec::with_capacity(total_elements);

        for output_idx in 0..total_elements {
            // Convert flat index to multi-dimensional coordinates
            let mut remaining = output_idx;
            let mut output_coords = vec![0usize; ndim];
            for i in 0..ndim {
                output_coords[i] = remaining / output_strides[i];
                remaining %= output_strides[i];
            }

            // Compute continuous input coordinates and interpolation weights
            let mut input_coords_f32 = vec![0.0f32; ndim];
            for dim in 0..ndim {
                let coord = Self::transform_coordinate(
                    output_coords[dim],
                    scales[dim],
                    input_shape[dim],
                    output_shape[dim],
                    coord_mode,
                );
                // Clamp to valid range for interpolation
                input_coords_f32[dim] = coord.max(0.0).min((input_shape[dim] - 1) as f32);
            }

            // N-linear interpolation: interpolate across all dimensions
            // For efficiency, we do this iteratively using 2^N corner values
            let value =
                Self::nlinear_sample(input_data, &input_coords_f32, input_shape, &input_strides);

            output_data.push(value);
        }

        let output_array = Array::from_shape_vec(output_shape.to_vec(), output_data)
            .expect("Failed to create output array");

        Tensor::from_array(output_array.into_dyn())
    }

    /// N-linear interpolation: sample at continuous coordinates
    fn nlinear_sample(
        input_data: &[f32],
        coords: &[f32],
        input_shape: &[usize],
        input_strides: &[usize],
    ) -> f32 {
        let ndim = coords.len();

        // For each dimension, compute floor/ceil indices and weight
        let mut floors = vec![0usize; ndim];
        let mut ceils = vec![0usize; ndim];
        let mut weights = vec![0.0f32; ndim];

        for dim in 0..ndim {
            let coord = coords[dim];
            floors[dim] = (coord.floor() as usize).min(input_shape[dim] - 1);
            ceils[dim] = (coord.ceil() as usize).min(input_shape[dim] - 1);
            // Weight for ceiling value (1 - weight for floor)
            weights[dim] = coord - floors[dim] as f32;
        }

        // Iterate over all 2^N corners and compute weighted sum
        let num_corners = 1 << ndim;
        let mut result = 0.0f32;

        for corner_idx in 0..num_corners {
            // Determine which index (floor or ceil) to use for each dimension
            let mut corner_coords = vec![0usize; ndim];
            let mut corner_weight = 1.0f32;

            for dim in 0..ndim {
                let use_ceil = (corner_idx >> dim) & 1 == 1;
                if use_ceil {
                    corner_coords[dim] = ceils[dim];
                    corner_weight *= weights[dim];
                } else {
                    corner_coords[dim] = floors[dim];
                    corner_weight *= 1.0 - weights[dim];
                }
            }

            // Compute flat index for this corner
            let flat_idx: usize = corner_coords
                .iter()
                .zip(input_strides.iter())
                .map(|(&coord, &stride)| coord * stride)
                .sum();

            result += input_data[flat_idx] * corner_weight;
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attributes::NodeAttributes;

    // =====================================================
    // ORT-validated Resize tests
    // These test cases use exact values from ONNX Runtime
    // =====================================================

    /// Test nearest neighbor with asymmetric + floor (Kokoro model configuration)
    /// ORT reference: Input [0,1,2,3] -> Output [0,0,1,1,2,2,3,3]
    #[test]
    fn test_resize_nearest_asymmetric_floor_2x() {
        let input = Tensor::from_vec_f32(vec![0.0, 1.0, 2.0, 3.0], vec![1, 1, 4]);
        let scales = Tensor::from_vec_f32(vec![1.0, 1.0, 2.0], vec![3]);
        let roi = Tensor::from_vec_f32(vec![], vec![0]);

        let mut attrs = NodeAttributes::new();
        attrs.add_string("mode".to_string(), "nearest".to_string());
        attrs.add_string(
            "coordinate_transformation_mode".to_string(),
            "asymmetric".to_string(),
        );
        attrs.add_string("nearest_mode".to_string(), "floor".to_string());

        let result = Resize::forward(&[input, roi, scales], &attrs);
        let data = result.as_slice();

        // ORT produces: [0, 0, 1, 1, 2, 2, 3, 3]
        let expected = vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
        assert_eq!(data.len(), expected.len(), "Output length mismatch");
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

    /// Test nearest neighbor with half_pixel + floor
    /// ORT reference: Input [0,1,2,3] -> Output [0,0,0,1,1,2,2,3]
    /// Note: Different from asymmetric!
    #[test]
    fn test_resize_nearest_half_pixel_floor_2x() {
        let input = Tensor::from_vec_f32(vec![0.0, 1.0, 2.0, 3.0], vec![1, 1, 4]);
        let scales = Tensor::from_vec_f32(vec![1.0, 1.0, 2.0], vec![3]);
        let roi = Tensor::from_vec_f32(vec![], vec![0]);

        let mut attrs = NodeAttributes::new();
        attrs.add_string("mode".to_string(), "nearest".to_string());
        attrs.add_string(
            "coordinate_transformation_mode".to_string(),
            "half_pixel".to_string(),
        );
        attrs.add_string("nearest_mode".to_string(), "floor".to_string());

        let result = Resize::forward(&[input, roi, scales], &attrs);
        let data = result.as_slice();

        // ORT produces: [0, 0, 0, 1, 1, 2, 2, 3]
        // This is different from asymmetric due to half_pixel transformation
        let expected = vec![0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0];
        assert_eq!(data.len(), expected.len(), "Output length mismatch");
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

    /// Test 3D tensor resize (multi-channel)
    /// ORT reference: Input [[1,2,3],[4,5,6]] -> Output [[1,1,2,2,3,3],[4,4,5,5,6,6]]
    #[test]
    fn test_resize_nearest_multichannel() {
        let input = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 2, 3]);
        let scales = Tensor::from_vec_f32(vec![1.0, 1.0, 2.0], vec![3]);
        let roi = Tensor::from_vec_f32(vec![], vec![0]);

        let mut attrs = NodeAttributes::new();
        attrs.add_string("mode".to_string(), "nearest".to_string());
        attrs.add_string(
            "coordinate_transformation_mode".to_string(),
            "asymmetric".to_string(),
        );
        attrs.add_string("nearest_mode".to_string(), "floor".to_string());

        let result = Resize::forward(&[input, roi, scales], &attrs);
        let data = result.as_slice();

        // ORT produces: [1,1,2,2,3,3, 4,4,5,5,6,6]
        let expected = vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0];
        assert_eq!(data.len(), expected.len(), "Output length mismatch");
        assert_eq!(result.shape(), &[1, 2, 6], "Shape should be [1,2,6]");
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

    /// Test exact F0.1 upsample configuration from Kokoro model
    /// ORT reference values verified
    #[test]
    fn test_resize_f0_1_upsample_config() {
        let input = Tensor::from_vec_f32(vec![0.651, 1.421, 1.348, 0.772, 0.504], vec![1, 1, 5]);
        let scales = Tensor::from_vec_f32(vec![1.0, 1.0, 2.0], vec![3]);
        let roi = Tensor::from_vec_f32(vec![], vec![0]);

        let mut attrs = NodeAttributes::new();
        attrs.add_string("mode".to_string(), "nearest".to_string());
        attrs.add_string(
            "coordinate_transformation_mode".to_string(),
            "asymmetric".to_string(),
        );
        attrs.add_string("nearest_mode".to_string(), "floor".to_string());

        let result = Resize::forward(&[input, roi, scales], &attrs);
        let data = result.as_slice();

        // ORT produces each value duplicated
        let expected = vec![
            0.651, 0.651, 1.421, 1.421, 1.348, 1.348, 0.772, 0.772, 0.504, 0.504,
        ];
        assert_eq!(data.len(), expected.len(), "Output length mismatch");
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

    /// Test nearest neighbor with half_pixel + round_prefer_floor (ONNX default)
    #[test]
    fn test_resize_nearest_half_pixel_round_prefer_floor() {
        let input = Tensor::from_vec_f32(vec![0.0, 1.0, 2.0, 3.0], vec![1, 1, 4]);
        let scales = Tensor::from_vec_f32(vec![1.0, 1.0, 2.0], vec![3]);
        let roi = Tensor::from_vec_f32(vec![], vec![0]);

        let mut attrs = NodeAttributes::new();
        attrs.add_string("mode".to_string(), "nearest".to_string());
        attrs.add_string(
            "coordinate_transformation_mode".to_string(),
            "half_pixel".to_string(),
        );
        attrs.add_string("nearest_mode".to_string(), "round_prefer_floor".to_string());

        let result = Resize::forward(&[input, roi, scales], &attrs);
        let data = result.as_slice();

        // ORT produces: [0, 0, 1, 1, 2, 2, 3, 3]
        let expected = vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
        assert_eq!(data.len(), expected.len(), "Output length mismatch");
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

    // =====================================================
    // Original tests (kept for coverage)
    // =====================================================

    /// Regression test: Linear interpolation should produce smooth outputs
    /// when upsampling step functions, not piecewise-constant like nearest-neighbor.
    #[test]
    fn test_resize_linear_interpolation() {
        // Create a step function: [0, 0, 1, 1]
        let input_data = vec![0.0f32, 0.0, 1.0, 1.0];
        let input = Tensor::from_vec_f32(input_data, vec![1, 1, 4]);

        // Scale 2x: expect smooth transition, not step
        let scales = Tensor::from_vec_f32(vec![1.0, 1.0, 2.0], vec![3]);
        let roi = Tensor::from_vec_f32(vec![], vec![0]); // Empty ROI

        // Set mode=linear
        let mut attrs = NodeAttributes::new();
        attrs.add_string("mode".to_string(), "linear".to_string());
        attrs.add_string(
            "coordinate_transformation_mode".to_string(),
            "half_pixel".to_string(),
        );

        let result = Resize::forward(&[input, roi, scales], &attrs);
        let data = result.as_slice();

        // Should produce 8 values with smooth transition
        assert_eq!(data.len(), 8, "2x upscale of 4 elements should give 8");

        // Check that we have intermediate values (not just 0 and 1)
        let has_intermediate = data.iter().any(|&v| v > 0.1 && v < 0.9);
        assert!(
            has_intermediate,
            "Linear interpolation should produce smooth transitions, not steps"
        );

        // Values should be monotonically non-decreasing (input is non-decreasing)
        for i in 1..data.len() {
            assert!(
                data[i] >= data[i - 1] - 0.01,
                "Output should be roughly non-decreasing for non-decreasing input"
            );
        }
    }

    #[test]
    fn test_resize_nearest_neighbor() {
        // Create input: [0, 1]
        let input_data = vec![0.0f32, 1.0];
        let input = Tensor::from_vec_f32(input_data, vec![1, 1, 2]);

        // Scale 2x
        let scales = Tensor::from_vec_f32(vec![1.0, 1.0, 2.0], vec![3]);
        let roi = Tensor::from_vec_f32(vec![], vec![0]);

        // Default mode is nearest
        let attrs = NodeAttributes::new();

        let result = Resize::forward(&[input, roi, scales], &attrs);
        let data = result.as_slice();

        assert_eq!(data.len(), 4, "2x upscale of 2 elements should give 4");

        // Nearest neighbor should only produce values 0 or 1
        for v in &data {
            assert!(
                *v == 0.0 || *v == 1.0,
                "Nearest neighbor should only produce exact input values"
            );
        }
    }
}
