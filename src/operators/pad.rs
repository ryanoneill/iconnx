/// Pad operator for Iconnx
///
/// Implements tensor padding with all ONNX modes
/// ONNX spec: https://onnx.ai/onnx/operators/onnx__Pad.html
use crate::tensor::Tensor;
use ndarray::{ArrayD, IxDyn};

/// Pad operator - adds padding to tensor edges
pub struct Pad;

/// Padding mode per ONNX specification
#[derive(Debug, Clone, Copy, PartialEq)]
enum PadMode {
    /// Pad with a constant value
    Constant,
    /// Pad with reflection of values (not including edge)
    Reflect,
    /// Pad with edge values
    Edge,
    /// Wrap-around padding (as if tensor forms a torus)
    Wrap,
}

impl Pad {
    /// Forward pass: pad tensor
    ///
    /// # Arguments
    /// * `inputs` - 2-4 inputs: [data, pads, constant_value (optional), axes (optional)]
    ///   - data: Input tensor to pad
    ///   - pads: 1D tensor with padding amounts [axis0_begin, axis1_begin, ..., axis0_end, axis1_end, ...]
    ///   - constant_value: Scalar tensor with padding value (only for constant mode)
    ///   - axes: Optional 1D tensor specifying which axes to pad
    ///
    /// # Attributes
    /// * `mode`: Padding mode - "constant" (default), "reflect", "edge", or "wrap"
    ///
    /// # Returns
    /// Padded tensor
    pub fn forward(
        inputs: &[Tensor],
        attributes: &crate::attributes::NodeAttributes,
    ) -> Tensor {
        assert!(
            inputs.len() >= 2 && inputs.len() <= 4,
            "Pad requires 2-4 inputs, got {}",
            inputs.len()
        );

        // Get mode attribute (default: "constant")
        let mode = match attributes.get_string("mode") {
            Some("constant") | None => PadMode::Constant,
            Some("reflect") => PadMode::Reflect,
            Some("edge") => PadMode::Edge,
            Some("wrap") => PadMode::Wrap,
            Some(m) => panic!("Unknown padding mode: {}", m),
        };

        let pads_tensor = &inputs[1];

        // Get constant_value (only used for constant mode)
        let constant_value_f32: f32 = if inputs.len() >= 3 && mode == PadMode::Constant {
            if inputs[2].is_float32() {
                inputs[2].as_slice()[0]
            } else if inputs[2].is_int64() {
                inputs[2].as_slice_i64()[0] as f32
            } else {
                inputs[2].as_slice_i32()[0] as f32
            }
        } else {
            0.0
        };

        let constant_value_i64 = constant_value_f32 as i64;

        // Extract padding amounts (ONNX spec: tensor(int64))
        let ndim = inputs[0].ndim();

        let (pads_begin, pads_end) = if pads_tensor.is_int64() {
            let pads_slice = pads_tensor.as_slice_i64();
            assert_eq!(
                pads_slice.len(),
                ndim * 2,
                "Pads must have {} elements (2 per axis), got {}",
                ndim * 2,
                pads_slice.len()
            );
            let pads_begin: Vec<usize> = pads_slice[0..ndim].iter().map(|&x| x as usize).collect();
            let pads_end: Vec<usize> = pads_slice[ndim..].iter().map(|&x| x as usize).collect();
            (pads_begin, pads_end)
        } else {
            // Fallback for float32 pads (shouldn't happen per spec, but handle it)
            let pads_slice = pads_tensor.as_slice();
            assert_eq!(
                pads_slice.len(),
                ndim * 2,
                "Pads must have {} elements (2 per axis), got {}",
                ndim * 2,
                pads_slice.len()
            );
            let pads_begin: Vec<usize> = pads_slice[0..ndim].iter().map(|&x| x as usize).collect();
            let pads_end: Vec<usize> = pads_slice[ndim..].iter().map(|&x| x as usize).collect();
            (pads_begin, pads_end)
        };

        // Use map_preserving_dtype to handle Float32, Int32, Int64
        inputs[0].map_preserving_dtype(
            |data| Self::pad_array(data, &pads_begin, &pads_end, mode, constant_value_f32),
            |data| Self::pad_array(data, &pads_begin, &pads_end, mode, constant_value_i64),
        )
    }

    /// Generic padding implementation for any numeric type
    fn pad_array<T: Clone + Copy + Default>(
        data: &ArrayD<T>,
        pads_begin: &[usize],
        pads_end: &[usize],
        mode: PadMode,
        constant_value: T,
    ) -> ArrayD<T> {
        let input_shape = data.shape();
        let ndim = input_shape.len();

        // Calculate output shape
        let output_shape: Vec<usize> = (0..ndim)
            .map(|i| input_shape[i] + pads_begin[i] + pads_end[i])
            .collect();

        // Create output array
        let total_elements: usize = output_shape.iter().product();
        let mut output_data = vec![constant_value; total_elements];

        // Ensure data is contiguous
        let data_contiguous = data.as_standard_layout();
        let data_slice = data_contiguous.as_slice().unwrap();

        // Compute strides for index conversion
        let input_strides = Self::compute_strides(input_shape);
        let output_strides = Self::compute_strides(&output_shape);

        // Fill output based on mode
        for (flat_out_idx, output_val) in output_data.iter_mut().enumerate() {
            // Convert flat output index to multi-dimensional index
            let mut output_indices = vec![0usize; ndim];
            let mut remaining = flat_out_idx;
            for (i, &stride) in output_strides.iter().enumerate() {
                output_indices[i] = remaining / stride;
                remaining %= stride;
            }

            // Map output index to input index based on mode
            let input_indices: Option<Vec<usize>> =
                Self::map_output_to_input(&output_indices, input_shape, pads_begin, pads_end, mode);

            if let Some(indices) = input_indices {
                // Convert input indices to flat index
                let flat_input_idx: usize = indices
                    .iter()
                    .zip(input_strides.iter())
                    .map(|(&idx, &stride)| idx * stride)
                    .sum();

                *output_val = data_slice[flat_input_idx];
            }
            // For constant mode, the value is already set to constant_value
        }

        ArrayD::from_shape_vec(IxDyn(&output_shape), output_data)
            .expect("Failed to create padded tensor")
    }

    /// Map an output index to the corresponding input index based on padding mode
    ///
    /// Returns None if the position should use constant_value (only for Constant mode)
    fn map_output_to_input(
        output_indices: &[usize],
        input_shape: &[usize],
        pads_begin: &[usize],
        _pads_end: &[usize],
        mode: PadMode,
    ) -> Option<Vec<usize>> {
        let ndim = input_shape.len();
        let mut input_indices = Vec::with_capacity(ndim);

        for i in 0..ndim {
            let out_idx = output_indices[i];
            let pad_begin = pads_begin[i];
            let in_size = input_shape[i];

            let in_idx = match mode {
                PadMode::Constant => {
                    // For constant mode, return None if outside input range
                    if out_idx < pad_begin || out_idx >= pad_begin + in_size {
                        return None;
                    }
                    out_idx - pad_begin
                }
                PadMode::Edge => {
                    // Clamp to edge values
                    if out_idx < pad_begin {
                        0
                    } else if out_idx >= pad_begin + in_size {
                        in_size - 1
                    } else {
                        out_idx - pad_begin
                    }
                }
                PadMode::Reflect => {
                    // Reflect without including the edge value
                    let relative = out_idx as isize - pad_begin as isize;
                    Self::reflect_index(relative, in_size)
                }
                PadMode::Wrap => {
                    // Wrap around (modulo)
                    let relative = out_idx as isize - pad_begin as isize;
                    Self::wrap_index(relative, in_size)
                }
            };

            input_indices.push(in_idx);
        }

        Some(input_indices)
    }

    /// Reflect an index into valid range
    ///
    /// For reflect mode, the padding is the reflection of the vector mirrored
    /// on the first and last values of the vector along each axis.
    fn reflect_index(relative: isize, size: usize) -> usize {
        if size == 1 {
            return 0;
        }

        let size_i = size as isize;
        let mut idx = relative;

        // Handle negative indices by reflecting
        while idx < 0 {
            idx = -idx;
        }

        // Handle indices >= size by reflecting back
        // The period is 2*(size-1) for reflection
        let period = 2 * (size_i - 1);
        if period > 0 {
            idx %= period;
            if idx >= size_i {
                idx = period - idx;
            }
        }

        idx as usize
    }

    /// Wrap an index into valid range (modulo)
    fn wrap_index(relative: isize, size: usize) -> usize {
        let size_i = size as isize;
        let mut idx = relative % size_i;
        if idx < 0 {
            idx += size_i;
        }
        idx as usize
    }

    /// Compute strides for a given shape
    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }
}
