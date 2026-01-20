/// Slice operator for Iconnx
///
/// Extract range [start:end:step] along axes
/// ONNX spec: https://onnx.ai/onnx/operators/onnx__Slice.html
use crate::tensor::Tensor;
use ndarray::ArrayD;

/// Slice operator - extract tensor slices
pub struct Slice;

impl Slice {
    /// Forward pass: slice tensor
    ///
    /// # Arguments
    /// * `inputs` - [data, starts, ends, axes, steps]
    ///   - data: input tensor
    ///   - starts: starting indices for each axis
    ///   - ends: ending indices for each axis
    ///   - axes: axes to slice along
    ///   - steps: step size for each axis
    ///
    /// # Returns
    /// Sliced tensor
    ///
    /// Note: This implementation uses ndarray's safe `.slice_each_axis()` method
    /// instead of unsafe `SliceInfo::new()`
    pub fn forward(
        inputs: &[Tensor],
        _attributes: &crate::attributes::NodeAttributes,
    ) -> Tensor {
        assert!(
            inputs.len() >= 3 && inputs.len() <= 5,
            "Slice requires 3-5 inputs [data, starts, ends, axes?, steps?], got {}",
            inputs.len()
        );

        // Extract slice parameters (Int64 or Float32)
        let starts: Vec<isize> = if inputs[1].is_int64() {
            inputs[1]
                .as_slice_i64()
                .iter()
                .map(|&x| x as isize)
                .collect()
        } else {
            inputs[1].as_slice().iter().map(|&x| x as isize).collect()
        };

        let ends: Vec<isize> = if inputs[2].is_int64() {
            inputs[2]
                .as_slice_i64()
                .iter()
                .map(|&x| x as isize)
                .collect()
        } else {
            inputs[2].as_slice().iter().map(|&x| x as isize).collect()
        };

        // axes is optional (defaults to [0, 1, 2, ...])
        // Handle negative axes (e.g., -1 = last axis)
        let ndim = inputs[0].ndim();
        let axes: Vec<usize> = if inputs.len() >= 4 {
            if inputs[3].is_int64() {
                inputs[3]
                    .as_slice_i64()
                    .iter()
                    .map(|&x| {
                        if x < 0 {
                            (ndim as i64 + x) as usize
                        } else {
                            x as usize
                        }
                    })
                    .collect()
            } else {
                inputs[3]
                    .as_slice()
                    .iter()
                    .map(|&x| {
                        let xi = x as i64;
                        if xi < 0 {
                            (ndim as i64 + xi) as usize
                        } else {
                            x as usize
                        }
                    })
                    .collect()
            }
        } else {
            // Default: all axes in order
            (0..ndim).collect()
        };

        // steps is optional (defaults to [1, 1, 1, ...])
        let steps: Vec<isize> = if inputs.len() >= 5 {
            if inputs[4].is_int64() {
                inputs[4]
                    .as_slice_i64()
                    .iter()
                    .map(|&x| x as isize)
                    .collect()
            } else {
                inputs[4].as_slice().iter().map(|&x| x as isize).collect()
            }
        } else {
            // Default: step=1 for all axes
            vec![1; axes.len()]
        };

        // Apply slicing - works on any dtype
        inputs[0].map_preserving_dtype(
            |data_f32| {
                // Float32 path
                Self::slice_array(data_f32, &starts, &ends, &axes, &steps)
            },
            |data_i64| {
                // Int64 path
                Self::slice_array(data_i64, &starts, &ends, &axes, &steps)
            },
        )
    }

    /// Helper: Slice an ndarray with given parameters
    fn slice_array<T: Clone>(
        data: &ArrayD<T>,
        starts: &[isize],
        ends: &[isize],
        axes: &[usize],
        steps: &[isize],
    ) -> ArrayD<T> {
        let result = data.slice_each_axis(|ax| {
            // Find if this axis is in the axes list
            for i in 0..axes.len() {
                if axes[i] == ax.axis.index() {
                    let axis_len = ax.len as isize;
                    let mut start = starts[i];
                    let mut end = ends[i];
                    let mut step = steps[i];

                    // Handle step=0 (invalid) - default to 1
                    if step == 0 {
                        step = 1;
                    }

                    // Clamp start to valid range
                    if start < 0 {
                        start = (axis_len + start).max(0);
                    }
                    start = start.min(axis_len);

                    // Clamp end to valid range (INT64_MAX means "to the end")
                    if end < 0 {
                        end = (axis_len + end).max(0);
                    }
                    if end > axis_len || end == i64::MAX as isize {
                        end = axis_len;
                    }

                    return ndarray::Slice::new(start, Some(end), step);
                }
            }

            // Not in the slice list - take full axis
            ndarray::Slice::new(0, None, 1)
        });

        result.to_owned()
    }
}
