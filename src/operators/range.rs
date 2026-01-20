/// Range operator for Iconnx
///
/// Implements sequence generation [start, start+step, ..., end)
/// ONNX spec: https://onnx.ai/onnx/operators/onnx__Range.html
use crate::tensor::Tensor;

/// Range operator - generates sequence from start to limit with step
pub struct Range;

impl Range {
    /// Forward pass: generate sequence [start, start+step, ..., limit)
    ///
    /// # Arguments
    /// * `inputs` - Exactly 3 inputs: [start, limit, delta]
    ///   - start: Scalar tensor with starting value
    ///   - limit: Scalar tensor with limit (exclusive)
    ///   - delta: Scalar tensor with step size
    ///
    /// # Returns
    /// 1D tensor containing sequence
    ///
    /// # Panics
    /// Panics if not exactly 3 inputs or inputs are not scalars
    pub fn forward(
        inputs: &[Tensor],
        _attributes: &crate::attributes::NodeAttributes,
    ) -> Tensor {
        assert_eq!(inputs.len(), 3, "Range requires exactly 3 inputs");

        assert!(
            inputs[0].ndim() == 0,
            "Start must be scalar, got {}D",
            inputs[0].ndim()
        );
        assert!(
            inputs[1].ndim() == 0,
            "Limit must be scalar, got {}D",
            inputs[1].ndim()
        );
        assert!(
            inputs[2].ndim() == 0,
            "Delta must be scalar, got {}D",
            inputs[2].ndim()
        );

        // Handle Int64, Int32, and Float32 ranges
        // Convert all to common dtype if mixed
        let all_int64 = inputs.iter().all(|t| t.is_int64());
        let all_int32 = inputs.iter().all(|t| t.is_int32());
        let all_float32 = inputs.iter().all(|t| t.is_float32());

        if all_int64 {
            let start = inputs[0].as_slice_i64()[0];
            let limit = inputs[1].as_slice_i64()[0];
            let delta = inputs[2].as_slice_i64()[0];

            assert!(delta != 0, "Delta cannot be zero");

            let mut data = Vec::new();
            let mut current = start;

            if delta > 0 {
                while current < limit {
                    data.push(current);
                    current += delta;
                }
            } else {
                while current > limit {
                    data.push(current);
                    current += delta;
                }
            }

            let len = data.len();
            Tensor::from_vec_i64(data, vec![len])
        } else if all_int32 {
            let start = inputs[0].as_slice_i32()[0];
            let limit = inputs[1].as_slice_i32()[0];
            let delta = inputs[2].as_slice_i32()[0];

            assert!(delta != 0, "Delta cannot be zero");

            let mut data = Vec::new();
            let mut current = start;

            if delta > 0 {
                while current < limit {
                    data.push(current);
                    current += delta;
                }
            } else {
                while current > limit {
                    data.push(current);
                    current += delta;
                }
            }

            let len = data.len();
            Tensor::from_vec_i32(data, vec![len])
        } else if all_float32 {
            // Float32 path
            let start = inputs[0].as_slice()[0];
            let limit = inputs[1].as_slice()[0];
            let delta = inputs[2].as_slice()[0];

            assert!(delta != 0.0, "Delta cannot be zero");

            let mut data = Vec::new();
            let mut current = start;

            if delta > 0.0 {
                while current < limit {
                    data.push(current);
                    current += delta;
                }
            } else {
                while current > limit {
                    data.push(current);
                    current += delta;
                }
            }

            let len = data.len();
            Tensor::from_vec(data, vec![len])
        } else {
            // Mixed dtypes - convert all to Float32
            let start = if inputs[0].is_int64() {
                inputs[0].as_slice_i64()[0] as f32
            } else if inputs[0].is_int32() {
                inputs[0].as_slice_i32()[0] as f32
            } else {
                inputs[0].as_slice()[0]
            };

            let limit = if inputs[1].is_int64() {
                inputs[1].as_slice_i64()[0] as f32
            } else if inputs[1].is_int32() {
                inputs[1].as_slice_i32()[0] as f32
            } else {
                inputs[1].as_slice()[0]
            };

            let delta = if inputs[2].is_int64() {
                inputs[2].as_slice_i64()[0] as f32
            } else if inputs[2].is_int32() {
                inputs[2].as_slice_i32()[0] as f32
            } else {
                inputs[2].as_slice()[0]
            };

            assert!(delta != 0.0, "Delta cannot be zero");

            let mut data = Vec::new();
            let mut current = start;

            if delta > 0.0 {
                while current < limit {
                    data.push(current);
                    current += delta;
                }
            } else {
                while current > limit {
                    data.push(current);
                    current += delta;
                }
            }

            let len = data.len();
            Tensor::from_vec(data, vec![len])
        }
    }
}
