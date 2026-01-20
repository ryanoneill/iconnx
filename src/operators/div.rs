/// Div operator for Iconnx
///
/// Implements element-wise division with broadcasting support
/// ONNX spec: https://onnx.ai/onnx/operators/onnx__Div.html
use crate::tensor::Tensor;

/// Div operator - performs element-wise division
pub struct Div;

impl Div {
    /// Forward pass: element-wise division (A / B)
    ///
    /// # Arguments
    /// * `inputs` - Exactly 2 input tensors [A, B]
    ///
    /// # Returns
    /// Tensor containing A / B (standard IEEE float division)
    ///
    /// # Panics
    /// Panics if not exactly 2 inputs
    pub fn forward(
        inputs: &[Tensor],
        _attributes: &crate::attributes::NodeAttributes,
    ) -> Tensor {
        assert_eq!(inputs.len(), 2, "Div requires exactly 2 inputs");

        // Use dispatch_multi to handle multiple dtypes
        Tensor::dispatch_multi(
            inputs,
            |arrays| {
                // Compute broadcast shape and broadcast inputs
                let max_rank = arrays[0].ndim().max(arrays[1].ndim());
                let mut broadcast_shape = vec![1; max_rank];
                for (i, &dim) in arrays[0].shape().iter().rev().enumerate() {
                    let idx = max_rank - 1 - i;
                    broadcast_shape[idx] = broadcast_shape[idx].max(dim);
                }
                for (i, &dim) in arrays[1].shape().iter().rev().enumerate() {
                    let idx = max_rank - 1 - i;
                    broadcast_shape[idx] = broadcast_shape[idx].max(dim);
                }
                let a_bc = arrays[0]
                    .broadcast(broadcast_shape.as_slice())
                    .unwrap()
                    .to_owned();
                let b_bc = arrays[1]
                    .broadcast(broadcast_shape.as_slice())
                    .unwrap()
                    .to_owned();

                // Standard IEEE float division (matches ORT behavior)
                &a_bc / &b_bc
            },
            |arrays| {
                let max_rank = arrays[0].ndim().max(arrays[1].ndim());
                let mut broadcast_shape = vec![1; max_rank];
                for (i, &dim) in arrays[0].shape().iter().rev().enumerate() {
                    let idx = max_rank - 1 - i;
                    broadcast_shape[idx] = broadcast_shape[idx].max(dim);
                }
                for (i, &dim) in arrays[1].shape().iter().rev().enumerate() {
                    let idx = max_rank - 1 - i;
                    broadcast_shape[idx] = broadcast_shape[idx].max(dim);
                }
                let a_bc = arrays[0]
                    .broadcast(broadcast_shape.as_slice())
                    .unwrap()
                    .to_owned();
                let b_bc = arrays[1]
                    .broadcast(broadcast_shape.as_slice())
                    .unwrap()
                    .to_owned();
                // Note: i64 division doesn't have the same edge case issues
                &a_bc / &b_bc
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attributes::NodeAttributes;

    #[test]
    fn test_div_ieee_float_division() {
        // Standard IEEE float division - matches ORT behavior
        let a = Tensor::from_vec_f32(vec![0.0, 0.0, 1.0], vec![3]);
        let b = Tensor::from_vec_f32(vec![0.0, 1e-12, 0.0], vec![3]);
        let result = Div::forward(&[a, b], &NodeAttributes::new());
        let data = result.as_slice();

        // 0/0 -> NaN (IEEE standard)
        assert!(data[0].is_nan(), "0/0 should be NaN per IEEE");
        // 0/small -> 0
        assert_eq!(data[1], 0.0, "0/small should be 0");
        // 1/0 -> Inf (IEEE standard)
        assert!(data[2].is_infinite() && data[2] > 0.0, "1/0 should be +Inf");
    }

    #[test]
    fn test_div_normal_values() {
        let a = Tensor::from_vec_f32(vec![6.0, -8.0, 10.0], vec![3]);
        let b = Tensor::from_vec_f32(vec![2.0, 4.0, -5.0], vec![3]);
        let result = Div::forward(&[a, b], &NodeAttributes::new());
        let data = result.as_slice();

        assert!((data[0] - 3.0).abs() < 1e-6, "6/2 should be 3");
        assert!((data[1] - (-2.0)).abs() < 1e-6, "-8/4 should be -2");
        assert!((data[2] - (-2.0)).abs() < 1e-6, "10/-5 should be -2");
    }
}
