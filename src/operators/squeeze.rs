/// Squeeze operator for Iconnx
///
/// Removes dimensions of size 1
/// ONNX spec: https://onnx.ai/onnx/operators/onnx__Squeeze.html
use crate::tensor::Tensor;

/// Squeeze operator - remove singleton dimensions
pub struct Squeeze;

impl Squeeze {
    /// Forward pass: remove dimensions of size 1
    ///
    /// # Arguments
    /// * `inputs` - [data_tensor] or [data_tensor, axes_tensor]
    ///
    /// # Returns
    /// Tensor with singleton dimensions removed
    pub fn forward(
        inputs: &[Tensor],
        _attributes: &crate::attributes::NodeAttributes,
    ) -> Tensor {
        assert!(
            !inputs.is_empty() && inputs.len() <= 2,
            "Squeeze requires 1 or 2 inputs"
        );

        let data_shape = inputs[0].shape();

        // Get axes to squeeze
        let axes_to_squeeze: Vec<usize> = if inputs.len() == 2 {
            // Axes provided as second input
            let axes_tensor = &inputs[1];
            if axes_tensor.is_int64() {
                axes_tensor
                    .as_slice_i64()
                    .iter()
                    .map(|&axis| {
                        // Handle negative axes
                        if axis < 0 {
                            (data_shape.len() as i64 + axis) as usize
                        } else {
                            axis as usize
                        }
                    })
                    .collect()
            } else {
                // Empty or wrong type - squeeze all
                (0..data_shape.len())
                    .filter(|&i| data_shape[i] == 1)
                    .collect()
            }
        } else {
            // No axes specified - squeeze all dimensions of size 1
            (0..data_shape.len())
                .filter(|&i| data_shape[i] == 1)
                .collect()
        };

        // Validate axes point to dimensions of size 1
        for &axis in &axes_to_squeeze {
            assert_eq!(
                data_shape[axis], 1,
                "Cannot squeeze axis {} with size {}",
                axis, data_shape[axis]
            );
        }

        // Build new shape by removing specified axes
        let mut new_shape = Vec::new();
        for (i, &dim) in data_shape.iter().enumerate() {
            if !axes_to_squeeze.contains(&i) {
                new_shape.push(dim);
            }
        }

        // Handle different dtypes
        match &inputs[0] {
            Tensor::Float32(data) => {
                if new_shape.is_empty() {
                    // All dimensions were squeezed, result is scalar
                    return Tensor::from_vec(vec![data.iter().next().copied().unwrap()], vec![]);
                }
                let result = data
                    .to_shape(new_shape)
                    .expect("Failed to squeeze")
                    .into_owned();
                Tensor::from_array(result)
            }
            Tensor::Int64(data) => {
                if new_shape.is_empty() {
                    // All dimensions were squeezed, result is scalar
                    return Tensor::from_vec_i64(
                        vec![data.iter().next().copied().unwrap()],
                        vec![],
                    );
                }
                let result = data
                    .to_shape(new_shape)
                    .expect("Failed to squeeze")
                    .into_owned();
                Tensor::from_array_i64(result)
            }
            Tensor::Int32(data) => {
                if new_shape.is_empty() {
                    // All dimensions were squeezed, result is scalar
                    return Tensor::from_vec_i32(
                        vec![data.iter().next().copied().unwrap()],
                        vec![],
                    );
                }
                let result = data
                    .to_shape(new_shape)
                    .expect("Failed to squeeze")
                    .into_owned();
                Tensor::from_array_i32(result)
            }
            _ => {
                // For other dtypes, convert to Float32
                let data_f32 = inputs[0].to_float32();
                Self::forward(&[data_f32], _attributes)
            }
        }
    }
}
