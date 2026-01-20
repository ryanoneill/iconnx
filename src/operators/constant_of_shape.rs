/// ConstantOfShape operator for Iconnx
///
/// Implements tensor creation filled with constant value
/// ONNX spec: https://onnx.ai/onnx/operators/onnx__ConstantOfShape.html
use crate::tensor::Tensor;

/// ConstantOfShape operator - creates tensor of given shape filled with constant value
pub struct ConstantOfShape;

impl ConstantOfShape {
    /// Forward pass: create tensor of given shape filled with value
    ///
    /// # Arguments
    /// * `inputs` - Exactly 2 inputs: [shape_tensor, value_tensor]
    ///   - shape_tensor: 1D tensor containing target shape (as f32, will be cast to usize)
    ///   - value_tensor: Scalar tensor containing fill value
    ///
    /// # Returns
    /// Tensor of given shape filled with value
    ///
    /// # Panics
    /// Panics if not exactly 2 inputs, shape is not 1D, or value is not scalar
    pub fn forward(
        inputs: &[Tensor],
        attributes: &crate::attributes::NodeAttributes,
    ) -> Tensor {
        assert_eq!(
            inputs.len(),
            1,
            "ConstantOfShape requires exactly 1 input [shape]"
        );

        let shape_tensor = &inputs[0];

        // Extract shape (Int64 or Float32)
        // Handle both 1D [N] and 2D [1, N] shapes (squeeze if needed)
        let shape: Vec<usize> = if shape_tensor.ndim() == 1 {
            // Standard 1D shape
            if shape_tensor.is_int64() {
                shape_tensor
                    .as_slice_i64()
                    .iter()
                    .map(|&x| x as usize)
                    .collect()
            } else {
                shape_tensor
                    .as_slice()
                    .iter()
                    .map(|&x| x as usize)
                    .collect()
            }
        } else if shape_tensor.ndim() == 2 {
            // 2D: flatten to 1D (works for both [1, N] and [N, 1])
            if shape_tensor.is_int64() {
                shape_tensor
                    .as_slice_i64()
                    .iter()
                    .map(|&x| x as usize)
                    .collect()
            } else {
                shape_tensor
                    .as_slice()
                    .iter()
                    .map(|&x| x as usize)
                    .collect()
            }
        } else {
            panic!(
                "Shape tensor must be 1D or 2D, got {}D with shape {:?}",
                shape_tensor.ndim(),
                shape_tensor.shape()
            );
        };

        // Get fill value from attributes (default 0.0)
        // Value tensor determines output dtype
        if let Some(value_tensor) = attributes.get_tensor("value") {
            let total_elements: usize = shape.iter().product();

            if value_tensor.is_int64() {
                let fill_value = value_tensor.as_slice_i64()[0];
                Tensor::from_vec_i64(vec![fill_value; total_elements], shape)
            } else if value_tensor.is_int32() {
                let fill_value = value_tensor.as_slice_i32()[0];
                Tensor::from_vec_i32(vec![fill_value; total_elements], shape)
            } else {
                let fill_value = value_tensor.as_slice()[0];
                Tensor::from_vec(vec![fill_value; total_elements], shape)
            }
        } else {
            // ONNX default: Float32 zeros
            let total_elements: usize = shape.iter().product();
            Tensor::from_vec(vec![0.0; total_elements], shape)
        }
    }
}
