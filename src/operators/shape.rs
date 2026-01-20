/// Shape operator for Iconnx
///
/// Get shape as 1D tensor
/// ONNX spec: https://onnx.ai/onnx/operators/onnx__Shape.html
use crate::tensor::Tensor;

/// Shape operator - returns shape as 1D tensor
pub struct Shape;

impl Shape {
    /// Forward pass: get shape
    ///
    /// # Arguments
    /// * `inputs` - Single input tensor (any dtype)
    /// * `attributes` - ONNX attributes:
    ///   - start: Starting axis (default: 0). Negative values count from end.
    ///   - end: Ending axis (exclusive, default: None = all remaining). Negative values count from end.
    ///
    /// # Returns
    /// 1D Int64 tensor containing shape dimensions [start:end]
    pub fn forward(
        inputs: &[Tensor],
        attributes: &crate::attributes::NodeAttributes,
    ) -> Tensor {
        assert_eq!(inputs.len(), 1, "Shape requires exactly 1 input");

        // Get full shape (works for any dtype)
        let full_shape = inputs[0].shape();
        let ndim = full_shape.len() as i64;

        // Get start attribute (default: 0)
        let start_attr = attributes.get_int("start").unwrap_or(0);
        let start = if start_attr < 0 {
            (ndim + start_attr).max(0) as usize
        } else {
            (start_attr as usize).min(full_shape.len())
        };

        // Get end attribute (default: ndim = all remaining)
        let end_attr = attributes.get_int("end");
        let end = match end_attr {
            Some(e) if e < 0 => (ndim + e).max(0) as usize,
            Some(e) => (e as usize).min(full_shape.len()),
            None => full_shape.len(),
        };

        // Slice the shape [start:end]
        let sliced_shape = &full_shape[start..end.max(start)];

        // Convert shape to i64 vector (ONNX Shape returns Int64)
        let shape_vec: Vec<i64> = sliced_shape.iter().map(|&dim| dim as i64).collect();

        Tensor::from_vec_i64(shape_vec, vec![sliced_shape.len()])
    }
}
