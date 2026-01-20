/// Transpose operator for Iconnx
///
/// Permutes tensor dimensions
/// ONNX spec: https://onnx.ai/onnx/operators/onnx__Transpose.html
use crate::tensor::Tensor;

/// Transpose operator - permutes dimensions
pub struct Transpose;

impl Transpose {
    /// Forward pass: transpose tensor dimensions
    ///
    /// # Arguments
    /// * `inputs` - Either [data] or [data, perm]
    ///   - data: Tensor to transpose
    ///   - perm (optional): Permutation order as 1D tensor
    ///
    /// If no permutation provided, reverses all axes (e.g., 2D matrix transpose)
    ///
    /// # Returns
    /// Transposed tensor
    pub fn forward(
        inputs: &[Tensor],
        attributes: &crate::attributes::NodeAttributes,
    ) -> Tensor {
        assert_eq!(inputs.len(), 1, "Transpose requires exactly 1 input");

        let data = &inputs[0];

        // Get permutation from attributes (ONNX spec)
        let perm: Vec<usize> = if let Some(perm_attr) = attributes.get_ints("perm") {
            // Use permutation from attributes
            perm_attr.iter().map(|&v| v as usize).collect()
        } else {
            // Default: reverse all axes
            (0..data.ndim()).rev().collect()
        };

        #[cfg(debug_assertions)]
        {}

        // Perform transpose
        let transposed = data.to_array().clone().permuted_axes(perm);

        #[cfg(debug_assertions)]
        {}

        Tensor::from_array(transposed)
    }
}
