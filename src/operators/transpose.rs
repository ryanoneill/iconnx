/// Transpose operator for Iconnx
///
/// Permutes tensor dimensions. Memcpy-class op — preserves dtype.
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
    /// If no permutation provided, reverses all axes (e.g., 2D matrix transpose).
    /// Dtype-polymorphic — output preserves input dtype (Transpose is
    /// pure shape relabel, no math). WS-4 added Int8/UInt8 arms so the
    /// constant-folding pass can transpose quantized weight initializers
    /// in DistilBERT-INT8 without panicking via the f32-strict
    /// `to_array()` accessor.
    pub fn forward(
        inputs: &[Tensor],
        attributes: &crate::attributes::NodeAttributes,
    ) -> Tensor {
        assert_eq!(inputs.len(), 1, "Transpose requires exactly 1 input");

        let data = &inputs[0];

        let perm: Vec<usize> = if let Some(perm_attr) = attributes.get_ints("perm") {
            perm_attr.iter().map(|&v| v as usize).collect()
        } else {
            (0..data.ndim()).rev().collect()
        };

        match data {
            Tensor::Float32(arr) => {
                Tensor::from_array(arr.clone().permuted_axes(perm))
            }
            Tensor::Float64(arr) => {
                Tensor::Float64(arr.clone().permuted_axes(perm))
            }
            Tensor::Int64(arr) => {
                Tensor::from_array_i64(arr.clone().permuted_axes(perm))
            }
            Tensor::Int32(arr) => {
                Tensor::from_array_i32(arr.clone().permuted_axes(perm))
            }
            Tensor::Bool(arr) => {
                Tensor::Bool(arr.clone().permuted_axes(perm))
            }
            Tensor::Int8(arr) => {
                Tensor::Int8(arr.clone().permuted_axes(perm))
            }
            Tensor::UInt8(arr) => {
                Tensor::UInt8(arr.clone().permuted_axes(perm))
            }
            Tensor::Float16(arr) => {
                Tensor::Float16(arr.clone().permuted_axes(perm))
            }
            Tensor::BFloat16(arr) => {
                Tensor::BFloat16(arr.clone().permuted_axes(perm))
            }
        }
    }
}
