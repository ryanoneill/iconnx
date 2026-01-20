/// Softmax operator for Iconnx
///
/// Computes softmax: exp(x) / sum(exp(x))
/// ONNX spec: https://onnx.ai/onnx/operators/onnx__Softmax.html
use crate::tensor::Tensor;
use ndarray::{ArrayD, Axis};

/// Softmax operator - normalized exponential function
pub struct Softmax;

impl Softmax {
    /// Forward pass: softmax activation
    ///
    /// # Arguments
    /// * `inputs` - Single input tensor
    /// * `attributes` - ONNX attributes:
    ///   - axis: Axis along which to compute softmax (default: -1 = last axis)
    ///
    /// # Returns
    /// Tensor with softmax applied along specified axis
    pub fn forward(
        inputs: &[Tensor],
        attributes: &crate::attributes::NodeAttributes,
    ) -> Tensor {
        assert_eq!(inputs.len(), 1, "Softmax requires exactly 1 input");

        let data = inputs[0].to_array();

        // Get axis from attributes (ONNX default is -1 = last axis)
        let axis_attr = attributes.get_int("axis").unwrap_or(-1);
        let axis = if axis_attr < 0 {
            (data.ndim() as i64 + axis_attr) as usize
        } else {
            axis_attr as usize
        };

        let result = Self::softmax_along_axis(data, axis);

        Tensor::from_array(result)
    }

    fn softmax_along_axis(data: &ArrayD<f32>, axis: usize) -> ArrayD<f32> {
        // Subtract max for numerical stability
        let max = data.fold_axis(Axis(axis), f32::NEG_INFINITY, |&a, &b| a.max(b));
        let shifted = data - &max.insert_axis(Axis(axis));

        // Compute exp
        let exp = shifted.mapv(|x| x.exp());

        // Sum along axis
        let sum = exp.sum_axis(Axis(axis));

        // Divide
        &exp / &sum.insert_axis(Axis(axis))
    }
}
