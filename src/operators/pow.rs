/// Pow operator for Iconnx
///
/// Element-wise exponentiation: x^y
/// ONNX spec: https://onnx.ai/onnx/operators/onnx__Pow.html
use crate::tensor::Tensor;

/// Pow operator - element-wise power
pub struct Pow;

impl Pow {
    /// Forward pass: element-wise x^y with broadcasting
    ///
    /// # Arguments
    /// * `inputs` - [base, exponent] (both tensors)
    ///
    /// # Returns
    /// Tensor containing base^exponent element-wise
    pub fn forward(
        inputs: &[Tensor],
        _attributes: &crate::attributes::NodeAttributes,
    ) -> Tensor {
        assert_eq!(inputs.len(), 2, "Pow requires exactly 2 inputs");

        Tensor::dispatch_multi(
            inputs,
            |arrays| {
                // F32: element-wise power with broadcasting
                let base = arrays[0];
                let exp_bc = arrays[1].broadcast(base.raw_dim()).unwrap();
                ndarray::Array::from_shape_fn(base.raw_dim(), |idx| {
                    let b = base[&idx];
                    let e = exp_bc[&idx];
                    b.powf(e)
                })
            },
            |arrays| {
                // I64: element-wise power with broadcasting
                let base = arrays[0];
                let exp_bc = arrays[1].broadcast(base.raw_dim()).unwrap();
                ndarray::Array::from_shape_fn(base.raw_dim(), |idx| {
                    let b = base[&idx] as f64;
                    let e = exp_bc[&idx] as i32;
                    b.powi(e) as i64
                })
            },
        )
    }
}
