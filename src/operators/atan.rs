/// Atan operator for Iconnx
///
/// Element-wise arctangent
/// ONNX spec: https://onnx.ai/onnx/operators/onnx__Atan.html
use crate::tensor::Tensor;

/// Atan operator - element-wise arctangent
pub struct Atan;

impl Atan {
    /// Forward pass: element-wise atan
    ///
    /// # Arguments
    /// * `inputs` - Single input tensor
    ///
    /// # Returns
    /// Tensor with atan applied element-wise
    ///
    /// # Numerical Stability
    /// NaN inputs (typically from 0/0 division in atan2-style patterns)
    /// return 0.0, matching the atan2(0,0) = 0 convention.
    pub fn forward(
        inputs: &[Tensor],
        _attributes: &crate::attributes::NodeAttributes,
    ) -> Tensor {
        assert_eq!(inputs.len(), 1, "Atan requires exactly 1 input");

        let data = inputs[0].to_array();
        // Handle NaN inputs: return 0 (matches atan2(0,0) convention)
        let result = data.mapv(|x| if x.is_nan() { 0.0 } else { x.atan() });

        Tensor::from_array(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attributes::NodeAttributes;

    /// Regression test: NaN inputs should return 0, matching atan2(0,0) convention.
    /// This prevents NaN propagation in phase angle computations using div+atan patterns.
    #[test]
    fn test_atan_nan_returns_zero() {
        let input = Tensor::from_vec_f32(vec![f32::NAN, 0.0, 1.0, f32::INFINITY], vec![4]);
        let result = Atan::forward(&[input], &NodeAttributes::new());
        let data = result.as_slice();

        // NaN input -> 0.0 (by convention)
        assert_eq!(data[0], 0.0, "atan(NaN) should return 0.0");

        // Normal cases should still work
        assert!(
            (data[1] - 0.0f32.atan()).abs() < 1e-6,
            "atan(0) should work"
        );
        assert!(
            (data[2] - 1.0f32.atan()).abs() < 1e-6,
            "atan(1) should work"
        );
        assert!(
            (data[3] - std::f32::consts::FRAC_PI_2).abs() < 1e-6,
            "atan(inf) should be π/2"
        );
    }

    #[test]
    fn test_atan_preserves_sign() {
        let input = Tensor::from_vec_f32(vec![-1.0, 0.0, 1.0], vec![3]);
        let result = Atan::forward(&[input], &NodeAttributes::new());
        let data = result.as_slice();

        assert!(data[0] < 0.0, "atan(-1) should be negative");
        assert_eq!(data[1], 0.0, "atan(0) should be 0");
        assert!(data[2] > 0.0, "atan(1) should be positive");
    }
}
