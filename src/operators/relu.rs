//! ONNX Relu operator: y = max(0, x), elementwise, shape-preserving.
//!
//! Stable since opset 6; takes a single input, emits a single output,
//! has no attributes. Non-negative, monotone non-decreasing, idempotent
//! (`Relu(Relu(x)) == Relu(x)`). Propagates NaN per IEEE 754 `fmax`
//! semantics (`fmax(NaN, 0) = NaN`).
//!
//! CPU reference implementation. The production GPU path is
//! `cuda::kernels::gpu_relu` — this file exists so the graph executor,
//! constant-folding pass, and fusion equivalence tests have a CPU
//! reference to compare against.

use crate::attributes::NodeAttributes;
use crate::tensor::Tensor;

/// ONNX Relu.
pub struct Relu;

impl Relu {
    /// Apply Relu elementwise to `inputs[0]`. ONNX Relu takes exactly
    /// one input and no attributes, so `_attrs` is ignored.
    pub fn forward(inputs: &[Tensor], _attrs: &NodeAttributes) -> Tensor {
        let input = &inputs[0];
        let data = input.to_array();
        let result = data.mapv(relu_f32);
        Tensor::from_array(result)
    }
}

/// `max(0, x)` with `fmax`-style NaN propagation.
///
/// `f32::max` follows the IEEE 754 2008 `maxNum` rule — it returns the
/// non-NaN operand when exactly one is NaN. That would silently mask
/// upstream NaNs in Relu outputs, which is the opposite of what ONNX
/// reference implementations do (ORT / onnxruntime propagate NaN). We
/// branch on `is_nan` explicitly so `relu(NaN) == NaN`.
pub(crate) fn relu_f32(x: f32) -> f32 {
    if x.is_nan() {
        f32::NAN
    } else if x > 0.0 {
        x
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Spot values ---------------------------------------------------

    #[test]
    fn relu_negative_is_zero() {
        assert_eq!(relu_f32(-1.0), 0.0);
    }

    #[test]
    fn relu_zero_is_zero() {
        assert_eq!(relu_f32(0.0), 0.0);
    }

    #[test]
    fn relu_positive_is_input() {
        assert_eq!(relu_f32(1.0), 1.0);
    }

    #[test]
    fn relu_positive_infinity_is_positive_infinity() {
        assert_eq!(relu_f32(f32::INFINITY), f32::INFINITY);
    }

    #[test]
    fn relu_negative_infinity_is_zero() {
        assert_eq!(relu_f32(f32::NEG_INFINITY), 0.0);
    }

    #[test]
    fn relu_nan_is_nan() {
        // IEEE 754: max(NaN, anything) is NaN under `maximum` semantics.
        // ORT does the same, so we mirror it rather than silently
        // dropping the NaN via f32::max's maxNum rule.
        assert!(relu_f32(f32::NAN).is_nan());
    }

    // ---- Property tests (grid over [-6, 6] at step 0.001) -------------

    fn grid() -> Vec<f32> {
        let mut v = Vec::with_capacity(12_001);
        let mut x = -6.0_f32;
        for _ in 0..=12_000 {
            v.push(x);
            x += 0.001;
        }
        v
    }

    #[test]
    fn relu_is_non_negative_on_grid() {
        for x in grid() {
            let y = relu_f32(x);
            assert!(
                y >= 0.0,
                "relu({}) = {} is negative; Relu output must be non-negative",
                x,
                y
            );
        }
    }

    #[test]
    fn relu_is_monotonic_non_decreasing_on_grid() {
        let g = grid();
        for pair in g.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            let (ra, rb) = (relu_f32(a), relu_f32(b));
            assert!(
                rb >= ra,
                "relu is not monotone non-decreasing at ({}, {}): relu({}) = {}, relu({}) = {}",
                a,
                b,
                a,
                ra,
                b,
                rb,
            );
        }
    }

    #[test]
    fn relu_is_idempotent_on_grid() {
        for x in grid() {
            let r1 = relu_f32(x);
            let r2 = relu_f32(r1);
            assert_eq!(
                r1, r2,
                "relu(relu({})) = {} differs from relu({}) = {}",
                x, r2, x, r1
            );
        }
    }

    // ---- Tensor wrapper ----------------------------------------------

    #[test]
    fn relu_forward_on_tensor_preserves_shape_and_clips_negatives() {
        let input = Tensor::from_vec(vec![-2.0, -0.5, 0.0, 0.5, 2.0], vec![5]);
        let attrs = NodeAttributes::new();
        let output = Relu::forward(&[input], &attrs);
        let out_vals = output.to_array();
        assert_eq!(out_vals.shape(), &[5]);
        assert_eq!(out_vals[[0]], 0.0);
        assert_eq!(out_vals[[1]], 0.0);
        assert_eq!(out_vals[[2]], 0.0);
        assert_eq!(out_vals[[3]], 0.5);
        assert_eq!(out_vals[[4]], 2.0);
    }
}
