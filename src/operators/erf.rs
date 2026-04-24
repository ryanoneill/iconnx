/// Erf activation operator for Iconnx.
///
/// erf(x) = (2/√π) · ∫₀ˣ e^(-t²) dt
/// ONNX spec: https://onnx.ai/onnx/operators/onnx__Erf.html
///
/// Rust's `std` does not expose erf on stable. Implemented here via
/// the Abramowitz & Stegun 7.1.26 polynomial approximation; max
/// absolute error ~1.5e-7, well inside iconnx's 1e-5 fp32 tolerance
/// budget. iconnx has zero math-crate dependencies; introducing libm
/// for a single op is speculative infrastructure.
use crate::tensor::Tensor;

/// Erf operator — elementwise Gauss error function.
pub struct Erf;

impl Erf {
    /// Forward pass: element-wise erf.
    pub fn forward(
        inputs: &[Tensor],
        _attributes: &crate::attributes::NodeAttributes,
    ) -> Tensor {
        assert_eq!(inputs.len(), 1, "Erf requires exactly 1 input");
        let data = inputs[0].to_array();
        let result = data.mapv(erf_f32);
        Tensor::from_array(result)
    }
}

/// Abramowitz & Stegun 7.1.26. |error| ≤ 1.5e-7 on all finite inputs.
pub(crate) fn erf_f32(x: f32) -> f32 {
    let sign = x.signum();
    let a = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let y = 1.0
        - (((((1.061_405_4 * t - 1.453_152_1) * t) + 1.421_413_8) * t - 0.284_496_72) * t
            + 0.254_829_6)
            * t
            * (-a * a).exp();
    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Spot values ---------------------------------------------------

    #[test]
    fn erf_zero_is_zero() {
        assert_eq!(erf_f32(0.0), 0.0);
    }

    #[test]
    fn erf_one_matches_reference() {
        // erf(1) ≈ 0.8427007929
        assert!((erf_f32(1.0) - 0.842_700_8).abs() < 1e-6);
    }

    #[test]
    fn erf_negative_one_matches_reference() {
        assert!((erf_f32(-1.0) + 0.842_700_8).abs() < 1e-6);
    }

    #[test]
    fn erf_positive_infinity_is_one() {
        assert_eq!(erf_f32(f32::INFINITY), 1.0);
    }

    #[test]
    fn erf_negative_infinity_is_negative_one() {
        assert_eq!(erf_f32(f32::NEG_INFINITY), -1.0);
    }

    #[test]
    fn erf_nan_is_nan() {
        assert!(erf_f32(f32::NAN).is_nan());
    }

    // ---- A&S tolerance locks ------------------------------------------

    #[test]
    fn erf_matches_reference_at_half() {
        // High-precision reference: erf(0.5) ≈ 0.5204998778130465
        let reference = 0.520_499_9_f32;
        assert!(
            (erf_f32(0.5) - reference).abs() < 2e-7,
            "erf(0.5) = {} deviates from reference {} beyond 2e-7; \
             the A&S polynomial coefficients may have been transcribed wrong",
            erf_f32(0.5),
            reference,
        );
        assert!((erf_f32(-0.5) + reference).abs() < 2e-7);
    }

    #[test]
    fn erf_matches_reference_at_two_and_a_half() {
        // High-precision reference: erf(2.5) ≈ 0.9995930479825550
        let reference = 0.999_593_1_f32;
        assert!(
            (erf_f32(2.5) - reference).abs() < 2e-7,
            "erf(2.5) = {} deviates from reference {} beyond 2e-7",
            erf_f32(2.5),
            reference,
        );
        assert!((erf_f32(-2.5) + reference).abs() < 2e-7);
    }

    // ---- Property tests over a hand-rolled grid -----------------------

    fn grid() -> impl Iterator<Item = f32> {
        // [-6, 6] at step 0.001 → ~12,001 inputs.
        (0..=12_000).map(|i| -6.0 + (i as f32) * 0.001)
    }

    #[test]
    fn erf_is_odd_symmetric_on_grid() {
        for x in grid() {
            let sum = erf_f32(-x) + erf_f32(x);
            assert!(
                sum.abs() < 1e-6,
                "odd-symmetry violated at x={}: erf(-x) + erf(x) = {}",
                x,
                sum,
            );
        }
    }

    #[test]
    fn erf_is_bounded_on_grid() {
        for x in grid() {
            let v = erf_f32(x);
            assert!(
                v.abs() <= 1.0 + 1e-6,
                "boundedness violated at x={}: |erf(x)| = {} > 1",
                x,
                v.abs(),
            );
        }
    }

    #[test]
    fn erf_is_monotonic_on_grid() {
        let mut prev = erf_f32(-6.0);
        for x in grid().skip(1) {
            let curr = erf_f32(x);
            assert!(
                curr >= prev - 1e-6,
                "monotonicity violated at x={}: erf previous={}, current={}",
                x,
                prev,
                curr,
            );
            prev = curr;
        }
    }

    // ---- Tensor wrapper ----------------------------------------------

    #[test]
    fn erf_forward_on_tensor_preserves_shape_and_values() {
        let input = Tensor::from_vec(vec![0.0, 0.5, 1.0, -0.5, -1.0], vec![5]);
        let attrs = crate::attributes::NodeAttributes::new();
        let output = Erf::forward(&[input], &attrs);
        let out_vals = output.to_array();
        assert_eq!(out_vals.shape(), &[5]);
        assert!((out_vals[[0]] - 0.0_f32).abs() < 1e-6);
        assert!((out_vals[[1]] - 0.520_499_9).abs() < 2e-7);
        assert!((out_vals[[2]] - 0.842_700_8).abs() < 1e-6);
        assert!((out_vals[[3]] + 0.520_499_9).abs() < 2e-7);
        assert!((out_vals[[4]] + 0.842_700_8).abs() < 1e-6);
    }
}
