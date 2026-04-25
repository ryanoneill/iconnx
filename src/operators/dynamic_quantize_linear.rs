//! ONNX DynamicQuantizeLinear: per-tensor dynamic quantization to UINT8.
//!
//! Outputs `(y, y_scale, y_zero_point)`:
//!   * y_scale = (max(x) - min(x)) / 255
//!   * intermediate_zp = round_ties_even(0 - min(x) / y_scale)
//!   * y_zero_point = clip(intermediate_zp, 0, 255) as u8
//!   * y[i] = saturate(round_ties_even(x[i] / y_scale) + y_zero_point) as u8
//!
//! Edge cases:
//!   * min == max → y_scale = 0, y_zero_point = 0, all y[i] = 0 (ORT
//!     convention; ONNX spec defers to implementation).
//!   * NaN values are skipped in the min/max scan (ORT convention).
//!
//! WS-4 M4.5. Multi-output op — caller routes through Split-style
//! dispatch, not single-output `execute_operator`.

use crate::attributes::NodeAttributes;
use crate::tensor::Tensor;
use ndarray::ArrayD;

/// ONNX DynamicQuantizeLinear operator.
pub struct DynamicQuantizeLinear;

impl DynamicQuantizeLinear {
    /// Forward pass: returns three tensors `[y, y_scale, y_zero_point]`.
    ///
    /// `inputs[0]` must be `Tensor::Float32`. Per ONNX spec, the qmin/qmax
    /// range is `[0, 255]` (always UINT8) and rounding is round-half-to-even.
    /// `min(x)` and `max(x)` are computed against an initial range that
    /// includes 0.0 so the resulting `y_zero_point` can always represent
    /// zero (matches ORT behaviour).
    pub fn forward(inputs: &[Tensor], _attrs: &NodeAttributes) -> Vec<Tensor> {
        assert_eq!(
            inputs.len(),
            1,
            "DynamicQuantizeLinear takes exactly 1 input"
        );
        let x = match &inputs[0] {
            Tensor::Float32(arr) => arr.clone(),
            other => panic!(
                "DynamicQuantizeLinear: input must be Float32, got {}",
                other.dtype()
            ),
        };
        let shape: Vec<usize> = x.shape().to_vec();

        // ONNX semantics: include 0.0 in the range so y_zero_point can
        // represent it (matches ORT behaviour).
        //
        // Scan in f32 (input precision) but accumulate min/max as f64 so
        // the downstream scale/zp computation has the same precision as
        // the ONNX reference implementation. numpy.minimum(x.min(), 0.0)
        // promotes to float64 because the Python literal 0.0 is f64;
        // matching that promotion is the only way to hit the documented
        // round-half-to-even boundary cases (e.g. x=[-1,0,1] → zp=128).
        let mut min_x = 0.0_f64;
        let mut max_x = 0.0_f64;
        for &v in x.iter() {
            if v.is_nan() {
                continue;
            }
            let vd = v as f64;
            if vd < min_x {
                min_x = vd;
            }
            if vd > max_x {
                max_x = vd;
            }
        }

        let qmin: f64 = 0.0;
        let qmax: f64 = 255.0;

        // Edge case: constant tensor → scale = 0 sentinel, all y = 0.
        if min_x == max_x {
            let zeros: Vec<u8> = vec![0_u8; x.len()];
            let y_arr = ArrayD::from_shape_vec(ndarray::IxDyn(&shape), zeros)
                .expect("shape match for zeros");
            return vec![
                Tensor::UInt8(y_arr),
                Tensor::Float32(ArrayD::from_elem(ndarray::IxDyn(&[]), 0.0_f32)),
                Tensor::UInt8(ArrayD::from_elem(ndarray::IxDyn(&[]), 0_u8)),
            ];
        }

        // Scale and zero-point are computed in f64 to match ONNX
        // reference; the scale tensor is then narrowed to f32 for the
        // output (its dtype per spec).
        let y_scale_f64: f64 = (max_x - min_x) / (qmax - qmin);
        let intermediate_zp: f64 = (qmin - min_x / y_scale_f64).round_ties_even();
        let y_zero_point: u8 = intermediate_zp.clamp(qmin, qmax) as u8;
        let y_scale: f32 = y_scale_f64 as f32;

        // Per-element quantization is done in f64 against the f64 scale
        // for consistency with the zp boundary above. The result lies in
        // `[0, 255]` so the precision overhead is irrelevant.
        let y_data: Vec<u8> = x
            .iter()
            .map(|&xi| {
                let q = ((xi as f64) / y_scale_f64).round_ties_even() + y_zero_point as f64;
                q.clamp(qmin, qmax) as u8
            })
            .collect();

        let y_arr = ArrayD::from_shape_vec(ndarray::IxDyn(&shape), y_data)
            .expect("shape match for quantized output");
        vec![
            Tensor::UInt8(y_arr),
            Tensor::Float32(ArrayD::from_elem(ndarray::IxDyn(&[]), y_scale)),
            Tensor::UInt8(ArrayD::from_elem(ndarray::IxDyn(&[]), y_zero_point)),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dynamic_quantize_linear_simple_positive() {
        // x = [0.0, 2.0, 4.0]; max=4, min=0;
        // y_scale = 4/255 ≈ 0.01568627
        // intermediate_zp = round(0 - 0/0.01568627) = 0; clipped → 0
        // y = saturate(round(x / 0.01568627) + 0) = [0, 128, 255]
        let x = Tensor::from_vec_f32(vec![0.0_f32, 2.0, 4.0], vec![3]);
        let outs = DynamicQuantizeLinear::forward(&[x], &NodeAttributes::new());
        assert_eq!(outs.len(), 3);
        if let Tensor::UInt8(arr) = &outs[0] {
            assert_eq!(arr.iter().copied().collect::<Vec<_>>(), vec![0_u8, 128, 255]);
        } else {
            panic!("expected UInt8 output 0")
        }
        if let Tensor::Float32(s) = &outs[1] {
            assert!((s[[]] - 4.0_f32 / 255.0).abs() < 1e-6);
        } else {
            panic!("expected Float32 scale")
        }
        if let Tensor::UInt8(zp) = &outs[2] {
            assert_eq!(zp[[]], 0_u8);
        } else {
            panic!("expected UInt8 zero_point")
        }
    }

    #[test]
    fn dynamic_quantize_linear_with_negative_input() {
        // x = [-1.0, 0.0, 1.0]; max=1, min=-1; y_scale = 2/255
        // intermediate_zp = round(0 - (-1)/(2/255)) = round(127.5) = 128 (round-half-to-even)
        let x = Tensor::from_vec_f32(vec![-1.0_f32, 0.0, 1.0], vec![3]);
        let outs = DynamicQuantizeLinear::forward(&[x], &NodeAttributes::new());
        if let Tensor::UInt8(zp) = &outs[2] {
            assert_eq!(zp[[]], 128_u8, "round-half-to-even at 127.5 should yield 128");
        } else {
            panic!()
        }
    }

    #[test]
    fn dynamic_quantize_linear_constant_input_zero_range() {
        // All-zero input: max=min=0, y_scale would be 0/255=0. ONNX spec
        // doesn't fully define this but ORT yields y_scale=0, zp=0, y=[0,...].
        let x = Tensor::from_vec_f32(vec![0.0_f32; 5], vec![5]);
        let outs = DynamicQuantizeLinear::forward(&[x], &NodeAttributes::new());
        if let Tensor::UInt8(arr) = &outs[0] {
            assert_eq!(arr.iter().copied().collect::<Vec<_>>(), vec![0_u8; 5]);
        } else {
            panic!()
        }
    }
}
