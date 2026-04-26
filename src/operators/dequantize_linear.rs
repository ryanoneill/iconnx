//! ONNX DequantizeLinear: `y = (x - zero_point) * scale`.
//!
//! Spec: input dtype is one of INT8, UINT8, INT32. `scale` is FP32, scalar
//! (per-tensor) or 1-D (per-axis). `zero_point` (optional) is the same dtype
//! as `x`. Output is FP32, same shape as `x`. Per-axis broadcast applies the
//! `i`-th scale/zp to every element whose `axis`-coordinate is `i`.
//!
//! WS-4 M4.4. Sister ops: `DynamicQuantizeLinear` (M4.5), `MatMulInteger`
//! (M4.6).

use ndarray::{ArrayD, IxDyn};

use crate::attributes::NodeAttributes;
use crate::tensor::Tensor;

/// ONNX DequantizeLinear operator.
pub struct DequantizeLinear;

impl DequantizeLinear {
    /// Forward pass: `y = (x - zero_point) * scale` cast to FP32.
    ///
    /// Inputs:
    /// * `inputs[0]` — quantized tensor (`Int8` / `UInt8` / `Int32`).
    /// * `inputs[1]` — FP32 `scale`. Scalar (per-tensor) or 1-D vector
    ///   (per-axis along `axis`).
    /// * `inputs[2]` — optional zero point (same dtype as `inputs[0]`,
    ///   same shape as `scale`). Defaults to zero when absent.
    ///
    /// Attribute:
    /// * `axis` (default `1`, supports negatives) — only consulted in the
    ///   per-axis case.
    pub fn forward(inputs: &[Tensor], attrs: &NodeAttributes) -> Tensor {
        assert!(
            inputs.len() == 2 || inputs.len() == 3,
            "DequantizeLinear requires 2 or 3 inputs, got {}",
            inputs.len()
        );

        let x = &inputs[0];
        let scale = &inputs[1];
        let zp = inputs.get(2);

        // Output shape == input shape.
        let out_shape: Vec<usize> = x.shape().to_vec();

        // `scale` must be FP32 by spec; we panic loudly rather than silently
        // accept other dtypes.
        let scale_vec: Vec<f32> = match scale {
            Tensor::Float32(_) => scale.as_slice(),
            other => panic!(
                "DequantizeLinear: scale must be Float32, got {}",
                other.dtype()
            ),
        };

        // Per-tensor when scale is a scalar (rank 0 or single-element 1-D);
        // per-axis otherwise.
        let scale_rank = scale.shape().len();
        let scale_len = scale_vec.len();
        let per_tensor = scale_rank == 0 || (scale_rank == 1 && scale_len == 1);

        // Resolve `axis` only when per-axis.
        let axis: usize = if per_tensor {
            0 // Unused — keeps later expansion code paths uniform.
        } else {
            let raw = attrs.get_int("axis").unwrap_or(1);
            let ndim = x.ndim() as i64;
            let resolved = if raw < 0 { raw + ndim } else { raw };
            assert!(
                (0..ndim).contains(&resolved),
                "DequantizeLinear: axis {} out of range for ndim {}",
                raw,
                ndim
            );
            resolved as usize
        };

        // Validate per-axis scale length matches the corresponding axis dim.
        if !per_tensor {
            let expected = out_shape[axis];
            assert_eq!(
                scale_len, expected,
                "DequantizeLinear: per-axis scale length {} doesn't match \
                 input shape[{}] = {}",
                scale_len, axis, expected
            );
        }

        let total: usize = out_shape.iter().product::<usize>().max(1);
        let scale_per_elem: Vec<f32> = if per_tensor {
            // Single broadcast value covers every output element.
            vec![scale_vec[0]; total]
        } else {
            broadcast_axis_to_full_shape(&scale_vec, axis, &out_shape)
        };

        // Zero point: collect into a per-element f32 vector. Default-zero
        // when absent.
        let zp_per_elem: Vec<f32> = match zp {
            None => vec![0.0_f32; total],
            Some(zp_tensor) => {
                // ONNX DequantizeLinear requires zp.dtype == x.dtype and
                // zp.shape == scale.shape (mirroring the GPU path's
                // validation in src/cuda/ops/quantize.rs). Enforce both
                // unconditionally — silent acceptance of malformed zp
                // would mask broken graphs that ORT rejects upfront.
                if zp_tensor.dtype() != x.dtype() {
                    panic!(
                        "DequantizeLinear: zero_point dtype {} must match \
                         input dtype {}",
                        zp_tensor.dtype(),
                        x.dtype()
                    );
                }
                let zp_shape = zp_tensor.shape();
                assert_eq!(
                    zp_shape,
                    scale.shape(),
                    "DequantizeLinear: zero_point shape {:?} must match \
                     scale shape {:?}",
                    zp_shape,
                    scale.shape()
                );
                let zp_per_tensor =
                    zp_shape.is_empty() || (zp_shape.len() == 1 && zp_shape == [1]);

                let zp_vec_f32: Vec<f32> = match zp_tensor {
                    Tensor::Int8(_) => zp_tensor
                        .as_slice_i8()
                        .into_iter()
                        .map(|v| v as f32)
                        .collect(),
                    Tensor::UInt8(_) => zp_tensor
                        .as_slice_u8()
                        .into_iter()
                        .map(|v| v as f32)
                        .collect(),
                    Tensor::Int32(_) => zp_tensor
                        .as_slice_i32()
                        .into_iter()
                        .map(|v| v as f32)
                        .collect(),
                    _ => unreachable!(
                        "zero_point dtype already validated to match x.dtype \
                         (Int8/UInt8/Int32) above"
                    ),
                };

                if zp_per_tensor {
                    vec![zp_vec_f32[0]; total]
                } else {
                    broadcast_axis_to_full_shape(&zp_vec_f32, axis, &out_shape)
                }
            }
        };

        // Walk the input as f32 and compute (x - zp) * scale.
        let x_f32_iter: Vec<f32> = match x {
            Tensor::Int8(_) => x.as_slice_i8().into_iter().map(|v| v as f32).collect(),
            Tensor::UInt8(_) => x.as_slice_u8().into_iter().map(|v| v as f32).collect(),
            Tensor::Int32(_) => x.as_slice_i32().into_iter().map(|v| v as f32).collect(),
            other => panic!(
                "DequantizeLinear: input dtype {} unsupported (must be Int8, \
                 UInt8, or Int32)",
                other.dtype()
            ),
        };

        let result: Vec<f32> = x_f32_iter
            .iter()
            .zip(zp_per_elem.iter())
            .zip(scale_per_elem.iter())
            .map(|((&xi, &zpi), &si)| (xi - zpi) * si)
            .collect();

        let array = ArrayD::from_shape_vec(IxDyn(&out_shape), result)
            .expect("DequantizeLinear: shape and data length agree by construction");
        Tensor::Float32(array)
    }
}

/// Broadcast a 1-D vector of length `full_shape[axis]` across the full output
/// shape. The element at flat output index `i` is the value of `scale_1d` at
/// the `axis`-coordinate of `i`.
fn broadcast_axis_to_full_shape(
    scale_1d: &[f32],
    axis: usize,
    full_shape: &[usize],
) -> Vec<f32> {
    let total: usize = full_shape.iter().product::<usize>().max(1);
    let outer: usize = full_shape[..axis].iter().product::<usize>().max(1);
    let axis_size = full_shape[axis];
    let inner: usize = full_shape[axis + 1..].iter().product::<usize>().max(1);
    debug_assert_eq!(
        scale_1d.len(),
        axis_size,
        "broadcast_axis_to_full_shape: scale length {} doesn't match \
         full_shape[{}] = {}",
        scale_1d.len(),
        axis,
        axis_size
    );
    debug_assert_eq!(outer * axis_size * inner, total);

    let mut out = Vec::with_capacity(total);
    for _o in 0..outer {
        for &v in scale_1d.iter().take(axis_size) {
            for _i in 0..inner {
                out.push(v);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dequantize_per_tensor_int8_no_zero_point() {
        // x = [-128, 0, 127], scale = 0.5 → y = [-64, 0, 63.5]
        let x = Tensor::from_vec_i8(vec![-128_i8, 0, 127], vec![3]);
        let scale = Tensor::from_vec_f32(vec![0.5_f32], vec![]);
        let out = DequantizeLinear::forward(&[x, scale], &NodeAttributes::new());
        assert_eq!(out.shape(), &[3]);
        let v = out.as_slice();
        assert!((v[0] - (-64.0)).abs() < 1e-6);
        assert!((v[1] - 0.0).abs() < 1e-6);
        assert!((v[2] - 63.5).abs() < 1e-6);
    }

    #[test]
    fn dequantize_per_channel_int8_axis_0() {
        // x = [[1, 2], [3, 4]] (2x2), scale = [0.1, 0.2] (axis=0) →
        // y[0,:] = [0.1, 0.2]; y[1,:] = [0.6, 0.8]
        let x = Tensor::from_vec_i8(vec![1_i8, 2, 3, 4], vec![2, 2]);
        let scale = Tensor::from_vec_f32(vec![0.1_f32, 0.2], vec![2]);
        let mut attrs = NodeAttributes::new();
        attrs.add_int("axis".into(), 0);
        let out = DequantizeLinear::forward(&[x, scale], &attrs);
        assert_eq!(out.shape(), &[2, 2]);
        let v = out.as_slice();
        assert!((v[0] - 0.1).abs() < 1e-6); // [0,0]
        assert!((v[1] - 0.2).abs() < 1e-6); // [0,1]
        assert!((v[2] - 0.6).abs() < 1e-6); // [1,0]
        assert!((v[3] - 0.8).abs() < 1e-6); // [1,1]
    }

    #[test]
    fn dequantize_per_tensor_uint8_with_zero_point() {
        // y = (x - zp) * scale; x = [0, 128, 255], zp = 128, scale = 0.5
        // → y = [-64, 0, 63.5]
        let x = Tensor::from_vec_u8(vec![0_u8, 128, 255], vec![3]);
        let scale = Tensor::from_vec_f32(vec![0.5_f32], vec![]);
        let zp = Tensor::from_vec_u8(vec![128_u8], vec![]);
        let out = DequantizeLinear::forward(&[x, scale, zp], &NodeAttributes::new());
        let v = out.as_slice();
        assert!((v[0] - (-64.0)).abs() < 1e-6);
        assert!((v[1] - 0.0).abs() < 1e-6);
        assert!((v[2] - 63.5).abs() < 1e-6);
    }

    #[test]
    fn dequantize_int32_input_for_matmul_integer_output() {
        // INT32 input is the MatMulInteger output → Float32 path. zp default 0.
        let x = Tensor::from_vec_i32(vec![100_i32, -50, 200], vec![3]);
        let scale = Tensor::from_vec_f32(vec![0.01_f32], vec![]);
        let out = DequantizeLinear::forward(&[x, scale], &NodeAttributes::new());
        let v = out.as_slice();
        assert!((v[0] - 1.0).abs() < 1e-6);
        assert!((v[1] - (-0.5)).abs() < 1e-6);
        assert!((v[2] - 2.0).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "zero_point dtype")]
    fn dequantize_zp_dtype_mismatch_panics() {
        // ONNX requires zp.dtype == x.dtype. Mismatched dtypes must panic
        // rather than silently coerce.
        let x = Tensor::from_vec_i8(vec![1_i8, 2, 3], vec![3]);
        let scale = Tensor::from_vec_f32(vec![0.1_f32], vec![]);
        let zp = Tensor::from_vec_u8(vec![5_u8], vec![]); // u8, not i8
        let _ = DequantizeLinear::forward(&[x, scale, zp], &NodeAttributes::new());
    }

    #[test]
    #[should_panic(expected = "zero_point shape")]
    fn dequantize_zp_shape_mismatch_panics() {
        // zp.shape must equal scale.shape. Per-tensor scalar scale + 1-D
        // zp is rejected (not silently broadcast).
        let x = Tensor::from_vec_i8(vec![1_i8, 2, 3, 4], vec![4]);
        let scale = Tensor::from_vec_f32(vec![0.1_f32], vec![]); // scalar
        let zp = Tensor::from_vec_i8(vec![0_i8, 1], vec![2]); // 1-D — mismatch
        let _ = DequantizeLinear::forward(&[x, scale, zp], &NodeAttributes::new());
    }
}
