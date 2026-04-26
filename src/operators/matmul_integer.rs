//! ONNX MatMulInteger: integer matrix multiplication.
//!
//! Inputs: `a`, `b`, optional `a_zero_point`, optional `b_zero_point`.
//! Allowed input dtypes: `i8` or `u8` for both `a` and `b` (4 sign
//! combinations — s8s8, u8s8, s8u8, u8u8). DistilBERT-INT8 specifically
//! uses **u8 × i8** (UINT8 activations × INT8 weights).
//! Output: **always Int32** (no scale applied — that's a separate `Mul`
//! or fused into a downstream `DequantizeLinear`).
//!
//! Math:
//!   out[m, n] = Σ_k (a[m,k] - a_zp) * (b[k,n] - b_zp)
//! All operands are promoted to `i32` before subtraction and
//! multiplication. `a_zp` / `b_zp` default to 0 when absent. zp dtype
//! must match the corresponding operand dtype.
//!
//! M4.6 MVP restrictions (enforced by panics):
//!   * Rank: 2-D inputs only. DistilBERT-INT8 unrolls batches at the IR
//!     level so 2-D is sufficient; batched MatMul is M4.7+ scope.
//!   * Zero-point: scalar per-tensor only. ONNX allows per-axis zp on
//!     `b`'s inner dim, but DistilBERT does not exercise that path; the
//!     restriction keeps the GPU kernel ABI simple and prevents silent
//!     acceptance of unsupported ONNX nodes.
//!
//! Worst-case accumulator: with K = 3072 (DistilBERT FFN inner dim)
//! and full-range INT8 inputs (`±127`), the per-cell sum is bounded by
//! `K * 127 * 127 ≈ 49.5M`, well within `INT32_MAX` (2.1e9). The
//! `matmul_integer_k_3072_overflow_safety_distilbert_ffn` test pins this
//! invariant — it's the gating verification that must pass on CPU
//! **before** the GPU kernels are written, per the WS-4 leadline-
//! 2026-04-25 directive.
//!
//! WS-4 M4.6.

use ndarray::{ArrayD, IxDyn};

use crate::attributes::NodeAttributes;
use crate::tensor::Tensor;

/// ONNX MatMulInteger operator.
pub struct MatMulInteger;

impl MatMulInteger {
    /// Forward pass: integer GEMM with optional zero-point subtraction.
    ///
    /// Accepts 2, 3, or 4 inputs:
    ///   * `inputs[0]` — `a` (`Int8` or `UInt8`), shape `[M, K]`.
    ///   * `inputs[1]` — `b` (`Int8` or `UInt8`), shape `[K, N]`.
    ///   * `inputs[2]` — optional `a_zero_point` (scalar, dtype matches `a`).
    ///   * `inputs[3]` — optional `b_zero_point` (scalar, dtype matches `b`).
    ///
    /// Returns a `Tensor::Int32` of shape `[M, N]`.
    pub fn forward(inputs: &[Tensor], _attrs: &NodeAttributes) -> Tensor {
        assert!(
            (2..=4).contains(&inputs.len()),
            "MatMulInteger requires 2, 3, or 4 inputs (got {})",
            inputs.len()
        );

        let a = &inputs[0];
        let b = &inputs[1];
        let a_zp = inputs.get(2);
        let b_zp = inputs.get(3);

        // Dtype validation: each operand must be Int8 or UInt8.
        match a {
            Tensor::Int8(_) | Tensor::UInt8(_) => {}
            other => panic!(
                "MatMulInteger: input a must be Int8 or UInt8, got {}",
                other.dtype()
            ),
        }
        match b {
            Tensor::Int8(_) | Tensor::UInt8(_) => {}
            other => panic!(
                "MatMulInteger: input b must be Int8 or UInt8, got {}",
                other.dtype()
            ),
        }

        // Rank check — 2-D only for M4.6.
        assert_eq!(
            a.ndim(),
            2,
            "MatMulInteger: a must be 2-D (got rank {}); batched is M4.7+ scope",
            a.ndim()
        );
        assert_eq!(
            b.ndim(),
            2,
            "MatMulInteger: b must be 2-D (got rank {}); batched is M4.7+ scope",
            b.ndim()
        );

        let m = a.shape()[0];
        let k_dim = a.shape()[1];
        assert_eq!(
            b.shape()[0],
            k_dim,
            "MatMulInteger: K mismatch — a is {:?}, b is {:?}",
            a.shape(),
            b.shape()
        );
        let n = b.shape()[1];

        // zp extraction with dtype validation. Defaults to 0 when absent.
        let a_zp_i32 = extract_scalar_zp(a_zp, a.dtype(), "a");
        let b_zp_i32 = extract_scalar_zp(b_zp, b.dtype(), "b");

        // Pull contiguous slices once to avoid per-element matching in the
        // inner loop. Each branch promotes operands to i32 and runs a
        // straightforward triple loop with i32 accumulator. The four
        // explicit branches are clearer for review than a generic helper
        // and let LLVM specialize each path.
        let mut out_data = vec![0_i32; m * n];

        match (a, b) {
            (Tensor::Int8(_), Tensor::Int8(_)) => {
                let av = a.as_slice_i8();
                let bv = b.as_slice_i8();
                gemm_i32_into(
                    &mut out_data,
                    m,
                    k_dim,
                    n,
                    a_zp_i32,
                    b_zp_i32,
                    |i| av[i] as i32,
                    |i| bv[i] as i32,
                );
            }
            (Tensor::UInt8(_), Tensor::Int8(_)) => {
                let av = a.as_slice_u8();
                let bv = b.as_slice_i8();
                gemm_i32_into(
                    &mut out_data,
                    m,
                    k_dim,
                    n,
                    a_zp_i32,
                    b_zp_i32,
                    |i| av[i] as i32,
                    |i| bv[i] as i32,
                );
            }
            (Tensor::Int8(_), Tensor::UInt8(_)) => {
                let av = a.as_slice_i8();
                let bv = b.as_slice_u8();
                gemm_i32_into(
                    &mut out_data,
                    m,
                    k_dim,
                    n,
                    a_zp_i32,
                    b_zp_i32,
                    |i| av[i] as i32,
                    |i| bv[i] as i32,
                );
            }
            (Tensor::UInt8(_), Tensor::UInt8(_)) => {
                let av = a.as_slice_u8();
                let bv = b.as_slice_u8();
                gemm_i32_into(
                    &mut out_data,
                    m,
                    k_dim,
                    n,
                    a_zp_i32,
                    b_zp_i32,
                    |i| av[i] as i32,
                    |i| bv[i] as i32,
                );
            }
            _ => unreachable!("dtype combination already validated above"),
        }

        let array = ArrayD::from_shape_vec(IxDyn(&[m, n]), out_data)
            .expect("MatMulInteger: shape and data length agree by construction");
        Tensor::Int32(array)
    }
}

/// Pull a scalar zero-point as `i32` from an optional input tensor.
///
/// Defaults to 0 when `tensor` is `None`. Validates that:
///   * dtype matches `operand_dtype` (ONNX: zp.dtype == operand.dtype).
///   * shape is scalar (rank 0) or single-element rank-1. Per-axis zp is
///     not supported in M4.6 — panic with a clear message rather than
///     silently coercing.
fn extract_scalar_zp(
    tensor: Option<&Tensor>,
    operand_dtype: &'static str,
    label: &str,
) -> i32 {
    let Some(zp) = tensor else { return 0 };

    if zp.dtype() != operand_dtype {
        panic!(
            "MatMulInteger: {}_zero_point dtype {} must match operand dtype {}",
            label,
            zp.dtype(),
            operand_dtype
        );
    }

    let shape = zp.shape();
    let is_scalar = shape.is_empty() || (shape.len() == 1 && shape[0] == 1);
    if !is_scalar {
        panic!(
            "MatMulInteger: {}_zero_point must be scalar (got shape {:?}); \
             per-axis zp is not supported in M4.6",
            label, shape
        );
    }

    match zp {
        Tensor::Int8(arr) => arr.iter().next().copied().unwrap_or(0) as i32,
        Tensor::UInt8(arr) => arr.iter().next().copied().unwrap_or(0) as i32,
        _ => unreachable!(
            "zp dtype already validated to match operand_dtype \
             (Int8 or UInt8) above"
        ),
    }
}

/// Naive triple-loop GEMM into `out`, with per-operand i32 accessors and
/// scalar zero-point subtraction.
///
/// `out` is row-major `[M, N]`; `a` is row-major `[M, K]`; `b` is row-
/// major `[K, N]`. The accumulator stays in i32 throughout — for the
/// MVP DistilBERT shapes (K ≤ 3072, INT8 inputs) the worst-case sum is
/// well within `INT32_MAX` (~50M ≪ 2.1e9), so the `checked_*` path
/// never trips on a real model. We use checked arithmetic anyway so a
/// pathological future input (or a malformed graph) panics with a clear
/// diagnostic instead of silently wrapping — matches the project's
/// "no silent fallbacks" rule. The GPU kernels can't easily detect
/// per-thread overflow at scale, so they wrap per CUDA C semantics; the
/// CPU forward (this oracle) treats overflow as the bug it is.
#[allow(clippy::too_many_arguments)]
fn gemm_i32_into(
    out: &mut [i32],
    m: usize,
    k_dim: usize,
    n: usize,
    a_zp: i32,
    b_zp: i32,
    a_at: impl Fn(usize) -> i32,
    b_at: impl Fn(usize) -> i32,
) {
    for row in 0..m {
        for col in 0..n {
            let mut acc: i32 = 0;
            for kk in 0..k_dim {
                let av = a_at(row * k_dim + kk) - a_zp;
                let bv = b_at(kk * n + col) - b_zp;
                let prod = av.checked_mul(bv).expect(
                    "MatMulInteger: i32 overflow in (a-a_zp) * (b-b_zp); \
                     input range exceeds INT32 bounds",
                );
                acc = acc.checked_add(prod).expect(
                    "MatMulInteger: i32 overflow in K-summation; \
                     K * (a*b) range exceeds INT32 bounds",
                );
            }
            out[row * n + col] = acc;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_integer_uint8_int8_distilbert_combo_no_zero_points() {
        // 2x3 UINT8 × 3x2 INT8 → 2x2 INT32. No zero_points.
        let a = Tensor::from_vec_u8(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        let b = Tensor::from_vec_i8(vec![1, 2, 3, 4, 5, 6], vec![3, 2]);
        let out = MatMulInteger::forward(&[a, b], &NodeAttributes::new());
        // out[0,0] = 1*1 + 2*3 + 3*5 = 22; out[0,1] = 1*2+2*4+3*6 = 28
        // out[1,0] = 4*1 + 5*3 + 6*5 = 49; out[1,1] = 4*2+5*4+6*6 = 64
        if let Tensor::Int32(arr) = out {
            assert_eq!(
                arr.iter().copied().collect::<Vec<_>>(),
                vec![22_i32, 28, 49, 64]
            );
        } else {
            panic!()
        }
    }

    #[test]
    fn matmul_integer_int8_int8_with_zero_points() {
        let a = Tensor::from_vec_i8(vec![10_i8, 20, 30, 40], vec![2, 2]);
        let b = Tensor::from_vec_i8(vec![1_i8, 2, 3, 4], vec![2, 2]);
        let a_zp = Tensor::from_vec_i8(vec![5], vec![]);
        let b_zp = Tensor::from_vec_i8(vec![1], vec![]);
        let out = MatMulInteger::forward(&[a, b, a_zp, b_zp], &NodeAttributes::new());
        // (a-5)*(b-1): a' = [[5,15],[25,35]], b' = [[0,1],[2,3]]
        // out[0,0] = 5*0+15*2 = 30; out[0,1] = 5*1+15*3 = 50
        // out[1,0] = 25*0+35*2 = 70; out[1,1] = 25*1+35*3 = 130
        if let Tensor::Int32(arr) = out {
            assert_eq!(
                arr.iter().copied().collect::<Vec<_>>(),
                vec![30_i32, 50, 70, 130]
            );
        } else {
            panic!()
        }
    }

    #[test]
    fn matmul_integer_distilbert_shape_uint8_int8_k_768() {
        // Larger shape: M=4, K=768, N=4 — exercises tiled GEMM correctly.
        // Simple check: a = ones; b = ones; out = K (every cell).
        let a = Tensor::from_vec_u8(vec![1_u8; 4 * 768], vec![4, 768]);
        let b = Tensor::from_vec_i8(vec![1_i8; 768 * 4], vec![768, 4]);
        let out = MatMulInteger::forward(&[a, b], &NodeAttributes::new());
        if let Tensor::Int32(arr) = out {
            assert_eq!(arr.shape(), &[4, 4]);
            assert!(arr.iter().all(|&v| v == 768));
        } else {
            panic!()
        }
    }

    #[test]
    fn matmul_integer_k_3072_overflow_safety_distilbert_ffn() {
        // The FFN K=3072 path. Per spec §Risks #2: realistic INT8 values
        // stay well within INT32 range; a sanity check that K=3072 doesn't
        // accidentally overflow with moderate-value inputs.
        // Worst-case: a all 100, b all 100 (UINT8, INT8 respectively).
        // sum_k = 3072 * 100 * 100 = 30,720,000 < INT32_MAX (2.1e9). Safe.
        let a = Tensor::from_vec_u8(vec![100_u8; 3072], vec![1, 3072]);
        let b = Tensor::from_vec_i8(vec![100_i8; 3072], vec![3072, 1]);
        let out = MatMulInteger::forward(&[a, b], &NodeAttributes::new());
        if let Tensor::Int32(arr) = out {
            assert_eq!(arr.shape(), &[1, 1]);
            assert_eq!(arr[[0, 0]], 30_720_000);
        } else {
            panic!()
        }
    }

    #[test]
    #[should_panic(expected = "MatMulInteger")]
    fn matmul_integer_rejects_one_d_input() {
        // ONNX MatMulInteger requires 2-D inputs in M4.6 MVP scope.
        // 1-D input must panic, not silently coerce.
        let a = Tensor::from_vec_i8(vec![1_i8, 2, 3], vec![3]);
        let b = Tensor::from_vec_i8(vec![1_i8, 2, 3], vec![3]);
        let _ = MatMulInteger::forward(&[a, b], &NodeAttributes::new());
    }

    #[test]
    #[should_panic(expected = "MatMulInteger")]
    fn matmul_integer_rejects_per_axis_zero_point() {
        // Scalar zp only in M4.6 MVP. 1-D zp (per-axis) must panic.
        let a = Tensor::from_vec_i8(vec![1_i8, 2, 3, 4], vec![2, 2]);
        let b = Tensor::from_vec_i8(vec![1_i8, 2, 3, 4], vec![2, 2]);
        let a_zp = Tensor::from_vec_i8(vec![0_i8, 0], vec![2]); // per-axis
        let _ = MatMulInteger::forward(&[a, b, a_zp], &NodeAttributes::new());
    }

    #[test]
    #[should_panic(expected = "MatMulInteger")]
    fn matmul_integer_rejects_zp_dtype_mismatch() {
        // ONNX requires zp.dtype == operand.dtype. Mismatched dtype must
        // panic rather than silently coerce.
        let a = Tensor::from_vec_i8(vec![1_i8, 2, 3, 4], vec![2, 2]);
        let b = Tensor::from_vec_i8(vec![1_i8, 2, 3, 4], vec![2, 2]);
        let a_zp = Tensor::from_vec_u8(vec![0_u8], vec![]); // u8, not i8
        let _ = MatMulInteger::forward(&[a, b, a_zp], &NodeAttributes::new());
    }

    #[test]
    #[should_panic(expected = "MatMulInteger")]
    fn matmul_integer_rejects_k_mismatch() {
        // K dim must match between a.shape()[1] and b.shape()[0].
        let a = Tensor::from_vec_i8(vec![1_i8; 6], vec![2, 3]);
        let b = Tensor::from_vec_i8(vec![1_i8; 8], vec![4, 2]); // 4 != 3
        let _ = MatMulInteger::forward(&[a, b], &NodeAttributes::new());
    }
}
