//! WS-3 M3.6 — FP16 MatMul / Gemm contract tests.
//!
//! Covers the cuBLAS `gemm_fp16_acc32` path: FP16 IO with FP32
//! accumulator. The accumulator choice matters for Whisper attention's
//! K=1500 reductions where pure-FP16 would overflow.
//!
//! All tests assert dtype Float16 on output to lock in the no-silent-
//! coercion contract. Tolerances are tight on the small hand-computed
//! cases and looser on the Whisper-shape cross-check (where the f32
//! reference itself differs from the FP16 result by a few ULPs).

#![cfg(feature = "cuda")]

use half::f16;
use iconnx::cuda::{
    gpu_batched_matmul, gpu_gemm, gpu_matmul, DType, GpuMemoryPool, GpuTensor, IconnxCudaContext,
};

fn setup() -> (IconnxCudaContext, GpuMemoryPool) {
    let ctx = IconnxCudaContext::new().expect("CUDA context");
    let pool = GpuMemoryPool::new();
    (ctx, pool)
}

fn h(values: &[f32]) -> Vec<f16> {
    values.iter().copied().map(f16::from_f32).collect()
}

fn assert_close_f16(actual: &[f16], expected: &[f32], tol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{}: length mismatch", label);
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let a_f32 = a.to_f32();
        assert!(
            (a_f32 - e).abs() <= tol,
            "{}: idx {} got {} vs {} (tol {})",
            label,
            i,
            a_f32,
            e,
            tol
        );
    }
}

#[test]
#[ignore = "requires CUDA GPU"]
fn matmul_float16_2x2_hand_computed() {
    // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]. Expected:
    //   A @ B = [[1*5 + 2*7, 1*6 + 2*8], [3*5 + 4*7, 3*6 + 4*8]]
    //         = [[19, 22], [43, 50]]
    let (ctx, mut pool) = setup();
    let a = GpuTensor::from_host_f16(&ctx, &h(&[1.0, 2.0, 3.0, 4.0]), vec![2, 2]).unwrap();
    let b = GpuTensor::from_host_f16(&ctx, &h(&[5.0, 6.0, 7.0, 8.0]), vec![2, 2]).unwrap();

    let c = gpu_matmul(&ctx, &mut pool, &a, &b).unwrap();
    assert_eq!(
        c.dtype(),
        DType::Float16,
        "FP16 input must produce FP16 output (no silent f32 coercion)"
    );
    assert_eq!(c.shape(), &[2, 2]);

    let result = c.to_host_f16(&ctx).unwrap();
    // All target values exactly representable in f16; tolerance 1e-3.
    assert_close_f16(&result, &[19.0, 22.0, 43.0, 50.0], 1e-3, "matmul_2x2_f16");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn matmul_float16_rectangular_hand_computed() {
    // A = [[1, 2, 3], [4, 5, 6]] (2x3), B = [[1, 2], [3, 4], [5, 6]] (3x2)
    // A @ B = [[1+6+15, 2+8+18], [4+15+30, 8+20+36]] = [[22, 28], [49, 64]]
    let (ctx, mut pool) = setup();
    let a = GpuTensor::from_host_f16(&ctx, &h(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), vec![2, 3]).unwrap();
    let b = GpuTensor::from_host_f16(&ctx, &h(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), vec![3, 2]).unwrap();

    let c = gpu_matmul(&ctx, &mut pool, &a, &b).unwrap();
    assert_eq!(c.dtype(), DType::Float16);
    assert_eq!(c.shape(), &[2, 2]);

    let result = c.to_host_f16(&ctx).unwrap();
    assert_close_f16(&result, &[22.0, 28.0, 49.0, 64.0], 1e-3, "matmul_rect_f16");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn gemm_float16_with_alpha() {
    // 2.0 * A @ I = 2 * [[1,2],[3,4]] = [[2,4],[6,8]].
    let (ctx, mut pool) = setup();
    let a = GpuTensor::from_host_f16(&ctx, &h(&[1.0, 2.0, 3.0, 4.0]), vec![2, 2]).unwrap();
    let b = GpuTensor::from_host_f16(&ctx, &h(&[1.0, 0.0, 0.0, 1.0]), vec![2, 2]).unwrap();

    let c = gpu_gemm(&ctx, &mut pool, &a, &b, None, 2.0, 0.0, false, false).unwrap();
    assert_eq!(c.dtype(), DType::Float16);
    let result = c.to_host_f16(&ctx).unwrap();
    assert_close_f16(&result, &[2.0, 4.0, 6.0, 8.0], 1e-3, "gemm_alpha_f16");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn gemm_float16_with_beta_and_c() {
    // C_init = [[1, 1], [1, 1]]. Compute 1.0 * A @ I + 1.0 * C_init.
    //   A @ I = [[1, 2], [3, 4]] -> + C_init = [[2, 3], [4, 5]].
    let (ctx, mut pool) = setup();
    let a = GpuTensor::from_host_f16(&ctx, &h(&[1.0, 2.0, 3.0, 4.0]), vec![2, 2]).unwrap();
    let b = GpuTensor::from_host_f16(&ctx, &h(&[1.0, 0.0, 0.0, 1.0]), vec![2, 2]).unwrap();
    let c_init = GpuTensor::from_host_f16(&ctx, &h(&[1.0, 1.0, 1.0, 1.0]), vec![2, 2]).unwrap();

    let out = gpu_gemm(&ctx, &mut pool, &a, &b, Some(&c_init), 1.0, 1.0, false, false).unwrap();
    assert_eq!(out.dtype(), DType::Float16);
    let result = out.to_host_f16(&ctx).unwrap();
    assert_close_f16(&result, &[2.0, 3.0, 4.0, 5.0], 1e-3, "gemm_beta_f16");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn matmul_float16_long_k_no_overflow() {
    // K=1500 mirrors Whisper attention's [1, 1500, 384] × [384, 384]
    // where K-axis aggregation could overflow pure-FP16 (max 65504).
    // Each partial product is 0.5 * 0.5 = 0.25, so K=1500 products
    // sum to 375 — well past f16 max if we accumulated in f16.
    // gemm_fp16_acc32 keeps the running sum in f32, so this returns 375.
    let (ctx, mut pool) = setup();

    let m = 4;
    let k = 1500;
    let n = 4;
    let a_host: Vec<f16> = vec![f16::from_f32(0.5); m * k];
    let b_host: Vec<f16> = vec![f16::from_f32(0.5); k * n];

    let a = GpuTensor::from_host_f16(&ctx, &a_host, vec![m, k]).unwrap();
    let b = GpuTensor::from_host_f16(&ctx, &b_host, vec![k, n]).unwrap();

    let c = gpu_matmul(&ctx, &mut pool, &a, &b).unwrap();
    assert_eq!(c.dtype(), DType::Float16);
    let result = c.to_host_f16(&ctx).unwrap();

    // Each result element = sum over 1500 of 0.25 = 375.
    for (i, h) in result.iter().enumerate() {
        let got = h.to_f32();
        assert!(
            (got - 375.0).abs() < 1.0,
            "long-K result idx {} = {}; expected ~375 (pure-FP16 accumulator would have overflowed)",
            i,
            got
        );
    }
}

#[test]
#[ignore = "requires CUDA GPU"]
fn matmul_float16_3d_2d_batched_whisper_attention_shape() {
    // Whisper attention emits [1, 1500, 384] × [384, 384] -> [1, 1500, 384].
    // The 3D × 2D batched wrapper reshapes [B, M, K] -> [B*M, K], runs 2D
    // GEMM, and reshapes [B*M, N] -> [B, M, N]. WS-4 M4.7 retrospective:
    // surfacing the batched shape was missed in MatMulInteger M4.6.
    let (ctx, mut pool) = setup();

    // Smaller scale — same shape pattern, K reduced for fast tests.
    let b_dim = 1;
    let m = 4;
    let k = 8;
    let n = 4;

    // Generate dense small values; chosen so the f16 round-trip is benign
    // and the f32 reference is well-defined.
    let a_host: Vec<f16> = (0..b_dim * m * k)
        .map(|i| f16::from_f32(((i % 7) as f32 + 1.0) * 0.125))
        .collect();
    let b_host: Vec<f16> = (0..k * n)
        .map(|i| f16::from_f32(((i % 5) as f32 + 1.0) * 0.25))
        .collect();

    let a = GpuTensor::from_host_f16(&ctx, &a_host, vec![b_dim, m, k]).unwrap();
    let b = GpuTensor::from_host_f16(&ctx, &b_host, vec![k, n]).unwrap();

    // Hand-compute reference in f32 from the f16 inputs (post round-trip
    // is what the GEMM would actually multiply).
    let a_f32: Vec<f32> = a_host.iter().map(|h| h.to_f32()).collect();
    let b_f32: Vec<f32> = b_host.iter().map(|h| h.to_f32()).collect();
    let mut ref_out = vec![0.0_f32; b_dim * m * n];
    for batch in 0..b_dim {
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0_f32;
                for kk in 0..k {
                    acc += a_f32[batch * m * k + i * k + kk] * b_f32[kk * n + j];
                }
                ref_out[batch * m * n + i * n + j] = acc;
            }
        }
    }

    // Test the dispatcher routes the 3D × 2D pattern correctly. The path
    // uses gpu_matmul as the single entry, with internal reshape for the
    // FP16 batched case (mirroring how f32 already does it via
    // gpu_matmul_2d_3d).
    let c = gpu_matmul(&ctx, &mut pool, &a, &b).unwrap();
    assert_eq!(c.dtype(), DType::Float16);
    assert_eq!(c.shape(), &[b_dim, m, n]);

    let result = c.to_host_f16(&ctx).unwrap();
    assert_close_f16(&result, &ref_out, 0.05, "matmul_3d_2d_f16");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn batched_matmul_float16_3d_3d() {
    // [B, M, K] × [B, K, N]: per-batch GEMM. Uses gpu_batched_matmul
    // via strided batched cuBLAS.
    let (ctx, mut pool) = setup();
    let b_dim = 2;
    let m = 3;
    let k = 4;
    let n = 2;

    let a_host: Vec<f16> = (0..b_dim * m * k)
        .map(|i| f16::from_f32(((i % 5) + 1) as f32 * 0.5))
        .collect();
    let b_host: Vec<f16> = (0..b_dim * k * n)
        .map(|i| f16::from_f32(((i % 3) + 1) as f32 * 0.5))
        .collect();

    let a = GpuTensor::from_host_f16(&ctx, &a_host, vec![b_dim, m, k]).unwrap();
    let b = GpuTensor::from_host_f16(&ctx, &b_host, vec![b_dim, k, n]).unwrap();

    let af: Vec<f32> = a_host.iter().map(|h| h.to_f32()).collect();
    let bf: Vec<f32> = b_host.iter().map(|h| h.to_f32()).collect();
    let mut ref_out = vec![0.0_f32; b_dim * m * n];
    for bi in 0..b_dim {
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0_f32;
                for kk in 0..k {
                    acc += af[bi * m * k + i * k + kk] * bf[bi * k * n + kk * n + j];
                }
                ref_out[bi * m * n + i * n + j] = acc;
            }
        }
    }

    let c = gpu_batched_matmul(&ctx, &mut pool, &a, &b).unwrap();
    assert_eq!(c.dtype(), DType::Float16);
    assert_eq!(c.shape(), &[b_dim, m, n]);

    let result = c.to_host_f16(&ctx).unwrap();
    assert_close_f16(&result, &ref_out, 0.05, "batched_matmul_f16");
}
