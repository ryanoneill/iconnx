//! WS-3.5 Y(3) — BFloat16 MatMul integration tests.
//!
//! Exercises `gpu_matmul` and `gpu_batched_matmul` on the BFloat16 path.
//! Mirrors `tests/matmul_fp16_test.rs` (WS-3 M3.6) one-for-one with the
//! `half::f16` → `half::bf16` substitution and a `1e-2` numerical
//! tolerance reflecting BF16's 7-bit mantissa (vs FP16's 10).
//!
//! The matmul-2D kernel routes to garboard `gemm_bf16` (cuBLAS
//! `CUBLAS_COMPUTE_32F`, sm_80+ HMMA tensor cores). The batched paths
//! fall back to a per-batch loop because `gemm_strided_batched_bf16`
//! is not yet exposed by garboard (Tracked Follow-Up #2 — perf gap, not
//! correctness gap).
//!
//! Hardware-gated: BF16 device kernels require sm_80 (Ampere). Tests
//! short-circuit cleanly on older hardware and are `#[ignore]`-marked
//! so they only run under `--include-ignored` on a CUDA host.

#![cfg(feature = "cuda")]

use half::bf16;
use iconnx::cuda::{gpu_batched_matmul, gpu_matmul, GpuMemoryPool, GpuTensor, IconnxCudaContext};

/// 2×3 × 3×4 BF16 matmul against a sparse identity-padded RHS so the
/// expected output is exact in BF16 (no rounding) and the assertion
/// can use a tight tolerance.
#[test]
#[ignore = "requires CUDA GPU"]
fn matmul_bfloat16_2x3_x_3x4() {
    let ctx = IconnxCudaContext::new().expect("ctx");
    if ctx.compute_capability().0 < 8 {
        return;
    }
    let mut pool = GpuMemoryPool::new();

    // A = [[1, 2, 3], [4, 5, 6]] (2×3)
    let a: Vec<bf16> = (1..=6).map(|i| bf16::from_f32(i as f32)).collect();
    // B = [[1,0,0,0],[0,1,0,0],[0,0,1,0]] (3×4) — identity-with-zero-pad
    let b: Vec<bf16> = vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
    ]
    .into_iter()
    .map(bf16::from_f32)
    .collect();

    let ga = GpuTensor::from_host_bf16(&ctx, &a, vec![2, 3]).unwrap();
    let gb = GpuTensor::from_host_bf16(&ctx, &b, vec![3, 4]).unwrap();
    let gc = gpu_matmul(&ctx, &mut pool, &ga, &gb).expect("matmul");
    assert_eq!(gc.shape(), &[2, 4]);

    let c = gc.to_host_bf16(&ctx).expect("download");
    // A @ B = [[1,2,3,0],[4,5,6,0]]
    let expected: Vec<bf16> = vec![1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0, 0.0]
        .into_iter()
        .map(bf16::from_f32)
        .collect();
    for (i, (got, want)) in c.iter().zip(expected.iter()).enumerate() {
        let diff = (got.to_f32() - want.to_f32()).abs();
        assert!(diff < 1e-2, "[{i}] got {got:?} want {want:?}");
    }
}

/// Mixed-dtype matmul (BF16 × FP16) must error loudly with
/// `CudaError::DtypeMismatch` (or carry both dtype names) — never silently
/// fall through to an f32 path. Mirrors the FP16/FP32 mismatch test.
#[test]
#[ignore = "requires CUDA GPU"]
fn matmul_bfloat16_dtype_mismatch_errors_loudly() {
    let ctx = IconnxCudaContext::new().expect("ctx");
    if ctx.compute_capability().0 < 8 {
        return;
    }
    let mut pool = GpuMemoryPool::new();

    let a: Vec<bf16> = vec![bf16::from_f32(1.0); 4];
    let b: Vec<half::f16> = vec![half::f16::from_f32(1.0); 4];
    let ga = GpuTensor::from_host_bf16(&ctx, &a, vec![2, 2]).unwrap();
    let gb = GpuTensor::from_host_f16(&ctx, &b, vec![2, 2]).unwrap();
    let result = gpu_matmul(&ctx, &mut pool, &ga, &gb);
    let err = match result {
        Ok(_) => panic!("BF16 × FP16 must error — no silent f32 fallthrough"),
        Err(e) => e,
    };
    let msg = format!("{err:?}");
    assert!(
        msg.contains("DtypeMismatch") || msg.contains("bfloat16") || msg.contains("float16"),
        "error must mention dtype mismatch: {msg}"
    );
}

/// Batched BF16 matmul: 2 batches × 2×3 × 3×2. Exercises the per-batch
/// loop fallback (garboard's `gemm_strided_batched_bf16` is not yet
/// available — Follow-Up #2). Numerically equivalent to the f32 path.
#[test]
#[ignore = "requires CUDA GPU"]
fn batched_matmul_bfloat16_per_batch_loop() {
    let ctx = IconnxCudaContext::new().expect("ctx");
    if ctx.compute_capability().0 < 8 {
        return;
    }
    let mut pool = GpuMemoryPool::new();

    // 2 batches, A_i = [[1,2,3],[4,5,6]], B_i = identity-pad as above.
    let a_one: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_one: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // 3×2 identity-pad
    let a_data: Vec<bf16> = [a_one.as_slice(), a_one.as_slice()]
        .concat()
        .into_iter()
        .map(bf16::from_f32)
        .collect();
    let b_data: Vec<bf16> = [b_one.as_slice(), b_one.as_slice()]
        .concat()
        .into_iter()
        .map(bf16::from_f32)
        .collect();

    let ga = GpuTensor::from_host_bf16(&ctx, &a_data, vec![2, 2, 3]).unwrap();
    let gb = GpuTensor::from_host_bf16(&ctx, &b_data, vec![2, 3, 2]).unwrap();
    let gc = gpu_batched_matmul(&ctx, &mut pool, &ga, &gb).expect("batched matmul");
    assert_eq!(gc.shape(), &[2, 2, 2]);

    let c = gc.to_host_bf16(&ctx).expect("download");
    // Each batch: A @ B = [[1,2],[4,5]]
    let expected: Vec<bf16> = vec![1.0, 2.0, 4.0, 5.0, 1.0, 2.0, 4.0, 5.0]
        .into_iter()
        .map(bf16::from_f32)
        .collect();
    for (i, (got, want)) in c.iter().zip(expected.iter()).enumerate() {
        let diff = (got.to_f32() - want.to_f32()).abs();
        assert!(diff < 1e-2, "[{i}] got {got:?} want {want:?}");
    }
}
