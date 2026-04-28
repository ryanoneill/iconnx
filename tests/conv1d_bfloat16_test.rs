//! WS-3.5 Y(3) — BFloat16 Conv1D integration tests.
//!
//! Exercises `gpu_conv1d` (im2col + GEMM via `gemm_bf16`) on the BF16
//! path. Mirrors `tests/conv_fp16_test.rs` (WS-3 M3.6) one-for-one with
//! `half::f16` → `half::bf16` substitution.
//!
//! The 1D conv pipeline is:
//!   im2col_1d_bf16_kernel → gpu_gemm (BF16 → gemm_bf16) → add_bias_bf16_kernel
//!
//! Coverage:
//! - identity_kernel: BF16 1D conv with kernel=[1.0] preserves input
//!   exactly (lossless on representable values).
//! - smoothing_kernel: BF16 1D conv with K=3 averaging-style kernel
//!   exercises the f32-accumulator promotion path (BF16's 7-bit mantissa
//!   would accumulate visible error without f32 promotion).
//!
//! Note: a grouped-conv-rejection test was considered but skipped — it
//! lives at `executor/conv.rs::dispatch_conv` (plain dispatch logic, no
//! async surface to test), and the FP16 sibling has no equivalent test
//! to mirror. The smoothing test is a more numerically-valuable
//! substitute that exercises a non-trivial K-dim reduction.

#![cfg(feature = "cuda")]

use half::bf16;
use iconnx::cuda::{
    gpu_conv1d, Conv1dParams, ConvKernelCache, GpuMemoryPool, GpuTensor, IconnxCudaContext,
};

/// Identity kernel ([1.0]) on a [1, 1, 3] BF16 input → output equals
/// input. Smallest possible BF16 1D conv that still walks every kernel
/// in the pipeline (im2col_1d_bf16 → gemm_bf16 → no bias-add).
#[test]
#[ignore = "requires CUDA GPU"]
fn conv1d_bfloat16_identity_kernel() {
    let ctx = IconnxCudaContext::new().expect("ctx");
    if ctx.compute_capability().0 < 8 {
        return;
    }
    let cache = ConvKernelCache::new(&ctx).expect("conv kernel cache");
    let mut pool = GpuMemoryPool::new();

    let input: Vec<bf16> = vec![1.0, 2.0, 3.0]
        .into_iter()
        .map(bf16::from_f32)
        .collect();
    let weight: Vec<bf16> = vec![bf16::from_f32(1.0)];
    let bias: Vec<bf16> = vec![bf16::from_f32(0.0)];

    let g_in = GpuTensor::from_host_bf16(&ctx, &input, vec![1, 1, 3]).unwrap();
    let g_w = GpuTensor::from_host_bf16(&ctx, &weight, vec![1, 1, 1]).unwrap();
    let g_b = GpuTensor::from_host_bf16(&ctx, &bias, vec![1]).unwrap();

    let params = Conv1dParams {
        kernel_size: 1,
        stride: 1,
        padding: 0,
        dilation: 1,
    };
    let out = gpu_conv1d(&ctx, &cache, &mut pool, &g_in, &g_w, Some(&g_b), &params)
        .expect("conv1d");
    assert_eq!(out.shape(), &[1, 1, 3]);

    let back = out.to_host_bf16(&ctx).expect("download");
    let expected: Vec<bf16> = vec![1.0, 2.0, 3.0]
        .into_iter()
        .map(bf16::from_f32)
        .collect();
    for (i, (got, want)) in back.iter().zip(expected.iter()).enumerate() {
        let diff = (got.to_f32() - want.to_f32()).abs();
        assert!(diff < 1e-2, "[{i}] got {got:?} want {want:?}");
    }
}

/// 1D conv over a 5-tap input with a 3-tap [1, 2, 1] kernel — exercises
/// the gemm_bf16 path on a non-trivial reduction so any kernel-side
/// dtype mismatch lights up. Tolerance widened to 5e-2 to account for
/// BF16's 7-bit mantissa across the K=3 reduction.
#[test]
#[ignore = "requires CUDA GPU"]
fn conv1d_bfloat16_smoothing_kernel() {
    let ctx = IconnxCudaContext::new().expect("ctx");
    if ctx.compute_capability().0 < 8 {
        return;
    }
    let cache = ConvKernelCache::new(&ctx).expect("conv kernel cache");
    let mut pool = GpuMemoryPool::new();

    // Input [1, 1, 5] = [1, 2, 3, 4, 5]; kernel [1, 1, 3] = [1, 2, 1];
    // expected output (no padding) = [1·1+2·2+3·1, 2·1+3·2+4·1, 3·1+4·2+5·1]
    //                              = [8, 12, 16]
    let input: Vec<bf16> = (1..=5).map(|i| bf16::from_f32(i as f32)).collect();
    let weight: Vec<bf16> = vec![1.0, 2.0, 1.0]
        .into_iter()
        .map(bf16::from_f32)
        .collect();

    let g_in = GpuTensor::from_host_bf16(&ctx, &input, vec![1, 1, 5]).unwrap();
    let g_w = GpuTensor::from_host_bf16(&ctx, &weight, vec![1, 1, 3]).unwrap();

    let params = Conv1dParams {
        kernel_size: 3,
        stride: 1,
        padding: 0,
        dilation: 1,
    };
    let out = gpu_conv1d(&ctx, &cache, &mut pool, &g_in, &g_w, None, &params)
        .expect("conv1d");
    assert_eq!(out.shape(), &[1, 1, 3]);

    let back = out.to_host_bf16(&ctx).expect("download");
    let expected = [8.0_f32, 12.0, 16.0];
    for (i, (got, want)) in back.iter().zip(expected.iter()).enumerate() {
        let diff = (got.to_f32() - want).abs();
        assert!(diff < 5e-2, "[{i}] got {got:?} want {want}");
    }
}
