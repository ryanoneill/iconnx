//! WS-3.5 Y(2) sub-2c — BFloat16 LayerNorm with f32 accumulator.
//!
//! Inputs/outputs BF16; gamma/beta stay in f32 per spec §2 (no FP16
//! LayerNorm to mirror; principled choice for a wider-exponent dtype
//! is f32 affine params).

#![cfg(feature = "cuda")]

use half::bf16;
use iconnx::cuda::{
    gpu_layer_norm, GpuMemoryPool, GpuTensor, IconnxCudaContext, ReductionKernelCache,
};

#[test]
#[ignore = "requires CUDA GPU"]
fn layer_norm_bfloat16_zero_mean_unit_var() {
    let ctx = IconnxCudaContext::new().expect("ctx");
    if ctx.compute_capability().0 < 8 {
        return;
    }
    let kernels = ReductionKernelCache::new(&ctx).expect("kernels");
    let mut pool = GpuMemoryPool::new();

    // Input row [-1, 0, 1] → mean=0, var=2/3, normalized ≈ [-1.225, 0, 1.225].
    // gamma=[1,1,1], beta=[0,0,0] — output equals normalized.
    let host: Vec<bf16> = vec![-1.0_f32, 0.0, 1.0]
        .into_iter()
        .map(bf16::from_f32)
        .collect();
    let g = GpuTensor::from_host_bf16(&ctx, &host, vec![1, 3]).expect("upload");
    let gamma_host: Vec<f32> = vec![1.0, 1.0, 1.0];
    let beta_host: Vec<f32> = vec![0.0, 0.0, 0.0];
    let gamma = GpuTensor::from_host_f32(&ctx, &gamma_host, vec![3]).expect("upload gamma");
    let beta = GpuTensor::from_host_f32(&ctx, &beta_host, vec![3]).expect("upload beta");

    let result = gpu_layer_norm(&ctx, &kernels, &mut pool, &g, &gamma, &beta, 1e-5)
        .expect("layer_norm");
    let back = result.to_host_bf16(&ctx).expect("download");

    let expected = [-1.2247_f32, 0.0, 1.2247];
    for (got, want) in back.iter().zip(expected.iter()) {
        let diff = (got.to_f32() - *want).abs();
        // BF16 LayerNorm tolerance accounts for f32-accumulator round-trip
        // plus rsqrtf approximation; 5e-2 is the standard for reductions.
        assert!(diff < 5e-2, "got {got:?} want {want} (BF16 5e-2 reduction tol)");
    }
}

#[test]
#[ignore = "requires CUDA GPU"]
fn layer_norm_bfloat16_with_affine() {
    let ctx = IconnxCudaContext::new().expect("ctx");
    if ctx.compute_capability().0 < 8 {
        return;
    }
    let kernels = ReductionKernelCache::new(&ctx).expect("kernels");
    let mut pool = GpuMemoryPool::new();

    // Same input row [-1, 0, 1]; gamma=[2,2,2] doubles, beta=[1,1,1] shifts.
    // Expected ≈ [2*(-1.2247) + 1, 0+1, 2*1.2247 + 1] = [-1.4494, 1.0, 3.4494].
    let host: Vec<bf16> = vec![-1.0_f32, 0.0, 1.0]
        .into_iter()
        .map(bf16::from_f32)
        .collect();
    let g = GpuTensor::from_host_bf16(&ctx, &host, vec![1, 3]).expect("upload");
    let gamma = GpuTensor::from_host_f32(&ctx, &[2.0_f32, 2.0, 2.0], vec![3])
        .expect("upload gamma");
    let beta = GpuTensor::from_host_f32(&ctx, &[1.0_f32, 1.0, 1.0], vec![3])
        .expect("upload beta");

    let result = gpu_layer_norm(&ctx, &kernels, &mut pool, &g, &gamma, &beta, 1e-5)
        .expect("layer_norm");
    let back = result.to_host_bf16(&ctx).expect("download");

    let expected = [-1.4494_f32, 1.0, 3.4494];
    for (got, want) in back.iter().zip(expected.iter()) {
        let diff = (got.to_f32() - *want).abs();
        assert!(diff < 5e-2, "got {got:?} want {want} (BF16 5e-2 reduction tol)");
    }
}
