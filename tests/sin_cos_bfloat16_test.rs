//! WS-3.5 Y(2) sub-2e — BFloat16 Sin / Cos via f32 round-trip.
//!
//! No native bf16 sin/cos intrinsic; the kernel rounds through f32.
//! Whisper-BF16 positional encoding produces sin/cos pairs over the
//! same input — both ops dispatch through this path.

#![cfg(feature = "cuda")]

use half::bf16;
use iconnx::cuda::{gpu_cos, gpu_sin, GpuMemoryPool, GpuTensor, IconnxCudaContext, KernelCache};

#[test]
#[ignore = "requires CUDA GPU"]
fn sin_bfloat16_at_zero_and_pi_over_2() {
    let ctx = IconnxCudaContext::new().expect("ctx");
    if ctx.compute_capability().0 < 8 {
        return;
    }
    let kernels = KernelCache::new(&ctx).expect("kernels");
    let mut pool = GpuMemoryPool::new();

    let host = vec![
        bf16::from_f32(0.0),
        bf16::from_f32(std::f32::consts::PI / 2.0),
    ];
    let g = GpuTensor::from_host_bf16(&ctx, &host, vec![2]).expect("upload");
    let result = gpu_sin(&ctx, &kernels, &mut pool, &g).expect("sin");
    let back = result.to_host_bf16(&ctx).expect("download");
    assert!((back[0].to_f32() - 0.0).abs() < 1e-2);
    assert!((back[1].to_f32() - 1.0).abs() < 1e-2);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn cos_bfloat16_at_zero_and_pi_over_2() {
    let ctx = IconnxCudaContext::new().expect("ctx");
    if ctx.compute_capability().0 < 8 {
        return;
    }
    let kernels = KernelCache::new(&ctx).expect("kernels");
    let mut pool = GpuMemoryPool::new();

    let host = vec![
        bf16::from_f32(0.0),
        bf16::from_f32(std::f32::consts::PI / 2.0),
    ];
    let g = GpuTensor::from_host_bf16(&ctx, &host, vec![2]).expect("upload");
    let result = gpu_cos(&ctx, &kernels, &mut pool, &g).expect("cos");
    let back = result.to_host_bf16(&ctx).expect("download");
    // cos(0) = 1, cos(π/2) ≈ 0 (BF16 PI/2 rounding shifts result a hair).
    assert!((back[0].to_f32() - 1.0).abs() < 1e-2);
    assert!(back[1].to_f32().abs() < 5e-2);
}
