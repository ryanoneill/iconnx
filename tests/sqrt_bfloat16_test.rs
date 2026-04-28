//! WS-3.5 Y(2) sub-2b — BFloat16 unary Sqrt (f32 round-trip).

#![cfg(feature = "cuda")]

use half::bf16;
use iconnx::cuda::{gpu_sqrt, GpuMemoryPool, GpuTensor, IconnxCudaContext, KernelCache};

#[test]
#[ignore = "requires CUDA GPU"]
fn sqrt_bfloat16_elementwise() {
    let ctx = IconnxCudaContext::new().expect("ctx");
    if ctx.compute_capability().0 < 8 {
        return;
    }
    let kernels = KernelCache::new(&ctx).expect("kernels");
    let mut pool = GpuMemoryPool::new();

    let a: Vec<bf16> = vec![4.0, 9.0, 16.0, 0.0]
        .into_iter()
        .map(bf16::from_f32)
        .collect();
    let ga = GpuTensor::from_host_bf16(&ctx, &a, vec![4]).expect("upload");
    let gc = gpu_sqrt(&ctx, &kernels, &mut pool, &ga).expect("sqrt");
    let c = gc.to_host_bf16(&ctx).expect("download");
    let expected = [2.0, 3.0, 4.0, 0.0];
    for (got, want) in c.iter().zip(expected.iter()) {
        let diff = (got.to_f32() - *want).abs();
        assert!(diff < 1e-2, "got {got:?} want {want} (BF16 1e-2 tol)");
    }
}

#[test]
#[ignore = "requires CUDA GPU"]
fn sqrt_bfloat16_negative_clamps_to_zero() {
    let ctx = IconnxCudaContext::new().expect("ctx");
    if ctx.compute_capability().0 < 8 {
        return;
    }
    let kernels = KernelCache::new(&ctx).expect("kernels");
    let mut pool = GpuMemoryPool::new();

    let a: Vec<bf16> = vec![-1.0_f32, -4.0_f32]
        .into_iter()
        .map(bf16::from_f32)
        .collect();
    let ga = GpuTensor::from_host_bf16(&ctx, &a, vec![2]).expect("upload");
    let gc = gpu_sqrt(&ctx, &kernels, &mut pool, &ga).expect("sqrt");
    let c = gc.to_host_bf16(&ctx).expect("download");
    // f32 sqrt_kernel returns 0.0 for negatives; BF16 mirrors policy.
    for got in c.iter() {
        assert_eq!(got.to_f32(), 0.0, "sqrt of negative must clamp to 0");
    }
}
