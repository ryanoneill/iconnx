//! WS-3.5 Y(2) sub-2c — BFloat16 Softmax with f32 accumulator.

#![cfg(feature = "cuda")]

use half::bf16;
use iconnx::cuda::{
    gpu_softmax, GpuMemoryPool, GpuTensor, IconnxCudaContext, ReductionKernelCache,
};

#[test]
#[ignore = "requires CUDA GPU"]
fn softmax_bfloat16_uniform_input() {
    let ctx = IconnxCudaContext::new().expect("ctx");
    if ctx.compute_capability().0 < 8 {
        return;
    }
    let kernels = ReductionKernelCache::new(&ctx).expect("kernels");
    let mut pool = GpuMemoryPool::new();

    // Uniform input → softmax = 1/n at every position.
    let host: Vec<bf16> = vec![1.0; 4].into_iter().map(bf16::from_f32).collect();
    let g = GpuTensor::from_host_bf16(&ctx, &host, vec![4]).expect("upload");
    let result = gpu_softmax(&ctx, &kernels, &mut pool, &g).expect("softmax");
    let back = result.to_host_bf16(&ctx).expect("download");
    for got in back.iter() {
        let diff = (got.to_f32() - 0.25).abs();
        assert!(diff < 1e-2, "got {got:?} want 0.25 (BF16 1e-2 tol)");
    }
}

#[test]
#[ignore = "requires CUDA GPU"]
fn softmax_bfloat16_sums_to_one() {
    let ctx = IconnxCudaContext::new().expect("ctx");
    if ctx.compute_capability().0 < 8 {
        return;
    }
    let kernels = ReductionKernelCache::new(&ctx).expect("kernels");
    let mut pool = GpuMemoryPool::new();

    // Hand-picked logits — sum of softmax must equal 1 (within tol).
    let host: Vec<bf16> = vec![1.0_f32, 2.0, 3.0, 4.0]
        .into_iter()
        .map(bf16::from_f32)
        .collect();
    let g = GpuTensor::from_host_bf16(&ctx, &host, vec![4]).expect("upload");
    let result = gpu_softmax(&ctx, &kernels, &mut pool, &g).expect("softmax");
    let back = result.to_host_bf16(&ctx).expect("download");
    let total: f32 = back.iter().map(|v| v.to_f32()).sum();
    assert!(
        (total - 1.0).abs() < 5e-2,
        "softmax must sum to 1.0; got {total}"
    );
}
