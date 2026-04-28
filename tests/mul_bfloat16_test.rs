//! WS-3.5 Y(2) sub-2b — BFloat16 elementwise Mul via __hmul.

#![cfg(feature = "cuda")]

use half::bf16;
use iconnx::cuda::{gpu_mul, GpuMemoryPool, GpuTensor, IconnxCudaContext, KernelCache};

#[test]
#[ignore = "requires CUDA GPU"]
fn mul_bfloat16_elementwise() {
    let ctx = IconnxCudaContext::new().expect("ctx");
    if ctx.compute_capability().0 < 8 {
        return;
    }
    let kernels = KernelCache::new(&ctx).expect("kernels");
    let mut pool = GpuMemoryPool::new();

    let a: Vec<bf16> = vec![2.0, 3.0, 4.0, 5.0]
        .into_iter()
        .map(bf16::from_f32)
        .collect();
    let b: Vec<bf16> = vec![1.5, 2.0, 0.5, 0.0]
        .into_iter()
        .map(bf16::from_f32)
        .collect();
    let ga = GpuTensor::from_host_bf16(&ctx, &a, vec![4]).expect("upload a");
    let gb = GpuTensor::from_host_bf16(&ctx, &b, vec![4]).expect("upload b");
    let gc = gpu_mul(&ctx, &kernels, &mut pool, &ga, &gb).expect("mul");
    let c = gc.to_host_bf16(&ctx).expect("download");
    let expected = [3.0, 6.0, 2.0, 0.0];
    for (got, want) in c.iter().zip(expected.iter()) {
        let diff = (got.to_f32() - *want).abs();
        assert!(diff < 1e-2, "got {got:?} want {want} (BF16 1e-2 tol)");
    }
}
