//! WS-3.5 Y(2) sub-2b — BFloat16 unary Erf (f32 round-trip).

#![cfg(feature = "cuda")]

use half::bf16;
use iconnx::cuda::{gpu_erf, GpuMemoryPool, GpuTensor, IconnxCudaContext, KernelCache};

#[test]
#[ignore = "requires CUDA GPU"]
fn erf_bfloat16_elementwise() {
    let ctx = IconnxCudaContext::new().expect("ctx");
    if ctx.compute_capability().0 < 8 {
        return;
    }
    let kernels = KernelCache::new(&ctx).expect("kernels");
    let mut pool = GpuMemoryPool::new();

    // erf(0) = 0, erf(1) ≈ 0.8427, erf(-1) ≈ -0.8427, erf(2) ≈ 0.9953
    let a: Vec<bf16> = vec![0.0_f32, 1.0, -1.0, 2.0]
        .into_iter()
        .map(bf16::from_f32)
        .collect();
    let ga = GpuTensor::from_host_bf16(&ctx, &a, vec![4]).expect("upload");
    let gc = gpu_erf(&ctx, &kernels, &mut pool, &ga).expect("erf");
    let c = gc.to_host_bf16(&ctx).expect("download");
    let expected = [0.0_f32, 0.8427, -0.8427, 0.9953];
    for (got, want) in c.iter().zip(expected.iter()) {
        let diff = (got.to_f32() - *want).abs();
        assert!(diff < 1e-2, "got {got:?} want {want} (BF16 1e-2 tol)");
    }
}
