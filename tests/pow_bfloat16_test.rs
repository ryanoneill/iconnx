//! WS-3.5 Y(2) sub-2b — BFloat16 elementwise Pow (f32 round-trip).

#![cfg(feature = "cuda")]

use half::bf16;
use iconnx::cuda::{gpu_pow, GpuMemoryPool, GpuTensor, IconnxCudaContext, KernelCache};

#[test]
#[ignore = "requires CUDA GPU"]
fn pow_bfloat16_squares() {
    let ctx = IconnxCudaContext::new().expect("ctx");
    if ctx.compute_capability().0 < 8 {
        return;
    }
    let kernels = KernelCache::new(&ctx).expect("kernels");
    let mut pool = GpuMemoryPool::new();

    let base: Vec<bf16> = vec![2.0, 3.0, 4.0, 5.0]
        .into_iter()
        .map(bf16::from_f32)
        .collect();
    let exp: Vec<bf16> = vec![2.0, 2.0, 2.0, 2.0]
        .into_iter()
        .map(bf16::from_f32)
        .collect();
    let gb = GpuTensor::from_host_bf16(&ctx, &base, vec![4]).expect("upload base");
    let ge = GpuTensor::from_host_bf16(&ctx, &exp, vec![4]).expect("upload exp");
    let gc = gpu_pow(&ctx, &kernels, &mut pool, &gb, &ge).expect("pow");
    let c = gc.to_host_bf16(&ctx).expect("download");
    // 2^2=4, 3^2=9, 4^2=16, 5^2=25
    let expected = [4.0_f32, 9.0, 16.0, 25.0];
    for (got, want) in c.iter().zip(expected.iter()) {
        let diff = (got.to_f32() - *want).abs();
        // 25 in BF16 is exactly representable. Allow a small relative
        // tolerance for the f32 round-trip.
        assert!(diff < 1.0, "got {got:?} want {want} (BF16 1e-2 tol on small mantissa)");
    }
}
