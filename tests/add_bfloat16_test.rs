//! WS-3.5 Y(2) sub-2b — BFloat16 elementwise Add via __hadd.
//!
//! Tolerance 1e-2 reflects BF16's 7-bit mantissa (4× FP16's 1e-3 due to
//! 7-bit vs 10-bit mantissa).
//!
//! Skipped (early-return) on pre-sm_80 hardware.

#![cfg(feature = "cuda")]

use half::bf16;
use iconnx::cuda::{
    gpu_add, GpuMemoryPool, GpuTensor, IconnxCudaContext, KernelCache, OpsKernelCache,
};

#[test]
#[ignore = "requires CUDA GPU"]
fn add_bfloat16_elementwise() {
    let ctx = IconnxCudaContext::new().expect("ctx");
    if ctx.compute_capability().0 < 8 {
        return;
    }
    let kernels = KernelCache::new(&ctx).expect("kernels");
    let _ops = OpsKernelCache::new(&ctx).expect("ops cache");
    let mut pool = GpuMemoryPool::new();

    let a: Vec<bf16> = vec![1.0, 2.0, 3.0, 4.0]
        .into_iter()
        .map(bf16::from_f32)
        .collect();
    let b: Vec<bf16> = vec![0.5, 0.5, 0.5, 0.5]
        .into_iter()
        .map(bf16::from_f32)
        .collect();
    let ga = GpuTensor::from_host_bf16(&ctx, &a, vec![4]).expect("upload a");
    let gb = GpuTensor::from_host_bf16(&ctx, &b, vec![4]).expect("upload b");
    let gc = gpu_add(&ctx, &kernels, &mut pool, &ga, &gb).expect("add");
    let c = gc.to_host_bf16(&ctx).expect("download");
    let expected: Vec<bf16> = vec![1.5, 2.5, 3.5, 4.5]
        .into_iter()
        .map(bf16::from_f32)
        .collect();
    for (got, want) in c.iter().zip(expected.iter()) {
        let diff = (got.to_f32() - want.to_f32()).abs();
        assert!(diff < 1e-2, "got {got:?} want {want:?} (BF16 1e-2 tol)");
    }
}
