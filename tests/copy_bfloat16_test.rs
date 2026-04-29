//! WS-3.5 Y(2) sub-2a — BFloat16 copy dispatch.
//!
//! Mirrors the FP16 copy test pattern. WS-1 M1.4's gpu_copy is per-dtype
//! dispatch with explicit kernels; BF16 follows the same shape.
//!
//! Skipped (early-return) on pre-sm_80 hardware.

#![cfg(feature = "cuda")]

use half::bf16;
use iconnx::cuda::{gpu_copy, GpuMemoryPool, GpuTensor, IconnxCudaContext, OpsKernelCache};

#[test]
#[ignore = "requires CUDA GPU"]
fn copy_bfloat16_identity() {
    let ctx = IconnxCudaContext::new().expect("ctx");
    if ctx.compute_capability().0 < 8 {
        return;
    }
    let cache = OpsKernelCache::new(&ctx).expect("cache");
    let mut pool = GpuMemoryPool::new();

    let host: Vec<bf16> = (0..6).map(|i| bf16::from_f32(i as f32 + 0.5)).collect();
    let g = GpuTensor::from_host_bf16(&ctx, &host, vec![2, 3]).expect("upload");
    let result = gpu_copy(&ctx, &cache, &mut pool, &g).expect("copy");
    assert_eq!(result.shape(), &[2, 3]);
    let back = result.to_host_bf16(&ctx).expect("download");
    assert_eq!(back, host);
}
