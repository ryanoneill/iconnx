//! WS-3.5 Y(2) sub-2a — BFloat16 expand dispatch.
//!
//! Mirrors the FP16 expand test pattern. Memcpy-class BF16 expand uses
//! the byte-identical `unsigned short*` kernel signature.
//!
//! Skipped (early-return) on pre-sm_80 hardware.

#![cfg(feature = "cuda")]

use half::bf16;
use iconnx::cuda::{gpu_expand, GpuMemoryPool, GpuTensor, IconnxCudaContext, OpsKernelCache};

#[test]
#[ignore = "requires CUDA GPU"]
fn expand_bfloat16_broadcast_singleton() {
    let ctx = IconnxCudaContext::new().expect("ctx");
    if ctx.compute_capability().0 < 8 {
        return;
    }
    let cache = OpsKernelCache::new(&ctx).expect("cache");
    let mut pool = GpuMemoryPool::new();

    // Input shape [1, 3] = [[1.0, 2.0, 3.0]] — broadcast to [2, 3].
    let host = vec![bf16::from_f32(1.0), bf16::from_f32(2.0), bf16::from_f32(3.0)];
    let g = GpuTensor::from_host_bf16(&ctx, &host, vec![1, 3]).expect("upload");

    let result = gpu_expand(&ctx, &cache, &mut pool, &g, vec![2, 3]).expect("expand");
    assert_eq!(result.shape(), &[2, 3]);
    let back = result.to_host_bf16(&ctx).expect("download");
    let expected = vec![
        bf16::from_f32(1.0),
        bf16::from_f32(2.0),
        bf16::from_f32(3.0),
        bf16::from_f32(1.0),
        bf16::from_f32(2.0),
        bf16::from_f32(3.0),
    ];
    assert_eq!(back, expected);
}
