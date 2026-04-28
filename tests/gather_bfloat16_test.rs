//! WS-3.5 Y(2) sub-2a — BFloat16 gather dispatch.
//!
//! Mirrors the FP16 gather test pattern. Memcpy-class BF16 gather uses
//! the byte-identical `unsigned short*` kernel signature.
//!
//! Skipped (early-return) on pre-sm_80 hardware.

#![cfg(feature = "cuda")]

use half::bf16;
use iconnx::cuda::{gpu_gather, GpuMemoryPool, GpuTensor, IconnxCudaContext, OpsKernelCache};

#[test]
#[ignore = "requires CUDA GPU"]
fn gather_bfloat16_axis0() {
    let ctx = IconnxCudaContext::new().expect("ctx");
    if ctx.compute_capability().0 < 8 {
        return;
    }
    let cache = OpsKernelCache::new(&ctx).expect("cache");
    let mut pool = GpuMemoryPool::new();

    // Data: 4x2 = [[0, 1], [2, 3], [4, 5], [6, 7]]
    let host: Vec<bf16> = (0..8).map(|i| bf16::from_f32(i as f32)).collect();
    let g = GpuTensor::from_host_bf16(&ctx, &host, vec![4, 2]).expect("upload");

    // Indices: [0, 2] — gather rows 0 and 2.
    let indices: Vec<i64> = vec![0, 2];
    let result = gpu_gather(&ctx, &cache, &mut pool, &g, &indices, &[2]).expect("gather");
    assert_eq!(result.shape(), &[2, 2]);
    let back = result.to_host_bf16(&ctx).expect("download");
    // Expected: [[0, 1], [4, 5]]
    let expected = vec![
        bf16::from_f32(0.0),
        bf16::from_f32(1.0),
        bf16::from_f32(4.0),
        bf16::from_f32(5.0),
    ];
    assert_eq!(back, expected);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn gather_bfloat16_negative_index_wraps() {
    let ctx = IconnxCudaContext::new().expect("ctx");
    if ctx.compute_capability().0 < 8 {
        return;
    }
    let cache = OpsKernelCache::new(&ctx).expect("cache");
    let mut pool = GpuMemoryPool::new();

    // Data: 3x1 = [[0], [1], [2]]
    let host: Vec<bf16> = (0..3).map(|i| bf16::from_f32(i as f32)).collect();
    let g = GpuTensor::from_host_bf16(&ctx, &host, vec![3, 1]).expect("upload");

    // Index -1 wraps to dim0_size-1 = 2 (last row).
    let indices: Vec<i64> = vec![-1];
    let result = gpu_gather(&ctx, &cache, &mut pool, &g, &indices, &[1]).expect("gather");
    let back = result.to_host_bf16(&ctx).expect("download");
    assert_eq!(back, vec![bf16::from_f32(2.0)]);
}
