//! WS-3.5 Y(2) sub-2a — BFloat16 transpose dispatch.
//!
//! Mirrors the FP16 transpose test pattern. Memcpy-class BF16 transpose
//! is a pure index shuffle so byte-level identity is sufficient — same
//! `unsigned short*` C-side parameter type as FP16 in the kernel.
//!
//! Skipped (early-return) on pre-sm_80 hardware.

#![cfg(feature = "cuda")]

use half::bf16;
use iconnx::cuda::{gpu_transpose_nd, GpuMemoryPool, GpuTensor, IconnxCudaContext, OpsKernelCache};

#[test]
#[ignore = "requires CUDA GPU"]
fn transpose_bfloat16_2d() {
    let ctx = IconnxCudaContext::new().expect("ctx");
    if ctx.compute_capability().0 < 8 {
        return;
    }
    let cache = OpsKernelCache::new(&ctx).expect("cache");
    let mut pool = GpuMemoryPool::new();

    // Input: 2x3 = [[0, 1, 2], [3, 4, 5]]
    let host: Vec<bf16> = (0..6).map(|i| bf16::from_f32(i as f32)).collect();
    let g = GpuTensor::from_host_bf16(&ctx, &host, vec![2, 3]).expect("upload");
    let perm: Vec<i64> = vec![1, 0];
    let t = gpu_transpose_nd(&ctx, &cache, &mut pool, &g, &perm).expect("transpose");
    assert_eq!(t.shape(), &[3, 2]);
    let back = t.to_host_bf16(&ctx).expect("download");
    // Transposed: [[0, 3], [1, 4], [2, 5]]
    let expected = vec![
        bf16::from_f32(0.0),
        bf16::from_f32(3.0),
        bf16::from_f32(1.0),
        bf16::from_f32(4.0),
        bf16::from_f32(2.0),
        bf16::from_f32(5.0),
    ];
    assert_eq!(back, expected);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn transpose_bfloat16_3d() {
    let ctx = IconnxCudaContext::new().expect("ctx");
    if ctx.compute_capability().0 < 8 {
        return;
    }
    let cache = OpsKernelCache::new(&ctx).expect("cache");
    let mut pool = GpuMemoryPool::new();

    // 2x2x3 input — perm [0, 2, 1] swaps last two dims.
    let host: Vec<bf16> = (0..12).map(|i| bf16::from_f32(i as f32)).collect();
    let g = GpuTensor::from_host_bf16(&ctx, &host, vec![2, 2, 3]).expect("upload");
    let perm: Vec<i64> = vec![0, 2, 1];
    let t = gpu_transpose_nd(&ctx, &cache, &mut pool, &g, &perm).expect("transpose");
    assert_eq!(t.shape(), &[2, 3, 2]);
}
