//! WS-3.5 Y(2) sub-2a — BFloat16 concat dispatch.
//!
//! Mirrors the FP16 concat test pattern. Memcpy-class BF16 concat uses
//! the byte-identical `unsigned short*` kernel signature.
//!
//! Skipped (early-return) on pre-sm_80 hardware.

#![cfg(feature = "cuda")]

use half::bf16;
use iconnx::cuda::{gpu_concat, GpuMemoryPool, GpuTensor, IconnxCudaContext, OpsKernelCache};

#[test]
#[ignore = "requires CUDA GPU"]
fn concat_bfloat16_axis0() {
    let ctx = IconnxCudaContext::new().expect("ctx");
    if ctx.compute_capability().0 < 8 {
        return;
    }
    let cache = OpsKernelCache::new(&ctx).expect("cache");
    let mut pool = GpuMemoryPool::new();

    let a_host: Vec<bf16> = (0..4).map(|i| bf16::from_f32(i as f32)).collect();
    let b_host: Vec<bf16> = (4..8).map(|i| bf16::from_f32(i as f32)).collect();
    let a = GpuTensor::from_host_bf16(&ctx, &a_host, vec![2, 2]).expect("upload a");
    let b = GpuTensor::from_host_bf16(&ctx, &b_host, vec![2, 2]).expect("upload b");

    let result = gpu_concat(&ctx, &cache, &mut pool, &[&a, &b], 0).expect("concat");
    assert_eq!(result.shape(), &[4, 2]);
    let back = result.to_host_bf16(&ctx).expect("download");
    let expected: Vec<bf16> = (0..8).map(|i| bf16::from_f32(i as f32)).collect();
    assert_eq!(back, expected);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn concat_bfloat16_axis1() {
    let ctx = IconnxCudaContext::new().expect("ctx");
    if ctx.compute_capability().0 < 8 {
        return;
    }
    let cache = OpsKernelCache::new(&ctx).expect("cache");
    let mut pool = GpuMemoryPool::new();

    // a = [[0, 1], [2, 3]], b = [[4, 5], [6, 7]]
    let a_host: Vec<bf16> = (0..4).map(|i| bf16::from_f32(i as f32)).collect();
    let b_host: Vec<bf16> = (4..8).map(|i| bf16::from_f32(i as f32)).collect();
    let a = GpuTensor::from_host_bf16(&ctx, &a_host, vec![2, 2]).expect("upload a");
    let b = GpuTensor::from_host_bf16(&ctx, &b_host, vec![2, 2]).expect("upload b");

    let result = gpu_concat(&ctx, &cache, &mut pool, &[&a, &b], 1).expect("concat");
    assert_eq!(result.shape(), &[2, 4]);
    let back = result.to_host_bf16(&ctx).expect("download");
    // Expected: [[0, 1, 4, 5], [2, 3, 6, 7]]
    let expected = vec![
        bf16::from_f32(0.0),
        bf16::from_f32(1.0),
        bf16::from_f32(4.0),
        bf16::from_f32(5.0),
        bf16::from_f32(2.0),
        bf16::from_f32(3.0),
        bf16::from_f32(6.0),
        bf16::from_f32(7.0),
    ];
    assert_eq!(back, expected);
}
