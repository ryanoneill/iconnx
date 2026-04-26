//! WS-3 M3.4 sub-1 — Float16 layout-op contract tests
//! (transpose, concat, gather — the memcpy-class subset Whisper-Tiny
//! exercises in its encoder).
//!
//! Each test seeds known FP16 inputs (round-trippable through `f16::from_f32`)
//! through the GPU kernel and compares to a hand-computed expected vector.
//! The kernels are pure index gymnastics — no arithmetic — so the values
//! must come back bit-identical.
//!
//! Lives in its own file because `tests.rs` is already past the project's
//! 1000-line guideline; new dtype coverage gets a dedicated file rather
//! than further bloat the main test module.
//!
//! Symbols are pulled from the parent module via `super::*` so the f16
//! dispatch arms are exercised through the same entry points as the
//! production callers.

use super::*;
use crate::cuda::memory_pool::GpuMemoryPool;
use crate::cuda::tensor::DType;
use half::f16;

fn setup() -> (IconnxCudaContext, OpsKernelCache, GpuMemoryPool) {
    let ctx = IconnxCudaContext::new().expect("Failed to create CUDA context");
    let cache = OpsKernelCache::new(&ctx).expect("Failed to compile kernels");
    let pool = GpuMemoryPool::new();
    (ctx, cache, pool)
}

fn h(values: &[f32]) -> Vec<f16> {
    values.iter().copied().map(f16::from_f32).collect()
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_transpose_nd_float16_2d_perm() {
    // [2, 3] -> transpose with perm [1, 0] -> [3, 2]. Bit-identical
    // memcpy — every f16 value is small-magnitude and exactly representable.
    let (ctx, cache, mut pool) = setup();

    let host = h(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let input = GpuTensor::from_host_f16(&ctx, &host, vec![2, 3]).unwrap();

    let output = gpu_transpose_nd(&ctx, &cache, &mut pool, &input, &[1, 0]).unwrap();
    assert_eq!(output.dtype(), DType::Float16);
    assert_eq!(output.shape(), &[3, 2]);

    let result = output.to_host_f16(&ctx).unwrap();
    let expected = h(&[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    assert_eq!(result, expected);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_transpose_nd_float16_3d_perm() {
    // [2, 2, 3] -> perm [2, 0, 1] -> [3, 2, 2]. Mirrors the f32 4D-perm
    // pattern but uses 3D to stay within the kernel's `out_coords[6]`
    // budget without needing extra inputs.
    let (ctx, cache, mut pool) = setup();

    let host = h(&[
        // [0, :, :]
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // [1, :, :]
        7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ]);
    let input = GpuTensor::from_host_f16(&ctx, &host, vec![2, 2, 3]).unwrap();

    let output = gpu_transpose_nd(&ctx, &cache, &mut pool, &input, &[2, 0, 1]).unwrap();
    assert_eq!(output.shape(), &[3, 2, 2]);

    // Expected: out[k, i, j] = inp[i, j, k]
    let mut expected = Vec::with_capacity(12);
    for k in 0..3 {
        for i in 0..2 {
            for j in 0..2 {
                expected.push(host[i * 6 + j * 3 + k]);
            }
        }
    }
    let result = output.to_host_f16(&ctx).unwrap();
    assert_eq!(result, expected);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_concat_float16_axis0() {
    // Axis-0 concat of two [1, 3] FP16 tensors -> [2, 3].
    let (ctx, cache, mut pool) = setup();

    let a = GpuTensor::from_host_f16(&ctx, &h(&[1.0, 2.0, 3.0]), vec![1, 3]).unwrap();
    let b = GpuTensor::from_host_f16(&ctx, &h(&[4.0, 5.0, 6.0]), vec![1, 3]).unwrap();

    let output = gpu_concat(&ctx, &cache, &mut pool, &[&a, &b], 0).unwrap();
    assert_eq!(output.dtype(), DType::Float16);
    assert_eq!(output.shape(), &[2, 3]);

    let result = output.to_host_f16(&ctx).unwrap();
    assert_eq!(result, h(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_concat_float16_axis1() {
    // Axis-1 concat of two [2, 1] FP16 tensors -> [2, 2]. Whisper's
    // attention concatenates KV-cache extensions along the inner axis.
    let (ctx, cache, mut pool) = setup();

    let a = GpuTensor::from_host_f16(&ctx, &h(&[1.0, 2.0]), vec![2, 1]).unwrap();
    let b = GpuTensor::from_host_f16(&ctx, &h(&[3.0, 4.0]), vec![2, 1]).unwrap();

    let output = gpu_concat(&ctx, &cache, &mut pool, &[&a, &b], 1).unwrap();
    assert_eq!(output.shape(), &[2, 2]);

    let result = output.to_host_f16(&ctx).unwrap();
    assert_eq!(result, h(&[1.0, 3.0, 2.0, 4.0]));
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_gather_float16_embedding_lookup() {
    // Vocab=4, embed_dim=3 — the same shape pattern Whisper's encoder
    // hits when looking up positional/token FP16 embeddings.
    let (ctx, cache, mut pool) = setup();

    let table_data: Vec<f32> = (0..12).map(|x| x as f32).collect();
    let input = GpuTensor::from_host_f16(&ctx, &h(&table_data), vec![4, 3]).unwrap();

    // Indices [0, 2, 1] -> output [3, 3] = rows 0, 2, 1.
    let indices = vec![0i64, 2, 1];
    let output = gpu_gather(&ctx, &cache, &mut pool, &input, &indices, &[3]).unwrap();
    assert_eq!(output.dtype(), DType::Float16);
    assert_eq!(output.shape(), &[3, 3]);

    let result = output.to_host_f16(&ctx).unwrap();
    let expected = h(&[0.0, 1.0, 2.0, 6.0, 7.0, 8.0, 3.0, 4.0, 5.0]);
    assert_eq!(result, expected);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_gather_float16_negative_index() {
    // Negative-index handling — wraps via Python/ONNX convention.
    let (ctx, cache, mut pool) = setup();

    let input = GpuTensor::from_host_f16(
        &ctx,
        &h(&[10.0, 20.0, 30.0, 40.0]),
        vec![4],
    )
    .unwrap();

    let indices = vec![-1i64, -2, 0];
    let output = gpu_gather(&ctx, &cache, &mut pool, &input, &indices, &[3]).unwrap();
    assert_eq!(output.shape(), &[3]);

    let result = output.to_host_f16(&ctx).unwrap();
    assert_eq!(result, h(&[40.0, 30.0, 10.0]));
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_gather_from_gpu_float16() {
    // Parallel of the on-GPU-indices gather path — Whisper's graph routes
    // gather indices on-GPU rather than copying through host.
    let (ctx, cache, mut pool) = setup();

    let table_data: Vec<f32> = (0..6).map(|x| x as f32).collect();
    let input = GpuTensor::from_host_f16(&ctx, &h(&table_data), vec![3, 2]).unwrap();
    let indices_tensor =
        GpuTensor::from_host_i64(&ctx, &[2i64, 0, 1], vec![3]).unwrap();

    let output =
        gpu_gather_from_gpu(&ctx, &cache, &mut pool, &input, &indices_tensor).unwrap();
    assert_eq!(output.dtype(), DType::Float16);
    assert_eq!(output.shape(), &[3, 2]);

    let result = output.to_host_f16(&ctx).unwrap();
    let expected = h(&[4.0, 5.0, 0.0, 1.0, 2.0, 3.0]);
    assert_eq!(result, expected);
}
