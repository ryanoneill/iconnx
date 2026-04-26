//! WS-3 M3.5 — Cast FP16 ↔ Float32 / Int32 / Int64 contract tests.
//!
//! These tests assert both directions of the FP16 boundary cast and
//! the integer-pair arms. Numbers are chosen to round-trip exactly
//! through the f16 mantissa where the spec calls for bit-equality;
//! larger magnitudes use a small tolerance (~ULP).

#![cfg(feature = "cuda")]

use half::f16;
use iconnx::cuda::{gpu_cast, DType, GpuTensor, IconnxCudaContext, OpsKernelCache};

fn setup() -> (IconnxCudaContext, OpsKernelCache) {
    let ctx = IconnxCudaContext::new().expect("CUDA context");
    let cache = OpsKernelCache::new(&ctx).expect("kernel cache");
    (ctx, cache)
}

#[test]
#[ignore = "requires CUDA GPU"]
fn cast_float16_to_float32_round_trip() {
    let (ctx, cache) = setup();
    let host_f16 = vec![
        f16::from_f32(1.5),
        f16::from_f32(-2.0),
        f16::from_f32(0.0),
        f16::from_f32(3.5),
    ];
    let g_in = GpuTensor::from_host_f16(&ctx, &host_f16, vec![4]).unwrap();
    let g_out = gpu_cast(&ctx, &cache, &g_in, DType::Float32).unwrap();
    assert_eq!(g_out.dtype(), DType::Float32);
    let host_f32 = g_out.to_host_f32(&ctx).unwrap();
    assert_eq!(host_f32, vec![1.5, -2.0, 0.0, 3.5]);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn cast_float32_to_float16_round_half_to_even() {
    // 1.0 + 1/4096 = 1.000244... — below the smallest f16 mantissa step
    // around 1.0 (which is 1/1024). Round-half-to-even: rounds to 1.0.
    let (ctx, cache) = setup();
    let g_in = GpuTensor::from_host_f32(&ctx, &[1.0_f32 + 1.0 / 4096.0], vec![1]).unwrap();
    let g_out = gpu_cast(&ctx, &cache, &g_in, DType::Float16).unwrap();
    assert_eq!(g_out.dtype(), DType::Float16);
    let back = g_out.to_host_f16(&ctx).unwrap()[0].to_f32();
    assert!((back - 1.0).abs() < 1e-6, "got {} expected 1.0", back);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn cast_float32_to_float16_basic_values() {
    let (ctx, cache) = setup();
    // All values exactly representable in f16.
    let g_in = GpuTensor::from_host_f32(&ctx, &[1.5_f32, -2.0, 65504.0, 0.0], vec![4]).unwrap();
    let g_out = gpu_cast(&ctx, &cache, &g_in, DType::Float16).unwrap();
    let host = g_out.to_host_f16(&ctx).unwrap();
    assert_eq!(
        host,
        vec![
            f16::from_f32(1.5),
            f16::from_f32(-2.0),
            f16::from_f32(65504.0),
            f16::ZERO,
        ]
    );
}

#[test]
#[ignore = "requires CUDA GPU"]
fn cast_float16_to_int32_truncates() {
    let (ctx, cache) = setup();
    // FP16 -> Int32 follows ONNX spec (truncate toward zero).
    let g_in = GpuTensor::from_host_f16(
        &ctx,
        &[
            f16::from_f32(1.5),
            f16::from_f32(-2.4),
            f16::from_f32(0.9),
            f16::from_f32(-0.5),
        ],
        vec![4],
    )
    .unwrap();
    let g_out = gpu_cast(&ctx, &cache, &g_in, DType::Int32).unwrap();
    assert_eq!(g_out.dtype(), DType::Int32);
    let host = g_out.to_host_i32(&ctx).unwrap();
    assert_eq!(host, vec![1, -2, 0, 0]);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn cast_int32_to_float16_round_trip() {
    let (ctx, cache) = setup();
    let g_in = GpuTensor::from_host_i32(&ctx, &[1, -2, 0, 1024], vec![4]).unwrap();
    let g_out = gpu_cast(&ctx, &cache, &g_in, DType::Float16).unwrap();
    let host = g_out.to_host_f16(&ctx).unwrap();
    assert_eq!(
        host,
        vec![
            f16::from_f32(1.0),
            f16::from_f32(-2.0),
            f16::ZERO,
            f16::from_f32(1024.0),
        ]
    );
}

#[test]
#[ignore = "requires CUDA GPU"]
fn cast_float16_to_int64_truncates() {
    let (ctx, cache) = setup();
    let g_in = GpuTensor::from_host_f16(
        &ctx,
        &[f16::from_f32(3.7), f16::from_f32(-1.2), f16::from_f32(0.0)],
        vec![3],
    )
    .unwrap();
    let g_out = gpu_cast(&ctx, &cache, &g_in, DType::Int64).unwrap();
    let host = g_out.to_host_i64(&ctx).unwrap();
    assert_eq!(host, vec![3i64, -1, 0]);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn cast_int64_to_float16_round_trip() {
    let (ctx, cache) = setup();
    let g_in = GpuTensor::from_host_i64(&ctx, &[5i64, -10, 0, 2048], vec![4]).unwrap();
    let g_out = gpu_cast(&ctx, &cache, &g_in, DType::Float16).unwrap();
    let host = g_out.to_host_f16(&ctx).unwrap();
    assert_eq!(
        host,
        vec![
            f16::from_f32(5.0),
            f16::from_f32(-10.0),
            f16::ZERO,
            f16::from_f32(2048.0),
        ]
    );
}

#[test]
#[ignore = "requires CUDA GPU"]
fn cast_float16_same_type_returns_clone() {
    let (ctx, cache) = setup();
    let host_f16 = vec![f16::from_f32(1.5), f16::from_f32(-2.0)];
    let g_in = GpuTensor::from_host_f16(&ctx, &host_f16, vec![2]).unwrap();
    let g_out = gpu_cast(&ctx, &cache, &g_in, DType::Float16).unwrap();
    assert_eq!(g_out.dtype(), DType::Float16);
    assert_eq!(g_out.to_host_f16(&ctx).unwrap(), host_f16);
}
