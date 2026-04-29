//! WS-3.5 Y(2) sub-2d — BFloat16 Cast dtype-pair arms.
//!
//! 10 NEW dtype-pair arms wire BF16 into the Cast op:
//!   FP32 ↔ BF16 (primary BERT/Whisper boundary)
//!   FP16 ↔ BF16 (NET-NEW pair via f32 intermediate)
//!   Int32 ↔ BF16
//!   Int64 ↔ BF16
//! Plus explicit-unsupported (Bool, BFloat16) per M3.7b discipline.
//!
//! Skipped (early-return) on pre-sm_80 hardware.

#![cfg(feature = "cuda")]

use half::{bf16, f16};
use iconnx::cuda::{gpu_cast, DType, GpuTensor, IconnxCudaContext, OpsKernelCache};

#[test]
#[ignore = "requires CUDA GPU"]
fn cast_bfloat16_to_float32_lossless_for_representable() {
    let ctx = IconnxCudaContext::new().expect("ctx");
    if ctx.compute_capability().0 < 8 {
        return;
    }
    let cache = OpsKernelCache::new(&ctx).expect("cache");

    // 1.5, -2.0, 0.0 are exactly representable in BF16.
    let host = vec![bf16::from_f32(1.5), bf16::from_f32(-2.0), bf16::from_f32(0.0)];
    let g = GpuTensor::from_host_bf16(&ctx, &host, vec![3]).expect("upload");
    let result = gpu_cast(&ctx, &cache, &g, DType::Float32).expect("cast bf16 → f32");
    assert_eq!(result.dtype(), DType::Float32);
    let back = result.to_host_f32(&ctx).expect("download");
    assert_eq!(back, vec![1.5_f32, -2.0, 0.0]);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn cast_float32_to_bfloat16_round_to_nearest_even() {
    let ctx = IconnxCudaContext::new().expect("ctx");
    if ctx.compute_capability().0 < 8 {
        return;
    }
    let cache = OpsKernelCache::new(&ctx).expect("cache");

    let host = vec![1.5_f32, -2.0, 0.0];
    let g = GpuTensor::from_host_f32(&ctx, &host, vec![3]).expect("upload");
    let result = gpu_cast(&ctx, &cache, &g, DType::BFloat16).expect("cast f32 → bf16");
    assert_eq!(result.dtype(), DType::BFloat16);
    let back = result.to_host_bf16(&ctx).expect("download");
    let expected = vec![bf16::from_f32(1.5), bf16::from_f32(-2.0), bf16::from_f32(0.0)];
    assert_eq!(back, expected);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn cast_float16_to_bfloat16_via_f32_intermediate() {
    let ctx = IconnxCudaContext::new().expect("ctx");
    if ctx.compute_capability().0 < 8 {
        return;
    }
    let cache = OpsKernelCache::new(&ctx).expect("cache");

    // Both formats represent 1.5 and -2.0 exactly; lossless round-trip.
    let host = vec![f16::from_f32(1.5), f16::from_f32(-2.0)];
    let g = GpuTensor::from_host_f16(&ctx, &host, vec![2]).expect("upload");
    let result = gpu_cast(&ctx, &cache, &g, DType::BFloat16).expect("cast f16 → bf16");
    assert_eq!(result.dtype(), DType::BFloat16);
    let back = result.to_host_bf16(&ctx).expect("download");
    assert_eq!(back[0].to_f32(), 1.5);
    assert_eq!(back[1].to_f32(), -2.0);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn cast_bfloat16_to_float16_via_f32_intermediate() {
    let ctx = IconnxCudaContext::new().expect("ctx");
    if ctx.compute_capability().0 < 8 {
        return;
    }
    let cache = OpsKernelCache::new(&ctx).expect("cache");

    let host = vec![bf16::from_f32(1.5), bf16::from_f32(-2.0)];
    let g = GpuTensor::from_host_bf16(&ctx, &host, vec![2]).expect("upload");
    let result = gpu_cast(&ctx, &cache, &g, DType::Float16).expect("cast bf16 → f16");
    assert_eq!(result.dtype(), DType::Float16);
    let back = result.to_host_f16(&ctx).expect("download");
    assert_eq!(back[0].to_f32(), 1.5);
    assert_eq!(back[1].to_f32(), -2.0);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn cast_int32_to_bfloat16() {
    let ctx = IconnxCudaContext::new().expect("ctx");
    if ctx.compute_capability().0 < 8 {
        return;
    }
    let cache = OpsKernelCache::new(&ctx).expect("cache");

    // Small ints are exactly representable in BF16.
    let host = vec![1_i32, -2, 5, 0];
    let g = GpuTensor::from_host_i32(&ctx, &host, vec![4]).expect("upload");
    let result = gpu_cast(&ctx, &cache, &g, DType::BFloat16).expect("cast i32 → bf16");
    assert_eq!(result.dtype(), DType::BFloat16);
    let back = result.to_host_bf16(&ctx).expect("download");
    let expected: Vec<bf16> = vec![1.0_f32, -2.0, 5.0, 0.0]
        .into_iter()
        .map(bf16::from_f32)
        .collect();
    assert_eq!(back, expected);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn cast_bfloat16_to_int32_truncates_toward_zero() {
    let ctx = IconnxCudaContext::new().expect("ctx");
    if ctx.compute_capability().0 < 8 {
        return;
    }
    let cache = OpsKernelCache::new(&ctx).expect("cache");

    // 1.7 → 1, -2.5 → -2, 3.0 → 3 (BF16 mantissa rounds 1.7 to 1.6953125
    // but it still truncates to 1).
    let host = vec![bf16::from_f32(1.7), bf16::from_f32(-2.5), bf16::from_f32(3.0)];
    let g = GpuTensor::from_host_bf16(&ctx, &host, vec![3]).expect("upload");
    let result = gpu_cast(&ctx, &cache, &g, DType::Int32).expect("cast bf16 → i32");
    assert_eq!(result.dtype(), DType::Int32);
    let back = result.to_host_i32(&ctx).expect("download");
    assert_eq!(back, vec![1_i32, -2, 3]);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn cast_int64_to_bfloat16_and_back() {
    let ctx = IconnxCudaContext::new().expect("ctx");
    if ctx.compute_capability().0 < 8 {
        return;
    }
    let cache = OpsKernelCache::new(&ctx).expect("cache");

    let host = vec![10_i64, -100, 0, 42];
    let g = GpuTensor::from_host_i64(&ctx, &host, vec![4]).expect("upload");
    let to_bf = gpu_cast(&ctx, &cache, &g, DType::BFloat16).expect("cast i64 → bf16");
    let back = gpu_cast(&ctx, &cache, &to_bf, DType::Int64).expect("cast bf16 → i64");
    assert_eq!(back.dtype(), DType::Int64);
    let back_host = back.to_host_i64(&ctx).expect("download");
    // Small ints (≤ 256) are exactly representable in BF16, so round-trip is lossless.
    assert_eq!(back_host, host);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn cast_bfloat16_to_bool_explicitly_unsupported() {
    let ctx = IconnxCudaContext::new().expect("ctx");
    if ctx.compute_capability().0 < 8 {
        return;
    }
    let cache = OpsKernelCache::new(&ctx).expect("cache");

    let host = vec![bf16::from_f32(1.0)];
    let g = GpuTensor::from_host_bf16(&ctx, &host, vec![1]).expect("upload");
    let result = gpu_cast(&ctx, &cache, &g, DType::Bool);
    assert!(result.is_err(), "BF16 → Bool must error");
    let err = result.err().unwrap();
    let dbg = format!("{:?}", err);
    assert!(
        dbg.contains("UnsupportedDtype") || dbg.contains("bool"),
        "error must reference bool / UnsupportedDtype: {}",
        dbg
    );
}

#[test]
#[ignore = "requires CUDA GPU"]
fn cast_bool_to_bfloat16_explicitly_unsupported() {
    let ctx = IconnxCudaContext::new().expect("ctx");
    if ctx.compute_capability().0 < 8 {
        return;
    }
    let cache = OpsKernelCache::new(&ctx).expect("cache");

    let host = vec![true, false];
    let g = GpuTensor::from_host_bool(&ctx, &host, vec![2]).expect("upload");
    let result = gpu_cast(&ctx, &cache, &g, DType::BFloat16);
    assert!(result.is_err(), "Bool → BF16 must error");
    let err = result.err().unwrap();
    let dbg = format!("{:?}", err);
    assert!(
        dbg.contains("UnsupportedDtype") || dbg.contains("bool"),
        "error must reference bool / UnsupportedDtype: {}",
        dbg
    );
}
