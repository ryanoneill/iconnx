//! WS-3 M3.3 — GPU `DType::Float16` + `DType::Bool` round-trip tests.
//!
//! Mirrors `tests/gpu_dtype_int8_uint8_roundtrip_test.rs` exactly; the
//! only thing that changes is the dtype under test. Verifies htod/dtoh
//! parity for both new GPU dtype variants and confirms the `dtype()`
//! method returns the expected enum.

#![cfg(feature = "cuda")]

use half::f16;
use iconnx::cuda::{DType, GpuTensor, IconnxCudaContext};

#[test]
#[ignore = "requires CUDA GPU"]
fn gpu_tensor_float16_htod_dtoh_roundtrip() {
    let ctx = IconnxCudaContext::new().expect("CUDA context");
    let host = vec![
        f16::from_f32(1.5),
        f16::from_f32(-2.0),
        f16::from_f32(0.0),
        f16::from_f32(65504.0), // f16 max
        f16::from_f32(-65504.0),
    ];
    let g = GpuTensor::from_host_f16(&ctx, &host, vec![5]).expect("upload");
    assert_eq!(g.dtype(), DType::Float16);
    assert_eq!(g.shape(), &[5]);
    assert_eq!(g.len(), 5);
    let back = g.to_host_f16(&ctx).expect("download");
    assert_eq!(back, host);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn gpu_tensor_float16_zeros() {
    let ctx = IconnxCudaContext::new().expect("CUDA context");
    let g = GpuTensor::zeros_f16(&ctx, vec![3]).expect("zeros");
    assert_eq!(g.dtype(), DType::Float16);
    let back = g.to_host_f16(&ctx).expect("download");
    assert_eq!(back, vec![f16::ZERO; 3]);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn gpu_tensor_bool_htod_dtoh_roundtrip() {
    let ctx = IconnxCudaContext::new().expect("CUDA context");
    let host = vec![true, false, true, true, false, false, true];
    let g = GpuTensor::from_host_bool(&ctx, &host, vec![7]).expect("upload");
    assert_eq!(g.dtype(), DType::Bool);
    assert_eq!(g.shape(), &[7]);
    let back = g.to_host_bool(&ctx).expect("download");
    assert_eq!(back, host);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn gpu_tensor_bool_zeros() {
    let ctx = IconnxCudaContext::new().expect("CUDA context");
    let g = GpuTensor::zeros_bool(&ctx, vec![4]).expect("zeros");
    assert_eq!(g.dtype(), DType::Bool);
    let back = g.to_host_bool(&ctx).expect("download");
    assert_eq!(back, vec![false; 4]);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn gpu_tensor_dtype_size_bytes() {
    // Compile-time sanity: DType::Float16 = 2 bytes, DType::Bool = 1.
    assert_eq!(DType::Float16.size_bytes(), 2);
    assert_eq!(DType::Bool.size_bytes(), 1);
    assert_eq!(DType::Float16.name(), "float16");
    assert_eq!(DType::Bool.name(), "bool");
}
