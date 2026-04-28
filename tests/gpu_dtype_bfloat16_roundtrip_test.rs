//! WS-3.5 Y(1) — GPU `DType::BFloat16` round-trip tests.
//!
//! Mirrors `tests/gpu_dtype_float16_bool_roundtrip_test.rs` exactly; the
//! only thing that changes is the dtype under test (bf16 instead of f16).
//! Verifies htod/dtoh parity for the new GPU dtype variant and confirms
//! the `dtype()` method returns the expected enum.
//!
//! Skipped (early-return) on pre-sm_80 hardware — the hardware-capability
//! gate at construction returns `CudaError::UnsupportedHardware` there;
//! `tests/gpu_bfloat16_hardware_capability_test.rs` covers that path
//! explicitly.

#![cfg(feature = "cuda")]

use half::bf16;
use iconnx::cuda::{DType, GpuTensor, IconnxCudaContext};

#[test]
#[ignore = "requires CUDA GPU"]
fn gpu_tensor_bfloat16_htod_dtoh_roundtrip() {
    let ctx = IconnxCudaContext::new().expect("CUDA context");
    if ctx.compute_capability().0 < 8 {
        // pre-sm_80 — roundtrip test skipped, hardware-gate test covers rejection.
        return;
    }
    let host = vec![
        bf16::from_f32(1.5),
        bf16::from_f32(-2.0),
        bf16::from_f32(0.0),
    ];
    let g = GpuTensor::from_host_bf16(&ctx, &host, vec![3]).expect("upload");
    assert_eq!(g.dtype(), DType::BFloat16);
    assert_eq!(g.shape(), &[3]);
    assert_eq!(g.len(), 3);
    let back = g.to_host_bf16(&ctx).expect("download");
    assert_eq!(back, host);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn gpu_tensor_bfloat16_zeros_constructor() {
    let ctx = IconnxCudaContext::new().expect("CUDA context");
    if ctx.compute_capability().0 < 8 {
        return;
    }
    let g = GpuTensor::zeros_bf16(&ctx, vec![5]).expect("zeros");
    assert_eq!(g.dtype(), DType::BFloat16);
    let back = g.to_host_bf16(&ctx).expect("download");
    assert_eq!(back.len(), 5);
    for v in back {
        assert_eq!(v.to_f32(), 0.0);
    }
}
