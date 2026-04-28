//! WS-3.5 Y(1) — Hardware-capability gate for BFloat16 GPU tensors.
//!
//! BF16 native compute requires `sm_80+` (Ampere and later: A100, H100,
//! RTX 30/40-series). Older Turing (`sm_75`: RTX 20-series, T4) compiles
//! bf16 NVRTC but emulates with significant perf loss; some intrinsics
//! (`__bfloat162float`, `__hadd` on `__nv_bfloat16`) require `sm_80+`
//! outright. Volta (`sm_70`, V100) has no BF16 support at all.
//!
//! Enforcement seam: `GpuTensor::zeros_bf16` and `GpuTensor::from_host_bf16`
//! check the device's compute capability and return typed
//! `CudaError::UnsupportedHardware` on pre-sm_80 devices. Defense-in-depth
//! at the single ctor entry: downstream dispatch sites can assume any
//! `GpuTensor::BFloat16` they receive is hardware-capable.

#![cfg(feature = "cuda")]

use iconnx::cuda::{GpuTensor, IconnxCudaContext};

#[test]
#[ignore = "requires CUDA GPU"]
fn bfloat16_zeros_succeeds_on_sm_80_or_newer() {
    let ctx = IconnxCudaContext::new().expect("ctx");
    let (major, _minor) = ctx.compute_capability();
    if major < 8 {
        // sm_75 or older — the gate should error; this test scenario doesn't apply.
        return;
    }
    let t = GpuTensor::zeros_bf16(&ctx, vec![4]).expect("alloc on sm_80+");
    assert_eq!(t.shape(), &[4]);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn bfloat16_zeros_fails_loudly_on_pre_sm_80() {
    let ctx = IconnxCudaContext::new().expect("ctx");
    let (major, minor) = ctx.compute_capability();
    if major >= 8 {
        // sm_80+ — gate succeeds; the failure path doesn't apply.
        return;
    }
    let result = GpuTensor::zeros_bf16(&ctx, vec![4]);
    // Avoid `expect_err` here: `GpuTensor` deliberately does not implement
    // `Debug` (raw device pointers + dtype dispatch make a meaningful Debug
    // impl ill-defined). Pattern-match instead so we never need to format
    // the success-arm payload.
    let err = match result {
        Ok(_) => panic!("must reject BF16 alloc on pre-sm_80"),
        Err(e) => e,
    };
    let msg = format!("{err:?}");
    assert!(
        msg.contains("UnsupportedHardware") || msg.contains("bfloat16"),
        "error must mention BF16 hardware requirement: {msg}"
    );
    assert!(
        msg.contains(&format!("sm_{}{}", major, minor)) || msg.contains("sm_80"),
        "error must mention required sm_80 + observed CC: {msg}"
    );
}
