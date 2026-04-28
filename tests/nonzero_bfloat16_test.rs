//! WS-3.5 Y(2) sub-2a — BFloat16 NonZero dispatch.
//!
//! Per M3.7c, NonZero is dtype-input-polymorphic via host-side path:
//! the operator downloads, applies != 0, and emits Int64 indices. BF16
//! parallels the FP16 host-side `to_f32() != 0.0` check.
//!
//! Skipped (early-return) on pre-sm_80 hardware.

#![cfg(feature = "cuda")]

use half::bf16;
use iconnx::cuda::{gpu_nonzero, DType, GpuTensor, IconnxCudaContext};

#[test]
#[ignore = "requires CUDA GPU"]
fn nonzero_bfloat16_returns_int64_indices() {
    let ctx = IconnxCudaContext::new().expect("ctx");
    if ctx.compute_capability().0 < 8 {
        return;
    }

    // Input: [0.0, 1.5, 0.0, -2.0, 0.0] — non-zero at indices 1 and 3.
    let host = vec![
        bf16::from_f32(0.0),
        bf16::from_f32(1.5),
        bf16::from_f32(0.0),
        bf16::from_f32(-2.0),
        bf16::from_f32(0.0),
    ];
    let g = GpuTensor::from_host_bf16(&ctx, &host, vec![5]).expect("upload");

    let result = gpu_nonzero(&ctx, &g).expect("nonzero");
    // Output dtype is Int64 (NonZero always emits indices).
    assert_eq!(result.dtype(), DType::Int64);
    // Shape: [ndim, num_nonzero] = [1, 2].
    assert_eq!(result.shape(), &[1, 2]);
    let back = result.to_host_i64(&ctx).expect("download");
    assert_eq!(back, vec![1, 3]);
}
