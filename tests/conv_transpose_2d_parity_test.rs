//! WS-6 M6.3 Task 11 — ConvTranspose-2D vs live ORT (verification only).
//!
//! 2D ConvTranspose already dispatches via `gpu_conv_transpose_2d` in
//! `src/cuda/executor/conv.rs`. No roster model exercised it before WS-6.
//! This test verifies the DBNet upsample form: stride=2,
//! input [1,4,8,8] → output [1,2,16,16].
//!
//! The fixture `convtranspose2d_s2.onnx` has:
//!   - Graph input `x` [1,4,8,8]
//!   - Initializer `kernel` [4,2,2,2]  (in_channels=4, out_channels=2, kH=2, kW=2)
//!   - Node attributes: strides=[2,2], pads=[0,0,0,0]
//!   - Graph output `y` [1,2,16,16]
//!
//! No source change is expected: if this test FAILS, that indicates a real
//! defect in `gpu_conv_transpose_2d` that must be diagnosed and fixed
//! without tolerance relaxation.

#![cfg(feature = "cuda")]

#[path = "common/ort_parity.rs"]
mod ort_parity;

use std::collections::HashMap;

use iconnx::{GpuGraphExecutor, OnnxParser, Tensor};

use ort_parity::{assert_close, seeded, OrtSession};

/// Verify ConvTranspose-2D (stride=2) parity against ORT on the DBNet
/// upsample form: [1,4,8,8] input → [1,2,16,16] output.
#[test]
#[ignore = "requires CUDA GPU"]
fn conv_transpose_2d_s2_matches_ort() {
    let fixture = "tests/fixtures/ws6_op/convtranspose2d_s2.onnx";
    let p = std::path::Path::new(fixture);
    let model = OnnxParser::parse_file(p).expect("parse fixture");
    let exec = GpuGraphExecutor::from_model(&model).expect("from_model");
    let mut ort = OrtSession::open(p).expect("open ORT session");

    // Input shape [1,4,8,8]: 256 elements.
    let shape = vec![1usize, 4, 8, 8];
    let numel: usize = shape.iter().product(); // 256
    let data = seeded(numel, 0xDB_BEEF_CAFE_0002);

    let mut inputs = HashMap::new();
    inputs.insert(
        "x".to_string(),
        Tensor::from_vec_f32(data.clone(), shape.clone()),
    );
    let got = exec.run(inputs, vec!["y"]).expect("iconnx run");

    let (oref, oref_shape) = ort
        .run_f32(&[("x", &data, shape.clone())], "y")
        .expect("ort run");

    // Verify output shape is [1,2,16,16] before numerical check.
    assert_eq!(
        oref_shape,
        vec![1usize, 2, 16, 16],
        "ORT output shape mismatch: expected [1,2,16,16] got {:?}",
        oref_shape
    );
    let got_data = got["y"].as_slice();
    assert_eq!(
        got_data.len(),
        512,
        "iconnx output element count mismatch: expected 512 (=[1,2,16,16]) got {}",
        got_data.len()
    );

    assert_close(&got_data, &oref, 1e-3, 1e-3);
}
