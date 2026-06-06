//! WS-6 M6.3 Tasks 8-9 — grouped/depthwise Conv2D vs live ORT.
//!
//! iconnx's hand-rolled 2D Conv (`gpu_conv2d`, im2col + GEMM) ignores the
//! `group` attribute and asserts `in_channels == kernel_in_channels`, so any
//! grouped or depthwise conv PANICS. MobileNetV3 (both OCR models) uses
//! grouped/depthwise convs pervasively. The fix routes `group > 1` (or
//! dilated) 2D convs through garboard's cuDNN `ConvForwardConfig`, which
//! natively supports 2D NCHW + groups + dilation, while keeping the
//! `group == 1` non-dilated case on the existing im2col path.
//!
//! These three parity tests pin that fix against ONNX Runtime's CPU EP as the
//! oracle. Each parses a single-Conv fixture (kernels baked as initializers),
//! runs it through both iconnx's GPU executor and ORT on the same seeded
//! input, and asserts `allclose` at atol=rtol=1e-3. The spike was bit-exact
//! for the bias-free cases; the biased case adds only f32-add noise, well
//! within 1e-3.

#![cfg(feature = "cuda")]

// Live-ORT parity helpers, shared by the parity tests in this file. Declared
// ONCE at file scope so `clippy::duplicate_mod` is satisfied (declaring the
// same `#[path]` module inside multiple test fns would load the file as a
// module multiple times).
#[path = "common/ort_parity.rs"]
mod ort_parity;

use std::collections::HashMap;

use iconnx::{GpuGraphExecutor, OnnxParser, Tensor};

use ort_parity::{assert_close, seeded, OrtSession};

/// Run a single-Conv fixture (input `x`, output `y`) through both iconnx's
/// GPU executor and ORT on the same seeded input, and assert parity.
fn assert_conv_parity(fixture: &str, shape: Vec<usize>) {
    let p = std::path::Path::new(fixture);
    let model = OnnxParser::parse_file(p).expect("parse fixture");
    let exec = GpuGraphExecutor::from_model(&model).expect("from_model");
    let mut ort = OrtSession::open(p).expect("open ORT session");

    let numel: usize = shape.iter().product();
    let data = seeded(numel, 0xC02D);

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), Tensor::from_vec_f32(data.clone(), shape.clone()));
    let got = exec.run(inputs, vec!["y"]).expect("iconnx run");

    let (oref, _oref_shape) = ort
        .run_f32(&[("x", &data, shape.clone())], "y")
        .expect("ort run");

    assert_close(&got["y"].as_slice(), &oref, 1e-3, 1e-3);
}

/// Depthwise conv (group == in_channels): `conv_depthwise_g8.onnx`, group=8,
/// input [1,8,16,16], kernel [8,1,3,3], pad 1. Pre-fix this PANICS at the
/// `in_channels == kernel_in_channels` assert (8 != 1).
#[test]
#[ignore = "requires CUDA GPU"]
fn depthwise_conv2d_matches_ort() {
    assert_conv_parity(
        "tests/fixtures/ws6_op/conv_depthwise_g8.onnx",
        vec![1, 8, 16, 16],
    );
}

/// General grouped conv (1 < group < in_channels): `conv_grouped_g2.onnx`,
/// group=2, input [1,8,16,16], kernel [16,4,3,3], pad 1. Pre-fix this PANICS
/// at the `in_channels == kernel_in_channels` assert (8 != 4).
#[test]
#[ignore = "requires CUDA GPU"]
fn general_grouped_conv2d_matches_ort() {
    assert_conv_parity(
        "tests/fixtures/ws6_op/conv_grouped_g2.onnx",
        vec![1, 8, 16, 16],
    );
}

/// Grouped conv WITH bias: `conv_grouped_g2_biased.onnx`, group=2 + a [16]
/// bias initializer. Exercises the `add_bias` follow-up on top of the cuDNN
/// grouped-conv output (cuDNN's conv_forward does not fold bias).
#[test]
#[ignore = "requires CUDA GPU"]
fn biased_grouped_conv2d_matches_ort() {
    assert_conv_parity(
        "tests/fixtures/ws6_op/conv_grouped_g2_biased.onnx",
        vec![1, 8, 16, 16],
    );
}
