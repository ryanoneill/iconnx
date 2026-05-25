//! WS-6 M6.4 Task 12 — HardSigmoid GPU activation vs live ORT.
//!
//! HardSigmoid is defined as `clamp(alpha * x + beta, 0, 1)` where alpha and
//! beta are read from node attributes (ONNX defaults: alpha=0.2, beta=0.5).
//!
//! The fixture `hardsigmoid_a02_b05_1x64.onnx` has alpha=0.2, beta=0.5
//! embedded as node attributes. This test pins that iconnx reads the
//! attributes correctly — if alpha/beta were hardcoded to wrong values the
//! parity check against ORT would fail.

#![cfg(feature = "cuda")]

#[path = "common/ort_parity.rs"]
mod ort_parity;

use std::collections::HashMap;

use iconnx::{GpuGraphExecutor, OnnxParser, Tensor};

use ort_parity::{assert_close, seeded, OrtSession};

/// GPU HardSigmoid matches ORT for alpha=0.2, beta=0.5.
///
/// The fixture has shape [1, 64]; seeded N(0, 1) input (seed pins the
/// parity test to a deterministic sequence). The alpha/beta attribute-read
/// is exercised because parity fails if either is hardcoded incorrectly.
#[test]
#[ignore = "requires CUDA GPU"]
fn hardsigmoid_a02_b05_matches_ort() {
    let fixture = "tests/fixtures/ws6_op/hardsigmoid_a02_b05_1x64.onnx";
    let p = std::path::Path::new(fixture);
    let model = OnnxParser::parse_file(p).expect("parse fixture");
    let exec = GpuGraphExecutor::from_model(&model).expect("from_model");
    let mut ort = OrtSession::open(p).expect("open ORT session");

    let data = seeded(64, 0xABCD_1234_5678_9ABC);
    let shape = vec![1usize, 64];

    let mut inputs = HashMap::new();
    inputs.insert(
        "x".to_string(),
        Tensor::from_vec_f32(data.clone(), shape.clone()),
    );
    let got = exec.run(inputs, vec!["y"]).expect("iconnx run");
    let got_data = got["y"].as_slice();

    let (oref, _oref_shape) = ort
        .run_f32(&[("x", &data, shape)], "y")
        .expect("ort run");

    assert_close(&got_data, &oref, 1e-3, 1e-3);
}
