//! WS-6 M6.4 Task 12 — HardSwish GPU activation vs live ORT.
//!
//! HardSwish is defined as `x * clamp((x + 3) / 6, 0, 1)` with no
//! attributes. The fixture `hardswish_1x64.onnx` exercises the full
//! activation across a seeded N(0, 1) input of 64 elements.

#![cfg(feature = "cuda")]

#[path = "common/ort_parity.rs"]
mod ort_parity;

use std::collections::HashMap;

use iconnx::{GpuGraphExecutor, OnnxParser, Tensor};

use ort_parity::{assert_close, seeded, OrtSession};

/// GPU HardSwish matches ORT over a seeded 1×64 input.
#[test]
#[ignore = "requires CUDA GPU"]
fn hardswish_matches_ort() {
    let fixture = "tests/fixtures/ws6_op/hardswish_1x64.onnx";
    let p = std::path::Path::new(fixture);
    let model = OnnxParser::parse_file(p).expect("parse fixture");
    let exec = GpuGraphExecutor::from_model(&model).expect("from_model");
    let mut ort = OrtSession::open(p).expect("open ORT session");

    let data = seeded(64, 0xDEAD_BEEF_C0FF_EE42);
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
