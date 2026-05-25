//! WS-6 M6.3 Task 10 — input-form Clip (opset 11+) vs live ORT.
//!
//! ONNX opset 11 moved Clip's `min`/`max` from attributes to optional inputs
//! (inputs[1] and inputs[2]). Prior to the Task 10 fix, the GPU Clip arm read
//! only from attributes and silently used f32::MIN/MAX when they were absent —
//! causing the 18 ReLU6 Clip nodes in the OCR rec model to be no-ops.
//!
//! Two tests:
//!
//! 1. `clip_input_form_relu6_matches_ort` — opset-14 fixture with min=0 /
//!    max=6 as scalar initializers; verifies the input path reads them
//!    correctly and clamps.
//!
//! 2. `clip_attr_form_matches_ort` — opset-6 fixture with min/max as
//!    attributes; regression guard that the attribute fallback still works.

#![cfg(feature = "cuda")]

// Live-ORT parity helpers. Declared ONCE at file scope (clippy::duplicate_mod
// safe: each test binary that `#[path]`-includes this file gets its own module
// instance, which is the intended pattern for shared test fixtures).
#[path = "common/ort_parity.rs"]
mod ort_parity;

use std::collections::HashMap;

use iconnx::{GpuGraphExecutor, OnnxParser, Tensor};

use ort_parity::{assert_close, seeded, OrtSession};

/// Test 1: input-form Clip (opset 14, min=0/max=6 as scalar initializers).
///
/// The fixture `clip_input_form_relu6.onnx` has:
///   - Graph input `x`
///   - Scalar initializers `clip_min=0.0` and `clip_max=6.0` as inputs[1]/[2]
///   - Graph output `y`
///
/// The input vector spans values below 0, in range, and above 6 so the
/// spot-asserts below can verify clamping in both directions.
#[test]
#[ignore = "requires CUDA GPU"]
fn clip_input_form_relu6_matches_ort() {
    let fixture = "tests/fixtures/ws6_op/clip_input_form_relu6.onnx";
    let p = std::path::Path::new(fixture);
    let model = OnnxParser::parse_file(p).expect("parse fixture");
    let exec = GpuGraphExecutor::from_model(&model).expect("from_model");
    let mut ort = OrtSession::open(p).expect("open ORT session");

    // Shape [1, 16]: 16 elements. Use seeded Gaussian then manually pin
    // two elements to be clearly below 0 and above 6 for the spot-asserts.
    let mut data = seeded(16, 0xC11F_BEEF_DEAD_0001);
    // Pin element 0 well below 0 so it must clamp to 0.0.
    data[0] = -3.5_f32;
    // Pin element 1 well above 6 so it must clamp to 6.0.
    data[1] = 9.0_f32;

    let shape = vec![1usize, 16];

    let mut inputs = HashMap::new();
    inputs.insert(
        "x".to_string(),
        Tensor::from_vec_f32(data.clone(), shape.clone()),
    );
    let got = exec.run(inputs, vec!["y"]).expect("iconnx run");
    let got_data = got["y"].as_slice();

    let (oref, _oref_shape) = ort
        .run_f32(&[("x", &data, shape.clone())], "y")
        .expect("ort run");

    assert_close(&got_data, &oref, 1e-3, 1e-3);

    // Spot-asserts: confirm clamping actually occurred (not just ORT-parity
    // on a no-op path). Pre-fix, element 0 would be -3.5 and element 1 9.0.
    assert!(
        (got_data[0] - 0.0_f32).abs() < 1e-6,
        "element 0 (input=-3.5) should be clamped to 0.0 by ReLU6, got {}",
        got_data[0]
    );
    assert!(
        (got_data[1] - 6.0_f32).abs() < 1e-6,
        "element 1 (input=9.0) should be clamped to 6.0 by ReLU6, got {}",
        got_data[1]
    );
}

/// Test 2: attribute-form Clip (opset 6, min/max as node attributes).
///
/// The fixture `clip_attr_form.onnx` has min=0 / max=6 baked into the node
/// attributes. This is the legacy pre-opset-11 path; the test guards that the
/// attribute fallback still works correctly after the Task 10 change.
#[test]
#[ignore = "requires CUDA GPU"]
fn clip_attr_form_matches_ort() {
    let fixture = "tests/fixtures/ws6_op/clip_attr_form.onnx";
    let p = std::path::Path::new(fixture);
    let model = OnnxParser::parse_file(p).expect("parse fixture");
    let exec = GpuGraphExecutor::from_model(&model).expect("from_model");
    let mut ort = OrtSession::open(p).expect("open ORT session");

    let mut data = seeded(16, 0xA77F_CAFE_0102_0304);
    data[0] = -2.0_f32;
    data[1] = 7.5_f32;

    let shape = vec![1usize, 16];

    let mut inputs = HashMap::new();
    inputs.insert(
        "x".to_string(),
        Tensor::from_vec_f32(data.clone(), shape.clone()),
    );
    let got = exec.run(inputs, vec!["y"]).expect("iconnx run");
    let got_data = got["y"].as_slice();

    let (oref, _oref_shape) = ort
        .run_f32(&[("x", &data, shape.clone())], "y")
        .expect("ort run");

    assert_close(&got_data, &oref, 1e-3, 1e-3);

    // Spot-asserts to confirm attribute-form clamping is also live.
    assert!(
        (got_data[0] - 0.0_f32).abs() < 1e-6,
        "element 0 (input=-2.0) should be clamped to 0.0, got {}",
        got_data[0]
    );
    assert!(
        (got_data[1] - 6.0_f32).abs() < 1e-6,
        "element 1 (input=7.5) should be clamped to 6.0, got {}",
        got_data[1]
    );
}
