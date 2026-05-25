//! WS-6 M6.4 Task 13 — BatchNormalization GPU inference vs live ORT.
//!
//! Inference BN is a per-channel affine:
//!   a[c] = scale[c] / sqrt(var[c] + eps)
//!   b[c] = B[c] - mean[c] * a[c]
//!   y[i] = a[ch] * x[i] + b[ch],  ch = (i / spatial) % C
//!
//! The fixture `batchnorm_1x3x8x8.onnx` has shape [1, 3, 8, 8] (NCHW),
//! scale/B/mean/var as [3] initializers, and epsilon=1e-5 as a node attribute.

#![cfg(feature = "cuda")]

#[path = "common/ort_parity.rs"]
mod ort_parity;

use std::collections::HashMap;

use iconnx::{GpuGraphExecutor, OnnxParser, Tensor};

use ort_parity::{assert_close, seeded, OrtSession};

/// GPU BatchNormalization (inference, NCHW) matches ORT within 1e-3.
///
/// Uses a [1, 3, 8, 8] seeded N(0, 1) input — 192 elements. The
/// scale/B/mean/var [3] initializers and epsilon=1e-5 are baked into the
/// fixture. Parity fails if the per-channel affine math is wrong or if
/// the kernel's channel-index formula is incorrect.
#[test]
#[ignore = "requires CUDA GPU"]
fn batchnorm_1x3x8x8_matches_ort() {
    let fixture = "tests/fixtures/ws6_op/batchnorm_1x3x8x8.onnx";
    let p = std::path::Path::new(fixture);
    let model = OnnxParser::parse_file(p).expect("parse fixture");
    let exec = GpuGraphExecutor::from_model(&model).expect("from_model");
    let mut ort = OrtSession::open(p).expect("open ORT session");

    let data = seeded(192, 0xDEAD_BEEF_CAFE_1234);
    let shape = vec![1usize, 3, 8, 8];

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
