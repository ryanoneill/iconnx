//! WS-6 M6.4 Task 14 — Tile GPU inference vs live ORT.
//!
//! ONNX Tile semantics: `out_shape[d] = in_shape[d] * repeats[d]`;
//! element at output coord `o` maps to input coord `o[d] % in_shape[d]`.
//! The `repeats` tensor is an INT64 input read at execute time.
//!
//! The fixture `tile_1x3_r2x2.onnx` has:
//!   input  x: float [1, 3]
//!   input  repeats: int64 [2] = [2, 2]  (stored as initializer)
//!   output y: float [2, 6]

#![cfg(feature = "cuda")]

#[path = "common/ort_parity.rs"]
mod ort_parity;

use std::collections::HashMap;

use iconnx::{GpuGraphExecutor, OnnxParser, Tensor};

use ort_parity::{assert_close, OrtSession};

/// GPU Tile matches ORT within 1e-3 for a [1,3] → [2,6] tiling.
///
/// Input is `[1.0, 2.0, 3.0]` shape [1,3]; repeats is the initializer [2,2]
/// baked into the fixture. Expected output shape is [2,6] and values tile
/// the input row across both axes. Parity fails if the per-axis modulo
/// addressing or stride computation is wrong.
#[test]
#[ignore = "requires CUDA GPU"]
fn tile_1x3_r2x2_matches_ort() {
    let fixture = "tests/fixtures/ws6_op/tile_1x3_r2x2.onnx";
    let p = std::path::Path::new(fixture);
    let model = OnnxParser::parse_file(p).expect("parse fixture");
    let exec = GpuGraphExecutor::from_model(&model).expect("from_model");
    let mut ort = OrtSession::open(p).expect("open ORT session");

    let data = vec![1.0f32, 2.0, 3.0];
    let shape = vec![1usize, 3];

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), Tensor::from_vec_f32(data.clone(), shape.clone()));
    let got = exec.run(inputs, vec!["y"]).expect("iconnx run");
    let got_data = got["y"].as_slice();

    // Verify output shape is [2, 6].
    assert_eq!(
        got["y"].shape(),
        &[2usize, 6],
        "Tile: expected output shape [2,6], got {:?}",
        got["y"].shape()
    );

    let (oref, oref_shape) = ort
        .run_f32(&[("x", &data, shape)], "y")
        .expect("ort run");

    assert_eq!(
        oref_shape,
        vec![2usize, 6],
        "ORT Tile: expected output shape [2,6], got {:?}",
        oref_shape
    );

    assert_close(&got_data, &oref, 1e-3, 1e-3);
}
