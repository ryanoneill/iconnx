//! WS-6 M6.2 Task 5 — InputShapeMismatch guard.
//!
//! Verifies that `GpuGraphExecutor::run` rejects a runtime input whose shape
//! disagrees with the dims the compiled plan was lowered against, and returns
//! a typed `CudaError::InputShapeMismatch` instead of silently miscomputing.
//!
//! The guard fires BEFORE any GPU work, so no partial execution occurs.

#![cfg(feature = "cuda")]

use std::collections::HashMap;
use iconnx::{GpuGraphExecutor, NodeAttributes, Tensor};
use iconnx::cuda::CudaError;

/// A concrete-shape plan ([1, 4]) must reject a runtime input shaped [1, 8].
///
/// This is the always-on floor: even without dynamic-shape support, a wrong
/// shape is a loud `InputShapeMismatch` error rather than a wrong result.
#[test]
#[ignore = "requires CUDA GPU"]
fn concrete_plan_mismatched_runtime_shape_errors() {
    let mut exec = GpuGraphExecutor::new().expect("new");
    exec.add_input("x".into(), vec![Some(1), Some(4)]); // concrete dims
    exec.add_node(
        "n0",
        "Relu",
        vec!["x"],
        vec!["y"],
        NodeAttributes::new(),
    );
    exec.compile().expect("compile at [1,4]");

    let mut inputs = HashMap::new();
    inputs.insert(
        "x".to_string(),
        Tensor::from_vec_f32(vec![1.0; 8], vec![1, 8]),
    );
    let err = exec
        .run(inputs, vec!["y"])
        .expect_err("must guard, not miscompute");
    assert!(
        matches!(err, CudaError::InputShapeMismatch { .. }),
        "expected InputShapeMismatch, got {err:?}",
    );
}

/// A dynamic plan ([None, None]) must accept any runtime shape without error.
///
/// This confirms that `None` plan dims (dynamic) do not trigger the guard.
#[test]
#[ignore = "requires CUDA GPU"]
fn dynamic_plan_accepts_any_runtime_shape() {
    let mut exec = GpuGraphExecutor::new().expect("new");
    exec.add_input("x".into(), vec![None, None]); // fully dynamic
    exec.add_node(
        "n0",
        "Relu",
        vec!["x"],
        vec!["y"],
        NodeAttributes::new(),
    );
    exec.compile().expect("compile with dynamic dims");

    let mut inputs = HashMap::new();
    inputs.insert(
        "x".to_string(),
        Tensor::from_vec_f32(vec![1.0; 8], vec![2, 4]),
    );
    // Must not error — dynamic plan accepts any matching-rank shape.
    exec.run(inputs, vec!["y"]).expect("dynamic dims must not trigger guard");
}

/// A correct same-shape run must not trigger the guard.
///
/// Regression check: ensures the guard path is a true identity on the happy
/// path — no spurious rejection when shapes actually match.
#[test]
#[ignore = "requires CUDA GPU"]
fn same_shape_run_does_not_trigger_guard() {
    let mut exec = GpuGraphExecutor::new().expect("new");
    exec.add_input("x".into(), vec![Some(1), Some(4)]); // concrete dims
    exec.add_node(
        "n0",
        "Relu",
        vec!["x"],
        vec!["y"],
        NodeAttributes::new(),
    );
    exec.compile().expect("compile at [1,4]");

    let mut inputs = HashMap::new();
    inputs.insert(
        "x".to_string(),
        Tensor::from_vec_f32(vec![-1.0, 2.0, -3.0, 4.0], vec![1, 4]),
    );
    let outputs = exec
        .run(inputs, vec!["y"])
        .expect("same-shape run must not trigger guard");
    assert!(outputs.contains_key("y"), "output 'y' must be present");
}

/// A rank mismatch (2-D plan, 3-D runtime) must also trigger the guard.
///
/// The guard checks rank before checking element-count so a fully-wrong
/// shape doesn't silently pass through dim-by-dim comparison.
#[test]
#[ignore = "requires CUDA GPU"]
fn rank_mismatch_triggers_guard() {
    let mut exec = GpuGraphExecutor::new().expect("new");
    exec.add_input("x".into(), vec![Some(1), Some(4)]); // rank 2, concrete
    exec.add_node(
        "n0",
        "Relu",
        vec!["x"],
        vec!["y"],
        NodeAttributes::new(),
    );
    exec.compile().expect("compile at [1,4]");

    let mut inputs = HashMap::new();
    inputs.insert(
        "x".to_string(),
        // rank 3 — element count matches (1*1*4 = 4) but rank disagrees
        Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 4]),
    );
    let err = exec
        .run(inputs, vec!["y"])
        .expect_err("rank mismatch must trigger guard");
    assert!(
        matches!(err, CudaError::InputShapeMismatch { .. }),
        "expected InputShapeMismatch on rank mismatch, got {err:?}",
    );
}

/// WS-6 M6.2 Task 6 — dynamic-load Conv correct across two shapes vs ORT.
///
/// pdfocr's path: one plan loaded from `conv_dyn.onnx` (dynamic-dim model)
/// is executed at two different (H,W) without any re-plan in between.
/// The fixture is a single Conv, input `x` with dim_param on N, H, W (so
/// `input_shapes()` yields `[None, Some(1), None, None]`), kernel 1/9
/// box-blur, pads 1, so output spatial == input spatial.
///
/// Verifies the R2 dynamic-load MUST: the Task 5 guard must NOT fire (None
/// dims accept any runtime value), and the numerical result must match ORT
/// at atol=rtol=1e-3 for both shapes.
#[test]
#[ignore = "requires CUDA GPU"]
fn dynamic_load_correct_across_two_shapes_vs_ort() {
    #[path = "common/ort_parity.rs"]
    mod ort_parity;
    use std::collections::HashMap;
    use iconnx::{GpuGraphExecutor, OnnxParser, Tensor};

    let p = std::path::Path::new("tests/fixtures/ws6_op/conv_dyn.onnx");
    let model = OnnxParser::parse_file(p).expect("parse");
    let exec = GpuGraphExecutor::from_model(&model).expect("from_model");
    let mut ort = ort_parity::OrtSession::open(p).expect("ort");

    // ONE plan, executed at two different (H,W) — no re-plan in between.
    for (h, w) in [(8usize, 8usize), (12usize, 12usize)] {
        let n = h * w;
        let data: Vec<f32> = (0..n).map(|i| (i % 7) as f32 - 3.0).collect();
        let mut iin = HashMap::new();
        iin.insert("x".to_string(), Tensor::from_vec_f32(data.clone(), vec![1, 1, h, w]));
        let got = exec.run(iin, vec!["y"]).expect("iconnx run");
        let (oref, oref_shape) = ort.run_f32(&[("x", &data, vec![1, 1, h, w])], "y").expect("ort run");
        assert_eq!(oref_shape, vec![1, 1, h, w], "ORT output spatial should match input (pad=1,k=3)");
        ort_parity::assert_close(&got["y"].as_slice(), &oref, 1e-3, 1e-3);
    }
}
