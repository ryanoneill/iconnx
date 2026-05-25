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
