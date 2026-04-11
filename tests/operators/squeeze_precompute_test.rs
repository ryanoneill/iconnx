//! Cycle 2 — Squeeze precomputation tests.
//!
//! Verifies that Squeeze's axes parameter is pre-computed from
//! Constant Int64 nodes at graph-build time, and that the dispatch
//! path produces a correct metadata-only output.

#![cfg(feature = "cuda")]

use iconnx::attributes::NodeAttributes;
use iconnx::cuda::inference::testing::get_node_precomputed;
use iconnx::cuda::GpuGraphExecutor;
use iconnx::tensor::Tensor;
use std::collections::HashMap;

/// Helper: build attributes whose "value" attribute is a 1-D Int64 tensor.
fn int64_constant_attrs(values: &[i64]) -> NodeAttributes {
    let mut attrs = NodeAttributes::default();
    let tensor = Tensor::from_vec_i64(values.to_vec(), vec![values.len()]);
    attrs.add_tensor("value".to_string(), tensor);
    attrs
}

#[test]
fn precompute_hit_from_constant() {
    let mut executor = GpuGraphExecutor::new().expect("executor");

    executor.add_node(
        "axes_const",
        "Constant",
        vec![],
        vec!["axes"],
        int64_constant_attrs(&[0]),
    );

    executor.add_node(
        "squeeze",
        "Squeeze",
        vec!["x", "axes"],
        vec!["y"],
        NodeAttributes::default(),
    );

    let snapshot = get_node_precomputed(&executor, "squeeze").expect("squeeze node must exist");
    assert_eq!(
        snapshot.squeeze_axes,
        Some(vec![0]),
        "precomputed squeeze_axes must be Some([0]) for Constant-sourced axes"
    );
}

#[test]
fn precompute_miss_when_axes_not_in_cache() {
    let mut executor = GpuGraphExecutor::new().expect("executor");

    executor.add_node(
        "squeeze",
        "Squeeze",
        vec!["x", "runtime_axes"],
        vec!["y"],
        NodeAttributes::default(),
    );

    let snapshot = get_node_precomputed(&executor, "squeeze").expect("squeeze node must exist");
    assert_eq!(
        snapshot.squeeze_axes, None,
        "precomputed squeeze_axes must be None when input not in cache"
    );
}

#[test]
fn squeeze_with_precomputed_axes_produces_correct_output() {
    let mut executor = GpuGraphExecutor::new().expect("executor");

    // Constant axes [0]
    executor.add_node(
        "axes_const",
        "Constant",
        vec![],
        vec!["axes"],
        int64_constant_attrs(&[0]),
    );

    // Squeeze dim 0 of a [1, 4] input -> [4]
    executor.add_node(
        "squeeze",
        "Squeeze",
        vec!["x", "axes"],
        vec!["y"],
        NodeAttributes::default(),
    );

    // Sanity: precomputation must have fired for this test to be meaningful.
    let snapshot = get_node_precomputed(&executor, "squeeze").expect("squeeze node must exist");
    assert_eq!(
        snapshot.squeeze_axes,
        Some(vec![0]),
        "precomputed squeeze_axes must be Some([0]) -- this test's correctness \
         depends on the precomputed path firing"
    );

    let mut inputs = HashMap::new();
    inputs.insert(
        "x".to_string(),
        Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]),
    );

    let outputs = executor.run(inputs, vec!["y"]).expect("run");
    let y = outputs.get("y").expect("y output");

    assert_eq!(
        y.shape(),
        &[4usize],
        "Squeeze with axes=[0] on [1,4] input must produce shape [4]"
    );
    assert_eq!(
        y.as_slice(),
        &[1.0f32, 2.0, 3.0, 4.0],
        "Squeeze must preserve element values — it's metadata-only"
    );
}
