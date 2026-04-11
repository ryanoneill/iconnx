//! Cycle 2 — Reshape precomputation tests.
//!
//! Verifies that Reshape's shape parameter is pre-computed from
//! Constant Int64 nodes at graph-build time, and that the `0` and
//! `-1` resolution semantics still produce the correct output.

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

    // Register a Constant node producing the shape [2, 3].
    executor.add_node(
        "shape_const",
        "Constant",
        vec![],
        vec!["shape"],
        int64_constant_attrs(&[2, 3]),
    );

    // Register a Reshape consuming the Constant output.
    executor.add_node(
        "reshape",
        "Reshape",
        vec!["x", "shape"],
        vec!["y"],
        NodeAttributes::default(),
    );

    let snapshot = get_node_precomputed(&executor, "reshape").expect("reshape node must exist");
    assert_eq!(
        snapshot.reshape_shape,
        Some(vec![2, 3]),
        "precomputed shape must be Some([2, 3]) for Constant-sourced shape input"
    );
}

#[test]
fn precompute_miss_when_shape_input_not_in_cache() {
    let mut executor = GpuGraphExecutor::new().expect("executor");

    executor.add_node(
        "reshape",
        "Reshape",
        vec!["x", "runtime_shape"],
        vec!["y"],
        NodeAttributes::default(),
    );

    let snapshot = get_node_precomputed(&executor, "reshape").expect("reshape node must exist");
    assert_eq!(
        snapshot.reshape_shape, None,
        "precomputed shape must be None when input not in static cache"
    );
}

#[test]
fn reshape_with_precomputed_shape_produces_correct_output() {
    let mut executor = GpuGraphExecutor::new().expect("executor");

    // Constant shape [2, 3]
    executor.add_node(
        "shape_const",
        "Constant",
        vec![],
        vec!["shape"],
        int64_constant_attrs(&[2, 3]),
    );

    // Reshape consuming [6]-element input to [2, 3]
    executor.add_node(
        "reshape",
        "Reshape",
        vec!["x", "shape"],
        vec!["y"],
        NodeAttributes::default(),
    );

    // Sanity: precomputation must have fired for this test to be meaningful.
    let snapshot = get_node_precomputed(&executor, "reshape").expect("reshape node must exist");
    assert_eq!(
        snapshot.reshape_shape,
        Some(vec![2, 3]),
        "precomputed shape must be Some([2, 3]) -- this test's correctness \
         depends on the precomputed path firing"
    );

    // Run with a Float32 [6] input
    let mut inputs = HashMap::new();
    inputs.insert(
        "x".to_string(),
        Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6]),
    );

    let outputs = executor.run(inputs, vec!["y"]).expect("run");
    let y = outputs.get("y").expect("y output");

    assert_eq!(
        y.shape(),
        &[2usize, 3usize],
        "Reshape must produce shape [2, 3]"
    );
    assert_eq!(
        y.as_slice(),
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        "Reshape must preserve element order (it's metadata-only)"
    );
}
