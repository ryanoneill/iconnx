//! Cycle 2 — Unsqueeze precomputation tests.
//!
//! Each test verifies one aspect of the graph-build-time precomputation
//! for Unsqueeze. The first two tests inspect the precomputed field
//! directly via the testing facade; the third test runs an executor
//! end-to-end and verifies the output shape and values.

#![cfg(feature = "cuda")]

use iconnx::attributes::NodeAttributes;
use iconnx::cuda::inference::testing::get_node_precomputed;
use iconnx::cuda::GpuGraphExecutor;
use iconnx::tensor::Tensor;
use std::collections::HashMap;

/// Helper: build attributes whose "value" attribute is a 1-D Int64 tensor.
/// Used to construct Constant nodes in tests.
fn int64_constant_attrs(values: &[i64]) -> NodeAttributes {
    let mut attrs = NodeAttributes::default();
    let tensor = Tensor::from_vec_i64(values.to_vec(), vec![values.len()]);
    attrs.add_tensor("value".to_string(), tensor);
    attrs
}

#[test]
fn precompute_hit_from_constant() {
    let mut executor = GpuGraphExecutor::new().expect("executor");

    // Register a Constant node that produces the axes [0].
    executor.add_node(
        "axes_const",
        "Constant",
        vec![],
        vec!["axes"],
        int64_constant_attrs(&[0]),
    );

    // Register an Unsqueeze that consumes the Constant output.
    executor.add_node(
        "unsq",
        "Unsqueeze",
        vec!["x", "axes"],
        vec!["y"],
        NodeAttributes::default(),
    );

    // After add_node returns, the Unsqueeze's precomputed_unsqueeze_axes
    // field should contain [0] because the Constant was cached at
    // add_node time.
    let snapshot =
        get_node_precomputed(&executor, "unsq").expect("unsq node must exist in executor");
    assert_eq!(
        snapshot.unsqueeze_axes,
        Some(vec![0]),
        "precomputed axes must be Some([0]) when axes came from a Constant node \
         added earlier in topological order"
    );
}

#[test]
fn precompute_miss_when_axes_input_is_not_in_static_cache() {
    let mut executor = GpuGraphExecutor::new().expect("executor");

    // Register an Unsqueeze whose axes input comes from a name that was
    // NEVER added to the cache (no Constant, no initializer).
    executor.add_node(
        "unsq",
        "Unsqueeze",
        vec!["x", "runtime_axes"],
        vec!["y"],
        NodeAttributes::default(),
    );

    // Precomputation should return None — the cache lookup for
    // "runtime_axes" misses because no one ever added it.
    let snapshot =
        get_node_precomputed(&executor, "unsq").expect("unsq node must exist in executor");
    assert_eq!(
        snapshot.unsqueeze_axes, None,
        "precomputed axes must be None when the axes input is not in the static cache"
    );
}

#[test]
fn unsqueeze_with_precomputed_axes_produces_correct_output() {
    let mut executor = GpuGraphExecutor::new().expect("executor");

    // Register a Constant node that produces axes [0].
    executor.add_node(
        "axes_const",
        "Constant",
        vec![],
        vec!["axes"],
        int64_constant_attrs(&[0]),
    );

    // Register an Unsqueeze that consumes the Constant output.
    executor.add_node(
        "unsq",
        "Unsqueeze",
        vec!["x", "axes"],
        vec!["y"],
        NodeAttributes::default(),
    );

    // Verify that precomputation succeeded (sanity).
    let snapshot = get_node_precomputed(&executor, "unsq").expect("unsq node must exist");
    assert_eq!(
        snapshot.unsqueeze_axes,
        Some(vec![0]),
        "precomputed axes must be Some([0]) -- this test's correctness \
         depends on the precomputed path firing"
    );

    // Run with a Float32 [4] input. Unsqueeze with axes=[0] should
    // produce a [1, 4] output.
    let mut inputs = HashMap::new();
    inputs.insert(
        "x".to_string(),
        Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0], vec![4]),
    );

    let outputs = executor.run(inputs, vec!["y"]).expect("executor.run");
    let y = outputs.get("y").expect("y output missing");

    assert_eq!(
        y.shape(),
        &[1usize, 4usize],
        "Unsqueeze with axes=[0] on shape [4] must produce shape [1, 4]"
    );
    assert_eq!(
        y.as_slice(),
        vec![1.0f32, 2.0, 3.0, 4.0],
        "Unsqueeze must preserve element values — it's metadata-only"
    );
}
