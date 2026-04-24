//! Regression test for WS-2 YOLOS-tiny narrow fix: Div with Int64 inputs.
//!
//! ViT's dynamic interpolation computes target patch-grid sizes via
//! `Div(int64_dim, int64_scale)`. Pre-fix, gpu_div was f32-only and the
//! executor surfaced `"Div (output=...): data_f32 called on int64
//! tensor"` via the dispatch-level diagnostic wrap (b3515b1). This test
//! runs a minimal synthetic Int64 Div graph end-to-end so the
//! regression coverage doesn't depend on a 60MB external model.

#![cfg(feature = "cuda")]

use std::collections::HashMap;

use iconnx::attributes::NodeAttributes;
use iconnx::cuda::inference::GpuGraphExecutor;
use iconnx::tensor::Tensor;

#[test]
#[ignore = "requires CUDA GPU"]
fn div_int64_graph_flows_through_gpu_executor() {
    let mut executor = GpuGraphExecutor::new().expect("executor");

    // Minimal graph: Div(a_i64, b_i64) → out_i64.
    // Uses initializers as inputs so the full executor dispatch path runs
    // (including the dispatch-level op-name context wrap).
    executor
        .add_initializer(
            "a".into(),
            &Tensor::from_vec_i64(vec![100_i64, 50, 24, 7], vec![4]),
        )
        .expect("add_initializer a");
    executor
        .add_initializer(
            "b".into(),
            &Tensor::from_vec_i64(vec![25_i64, 10, 8, 2], vec![4]),
        )
        .expect("add_initializer b");
    executor.add_node(
        "div_node",
        "Div",
        vec!["a", "b"],
        vec!["out"],
        NodeAttributes::new(),
    );

    executor.compile().expect("compile");

    let outputs = executor
        .run(HashMap::new(), vec!["out"])
        .expect("run Int64 Div graph");
    let out = outputs.get("out").expect("out present");
    assert_eq!(out.shape(), &[4]);
    let vals = out.as_slice_i64();
    // Truncating signed division: 100/25=4, 50/10=5, 24/8=3, 7/2=3.
    assert_eq!(vals, vec![4, 5, 3, 3]);
}
