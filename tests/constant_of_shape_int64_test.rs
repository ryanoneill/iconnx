//! Regression test for WS-2 follow-up: ConstantOfShape with an Int64
//! `value` attribute (common after a Shape op) must flow through the
//! GPU executor's layout dispatch without hitting Tensor::as_slice()'s
//! "called on non-Float32 tensor" panic at tensor.rs:296.
//!
//! YOLOS-tiny's ViT embedding interpolation is the real-world case.
//! This test exercises a minimal synthetic graph so the regression
//! coverage doesn't depend on a 60MB external model.

#![cfg(feature = "cuda")]

use std::collections::HashMap;

use iconnx::attributes::NodeAttributes;
use iconnx::cuda::inference::GpuGraphExecutor;
use iconnx::tensor::Tensor;

#[test]
#[ignore = "requires CUDA GPU"]
fn constant_of_shape_int64_value_flows_through_gpu_executor() {
    let mut executor = GpuGraphExecutor::new().expect("executor");

    // Minimal graph: Shape(x) -> ConstantOfShape(value=Int64(42)).
    //
    //   x : Float32[2, 3] (input)
    //   shape_out : Int64[2] = Shape(x)       = [2, 3]
    //   out : Int64[2, 3] = ConstantOfShape(shape_out, value=42_i64)
    //
    // Pre-fix: dispatch_layout ConstantOfShape arm hit
    // `t.as_slice()[0]` where `t` is the Int64 value tensor → panic.
    executor.add_input("x".into(), vec![Some(2), Some(3)]);
    executor.add_node(
        "shape_node",
        "Shape",
        vec!["x"],
        vec!["shape_out"],
        NodeAttributes::new(),
    );

    let mut cof_attrs = NodeAttributes::new();
    cof_attrs.add_tensor(
        "value".to_string(),
        Tensor::from_vec_i64(vec![42_i64], vec![1]),
    );
    executor.add_node(
        "cof_node",
        "ConstantOfShape",
        vec!["shape_out"],
        vec!["out"],
        cof_attrs,
    );

    executor.compile().expect("compile");

    let mut inputs = HashMap::new();
    inputs.insert(
        "x".to_string(),
        Tensor::from_vec_f32(vec![0.0_f32; 6], vec![2, 3]),
    );
    let outputs = executor
        .run(inputs, vec!["out"])
        .expect("run ConstantOfShape(Int64) graph");

    let out = outputs.get("out").expect("out present");
    assert_eq!(out.shape(), &[2, 3]);
    let vals = out.as_slice_i64();
    assert_eq!(vals.len(), 6);
    assert!(
        vals.iter().all(|&v| v == 42),
        "expected all-42 Int64 output, got {:?}",
        vals
    );
}
