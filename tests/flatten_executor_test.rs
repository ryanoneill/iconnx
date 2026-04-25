//! WS-2 M2.4: Flatten through the GpuGraphExecutor. ResNet-50's
//! classifier head uses `Flatten(axis=1)` after `GlobalAveragePool`
//! to feed the final `Gemm`. Tests the GPU dispatch (which is a pure
//! `GpuTensor::reshape` — no kernel) plus the CPU forward via
//! Identity passthrough.

#![cfg(feature = "cuda")]

use std::collections::HashMap;

use iconnx::attributes::NodeAttributes;
use iconnx::cuda::inference::GpuGraphExecutor;
use iconnx::tensor::Tensor;

#[test]
#[ignore = "requires CUDA GPU"]
fn flatten_axis_1_resnet_classifier_head_pattern() {
    // [N, C, 1, 1] → axis=1 → [N, C]. Standard ResNet-50 classifier.
    let mut executor = GpuGraphExecutor::new().expect("executor");
    executor.add_input("x".into(), vec![Some(2), Some(5), Some(1), Some(1)]);

    let mut attrs = NodeAttributes::new();
    attrs.add_int("axis".into(), 1);
    executor.add_node(
        "flatten",
        "Flatten",
        vec!["x"],
        vec!["out"],
        attrs,
    );

    executor.compile().expect("compile");

    let mut inputs = HashMap::new();
    let x_data: Vec<f32> = (0..10).map(|i| i as f32).collect();
    inputs.insert(
        "x".to_string(),
        Tensor::from_vec_f32(x_data, vec![2, 5, 1, 1]),
    );

    let outputs = executor.run(inputs, vec!["out"]).expect("run flatten");
    let out = outputs.get("out").expect("out");
    assert_eq!(out.shape(), &[2, 5]);
    assert_eq!(out.as_slice(), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn flatten_axis_0_collapses_to_row_vector() {
    let mut executor = GpuGraphExecutor::new().expect("executor");
    executor.add_input("x".into(), vec![Some(2), Some(3), Some(4)]);

    let mut attrs = NodeAttributes::new();
    attrs.add_int("axis".into(), 0);
    executor.add_node(
        "flatten",
        "Flatten",
        vec!["x"],
        vec!["out"],
        attrs,
    );
    executor.compile().expect("compile");

    let mut inputs = HashMap::new();
    inputs.insert(
        "x".to_string(),
        Tensor::from_vec_f32((0..24).map(|i| i as f32).collect(), vec![2, 3, 4]),
    );
    let outputs = executor.run(inputs, vec!["out"]).expect("run flatten");
    assert_eq!(outputs.get("out").unwrap().shape(), &[1, 24]);
}
