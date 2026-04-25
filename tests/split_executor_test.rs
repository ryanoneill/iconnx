//! WS-2 M2.5: Split is iconnx's first genuine multi-output op (LSTM is
//! aliased — single tensor under multiple names). Verifies the
//! `Vec<GpuTensor>` dispatch path through `GpuGraphExecutor`: each
//! Split output is independently allocated and consumable downstream.

#![cfg(feature = "cuda")]

use std::collections::HashMap;

use iconnx::attributes::NodeAttributes;
use iconnx::cuda::inference::GpuGraphExecutor;
use iconnx::tensor::Tensor;

#[test]
#[ignore = "requires CUDA GPU"]
fn split_three_way_via_explicit_split_input_through_executor() {
    let mut executor = GpuGraphExecutor::new().expect("executor");

    // Input X: [6, 2] f32. Split along axis 0 with sizes [1, 3, 2].
    executor.add_input("x".into(), vec![Some(6), Some(2)]);
    executor
        .add_initializer(
            "split_sizes".into(),
            &Tensor::from_vec_i64(vec![1, 3, 2], vec![3]),
        )
        .expect("split sizes");

    let mut attrs = NodeAttributes::new();
    attrs.add_int("axis".into(), 0);
    executor.add_node(
        "split_node",
        "Split",
        vec!["x", "split_sizes"],
        vec!["o0", "o1", "o2"],
        attrs,
    );

    executor.compile().expect("compile");

    let mut inputs = HashMap::new();
    let x_data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    inputs.insert(
        "x".to_string(),
        Tensor::from_vec_f32(x_data, vec![6, 2]),
    );

    let outputs = executor
        .run(inputs, vec!["o0", "o1", "o2"])
        .expect("run Split graph");

    let o0 = outputs.get("o0").expect("o0 present");
    let o1 = outputs.get("o1").expect("o1 present");
    let o2 = outputs.get("o2").expect("o2 present");

    // Each output is its own tensor (not aliased) with the right shape +
    // contents.
    assert_eq!(o0.shape(), &[1, 2]);
    assert_eq!(o0.as_slice(), vec![0.0, 1.0]);

    assert_eq!(o1.shape(), &[3, 2]);
    assert_eq!(o1.as_slice(), vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

    assert_eq!(o2.shape(), &[2, 2]);
    assert_eq!(o2.as_slice(), vec![8.0, 9.0, 10.0, 11.0]);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn split_via_num_outputs_attribute_through_executor() {
    // No explicit split input; rely on num_outputs attribute (opset 18+).
    let mut executor = GpuGraphExecutor::new().expect("executor");

    executor.add_input("x".into(), vec![Some(6)]);

    let mut attrs = NodeAttributes::new();
    attrs.add_int("axis".into(), 0);
    attrs.add_int("num_outputs".into(), 3);
    executor.add_node(
        "split_node",
        "Split",
        vec!["x"],
        vec!["a", "b", "c"],
        attrs,
    );

    executor.compile().expect("compile");

    let mut inputs = HashMap::new();
    inputs.insert(
        "x".to_string(),
        Tensor::from_vec_f32((0..6).map(|i| i as f32).collect(), vec![6]),
    );

    let outputs = executor
        .run(inputs, vec!["a", "b", "c"])
        .expect("run Split via num_outputs");

    assert_eq!(outputs.get("a").unwrap().as_slice(), vec![0.0, 1.0]);
    assert_eq!(outputs.get("b").unwrap().as_slice(), vec![2.0, 3.0]);
    assert_eq!(outputs.get("c").unwrap().as_slice(), vec![4.0, 5.0]);
}
