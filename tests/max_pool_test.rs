//! WS-2 M2.2: MaxPool through GpuGraphExecutor. ResNet-50 stem
//! shape (3x3 kernel, stride 2, pads 1) is the primary target;
//! also covers the simpler 2x2 stride 2 case to exercise the
//! kernel without padding. Numerical match against CPU forward at
//! exact equality (max is integer-stable for integer-valued inputs).

#![cfg(feature = "cuda")]

use std::collections::HashMap;

use iconnx::attributes::NodeAttributes;
use iconnx::cuda::inference::GpuGraphExecutor;
use iconnx::tensor::Tensor;

fn maxpool_attrs(kernel: &[i64], strides: &[i64], pads: &[i64]) -> NodeAttributes {
    let mut a = NodeAttributes::new();
    a.add_ints("kernel_shape".into(), kernel.to_vec());
    a.add_ints("strides".into(), strides.to_vec());
    a.add_ints("pads".into(), pads.to_vec());
    a
}

#[test]
#[ignore = "requires CUDA GPU"]
fn max_pool_2x2_stride_2_no_pad_via_executor() {
    let mut executor = GpuGraphExecutor::new().expect("executor");
    executor.add_input("x".into(), vec![Some(1), Some(1), Some(4), Some(4)]);
    executor.add_node(
        "pool",
        "MaxPool",
        vec!["x"],
        vec!["out"],
        maxpool_attrs(&[2, 2], &[2, 2], &[0, 0, 0, 0]),
    );
    executor.compile().expect("compile");

    let mut inputs = HashMap::new();
    inputs.insert(
        "x".to_string(),
        Tensor::from_vec_f32((0..16).map(|i| i as f32).collect(), vec![1, 1, 4, 4]),
    );
    let outputs = executor.run(inputs, vec!["out"]).expect("run maxpool");
    let out = outputs.get("out").expect("out");
    assert_eq!(out.shape(), &[1, 1, 2, 2]);
    assert_eq!(out.as_slice(), vec![5.0, 7.0, 13.0, 15.0]);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn max_pool_3x3_stride_2_pad_1_resnet_stem() {
    let mut executor = GpuGraphExecutor::new().expect("executor");
    executor.add_input("x".into(), vec![Some(1), Some(1), Some(4), Some(4)]);
    executor.add_node(
        "pool",
        "MaxPool",
        vec!["x"],
        vec!["out"],
        maxpool_attrs(&[3, 3], &[2, 2], &[1, 1, 1, 1]),
    );
    executor.compile().expect("compile");

    let mut inputs = HashMap::new();
    inputs.insert(
        "x".to_string(),
        Tensor::from_vec_f32((0..16).map(|i| i as f32).collect(), vec![1, 1, 4, 4]),
    );
    let outputs = executor.run(inputs, vec!["out"]).expect("run maxpool");
    let out = outputs.get("out").expect("out");
    assert_eq!(out.shape(), &[1, 1, 2, 2]);
    // Same expected output as the CPU test for this kernel/shape combo.
    assert_eq!(out.as_slice(), vec![5.0, 7.0, 13.0, 15.0]);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn max_pool_resnet_stem_full_size_pattern() {
    // The actual ResNet-50 stem dimensions: 1×1×8×8 stand-in for the
    // [N, 64, 112, 112] → [N, 64, 56, 56] pattern.
    // floor((8 + 2 - 3)/2 + 1) = floor(7/2)+1 = 3+1 = 4.
    let mut executor = GpuGraphExecutor::new().expect("executor");
    executor.add_input("x".into(), vec![Some(1), Some(1), Some(8), Some(8)]);
    executor.add_node(
        "pool",
        "MaxPool",
        vec!["x"],
        vec!["out"],
        maxpool_attrs(&[3, 3], &[2, 2], &[1, 1, 1, 1]),
    );
    executor.compile().expect("compile");

    let mut inputs = HashMap::new();
    inputs.insert(
        "x".to_string(),
        Tensor::from_vec_f32((0..64).map(|i| i as f32).collect(), vec![1, 1, 8, 8]),
    );
    let outputs = executor.run(inputs, vec!["out"]).expect("run maxpool");
    let out = outputs.get("out").expect("out");
    assert_eq!(out.shape(), &[1, 1, 4, 4]);

    // Cross-check against CPU forward.
    let cpu_input =
        Tensor::from_vec_f32((0..64).map(|i| i as f32).collect(), vec![1, 1, 8, 8]);
    let cpu_out = iconnx::operators::max_pool::MaxPool::forward(
        &[cpu_input],
        &maxpool_attrs(&[3, 3], &[2, 2], &[1, 1, 1, 1]),
    );
    assert_eq!(out.as_slice(), cpu_out.as_slice());
}
