//! WS-2 YOLOS-tiny: Resize must accept the opset-13+ `sizes` input
//! (4th positional input), not just `scales` (3rd). ViT's dynamic
//! embedding interpolation uses the `sizes` branch with an empty
//! `scales` placeholder.
//!
//! Coverage per leadline's spec:
//!   1. `resize_with_sizes_input_respects_target_shape` — [1,3,24,24]
//!      upscaled via sizes [1,3,48,48], mode=linear → [1,3,48,48].
//!   2. `resize_with_scales_input_still_works` — existing scales path
//!      preserved (scales=[1,1,2,2] → [1,3,48,48]).
//!   3. `resize_with_both_empty_errors` — spec violation surfaces as
//!      an error rather than a silent pick.

#![cfg(feature = "cuda")]

use std::collections::HashMap;

use iconnx::attributes::NodeAttributes;
use iconnx::cuda::inference::GpuGraphExecutor;
use iconnx::tensor::Tensor;

fn linear_resize_attrs() -> NodeAttributes {
    let mut attrs = NodeAttributes::new();
    attrs.add_string("mode".to_string(), "linear".to_string());
    attrs.add_string(
        "coordinate_transformation_mode".to_string(),
        "half_pixel".to_string(),
    );
    attrs
}

#[test]
#[ignore = "requires CUDA GPU"]
fn resize_with_sizes_input_respects_target_shape() {
    let mut executor = GpuGraphExecutor::new().expect("executor");

    // Input X: [1, 3, 24, 24] f32.
    // roi: 0-element f32 placeholder.
    // scales: 0-element f32 placeholder (spec allows this when sizes is used).
    // sizes: [1, 3, 48, 48] i64 — explicit target shape.
    executor.add_input("x".into(), vec![Some(1), Some(3), Some(24), Some(24)]);
    executor
        .add_initializer("roi".into(), &Tensor::from_vec_f32(Vec::new(), vec![0]))
        .expect("roi");
    executor
        .add_initializer("scales".into(), &Tensor::from_vec_f32(Vec::new(), vec![0]))
        .expect("scales");
    executor
        .add_initializer(
            "sizes".into(),
            &Tensor::from_vec_i64(vec![1_i64, 3, 48, 48], vec![4]),
        )
        .expect("sizes");

    executor.add_node(
        "resize_node",
        "Resize",
        vec!["x", "roi", "scales", "sizes"],
        vec!["out"],
        linear_resize_attrs(),
    );

    executor.compile().expect("compile");

    let mut inputs = HashMap::new();
    let nelems = 3 * 24 * 24;
    let x_data: Vec<f32> = (0..nelems).map(|i| i as f32 * 0.001).collect();
    inputs.insert(
        "x".to_string(),
        Tensor::from_vec_f32(x_data, vec![1, 3, 24, 24]),
    );

    let outputs = executor
        .run(inputs, vec!["out"])
        .expect("run Resize with sizes input");
    let out = outputs.get("out").expect("out");
    assert_eq!(
        out.shape(),
        &[1, 3, 48, 48],
        "Resize must honor sizes input literally"
    );
}

#[test]
#[ignore = "requires CUDA GPU"]
fn resize_with_scales_input_still_works() {
    // Regression: the existing scales path must not be broken by the
    // sizes branch. Same target shape expressed via scales=[1,1,2,2].
    let mut executor = GpuGraphExecutor::new().expect("executor");

    executor.add_input("x".into(), vec![Some(1), Some(3), Some(24), Some(24)]);
    executor
        .add_initializer("roi".into(), &Tensor::from_vec_f32(Vec::new(), vec![0]))
        .expect("roi");
    executor
        .add_initializer(
            "scales".into(),
            &Tensor::from_vec_f32(vec![1.0_f32, 1.0, 2.0, 2.0], vec![4]),
        )
        .expect("scales");
    executor.add_node(
        "resize_node",
        "Resize",
        vec!["x", "roi", "scales"],
        vec!["out"],
        linear_resize_attrs(),
    );

    executor.compile().expect("compile");

    let mut inputs = HashMap::new();
    let nelems = 3 * 24 * 24;
    let x_data: Vec<f32> = (0..nelems).map(|i| i as f32 * 0.001).collect();
    inputs.insert(
        "x".to_string(),
        Tensor::from_vec_f32(x_data, vec![1, 3, 24, 24]),
    );

    let outputs = executor
        .run(inputs, vec!["out"])
        .expect("run Resize with scales input");
    let out = outputs.get("out").expect("out");
    assert_eq!(out.shape(), &[1, 3, 48, 48]);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn resize_with_both_empty_errors() {
    // ONNX spec: exactly one of scales / sizes must be provided. Both
    // empty is a corrupt graph — must surface an error (not silently
    // pass through as identity).
    let mut executor = GpuGraphExecutor::new().expect("executor");

    executor.add_input("x".into(), vec![Some(1), Some(3), Some(24), Some(24)]);
    executor
        .add_initializer("roi".into(), &Tensor::from_vec_f32(Vec::new(), vec![0]))
        .expect("roi");
    executor
        .add_initializer("scales".into(), &Tensor::from_vec_f32(Vec::new(), vec![0]))
        .expect("scales");
    executor
        .add_initializer("sizes".into(), &Tensor::from_vec_i64(Vec::new(), vec![0]))
        .expect("sizes");
    executor.add_node(
        "resize_node",
        "Resize",
        vec!["x", "roi", "scales", "sizes"],
        vec!["out"],
        linear_resize_attrs(),
    );

    executor.compile().expect("compile");

    let mut inputs = HashMap::new();
    let nelems = 3 * 24 * 24;
    let x_data: Vec<f32> = (0..nelems).map(|i| i as f32 * 0.001).collect();
    inputs.insert(
        "x".to_string(),
        Tensor::from_vec_f32(x_data, vec![1, 3, 24, 24]),
    );

    let result = executor.run(inputs, vec!["out"]);
    match result {
        Err(e) => {
            let msg = format!("{}", e);
            assert!(
                msg.contains("Resize") && (msg.contains("scales") || msg.contains("sizes")),
                "expected Resize scales/sizes spec-violation error, got: {}",
                msg
            );
        }
        Ok(_) => panic!("expected error for Resize with both scales and sizes empty"),
    }
}
