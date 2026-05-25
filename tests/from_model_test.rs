//! WS-6 M6.1 Task 1 — `validate_parsed_model(&OnnxModel)` entry tests.
//!
//! These tests exercise the newly-extracted post-parse validation entry so
//! callers that already hold a parsed model (e.g. `from_model`) can run a
//! fail-fast op-name pre-flight without re-parsing.

#![cfg(feature = "cuda")]

use iconnx::onnx_parser::OnnxParser;
use iconnx::validate::validate_parsed_model;
use iconnx::ModelIncompatibility;

/// POSITIVE case: Relu is in OP_SUPPORT_TABLE → validation must succeed.
#[test]
fn validate_parsed_model_accepts_supported_model() {
    let model =
        OnnxParser::parse_file("tests/fixtures/ws6_op/relu_1x4.onnx").expect("parse failed");
    assert!(
        validate_parsed_model(&model).is_ok(),
        "Relu must validate as supported"
    );
}

/// GPU equivalence test: `GpuGraphExecutor::from_model` must produce
/// byte-identical outputs to hand-wired construction and must apply Relu
/// correctly to a mixed positive/negative input.
#[test]
#[ignore = "requires CUDA GPU"]
fn from_model_matches_hand_wiring() {
    use std::collections::HashMap;
    use iconnx::{GpuGraphExecutor, OnnxParser, Tensor};

    let model =
        OnnxParser::parse_file("tests/fixtures/ws6_op/relu_1x4.onnx").expect("parse");

    // Path A: from_model
    let exec_a = GpuGraphExecutor::from_model(&model).expect("from_model");

    // Path B: hand-wiring (mirrors the from_model implementation exactly)
    let mut exec_b = GpuGraphExecutor::new().expect("new");
    for (n, s) in model.input_shapes() {
        exec_b.add_input(n, s);
    }
    for (name, t) in model.extract_weights().expect("weights") {
        exec_b.add_initializer(name, &t).expect("init");
    }
    for (i, node) in model
        .computation_graph()
        .expect("graph")
        .nodes()
        .iter()
        .enumerate()
    {
        exec_b.add_node(
            &format!("n{i}"),
            &node.op_type,
            node.inputs.iter().map(|s| s.as_str()).collect(),
            node.outputs.iter().map(|s| s.as_str()).collect(),
            node.attributes.clone(),
        );
    }

    let input = Tensor::from_vec_f32(vec![-1.0, 2.0, -3.0, 4.0], vec![1, 4]);
    let mut ia = HashMap::new();
    ia.insert("x".to_string(), input.clone());
    let mut ib = HashMap::new();
    ib.insert("x".to_string(), input);

    let oa = exec_a.run(ia, vec!["y"]).expect("run a");
    let ob = exec_b.run(ib, vec!["y"]).expect("run b");

    // as_slice() returns Vec<f32>
    assert_eq!(oa["y"].as_slice(), ob["y"].as_slice());
    assert_eq!(oa["y"].as_slice(), vec![0.0_f32, 2.0, 0.0, 4.0]);
}

/// NEGATIVE case: CustomOpFoo is absent from OP_SUPPORT_TABLE →
/// `validate_parsed_model` must surface `UnsupportedOp` with count >= 1
/// and non-empty sample_node_names.
#[test]
fn validate_parsed_model_rejects_unsupported_op() {
    let model = OnnxParser::parse_file("tests/fixtures/ws5_synthetic/unsupported_op.onnx")
        .expect("parse failed");
    match validate_parsed_model(&model) {
        Err(failure) => {
            let found = failure.incompatibilities.iter().any(|i| {
                matches!(
                    i,
                    ModelIncompatibility::UnsupportedOp {
                        op_type,
                        count,
                        sample_node_names,
                    } if op_type == "CustomOpFoo"
                        && count >= &1
                        && !sample_node_names.is_empty()
                )
            });
            assert!(
                found,
                "expected UnsupportedOp {{ op_type: \"CustomOpFoo\", count >= 1, \
                 sample_node_names non-empty }}; got: {:?}",
                failure.incompatibilities
            );
        }
        Ok(_) => panic!("expected unsupported-op validation failure but got Ok"),
    }
}
