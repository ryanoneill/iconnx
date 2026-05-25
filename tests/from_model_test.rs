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
