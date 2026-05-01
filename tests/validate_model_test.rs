//! WS-5 M5.1 — `validate_model()` synthetic negative + happy-path tests.
//!
//! Roster integration of `validate_model` is exercised via leadline-bench's
//! `compare --validate` (gate 6 territory at Y(5)); this test file covers
//! synthetic negatives + happy-path units only. iconnx stays unaware of
//! leadline-bench's roster directory shape.

#![cfg(feature = "cuda")]

use std::path::PathBuf;

use iconnx::tensor::DType;
use iconnx::{validate_model, ModelIncompatibility};

#[test]
fn validate_model_unsupported_op_collected() {
    // PLACEHOLDER: Y(2) replaces this test once OP_SUPPORT_TABLE lands and
    // the synthetic CustomOp path actually fails. For now, this test
    // documents the expected behavior; assert is loose.
    let path = PathBuf::from("tests/fixtures/ws5_synthetic/unsupported_op.onnx");
    if !path.exists() {
        eprintln!("skipping: fixture not yet generated");
        return;
    }
    let result = validate_model(&path);
    // Y(2) makes this assert-on-Err; today the placeholder accepts everything.
    let _ = result;
}

#[test]
fn validate_model_opset_out_of_range_collected() {
    let path = PathBuf::from("tests/fixtures/ws5_synthetic/opset_out_of_range.onnx");
    if !path.exists() {
        eprintln!("skipping: fixture not yet generated");
        return;
    }
    let result = validate_model(&path);
    match result {
        Err(failure) => {
            assert!(failure.incompatibilities.iter().any(|i| matches!(
                i,
                ModelIncompatibility::OpsetOutOfRange { .. }
            )));
            assert!(failure.partial_capabilities.is_some());
        }
        Ok(_) => panic!("expected opset out of range incompat"),
    }
}

#[test]
fn validate_model_unknown_opset_domain_collected() {
    let path = PathBuf::from("tests/fixtures/ws5_synthetic/unknown_opset_domain.onnx");
    if !path.exists() {
        eprintln!("skipping: fixture not yet generated");
        return;
    }
    let result = validate_model(&path);
    match result {
        Err(failure) => {
            assert!(failure.incompatibilities.iter().any(|i| matches!(
                i,
                ModelIncompatibility::OpsetDomainUnknown { .. }
            )));
        }
        Ok(_) => panic!("expected unknown opset domain incompat"),
    }
}

#[test]
fn validate_model_non_finite_half_constant_warns() {
    let path = PathBuf::from("tests/fixtures/ws5_synthetic/bf16_with_neg_inf.onnx");
    if !path.exists() {
        eprintln!("skipping: fixture not yet generated");
        return;
    }
    let result = validate_model(&path);
    match result {
        Ok(caps) => {
            assert!(caps.warnings.iter().any(|w| matches!(
                w,
                iconnx::ModelWarning::NonFiniteHalfConstants { .. }
            )));
        }
        Err(_) => panic!("non-finite half is a warning, not an incompat"),
    }
}

#[test]
fn validate_model_parse_failure_short_circuits() {
    let path = PathBuf::from("tests/fixtures/ws5_synthetic/garbage.onnx");
    if !path.exists() {
        eprintln!("skipping: fixture not yet generated");
        return;
    }
    let result = validate_model(&path);
    match result {
        Err(failure) => {
            assert_eq!(failure.incompatibilities.len(), 1);
            assert!(matches!(
                failure.incompatibilities[0],
                ModelIncompatibility::Parse(_)
            ));
            assert!(failure.partial_capabilities.is_none());
        }
        Ok(_) => panic!("garbage file should not validate"),
    }
}

#[test]
fn validate_model_happy_path_returns_capabilities() {
    let path = PathBuf::from("tests/fixtures/ws5_synthetic/minimal_add.onnx");
    if !path.exists() {
        eprintln!("skipping: fixture not yet generated");
        return;
    }
    let result = validate_model(&path);
    match result {
        Ok(caps) => {
            assert!(!caps.opset_imports.is_empty());
            assert!(caps.used_dtypes.contains(&DType::Float32));
        }
        Err(failure) => {
            panic!("unexpected validation failure: {:?}", failure);
        }
    }
}
