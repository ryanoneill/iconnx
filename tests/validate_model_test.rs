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
    // Y(2) tightened: validate_model now consults OP_SUPPORT_TABLE and
    // surfaces UnsupportedOp for any op_type with no dispatch arm. The
    // synthetic CustomOpFoo fixture (a single-node graph with op_type
    // "CustomOpFoo", absent from OP_SUPPORT_TABLE) must fail validation
    // with a matching incompatibility.
    let path = PathBuf::from("tests/fixtures/ws5_synthetic/unsupported_op.onnx");
    if !path.exists() {
        eprintln!("skipping: fixture not yet generated");
        return;
    }
    let result = validate_model(&path);
    match result {
        Err(failure) => {
            // Y(2) review fix: tighten beyond op_type-only — also assert
            // `count >= 1` and `sample_node_names` non-empty so the test
            // exercises the full UnsupportedOp variant payload, not just
            // the discriminant. A regression that drops sample_node_names
            // collection (e.g. via an off-by-one in the validate walk)
            // would slip past the looser `op_type == ..` check.
            let found = failure.incompatibilities.iter().any(|i| matches!(
                i,
                ModelIncompatibility::UnsupportedOp {
                    op_type,
                    count,
                    sample_node_names,
                } if op_type == "CustomOpFoo"
                    && *count >= 1
                    && !sample_node_names.is_empty()
            ));
            assert!(
                found,
                "expected UnsupportedOp {{ op_type: \"CustomOpFoo\", count >= 1, \
                 sample_node_names non-empty }}; got: {:?}",
                failure.incompatibilities
            );
        }
        Ok(_) => panic!("expected unsupported op incompat"),
    }
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

#[test]
fn validate_model_fp64_initializer_emits_dtype_downcast_warning() {
    // Issue #2 review feedback: confirm `Tensor::Float64 → DType::Float32`
    // surfaces a structured `ModelWarning::DTypeDowncast` (one per model,
    // de-duplicated via `already_warned_fp64` flag in validate_model).
    let path = PathBuf::from("tests/fixtures/ws5_synthetic/fp64_initializer.onnx");
    if !path.exists() {
        eprintln!("skipping: fixture not yet generated");
        return;
    }
    let result = validate_model(&path);
    let caps = match result {
        Ok(c) => c,
        Err(failure) => panic!("unexpected validation failure: {:?}", failure),
    };
    let downcast_warnings: Vec<_> = caps
        .warnings
        .iter()
        .filter(|w| matches!(w, iconnx::ModelWarning::DTypeDowncast { .. }))
        .collect();
    assert_eq!(
        downcast_warnings.len(),
        1,
        "expected exactly one DTypeDowncast warning (de-duplicated per model), got {}",
        downcast_warnings.len()
    );
    match downcast_warnings[0] {
        iconnx::ModelWarning::DTypeDowncast { from, to } => {
            assert_eq!(*from, "float64");
            assert_eq!(*to, DType::Float32);
        }
        _ => unreachable!("filtered above"),
    }
    // FP64 initializer gets reported under Float32 in `used_dtypes`.
    assert!(caps.used_dtypes.contains(&DType::Float32));
}

#[test]
fn validate_model_int64_input_surfaced_via_io_walk() {
    let path = PathBuf::from("tests/fixtures/ws5_synthetic/int64_input_fp32_init.onnx");
    if !path.exists() {
        eprintln!("skipping: fixture not yet generated");
        return;
    }
    let result = validate_model(&path);
    let caps = match result {
        Ok(c) => c,
        Err(failure) => failure.partial_capabilities.expect("partial caps populated"),
    };
    assert!(
        caps.used_dtypes.contains(&iconnx::tensor::DType::Int64),
        "INT64 input should be surfaced via IO-walk; used_dtypes = {:?}",
        caps.used_dtypes
    );
    assert!(
        caps.used_dtypes.contains(&iconnx::tensor::DType::Float32),
        "FP32 initializer should also surface; used_dtypes = {:?}",
        caps.used_dtypes
    );
}

#[test]
#[ignore = "requires CUDA GPU"]
fn validate_model_for_hardware_bf16_full_path_compose() {
    // Issue #1 review feedback: exercise the full
    // `validate_model_for_hardware` orchestration with a real
    // `IconnxCudaContext`. Uses the BF16 fixture (which has a BF16
    // initializer) so the dtype walk hits the hardware-floor branch.
    //
    // On sm_80+ the call should compose cleanly; the BF16 hardware floor
    // is met, and the only surfaced observation is the
    // `NonFiniteHalfConstants` warning (the fixture has a -inf bit pattern).
    // On sm_75 the same call would surface a `HardwareFloorUnmet`
    // incompatibility — we don't cc-gate this test (the harness only runs
    // it on a GPU box at all), but we document the expectation here so the
    // sub-sm_80 path is preserved by Y(2)+ refactors.
    use iconnx::{validate_model_for_hardware, IconnxCudaContext, ModelWarning};

    let path = PathBuf::from("tests/fixtures/ws5_synthetic/bf16_with_neg_inf.onnx");
    if !path.exists() {
        eprintln!("skipping: fixture not yet generated");
        return;
    }
    let ctx = IconnxCudaContext::new().expect("CUDA context required for this test");
    let (cc_major, _) = ctx.compute_capability();

    let result = validate_model_for_hardware(&path, &ctx);

    if cc_major >= 8 {
        // Expected sm_80+ path: BF16 floor met, only the non-finite warning.
        let caps = result.expect("BF16 fixture should validate on sm_80+");
        assert!(
            caps.warnings
                .iter()
                .any(|w| matches!(w, ModelWarning::NonFiniteHalfConstants { .. })),
            "expected NonFiniteHalfConstants warning on bf16_with_neg_inf fixture",
        );
    } else {
        // Sub-sm_80 path: BF16 floor unmet, validation fails with
        // partial_capabilities populated (collect-all preserves caps).
        let failure = result.expect_err("BF16 fixture must fail on sm_75");
        assert!(failure.incompatibilities.iter().any(|i| matches!(
            i,
            iconnx::ModelIncompatibility::HardwareFloorUnmet {
                dtype: DType::BFloat16,
                ..
            }
        )));
        assert!(failure.partial_capabilities.is_some());
    }
}
