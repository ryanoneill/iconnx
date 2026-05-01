//! Drift-prevention: assert that MIN_OPSET_SUPPORTED / MAX_OPSET_SUPPORTED
//! match the actual opset_imports range across the leadline-bench roster.
//!
//! Iconnx does not own the roster (leadline-bench does). This test is
//! anchored on the *expected derivation* — if the implementer changes the
//! constants without updating the roster comment, the discrepancy is
//! visible in code review. The test cannot programmatically read the
//! roster paths (cross-repo); it asserts the values match the documented
//! derivation context.
//!
//! Updates required when the roster opset range changes:
//! 1. Verify the new low/high in leadline-bench's 10 ModelProfile gates.
//! 2. Update MIN/MAX_OPSET_SUPPORTED in src/opset.rs.
//! 3. Update the EXPECTED_* constants below.
//! 4. Add a memory entry noting the version transition.

#![cfg(feature = "cuda")]

use iconnx::{MAX_OPSET_SUPPORTED, MIN_OPSET_SUPPORTED};

/// Documented expected lower bound from the leadline-bench roster (2026-04-29).
///
/// Contributed by 7 of 10 roster models: BERT, BERT-BF16, DistilBERT-INT8,
/// ResNet-50, Whisper encoder/decoder (FP32 + BF16).
const EXPECTED_MIN: i64 = 11;

/// Documented expected upper bound from the leadline-bench roster (2026-04-29).
///
/// Contributed by kokoro-v1.0.onnx (the only roster model declaring
/// opset >= 14).
const EXPECTED_MAX: i64 = 20;

#[test]
fn min_opset_supported_matches_documented_derivation() {
    assert_eq!(
        MIN_OPSET_SUPPORTED, EXPECTED_MIN,
        "MIN_OPSET_SUPPORTED drift detected. If extending support: \
         (1) verify roster lower bound, (2) update src/opset.rs, \
         (3) update EXPECTED_MIN above."
    );
}

#[test]
fn max_opset_supported_matches_documented_derivation() {
    assert_eq!(
        MAX_OPSET_SUPPORTED, EXPECTED_MAX,
        "MAX_OPSET_SUPPORTED drift detected. If extending support: \
         (1) verify roster upper bound, (2) update src/opset.rs, \
         (3) update EXPECTED_MAX above. Conservative MAX policy: only \
         claim what's been validated."
    );
}

#[test]
fn min_strictly_less_than_max() {
    const { assert!(MIN_OPSET_SUPPORTED < MAX_OPSET_SUPPORTED) };
}
