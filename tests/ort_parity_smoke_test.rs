//! Smoke test proving the live-ORT parity harness works end to end.
//!
//! This is the FIRST live `ort::` use in the codebase — all prior parity
//! tests replayed pre-recorded fixtures. We exercise ORT's CPU EP (the
//! parity oracle for WS-6) through [`ort_parity::OrtSession`] on the
//! committed `identity_2x2` fixture (Identity, x[2,2] -> y).
//!
//! The harness itself only needs ORT (CPU EP, no GPU), but it lives in a
//! `cuda`-gated test crate and is `#[ignore]`d so it runs in the WS-6
//! `--ignored` lane on this GPU box alongside the per-op parity tests.

#![cfg(feature = "cuda")]

#[path = "common/ort_parity.rs"]
mod ort_parity;

#[test]
#[ignore = "requires CUDA GPU"]
fn ort_runs_identity_graph() {
    // Load the committed Identity fixture (x[2,2] -> y).
    let mut sess = ort_parity::OrtSession::open(std::path::Path::new(
        "tests/fixtures/ws6_op/identity_2x2.onnx",
    ))
    .expect("ort session");
    let (out, shape) = sess
        .run_f32(&[("x", &[1.0f32, 2.0, 3.0, 4.0], vec![2usize, 2])], "y")
        .expect("run");
    assert_eq!(shape, vec![2, 2]);
    assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn assert_close_passes_within_tolerance() {
    // No GPU/ORT needed — pure numeric helper, so it runs in the default lane.
    ort_parity::assert_close(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], 1e-5, 1e-5);
    ort_parity::assert_close(&[1.000_001, 2.0], &[1.0, 2.0], 1e-3, 1e-3);
}

#[test]
#[should_panic(expected = "assert_close")]
fn assert_close_panics_outside_tolerance() {
    ort_parity::assert_close(&[1.0, 5.0], &[1.0, 2.0], 1e-5, 1e-5);
}

#[test]
fn seeded_is_deterministic_and_unit_scale() {
    let a = ort_parity::seeded(8, 42);
    let b = ort_parity::seeded(8, 42);
    assert_eq!(a, b, "same seed must produce identical sequences");
    assert_eq!(a.len(), 8);

    let c = ort_parity::seeded(8, 43);
    assert_ne!(a, c, "different seeds must produce different sequences");

    // N(0,1) values are finite and stay in a sane band for a small sample.
    for v in &a {
        assert!(v.is_finite(), "seeded values must be finite, got {v}");
        assert!(v.abs() < 8.0, "seeded N(0,1) value out of band: {v}");
    }
}
