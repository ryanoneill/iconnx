// Shared test-helper module: `#[path]`-included into multiple test binaries,
// each of which uses only a SUBSET of these helpers (e.g. the dynamic-shape
// guard test uses `OrtSession` + `assert_close` but not `seeded`, while the
// smoke test uses all three). Per-binary `dead_code` is therefore a false
// positive — this module-scoped allow is the canonical idiom for `tests/common`
// shared fixtures, NOT a silenced real warning.
#![allow(dead_code)]

//! Live-ORT parity test harness.
//!
//! This is the FIRST live `ort::` use in the codebase. All prior parity
//! tests replayed pre-recorded fixtures; WS-6 instead validates iconnx's
//! GPU outputs against ONNX Runtime *at test time*. ORT's CPU execution
//! provider is the oracle: it is deterministic, well-tested, and free of
//! the GPU-specific numerics we are trying to validate.
//!
//! Three pieces, reused by every downstream per-op and whole-model parity
//! test:
//!
//! * [`OrtSession`] — opens an ONNX file and runs named `f32` inputs
//!   through ORT, returning the requested output as `(data, shape)`.
//! * [`assert_close`] — NumPy `allclose` semantics (`|a-b| <= atol +
//!   rtol*|b|`) with a loud per-element diagnostic on failure.
//! * [`seeded`] — deterministic `N(0,1)` runtime inputs (Box-Muller over a
//!   small LCG), matching the WS dtype-fixture convention. Zeros is
//!   pathological for transformer attention, so fixtures use seeded noise.
//!
//! ## Confirmed `ort 2.0.0-rc.10` API (read from the vendored crate source)
//!
//! * `Session::builder()? -> SessionBuilder`
//! * `SessionBuilder::commit_from_file(path: impl AsRef<Path>) -> Result<Session>`
//!   (CPU EP is used because no GPU EP is registered on the builder)
//! * `Tensor::<f32>::from_array((shape, data))` where `shape: Vec<usize>`
//!   and `data: Vec<f32>` (the `(D: ToShape, Vec<T>)` impl).
//! * `session.run(inputs)` accepting `Vec<(Cow<str>, SessionInputValue)>`
//!   (the `ValueMap` named-input path) -> `Result<SessionOutputs>`.
//! * `SessionOutputs[&str] -> &DynValue`, then
//!   `DynValue::try_extract_tensor::<f32>() -> Result<(&Shape, &[f32])>`.
//! * `ort::tensor::Shape` derefs to `&[i64]` (dims).

use std::borrow::Cow;
use std::path::Path;

use ort::session::{Session, SessionInputValue};
use ort::value::Tensor;

/// A wrapper around an [`ort::session::Session`] running on ORT's CPU
/// execution provider — the parity oracle for WS-6.
pub struct OrtSession {
    session: Session,
}

impl OrtSession {
    /// Open an ONNX model file and commit a [`Session`] on the CPU EP.
    ///
    /// No execution provider is registered on the builder, so ORT falls
    /// back to the CPU EP — exactly the deterministic oracle we want.
    pub fn open(path: &Path) -> Result<Self, ort::Error> {
        let session = Session::builder()?.commit_from_file(path)?;
        Ok(Self { session })
    }

    /// Feed named `f32` inputs through the model and extract one named
    /// `f32` output as `(data, shape)`.
    ///
    /// Each input is `(name, data, shape)`; `data.len()` must equal the
    /// product of `shape`. The output is returned as a flat row-major
    /// `Vec<f32>` plus its shape as `Vec<usize>`.
    pub fn run_f32(
        &mut self,
        inputs: &[(&str, &[f32], Vec<usize>)],
        output: &str,
    ) -> Result<(Vec<f32>, Vec<usize>), ort::Error> {
        let mut input_values: Vec<(Cow<'_, str>, SessionInputValue<'_>)> =
            Vec::with_capacity(inputs.len());
        for (name, data, shape) in inputs {
            let tensor = Tensor::<f32>::from_array((shape.clone(), data.to_vec()))?;
            input_values.push((Cow::Owned((*name).to_owned()), tensor.into()));
        }

        let outputs = self.session.run(input_values)?;
        let value = outputs.get(output).ok_or_else(|| {
            ort::Error::new(format!(
                "model produced no output named `{output}` (available: {:?})",
                outputs.keys().collect::<Vec<_>>()
            ))
        })?;

        let (shape, data) = value.try_extract_tensor::<f32>()?;
        let shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        Ok((data.to_vec(), shape))
    }
}

/// Assert two `f32` slices match under NumPy `allclose` semantics:
/// `|got - reference| <= atol + rtol * |reference|`, elementwise.
///
/// On success, prints `max_abs_diff` and the index where it occurred so
/// the parity margin is visible in `--nocapture` runs. On failure, panics
/// with the offending index, both values, the diff, and the threshold.
pub fn assert_close(got: &[f32], reference: &[f32], atol: f32, rtol: f32) {
    assert_eq!(
        got.len(),
        reference.len(),
        "assert_close: length mismatch — got {} elements, reference has {}",
        got.len(),
        reference.len()
    );

    let mut max_abs_diff = 0.0f32;
    let mut max_idx = 0usize;
    for (i, (&g, &r)) in got.iter().zip(reference.iter()).enumerate() {
        let diff = (g - r).abs();
        if diff > max_abs_diff {
            max_abs_diff = diff;
            max_idx = i;
        }
        let threshold = atol + rtol * r.abs();
        assert!(
            diff <= threshold,
            "assert_close: mismatch at index {i}: got={g}, reference={r}, \
             abs_diff={diff} > threshold={threshold} (atol={atol}, rtol={rtol})"
        );
    }

    let worst = got.get(max_idx).copied().unwrap_or(0.0);
    let worst_ref = reference.get(max_idx).copied().unwrap_or(0.0);
    println!(
        "assert_close: PASS over {} elements — max_abs_diff={max_abs_diff} at index {max_idx} \
         (got={worst}, reference={worst_ref})",
        got.len()
    );
}

/// Generate `n` deterministic `N(0, 1)` samples from `seed`.
///
/// Uses Box-Muller over a small SplitMix64-seeded LCG so the same `(n,
/// seed)` always yields the same sequence with no external rng dependency.
/// Matches the WS dtype-fixture convention (seeded Gaussian noise, never
/// zeros, which is pathological for transformer attention).
pub fn seeded(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    // A SplitMix64 step gives a well-distributed u64 from the LCG state.
    let mut next_u64 = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    // Open-unit-interval uniform in (0, 1]: shift off zero so ln() is safe.
    let mut next_unit = || {
        let bits = next_u64() >> 11; // 53 random bits
        (bits as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0)
    };

    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let u1 = next_unit();
        let u2 = next_unit();
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = std::f64::consts::TAU * u2;
        out.push((radius * angle.cos()) as f32);
        if out.len() < n {
            out.push((radius * angle.sin()) as f32);
        }
    }
    out
}
