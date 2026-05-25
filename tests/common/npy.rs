// Shared test-helper module: `#[path]`-included into multiple test binaries,
// each of which uses only a SUBSET of these helpers (e.g. `ocr_parity_test`
// uses `load_npy_f32` + `skip_if_missing` but not `verify_sha256`, while
// `npy_loader_test` exercises all helpers). Per-binary `dead_code` is therefore
// a false positive — this module-scoped allow is the canonical idiom for
// `tests/common` shared fixtures, NOT a silenced real warning.
#![allow(dead_code)]

//! Npy-loading, SHA-256, and skip helpers for WS-6 whole-model parity tests.
//!
//! Three pieces, reused by every downstream per-model parity test:
//!
//! * [`load_npy_f32`] — reads an `.npy` file from disk and returns `(data,
//!   shape)` as `(Vec<f32>, Vec<usize>)`.
//! * [`sha256_hex`] — returns the lowercase SHA-256 hex digest of a file's
//!   bytes using the `sha2` crate (O-6: no hand-rolled crypto).
//! * [`skip_if_missing`] — given a list of path strings, prints a clear
//!   reason line and returns `true` if any are absent, so the caller can
//!   `return` early and keep the suite green while fixtures are outstanding.
//! * [`verify_sha256`] — convenience wrapper that calls [`sha256_hex`] and
//!   compares against an expected hex string.

use std::path::Path;

use sha2::{Digest, Sha256};

/// Load an `.npy` file at `path` and return its flat data and shape.
///
/// Supports any C-order f32 array.  The returned shape is in row-major
/// (C) order, matching NumPy's default.
///
/// # Errors
///
/// Returns an [`std::io::Error`] if the file cannot be read or the npy
/// header / data cannot be parsed.
pub fn load_npy_f32(path: &Path) -> std::io::Result<(Vec<f32>, Vec<usize>)> {
    let bytes = std::fs::read(path)?;
    let npy = npyz::NpyFile::new(&bytes[..])?;
    let shape: Vec<usize> = npy.shape().iter().map(|&d| d as usize).collect();
    let data: Vec<f32> = npy.into_vec::<f32>()?;
    Ok((data, shape))
}

/// Return the lowercase SHA-256 hex digest of the file at `path`.
///
/// Uses the `sha2` crate (O-6).  Reads the entire file into memory; this
/// is intentional — model files are O(10 MB) and fixture files are much
/// smaller; test-time memory is not a concern.
///
/// # Errors
///
/// Returns an [`std::io::Error`] if the file cannot be read.
pub fn sha256_hex(path: &Path) -> std::io::Result<String> {
    let bytes = std::fs::read(path)?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    let result = hasher.finalize();
    Ok(format!("{result:x}"))
}

/// Check that all `paths` exist; if any are missing, print a clear reason
/// line to stderr and return `true` so the caller can `return` early.
///
/// Returns `false` when every path is present (all clear, proceed with
/// the test).
///
/// # Example
///
/// ```ignore
/// if skip_if_missing(&["./model.onnx", "./input.npy"]) {
///     return;
/// }
/// ```
pub fn skip_if_missing(paths: &[&str]) -> bool {
    let missing: Vec<&str> = paths
        .iter()
        .copied()
        .filter(|p| !Path::new(p).exists())
        .collect();
    if missing.is_empty() {
        return false;
    }
    eprintln!(
        "SKIP — fixture file(s) not present (pdfocr has not delivered them yet): {}",
        missing.join(", ")
    );
    true
}

/// Compute the SHA-256 of `path` and compare it against `expected_hex`.
///
/// Returns `true` on match, `false` on mismatch (with a diagnostic to
/// stderr). Returns an `Err` if the file cannot be read.
///
/// Used by whole-model parity tests to detect a stale or wrong fixture
/// before running inference.
pub fn verify_sha256(path: &Path, expected_hex: &str) -> std::io::Result<bool> {
    let actual = sha256_hex(path)?;
    if actual == expected_hex {
        Ok(true)
    } else {
        eprintln!(
            "SHA-256 mismatch for {path:?}:\n  expected: {expected_hex}\n  actual:   {actual}"
        );
        Ok(false)
    }
}
