//! Self-tests for `tests/common/npy.rs` helpers.
//!
//! All tests here are GPU-free and fixture-free — they run in the default
//! lane without `--ignored`. They prove that:
//!
//! * `load_npy_f32` correctly round-trips a small f32 array written by
//!   `npyz::WriterBuilder` in the same test.
//! * `sha256_hex` produces the correct digest (cross-checked against the
//!   well-known SHA-256 of `"abc"`).
//! * `skip_if_missing` returns `true` for absent paths and `false` when all
//!   paths are present.
//!
//! These tests do NOT require a GPU, the OCR model files, or the preprocessed
//! `.npy` input fixtures — they are provable now.

#[path = "common/npy.rs"]
mod npy;

use npyz::WriterBuilder;
use std::io::Write;

/// Write a 1-D f32 array to a temp file via `npyz` and round-trip through
/// `load_npy_f32`.  Confirms data and shape survive the cycle.
#[test]
fn load_npy_f32_roundtrip_1d() {
    let expected_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    // Write via npyz to an in-memory buffer.
    let mut buf: Vec<u8> = vec![];
    {
        let mut writer = npyz::WriteOptions::new()
            .default_dtype()
            .shape(&[expected_data.len() as u64])
            .writer(&mut buf)
            .begin_nd()
            .expect("begin_nd");
        writer
            .extend(expected_data.iter())
            .expect("extend");
        writer.finish().expect("finish");
    }

    // Save to a temp file so `load_npy_f32` can read it.
    let tmp = tempfile::NamedTempFile::new().expect("tempfile");
    {
        let mut f = std::fs::File::create(tmp.path()).expect("create");
        f.write_all(&buf).expect("write");
    }

    let (data, shape) = npy::load_npy_f32(tmp.path()).expect("load_npy_f32");
    assert_eq!(shape, vec![6], "shape mismatch");
    assert_eq!(data, expected_data, "data mismatch");
}

/// Round-trip a 2-D f32 array (shape [2, 3]) through `load_npy_f32`.
#[test]
fn load_npy_f32_roundtrip_2d() {
    // 2×3 row-major: [[1,2,3],[4,5,6]]
    let expected_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape_nd: Vec<u64> = vec![2, 3];

    let mut buf: Vec<u8> = vec![];
    {
        let mut writer = npyz::WriteOptions::new()
            .default_dtype()
            .shape(&shape_nd)
            .writer(&mut buf)
            .begin_nd()
            .expect("begin_nd");
        writer
            .extend(expected_data.iter())
            .expect("extend");
        writer.finish().expect("finish");
    }

    let tmp = tempfile::NamedTempFile::new().expect("tempfile");
    {
        let mut f = std::fs::File::create(tmp.path()).expect("create");
        f.write_all(&buf).expect("write");
    }

    let (data, shape) = npy::load_npy_f32(tmp.path()).expect("load_npy_f32");
    assert_eq!(shape, vec![2usize, 3], "shape mismatch");
    assert_eq!(data, expected_data, "data mismatch");
}

/// SHA-256("abc") == ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
/// (NIST / RFC 3174 test vector).
#[test]
fn sha256_hex_known_vector_abc() {
    let tmp = tempfile::NamedTempFile::new().expect("tempfile");
    std::fs::write(tmp.path(), b"abc").expect("write");

    let digest = npy::sha256_hex(tmp.path()).expect("sha256_hex");
    assert_eq!(
        digest,
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        "SHA-256 of 'abc' mismatch"
    );
}

/// SHA-256 of the empty string is
/// e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
#[test]
fn sha256_hex_known_vector_empty() {
    let tmp = tempfile::NamedTempFile::new().expect("tempfile");
    std::fs::write(tmp.path(), b"").expect("write");

    let digest = npy::sha256_hex(tmp.path()).expect("sha256_hex");
    assert_eq!(
        digest,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "SHA-256 of empty string mismatch"
    );
}

/// `skip_if_missing` returns `true` for paths that do not exist.
#[test]
fn skip_if_missing_absent() {
    assert!(
        npy::skip_if_missing(&["/nonexistent/definitely/not/here.onnx"]),
        "should return true for absent path"
    );
}

/// `skip_if_missing` returns `false` when all paths are present.
#[test]
fn skip_if_missing_present() {
    let tmp = tempfile::NamedTempFile::new().expect("tempfile");
    let path = tmp.path().to_str().expect("path to str");
    assert!(
        !npy::skip_if_missing(&[path]),
        "should return false when path exists"
    );
}

/// `verify_sha256` returns `Ok(true)` on a matching digest.
#[test]
fn verify_sha256_match() {
    let tmp = tempfile::NamedTempFile::new().expect("tempfile");
    std::fs::write(tmp.path(), b"abc").expect("write");

    let ok = npy::verify_sha256(
        tmp.path(),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    )
    .expect("verify_sha256");
    assert!(ok, "expected Ok(true) for matching digest");
}

/// `verify_sha256` returns `Ok(false)` on a mismatching digest.
#[test]
fn verify_sha256_mismatch() {
    let tmp = tempfile::NamedTempFile::new().expect("tempfile");
    std::fs::write(tmp.path(), b"abc").expect("write");

    let ok = npy::verify_sha256(tmp.path(), "0000000000000000000000000000000000000000000000000000000000000000")
        .expect("verify_sha256");
    assert!(!ok, "expected Ok(false) for wrong digest");
}
