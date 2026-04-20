//! Shared helpers for integration tests.
//!
//! This module is reachable from individual integration-test binaries
//! via `#[path = "common/mod.rs"] mod common;` at the top of each test
//! file. The `#[path]` attribute ensures the helper resolves to
//! `tests/common/mod.rs` regardless of whether the file is compiled as a
//! standalone test binary or as a submodule of `tests/mod.rs`.

pub mod kokoro_model;
