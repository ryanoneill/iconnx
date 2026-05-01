# Contributing to iconnx

## Developer Pre-Merge Discipline

Before opening a PR that touches dispatch, op support, or numerical paths,
run the full WS-5 conformance gate locally:

```bash
cargo test --features cuda --test onnx_conformance
```

This exercises iconnx against the full vendored ONNX conformance test
suite using GPU dispatch (when a GPU is present) or parser-only-recognition
(in CI). UnexpectedFailure or UnexpectedPass results indicate regressions
or stale allowlist entries; resolve before merging.

Also run the roster sweep via leadline-bench's `compare`. Both gates run
together; neither is sufficient alone.

## Re-vendoring ONNX conformance data

The vendored ONNX conformance test data lives at
`tests/conformance_data/onnx-<version>/node/`. To pin a newer ONNX release:

```bash
./tests/conformance_data/regenerate.sh v1.19.0
```

Then update `CONFORMANCE_DATA_DIR` in `tests/onnx_conformance.rs` to point at
the new directory, run the harness to populate any new entries in
`tests/conformance_known_failures.toml`, and commit.
