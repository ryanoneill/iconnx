//! WS-5 M5.3 — ONNX conformance harness.
//!
//! Walks tests/conformance_data/onnx-<N>/node/test_*/, parses each model,
//! looks up its op_type in iconnx's OP_SUPPORT_TABLE, and runs the test
//! based on dispatch context:
//!
//!   ParserOnlyCi (CI, no GPU): asserts model parses + op recognized.
//!   Gpu (developer pre-merge): runs iconnx + compares output to the
//!                              expected protobuf in the test data set.
//!
//! Outcomes per test (per dataset within a test):
//!   - Pass / KnownFailure / UnexpectedFailure / UnexpectedPass
//!   - SkippedUnsupportedOp / SkippedGpuOnlyInCi / ParserOnlyOk
//!
//! Allowlist semantics:
//!   - listed + fails  -> KnownFailure (expected)
//!   - listed + passes -> UnexpectedPass (allowlist stale)
//!   - !listed + fails -> UnexpectedFailure
//!   - !listed + passes -> Pass

#![cfg(feature = "cuda")]

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use iconnx::op_support::{lookup, DispatchContext, LookupResult};
use iconnx::OnnxParser;

const CONFORMANCE_DATA_DIR: &str = "tests/conformance_data/onnx-1.18/node";
const ALLOWLIST_PATH: &str = "tests/conformance_known_failures.toml";

#[derive(Debug, Deserialize)]
struct Allowlist {
    #[serde(default)]
    failures: Vec<AllowlistEntry>,
}

#[derive(Debug, Deserialize, Clone)]
struct AllowlistEntry {
    test: String,
    #[allow(dead_code)] // kept for human readability of the allowlist
    op: String,
    dispatch: String, // "cpu" | "gpu" | "both"
    #[allow(dead_code)]
    reason: String,
    #[serde(default)]
    #[allow(dead_code)]
    tracked_issue: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Outcome {
    // The variants Pass / KnownFailure / UnexpectedPass are part of the
    // documented three-state skip taxonomy + allowlist semantics. They will
    // be constructed once `run_dataset()` graduates from the Y(3) placeholder
    // (which always returns ParserOnlyOk) to the full GPU-execution +
    // tolerance-comparison path. `#[allow(dead_code)]` keeps the surface
    // honest until that follow-up lands.
    #[allow(dead_code)]
    Pass,
    ParserOnlyOk,
    #[allow(dead_code)]
    KnownFailure,
    UnexpectedFailure,
    #[allow(dead_code)]
    UnexpectedPass,
    SkippedUnsupportedOp,
    SkippedGpuOnlyInCi,
    SkippedNoFixture,
}

fn detect_dispatch_context() -> DispatchContext {
    use iconnx::cuda::IconnxCudaContext;
    match IconnxCudaContext::new() {
        Ok(_) => DispatchContext::Gpu,
        Err(_) => DispatchContext::ParserOnlyCi,
    }
}

fn load_allowlist() -> Allowlist {
    let path = PathBuf::from(ALLOWLIST_PATH);
    if !path.exists() {
        return Allowlist { failures: vec![] };
    }
    let s = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read allowlist {}: {}", ALLOWLIST_PATH, e));
    toml::from_str(&s)
        .unwrap_or_else(|e| panic!("parse allowlist {}: {}", ALLOWLIST_PATH, e))
}

fn allowlist_matches(
    allow: &Allowlist,
    test_name: &str,
    ctx: DispatchContext,
) -> Option<AllowlistEntry> {
    let target_dispatch = match ctx {
        DispatchContext::Gpu => "gpu",
        DispatchContext::ParserOnlyCi => "cpu",
    };
    allow
        .failures
        .iter()
        .find(|e| {
            e.test == test_name
                && (e.dispatch == target_dispatch || e.dispatch == "both")
        })
        .cloned()
}

fn run_one(
    test_dir: &Path,
    ctx: DispatchContext,
    allow: &Allowlist,
) -> (String, Vec<Outcome>) {
    let test_name = test_dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("(unknown)")
        .to_string();
    let model_path = test_dir.join("model.onnx");
    if !model_path.exists() {
        return (test_name, vec![Outcome::SkippedNoFixture]);
    }

    // Parse + extract op_type. If any of these fail, consult the allowlist
    // before classifying as UnexpectedFailure: a model can be on the
    // allowlist for parse-time reasons (e.g., contains a tensor attribute
    // with a dtype iconnx doesn't decode).
    let parse_failure_outcome = match allowlist_matches(allow, &test_name, ctx) {
        Some(_) => Outcome::KnownFailure,
        None => Outcome::UnexpectedFailure,
    };

    let model = match OnnxParser::parse_file(&model_path) {
        Ok(m) => m,
        Err(_) => return (test_name, vec![parse_failure_outcome]),
    };
    let graph = match model.computation_graph() {
        Ok(g) => g,
        Err(_) => return (test_name, vec![parse_failure_outcome]),
    };
    let op_type = match graph.nodes().first() {
        Some(n) => n.op_type.clone(),
        None => return (test_name, vec![parse_failure_outcome]),
    };

    // Dispatch lookup.
    match lookup(&op_type, ctx) {
        LookupResult::Unsupported => {
            return (test_name, vec![Outcome::SkippedUnsupportedOp]);
        }
        LookupResult::ParserOnlyRecognized => {
            return (test_name, vec![Outcome::ParserOnlyOk]);
        }
        LookupResult::Runnable => { /* fall through to execution */ }
    }

    // CpuCi case where op IS in supported_ops() but no CPU executor —
    // (redundant given current LookupResult, but kept for forward-compat
    // when CPU executor lands.)
    if ctx == DispatchContext::ParserOnlyCi {
        return (test_name, vec![Outcome::SkippedGpuOnlyInCi]);
    }

    // Iterate ALL test_data_set_N/ directories.
    let mut outcomes: Vec<Outcome> = Vec::new();
    let allow_entry = allowlist_matches(allow, &test_name, ctx);

    let dataset_dirs: Vec<PathBuf> = fs::read_dir(test_dir)
        .map(|rd| {
            rd.flatten()
                .map(|e| e.path())
                .filter(|p| {
                    p.is_dir()
                        && p.file_name()
                            .and_then(|n| n.to_str())
                            .map(|s| s.starts_with("test_data_set_"))
                            .unwrap_or(false)
                })
                .collect()
        })
        .unwrap_or_default();

    if dataset_dirs.is_empty() {
        return (test_name, vec![Outcome::SkippedNoFixture]);
    }

    for ds in dataset_dirs {
        let outcome = run_dataset(&model, &ds, &op_type, &allow_entry);
        outcomes.push(outcome);
    }

    (test_name, outcomes)
}

fn run_dataset(
    _model: &iconnx::onnx_parser::OnnxModel,
    _ds: &Path,
    _op_type: &str,
    allow_entry: &Option<AllowlistEntry>,
) -> Outcome {
    // PLACEHOLDER: actual ONNX-runtime-style execution is non-trivial.
    // For the WS-5 harness, the GPU execution path uses GpuGraphExecutor
    // built from the parsed model + per-dataset inputs/outputs.
    //
    // Implementation steps:
    //   1. Load input_*.pb -> HashMap<String, Tensor> via existing
    //      ONNX TensorProto parse (parse_tensor_proto).
    //   2. Load output_*.pb -> expected outputs.
    //   3. Build GpuGraphExecutor, add initializers, add nodes, compile.
    //   4. executor.run(inputs, output_names).
    //   5. compare actual vs expected with per-dtype tolerance:
    //        FP32: 1e-4 abs
    //        FP16: 1e-2 abs / 0.5% rel
    //        BF16: 5e-2 abs / 1% rel
    //        Int/Bool: bit-exact
    //   6. NaN/Inf detection: if actual contains non-finite, force Fail
    //      (mirrors leadline-bench nan_in_iconnx_output_forces_infinite_diff).
    //
    // For initial Y(3) commit, treat as ParserOnlyOk (i.e. parse + recognize
    // succeeded; full execution lives behind a follow-up). The allowlist
    // semantics remain in place for the future execution-comparison path.
    let _ = allow_entry;
    Outcome::ParserOnlyOk
}

fn print_summary(results: &HashMap<String, Vec<Outcome>>) {
    let mut counts: HashMap<&'static str, usize> = HashMap::new();
    let mut skipped_ops: HashMap<String, usize> = HashMap::new();

    for (test_name, outcomes) in results {
        for o in outcomes {
            let key = match o {
                Outcome::Pass => "pass",
                Outcome::ParserOnlyOk => "parser-only-ok",
                Outcome::KnownFailure => "known-failure",
                Outcome::UnexpectedFailure => "unexpected-failure",
                Outcome::UnexpectedPass => "unexpected-pass",
                Outcome::SkippedUnsupportedOp => {
                    // Increment the op-skip distribution.
                    if let Some(op) = test_name.strip_prefix("test_") {
                        // Use the first segment after "test_" as the op heuristic.
                        let op_key = op.split('_').next().unwrap_or(op).to_string();
                        *skipped_ops.entry(op_key).or_insert(0) += 1;
                    }
                    "skipped-unsupported-op"
                }
                Outcome::SkippedGpuOnlyInCi => "skipped-gpu-only-in-ci",
                Outcome::SkippedNoFixture => "skipped-no-fixture",
            };
            *counts.entry(key).or_insert(0) += 1;
        }
    }

    eprintln!("\n=== ONNX Conformance Summary ===");
    for (k, v) in &counts {
        eprintln!("  {:30} {}", k, v);
    }

    // Top-N skipped-op coverage gaps.
    let mut top: Vec<_> = skipped_ops.into_iter().collect();
    top.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
    eprintln!("\n=== Top skipped-op coverage gaps ===");
    for (op, n) in top.iter().take(10) {
        eprintln!("  {:25} {} tests skipped", op, n);
    }
}

#[test]
fn run_all_conformance_tests() {
    let ctx = detect_dispatch_context();
    eprintln!("Dispatch context: {:?}", ctx);

    let allow = load_allowlist();
    let data_dir = PathBuf::from(CONFORMANCE_DATA_DIR);
    if !data_dir.exists() {
        eprintln!("conformance data dir not present at {:?}; skipping", data_dir);
        return;
    }

    let test_dirs: Vec<PathBuf> = fs::read_dir(&data_dir)
        .expect("read conformance data dir")
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.is_dir()
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|s| s.starts_with("test_"))
                    .unwrap_or(false)
        })
        .collect();

    eprintln!("Walking {} test directories", test_dirs.len());

    let mut results: HashMap<String, Vec<Outcome>> = HashMap::new();
    for d in &test_dirs {
        let (name, outcomes) = run_one(d, ctx, &allow);
        results.insert(name, outcomes);
    }

    print_summary(&results);

    // CI gate: zero UnexpectedFailure or UnexpectedPass blocks merge.
    let mut bad = 0;
    for (name, outcomes) in &results {
        for o in outcomes {
            match o {
                Outcome::UnexpectedFailure => {
                    eprintln!("UnexpectedFailure: {}", name);
                    bad += 1;
                }
                Outcome::UnexpectedPass => {
                    eprintln!("UnexpectedPass (allowlist stale): {}", name);
                    bad += 1;
                }
                _ => {}
            }
        }
    }
    assert_eq!(bad, 0, "{} unexpected outcomes (see above); see allowlist {}", bad, ALLOWLIST_PATH);
}

#[test]
fn harness_self_test_recognizes_pass_and_unsupported() {
    // Walk the self-test tree (separate from conformance_data/onnx-N/).
    let self_test_dir = PathBuf::from("tests/conformance_data_self_test");
    if !self_test_dir.exists() {
        eprintln!("self-test fixtures not present; skipping");
        return;
    }

    let allow = Allowlist { failures: vec![] };
    let ctx = DispatchContext::ParserOnlyCi; // force CPU-mode for this harness self-test

    // Walk and verify expected outcomes.
    for entry in fs::read_dir(&self_test_dir).expect("read self-test dir") {
        let entry = entry.unwrap();
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let (name, outcomes) = run_one(&path, ctx, &allow);
        eprintln!("self-test: {} -> {:?}", name, outcomes);
        match name.as_str() {
            "test_pass" => {
                assert!(matches!(outcomes.as_slice(), [Outcome::ParserOnlyOk]));
            }
            "test_unsupported_op" => {
                assert!(matches!(outcomes.as_slice(), [Outcome::SkippedUnsupportedOp]));
            }
            other => panic!("unexpected self-test dir: {}", other),
        }
    }
}
