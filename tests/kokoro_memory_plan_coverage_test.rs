//! Phase 4 A1 landing criteria (3) + (4): on Kokoro, the static
//! `memory_planner` must deterministically assign at least 70% of
//! arena-eligible tensors to slots.
//!
//! Pure-planner integration — A1 lands the planner as infrastructure,
//! not a runtime arena. We construct the full Kokoro `OptimizableGraph`
//! via the executor's builder shim and call `compile()` to force
//! `lower()`. The test then inspects `ExecutionPlan::memory_plan` to
//! verify:
//!
//! * **(4) Classification coverage ≥ 0.7.** The
//!   `kokoro_memory_plan_coverage_is_above_70_percent` test asserts
//!   `arena_assigned / arena_eligible ≥ 0.7` from
//!   [`MemoryPlan::classification_counts`]. Catches the "predicate has
//!   a silent bug that sends everything to pool" failure mode that
//!   binary correctness tests don't detect.
//!
//! * **(3) Determinism.** The
//!   `kokoro_memory_plan_is_deterministic_across_independent_lowerings`
//!   test builds the Kokoro graph twice via two separate executors,
//!   runs `lower()` on each, and asserts the resulting `MemoryPlan`
//!   instances are byte-identical via `Debug`-format comparison.
//!
//! Both tests are `#[ignore]`d on the default run (they require the
//! Kokoro ONNX file and a CUDA GPU). Run with:
//!
//! ```bash
//! cargo test --features cuda --release \
//!     --test kokoro_memory_plan_coverage_test -- --ignored --nocapture
//! ```

#![cfg(feature = "cuda")]

#[path = "common/mod.rs"]
#[allow(clippy::duplicate_mod)] // shared helper loaded via `#[path]` in multiple integration-test binaries
mod common;

use iconnx::cuda::GpuGraphExecutor;
use iconnx::cuda::inference::testing::get_memory_plan;
use iconnx::onnx_parser::OnnxParser;

/// Build a fresh `GpuGraphExecutor` loaded with the full Kokoro v1.0
/// graph + concrete seq_len=5 input shapes. Mirrors the loader used by
/// `tests/kokoro_gpu_fusion_test.rs::setup_kokoro_gpu_executor` so the
/// planner sees the same graph shape used in the end-to-end inference
/// tests.
fn setup_kokoro_gpu_executor() -> GpuGraphExecutor {
    let model_path = common::kokoro_model::kokoro_model_path();
    let model = OnnxParser::parse_file(&model_path).expect("parse kokoro-v1.0.onnx");
    let weights = model.extract_weights().expect("extract weights");
    let graph = model.computation_graph().expect("parse computation graph");
    let nodes = graph.nodes();

    let mut executor = GpuGraphExecutor::new().expect("create CUDA executor");

    // Concrete shapes — see kokoro_gpu_fusion_test for rationale.
    executor.add_input("tokens".to_string(), vec![Some(1), Some(5)]);
    executor.add_input("style".to_string(), vec![Some(1), Some(256)]);
    executor.add_input("speed".to_string(), vec![Some(1)]);

    for (name, tensor) in &weights {
        executor
            .add_initializer(name.clone(), tensor)
            .expect("add initializer");
    }

    for (idx, node) in nodes.iter().enumerate() {
        let fallback_name = format!("node_{}", idx);
        let node_name = node
            .outputs
            .first()
            .map(|s| s.as_str())
            .unwrap_or(&fallback_name);
        let inputs: Vec<&str> = node.inputs.iter().map(|s| s.as_str()).collect();
        let outputs: Vec<&str> = node.outputs.iter().map(|s| s.as_str()).collect();
        executor.add_node(
            node_name,
            &node.op_type,
            inputs,
            outputs,
            node.attributes.clone(),
        );
    }

    executor
}

/// Landing criterion (4): ≥ 70% of arena-eligible tensors get slots on
/// Kokoro. Fails loudly with a diagnostic pointing at the likely cause
/// if the rate drops below the threshold — this is the silent-bug
/// detector for predicate drift.
#[test]
#[ignore] // Requires Kokoro + CUDA.
fn kokoro_memory_plan_coverage_is_above_70_percent() {
    let executor = setup_kokoro_gpu_executor();
    let plan = get_memory_plan(&executor).expect("memory plan present after compile");
    let c = plan.classification_counts();

    eprintln!(
        "Kokoro memory plan: eligible={} assigned={} pool={} \
         arena_bytes={} slots={}",
        c.arena_eligible, c.arena_assigned, c.pool_classified,
        plan.arena_bytes, plan.slot_count,
    );

    assert!(
        c.arena_eligible >= 10,
        "expected ≥ 10 arena-eligible tensors on Kokoro; got {} — \
         likely a silent bug in is_planner_eligible or \
         infer_output_dtype_bytes",
        c.arena_eligible,
    );
    let rate = c.arena_assigned as f64 / c.arena_eligible as f64;
    assert!(
        rate >= 0.7,
        "Phase 4 A1 landing criterion (4) failed: \
         arena_assigned/arena_eligible = {}/{} = {:.2}; need ≥ 0.70. \
         Most likely cause: is_planner_eligible or is_fusion_skipped has \
         a silent bug filtering out tensors that should be assigned, or \
         the greedy allocator's compatibility check is too strict.",
        c.arena_assigned, c.arena_eligible, rate,
    );
}

/// Landing criterion (3): back-to-back independent `lower()` calls
/// produce byte-identical `MemoryPlan` output. Compares via
/// `Debug`-format snapshot — the only representation of the plan
/// available through the testing facade, and robust because the
/// planner's `slot_assignments` uses a `BTreeMap` specifically for
/// deterministic iteration.
#[test]
#[ignore] // Requires Kokoro + CUDA.
fn kokoro_memory_plan_is_deterministic_across_independent_lowerings() {
    let e1 = setup_kokoro_gpu_executor();
    let e2 = setup_kokoro_gpu_executor();

    let p1 = get_memory_plan(&e1).expect("plan 1");
    let p2 = get_memory_plan(&e2).expect("plan 2");

    assert_eq!(
        p1.arena_bytes, p2.arena_bytes,
        "MemoryPlan.arena_bytes must be deterministic across independent lowerings",
    );
    assert_eq!(
        p1.slot_count, p2.slot_count,
        "MemoryPlan.slot_count must be deterministic",
    );
    assert_eq!(
        p1.classification_counts().arena_eligible,
        p2.classification_counts().arena_eligible,
        "PlannerClassification.arena_eligible must be deterministic",
    );
    assert_eq!(
        p1.classification_counts().arena_assigned,
        p2.classification_counts().arena_assigned,
        "PlannerClassification.arena_assigned must be deterministic",
    );
    assert_eq!(
        p1.classification_counts().pool_classified,
        p2.classification_counts().pool_classified,
        "PlannerClassification.pool_classified must be deterministic",
    );

    // Full `Debug`-format equivalence is the strongest snapshot check —
    // guards slot offsets, slot assignments, and everything else
    // reachable through `Debug`. The slot_assignments BTreeMap
    // guarantees ordered iteration, so the formatted strings match
    // byte-for-byte when the planner's output is truly identical.
    assert_eq!(
        format!("{:?}", p1),
        format!("{:?}", p2),
        "MemoryPlan Debug representations must be byte-identical across independent lowerings",
    );
}
