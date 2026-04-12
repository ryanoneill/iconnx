# AddMulAdd Detection Gap Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the `AddMulAdd` fusion walker to match all 51 real instances on Kokoro (currently finds zero) by extending its static-value check to recognize `Constant`-node outputs, not just initializers.

**Architecture:** Extend the existing `precomputed_slice` pattern's intent. Add an inline `is_static` closure in `detect_fused_patterns` that checks both `weights.contains_key(name)` and `output_to_node[name].op_type == "Constant"`. Replace the AddMulAdd walker's two `weights.contains_key` calls with `is_static` calls. Add a `count_fused_patterns` testing facade helper so the Kokoro integration test can assert the hit count by variant. No kernel changes, no new fused patterns.

**Tech Stack:** Rust 2021, existing iconnx inference infrastructure (`src/cuda/inference/fusion/detection.rs`, `src/cuda/inference/testing.rs`, `tests/fusion_detection_test.rs`, `tests/kokoro_gpu_fusion_test.rs`). No new modules, no new dependencies.

**Prerequisites for the implementing agent:**
- Read the spec first: `docs/superpowers/specs/2026-04-11-addmuladd-detection-gap-design.md`.
- Per `/home/ryano/.claude/CLAUDE.md`: TDD where possible, commits after every substantive change, **all commits must be signed** (`git commit -S`), stop and ask if signing fails, warnings never ignored.
- `cargo test --features cuda` is the regression gate. Run after each task.
- `cargo clippy --features cuda -- -D warnings` must be clean before every commit.
- **NEVER run `git checkout <sha>`** to inspect commits. Use `git show SHA`, `git show SHA --stat`, or `git log --show-signature -1 SHA`. Before every commit, run `git branch --show-current` — it must print the cycle branch name (set up by the controller, e.g., `cycle3/addmuladd-detection-gap`).

---

## Scope Check

This plan implements **Cycle 3 only** — the AddMulAdd detection gap fix. Out of scope:

- `AddMulAddLeakyRelu` fusion pattern (22 instances per Leadline) — separate future cycle.
- Extending `is_static` to recognize Cast-of-Constant or other chains — separate cycle (Approach C from Cycle 2's brainstorming).
- cuDNN LayerNorm replacement — separate cycle.
- Any change to existing fusion walkers other than AddMulAdd.
- Any change to the fused kernel `gpu_fused_add_mul_add` or any other kernel.

Any work outside the files listed below is a scope violation and should be flagged to the controller before landing.

---

## File Structure

### Modified files

- **`src/cuda/inference/fusion/detection.rs`** — add `is_static` closure, replace two `weights.contains_key` calls in the AddMulAdd walker with `is_static` calls. Net: ~12 lines.
- **`src/cuda/inference/mod.rs`** — widen `fused_patterns` field visibility from private to `pub(crate)` so the testing module can read it. Net: 1 line modified.
- **`src/cuda/inference/testing.rs`** — add `PatternCount` struct and `count_fused_patterns` helper. Net: ~70 lines added.
- **`tests/fusion_detection_test.rs`** — add 4 new synthetic-graph tests for the AddMulAdd walker (hit-from-initializer, hit-from-Constant, reject-when-b-computed, reject-when-c-computed). Plus a new `float32_constant_attrs` helper. Net: ~160 lines added.
- **`tests/kokoro_gpu_fusion_test.rs`** — extend the existing `fusion_fires_on_full_kokoro_at_seq_len_5` test with a new Cycle 3 assertion block that calls `count_fused_patterns` and asserts `add_mul_add >= 40`. Net: ~35 lines added.

### New files

Just the results doc at the end of the cycle: `docs/performance/2026-04-11-cycle3-addmuladd-detection-gap-results.md`.

### Files NOT touched

- `src/cuda/inference/fusion/patterns.rs` — `FusedPattern::AddMulAdd` already exists from Cycle 1.
- `src/cuda/kernels/fused/**` — no kernel changes.
- `src/cuda/inference/gpu_event_timer.rs` — no changes.
- `run_with_profiling` and the dispatch sites in `src/cuda/inference/mod.rs` — no changes.
- Cycle 2's precomputation infrastructure (`precomputed_*` fields, `try_precompute_*` helpers) — no changes.

---

## Phase A: Fix the walker

### Task 1: Add `is_static` closure, update AddMulAdd walker, add unit tests (TDD red → green)

This is one focused commit: the walker change and its unit tests land together. Three of the four new tests exercise code paths that already work (regression guards); one exercises the new Constant-source behavior and is the red → green signal.

**Files:**
- Modify: `src/cuda/inference/fusion/detection.rs`
- Modify: `tests/fusion_detection_test.rs`

#### Step 1: Write the four new unit tests FIRST (red state)

Open `tests/fusion_detection_test.rs`. Scroll to the bottom — the new tests and their helper go at the end of the file, after the existing Cycle 1 tests.

First, add a new helper function after the existing `int64_constant_attrs` (if present) or after the other helpers near the top. If `int64_constant_attrs` doesn't exist in this file (it may live in Cycle 2's precompute test files instead), add both helpers here:

```rust
/// Build a NodeAttributes whose `"value"` attribute is an Int64 tensor.
/// Used to construct `Constant` Int64 nodes in tests.
#[allow(dead_code)]
fn int64_constant_attrs(values: &[i64]) -> NodeAttributes {
    let mut attrs = NodeAttributes::default();
    let tensor = Tensor::from_vec_i64(values.to_vec(), vec![values.len()]);
    attrs.add_tensor("value".to_string(), tensor);
    attrs
}

/// Build a NodeAttributes whose `"value"` attribute is a Float32 tensor.
/// Used to construct `Constant` Float32 nodes in tests (e.g., affine
/// transform scales and biases that modern ONNX exports emit as
/// Constant nodes rather than baking into initializers).
fn float32_constant_attrs(value: f32) -> NodeAttributes {
    let mut attrs = NodeAttributes::default();
    let tensor = Tensor::from_vec_f32(vec![value], vec![1]);
    attrs.add_tensor("value".to_string(), tensor);
    attrs
}
```

Verify by grepping whether `int64_constant_attrs` already exists in this file:

```bash
grep -n "fn int64_constant_attrs\|fn float32_constant_attrs" tests/fusion_detection_test.rs
```

If `int64_constant_attrs` already exists, skip it and just add `float32_constant_attrs`.

If the imports at the top of the file don't include `Tensor` and `NodeAttributes`, add them:

```rust
use iconnx::attributes::NodeAttributes;
use iconnx::tensor::Tensor;
```

(Check first; these may already be present.)

Now add the four new test functions at the end of the file:

```rust
#[test]
fn add_mul_add_detects_with_initializer_bc() {
    let ctx = IconnxCudaContext::new().expect("CUDA context");

    // Build the 3-op chain: (x + a) * b + c
    //   t0 = Add(x, a)
    //   t1 = Mul(t0, b)
    //   y  = Add(t1, c)
    let nodes = vec![
        node("add1", "Add", &["x", "a"], &["t0"]),
        node("mul1", "Mul", &["t0", "b"], &["t1"]),
        node("add2", "Add", &["t1", "c"], &["y"]),
    ];

    // b and c are initializers (in the weights map).
    let mut weights = HashMap::new();
    weights.insert(
        "b".to_string(),
        GpuTensor::from_host_f32(&ctx, &[2.0], vec![1]).expect("b"),
    );
    weights.insert(
        "c".to_string(),
        GpuTensor::from_host_f32(&ctx, &[3.0], vec![1]).expect("c"),
    );

    let (patterns, _skip) = detect_fused_patterns_for_tests(&nodes, &weights, &ctx);

    let info = patterns.get("add1").unwrap_or_else(|| {
        panic!(
            "add1 was not registered. Registered patterns: {:?}",
            patterns.keys().collect::<Vec<_>>()
        )
    });
    match &info.pattern {
        FusedPattern::AddMulAdd {
            x_input,
            a_input,
            b_input,
            c_input,
            output_name,
        } => {
            assert_eq!(x_input, "x");
            assert_eq!(a_input, "a");
            assert_eq!(b_input, "b");
            assert_eq!(c_input, "c");
            assert_eq!(output_name, "y");
        }
        other => panic!("expected AddMulAdd, got {:?}", other),
    }
}

#[test]
fn add_mul_add_detects_with_constant_bc() {
    let ctx = IconnxCudaContext::new().expect("CUDA context");

    // Build a graph with Constant Float32 nodes producing b and c,
    // followed by the 3-op chain (x + a) * b + c.
    let nodes = vec![
        ExecutionNodeForTests {
            name: "b_const".to_string(),
            op_type: "Constant".to_string(),
            inputs: vec![],
            outputs: vec!["b".to_string()],
            attributes: float32_constant_attrs(2.0),
        },
        ExecutionNodeForTests {
            name: "c_const".to_string(),
            op_type: "Constant".to_string(),
            inputs: vec![],
            outputs: vec!["c".to_string()],
            attributes: float32_constant_attrs(3.0),
        },
        node("add1", "Add", &["x", "a"], &["t0"]),
        node("mul1", "Mul", &["t0", "b"], &["t1"]),
        node("add2", "Add", &["t1", "c"], &["y"]),
    ];

    // Empty weights map — b and c come from Constant nodes, not initializers.
    let weights = HashMap::new();

    let (patterns, _skip) = detect_fused_patterns_for_tests(&nodes, &weights, &ctx);

    let info = patterns.get("add1").unwrap_or_else(|| {
        panic!(
            "add1 was not registered with Constant-sourced b/c. \
             This is the Cycle 3 fix: is_static should recognize \
             Constant node outputs, not just initializers. \
             Registered: {:?}",
            patterns.keys().collect::<Vec<_>>()
        )
    });
    assert!(
        matches!(&info.pattern, FusedPattern::AddMulAdd { .. }),
        "add1 registered but not as AddMulAdd: {:?}",
        info.pattern
    );
}

#[test]
fn add_mul_add_rejects_when_b_is_computed() {
    let ctx = IconnxCudaContext::new().expect("CUDA context");

    // Build a graph where `b` is produced by a non-Constant op (Shape),
    // so it's neither an initializer nor a Constant — it's a runtime
    // computed value. The walker should NOT fuse this because the
    // semantic intent is "static affine transform."
    let nodes = vec![
        node("shape_src", "Shape", &["shape_src_in"], &["b"]),
        node("add1", "Add", &["x", "a"], &["t0"]),
        node("mul1", "Mul", &["t0", "b"], &["t1"]),
        node("add2", "Add", &["t1", "c"], &["y"]),
    ];

    // c is an initializer; b is the output of Shape (runtime-computed).
    let mut weights = HashMap::new();
    weights.insert(
        "c".to_string(),
        GpuTensor::from_host_f32(&ctx, &[3.0], vec![1]).expect("c"),
    );

    let (patterns, _skip) = detect_fused_patterns_for_tests(&nodes, &weights, &ctx);

    let is_add_mul_add = patterns
        .get("add1")
        .map(|info| matches!(info.pattern, FusedPattern::AddMulAdd { .. }))
        .unwrap_or(false);
    assert!(
        !is_add_mul_add,
        "AddMulAdd must NOT fuse when b is a runtime-computed value. \
         The static-value check exists to preserve the 'static affine \
         transform' semantic. Got patterns: {:?}",
        patterns.keys().collect::<Vec<_>>()
    );
}

#[test]
fn add_mul_add_rejects_when_c_is_computed() {
    let ctx = IconnxCudaContext::new().expect("CUDA context");

    // Mirror of the b-is-computed case. c comes from a non-Constant op
    // (Shape); b is an initializer. The walker should still reject.
    let nodes = vec![
        node("shape_src", "Shape", &["shape_src_in"], &["c"]),
        node("add1", "Add", &["x", "a"], &["t0"]),
        node("mul1", "Mul", &["t0", "b"], &["t1"]),
        node("add2", "Add", &["t1", "c"], &["y"]),
    ];

    let mut weights = HashMap::new();
    weights.insert(
        "b".to_string(),
        GpuTensor::from_host_f32(&ctx, &[2.0], vec![1]).expect("b"),
    );

    let (patterns, _skip) = detect_fused_patterns_for_tests(&nodes, &weights, &ctx);

    let is_add_mul_add = patterns
        .get("add1")
        .map(|info| matches!(info.pattern, FusedPattern::AddMulAdd { .. }))
        .unwrap_or(false);
    assert!(
        !is_add_mul_add,
        "AddMulAdd must NOT fuse when c is a runtime-computed value. \
         Got patterns: {:?}",
        patterns.keys().collect::<Vec<_>>()
    );
}
```

#### Step 2: Verify imports and helper availability at the top of the file

Scroll to the top of `tests/fusion_detection_test.rs`. Confirm the following are imported:

```rust
#![cfg(feature = "cuda")]

use std::collections::HashMap;

use iconnx::attributes::NodeAttributes;
use iconnx::cuda::inference::testing::{
    detect_fused_patterns_for_tests, ExecutionNodeForTests, FusedPattern, FusedPatternInfo,
};
use iconnx::cuda::{GpuTensor, IconnxCudaContext};
use iconnx::tensor::Tensor;
```

If `FusedPattern` isn't already imported, add it. If `Tensor` or `NodeAttributes` aren't imported, add them. The `node` helper function should already exist in this file from Cycle 1; if not, add it near the existing helpers:

```rust
/// Build an `ExecutionNodeForTests` with the given name, op type, inputs,
/// and outputs. Attributes default to empty.
#[allow(dead_code)]
fn node(name: &str, op: &str, inputs: &[&str], outputs: &[&str]) -> ExecutionNodeForTests {
    ExecutionNodeForTests {
        name: name.to_string(),
        op_type: op.to_string(),
        inputs: inputs.iter().map(|s| s.to_string()).collect(),
        outputs: outputs.iter().map(|s| s.to_string()).collect(),
        attributes: NodeAttributes::default(),
    }
}
```

#### Step 3: Confirm red state — run only the new tests

```bash
cargo test --features cuda --test fusion_detection_test -- add_mul_add
```

Expected:
- `add_mul_add_detects_with_initializer_bc` — **PASS** (regression guard for existing behavior).
- `add_mul_add_detects_with_constant_bc` — **FAIL** with "add1 was not registered with Constant-sourced b/c. This is the Cycle 3 fix…". This is the red state.
- `add_mul_add_rejects_when_b_is_computed` — **PASS** (semantic guard, existing behavior).
- `add_mul_add_rejects_when_c_is_computed` — **PASS** (semantic guard, existing behavior).

If `add_mul_add_detects_with_initializer_bc` fails, the test's helper functions or imports are wrong — fix the test before proceeding. The walker itself should match the initializer case because that's exactly what Cycle 1's code does.

If `add_mul_add_detects_with_constant_bc` passes without the fix (unexpected), STOP and investigate — either the walker is already doing something surprising or the test isn't actually exercising the Constant path.

#### Step 4: Add the `is_static` closure in `detection.rs`

Open `src/cuda/inference/fusion/detection.rs`. Find the top of `detect_fused_patterns`. It currently looks like:

```rust
pub(crate) fn detect_fused_patterns(
    nodes: &[ExecutionNode],
    weights: &HashMap<String, GpuTensor>,
    ctx: &IconnxCudaContext,
) -> (HashMap<String, FusedPatternInfo>, HashSet<String>) {
    // Build lookup maps
    let output_to_node: HashMap<&str, &ExecutionNode> = nodes
        .iter()
        .flat_map(|n| n.outputs.iter().map(move |o| (o.as_str(), n)))
        .collect();

    // Count uses of each tensor output (for single-use check)
    let mut output_uses: HashMap<&str, usize> = HashMap::new();
    for node in nodes {
        for input in &node.inputs {
            *output_uses.entry(input.as_str()).or_insert(0) += 1;
        }
    }

    let mut fused_patterns: HashMap<String, FusedPatternInfo> = HashMap::new();
    let mut nodes_to_skip: HashSet<String> = HashSet::new();

    // Detect Add -> Sqrt -> Div pattern (normalization)
```

Insert the `is_static` closure between `let mut nodes_to_skip` and the first `// Detect` comment:

```rust
    let mut fused_patterns: HashMap<String, FusedPatternInfo> = HashMap::new();
    let mut nodes_to_skip: HashSet<String> = HashSet::new();

    // Predicate: is this tensor name a "static" value (known at graph-build
    // time)? Matches either (a) an initializer registered via
    // add_initializer (present in `weights`), or (b) the output of a
    // `Constant` node added earlier in topological order (looked up via
    // `output_to_node[name].op_type`).
    //
    // Used by the AddMulAdd walker's semantic-intent check: fuse only
    // affine transforms where b (scale) and c (bias) are static values.
    // Extends the original "initializer-only" check — Cycle 3 confirmed
    // Kokoro's affine scales/biases come from Constant Float32 nodes, not
    // initializers, so the walker was missing all 51 real instances.
    //
    // Not recognized as static: Cast-of-Constant, Unsqueeze-of-Constant,
    // or any other chain. A broader shape/value inference pass would be
    // needed for those.
    let is_static = |name: &str| -> bool {
        weights.contains_key(name)
            || output_to_node
                .get(name)
                .map(|n| n.op_type == "Constant")
                .unwrap_or(false)
    };

    // Detect Add -> Sqrt -> Div pattern (normalization)
```

#### Step 5: Replace the AddMulAdd walker's static-value check

Still in `src/cuda/inference/fusion/detection.rs`, find the AddMulAdd walker. The check is in the inner block where `b_input` and `c_input` have been extracted. The relevant block currently reads:

```rust
                    // Only fuse if b and c are weights (not computed values)
                    // This ensures the affine transform pattern (gamma, beta are weights)
                    if !weights.contains_key(&b_input) || !weights.contains_key(&c_input)
                    {
                        continue;
                    }
```

Replace with:

```rust
                    // Only fuse if b and c are static (initializers or
                    // Constant node outputs). This preserves the affine-
                    // transform semantic — gamma/beta are baked into the
                    // model — while recognizing that modern ONNX exports
                    // emit these as Constant Float32 nodes rather than
                    // initializers. See the `is_static` closure above.
                    if !is_static(&b_input) || !is_static(&c_input) {
                        continue;
                    }
```

Two-line change (plus updated comment). The `is_static` closure captured `weights` and `output_to_node` at the top of the function, so both are in scope.

#### Step 6: Build

```bash
cargo build --features cuda
```

Expected: clean compile. Likely first-attempt errors:
- Missing `FusedPattern` import in the test file — add it to the test file's use block.
- `is_static` closure lifetime issues — unlikely because it only captures references, but if clippy complains about closure elision, add explicit `|name: &str|` annotation (already in the template above).

#### Step 7: Run the new unit tests — verify green state

```bash
cargo test --features cuda --test fusion_detection_test -- add_mul_add
```

Expected: all four tests pass:
- `add_mul_add_detects_with_initializer_bc` ✓
- `add_mul_add_detects_with_constant_bc` ✓ (was red, now green — the Cycle 3 fix)
- `add_mul_add_rejects_when_b_is_computed` ✓
- `add_mul_add_rejects_when_c_is_computed` ✓

#### Step 8: Run the full fusion_detection_test suite

```bash
cargo test --features cuda --test fusion_detection_test
```

Expected: all existing tests (5 from Cycles 1-2) plus the 4 new tests pass. 9 total.

#### Step 9: Full regression check

```bash
cargo test --features cuda 2>&1 | grep -E "^test result" | tail -30
```

Expected: every test binary reports `0 failed`.

#### Step 10: Clippy and format

```bash
cargo clippy --features cuda -- -D warnings
cargo fmt --check
```

Expected: clean. If `cargo fmt --check` shows diffs, they should only be in files you touched — run `cargo fmt` on those files specifically. Pre-existing drift in unrelated files is not Cycle 3's responsibility (same note as Cycles 1.5a/2).

#### Step 11: Commit (signed)

```bash
git branch --show-current
# Must print: cycle3/addmuladd-detection-gap
```

```bash
git add src/cuda/inference/fusion/detection.rs tests/fusion_detection_test.rs
git commit -S -m "$(cat <<'EOF'
fix(fusion): recognize Constant nodes in AddMulAdd static-value check

Leadline's chain analysis found 51 Add->Mul->Add affine transform
instances in Kokoro, but the existing AddMulAdd walker matched zero.
Root cause: the walker's semantic-intent check required b and c to
be in the `weights` HashMap (initializers only), but Kokoro emits
affine scales/biases as Constant Float32 nodes.

Fix: add an inline `is_static` closure in detect_fused_patterns
that recognizes either (a) initializers in the weights HashMap or
(b) outputs of Constant nodes (looked up via the already-available
output_to_node map). Replace the AddMulAdd walker's two
`weights.contains_key` calls with `is_static` calls.

The fix preserves the "static affine transform" semantic — it
still rejects chains where b or c is a runtime-computed value like
a Shape output. It only broadens the definition of "static" from
"initializer" to "initializer or Constant node output," matching
the same pattern Cycle 2 established for Int64 shape precomputation.

Adds 4 new unit tests to tests/fusion_detection_test.rs:
  - add_mul_add_detects_with_initializer_bc (regression guard)
  - add_mul_add_detects_with_constant_bc (the Cycle 3 fix, was red)
  - add_mul_add_rejects_when_b_is_computed (semantic guard)
  - add_mul_add_rejects_when_c_is_computed (semantic guard mirror)

Kokoro integration verification lands in Task 2.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

If `git commit -S` fails with a signing error, STOP and report BLOCKED. Do NOT use `--no-gpg-sign`.

## Self-Review Checklist (Task 1)

- [ ] 4 new unit test functions added at the end of `tests/fusion_detection_test.rs`
- [ ] `float32_constant_attrs` helper added (and `int64_constant_attrs` if not already present)
- [ ] All 4 new tests pass after the fix
- [ ] `is_static` closure added near the top of `detect_fused_patterns`
- [ ] Two `weights.contains_key` calls in the AddMulAdd walker replaced with `is_static`
- [ ] No other walker is touched
- [ ] `cargo build --features cuda` clean
- [ ] `cargo test --features cuda` full suite green
- [ ] `cargo clippy --features cuda -- -D warnings` clean
- [ ] `git branch --show-current` shows the cycle branch
- [ ] Commit signed

---

## Phase B: Kokoro integration verification

### Task 2: Add `count_fused_patterns` helper and Kokoro assertion

**Files:**
- Modify: `src/cuda/inference/mod.rs` (one-line visibility change)
- Modify: `src/cuda/inference/testing.rs` (new `PatternCount` + `count_fused_patterns`)
- Modify: `tests/kokoro_gpu_fusion_test.rs` (new assertion block)

#### Step 1: Check `fused_patterns` field visibility

Open `src/cuda/inference/mod.rs`. Find the `GpuGraphExecutor` struct. Look for the `fused_patterns` field:

```bash
grep -n "fused_patterns:" src/cuda/inference/mod.rs
```

Expected: one match showing `fused_patterns: RefCell<HashMap<String, FusedPatternInfo>>,` (currently private, no `pub` prefix).

Change the declaration to:

```rust
    pub(crate) fused_patterns: RefCell<HashMap<String, FusedPatternInfo>>,
```

`pub(crate)` is the minimum needed — `testing.rs` is a child module of `cuda::inference`, so it can see `pub(crate)` fields in its parent. Other crate code already accesses `fused_patterns` via `self.fused_patterns`, so widening doesn't change behavior.

#### Step 2: Build to verify the visibility change compiles

```bash
cargo build --features cuda
```

Expected: clean. The visibility change is a one-line edit; compile should succeed immediately.

#### Step 3: Add the `PatternCount` struct and `count_fused_patterns` helper in `testing.rs`

Open `src/cuda/inference/testing.rs`. Scroll to the bottom. Append:

```rust

/// Snapshot of how many of each `FusedPattern` variant the detection
/// walker registered. Used by integration tests to verify specific
/// pattern detection works end-to-end on real graphs.
///
/// Every variant gets its own counter even if only some are asserted.
/// The unasserted ones serve as soft regression signal in the test's
/// diagnostic printout — a future change that silently drops DivRsqrt
/// from 65 to 50 would be visible in the test output.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PatternCount {
    pub div_rsqrt: usize,
    pub add_mul_add: usize,
    pub gelu: usize,
    pub mul_sin_pow_mul_add: usize,
    pub mul_add: usize,
    pub add_mul: usize,
    pub sub_mul: usize,
    pub div_mul: usize,
}

/// Count how many of each fused-pattern variant the detection walker
/// has registered on the given executor. Call AFTER a `run()` or
/// `run_with_profiling()` invocation has triggered `detect_fused_patterns`.
///
/// This is a test-only helper — production code should not need it.
///
/// Panics if `executor.fused_patterns` is currently mutably borrowed by
/// another thread (which cannot happen in normal test usage because the
/// walker is called inline inside `run_with_profiling` and completes
/// before returning).
pub fn count_fused_patterns(
    executor: &crate::cuda::inference::GpuGraphExecutor,
) -> PatternCount {
    let patterns = executor.fused_patterns.borrow();
    let mut count = PatternCount::default();
    for info in patterns.values() {
        match &info.pattern {
            FusedPattern::DivRsqrt { .. } => count.div_rsqrt += 1,
            FusedPattern::AddMulAdd { .. } => count.add_mul_add += 1,
            FusedPattern::Gelu { .. } => count.gelu += 1,
            FusedPattern::MulSinPowMulAdd { .. } => count.mul_sin_pow_mul_add += 1,
            FusedPattern::MulAdd { .. } => count.mul_add += 1,
            FusedPattern::AddMul { .. } => count.add_mul += 1,
            FusedPattern::SubMul { .. } => count.sub_mul += 1,
            FusedPattern::DivMul { .. } => count.div_mul += 1,
        }
    }
    count
}
```

The `FusedPattern` type is already imported via the existing `pub use crate::cuda::inference::fusion::{FusedPattern, FusedPatternInfo};` at the top of the file (lines 23-27 of the current `testing.rs`).

#### Step 4: Build and verify

```bash
cargo build --features cuda
```

Expected: clean. If rustc complains about `executor.fused_patterns` being private, Step 1 wasn't applied correctly — revisit.

#### Step 5: Add the Kokoro integration assertion block

Open `tests/kokoro_gpu_fusion_test.rs`. Find the existing `fusion_fires_on_full_kokoro_at_seq_len_5` test function. It currently ends with the Cycle 2 precompute hit-rate assertions and a final closing `}`.

Insert the new Cycle 3 assertion block AFTER the existing Cycle 2 assertions, BEFORE the final `}`:

```rust

    // === Cycle 3: AddMulAdd detection gap fix ===
    //
    // Pre-Cycle-3, the AddMulAdd walker required b and c to be
    // initializers (weights.contains_key) and found 0 matches on
    // Kokoro, despite Leadline's chain analysis identifying 51 real
    // `Add -> Mul -> Add` instances in the graph. Cycle 3 extends
    // the walker's static-value check to also accept Constant-node
    // outputs.
    //
    // Expected hit count: close to 51, but potentially lower because:
    //   - Some chains may have multi-use intermediate outputs
    //     (the walker requires single-use)
    //   - Some may be subsumed by a future AddMulAddLeakyRelu variant
    //     (not yet implemented as of Cycle 3)
    //
    // The assertion threshold is 40, comfortably below 51 to absorb
    // these exclusions. A count between 30 and 40 would indicate the
    // walker is mostly working but has additional exclusions worth
    // investigating (future cycle). A count below 30 would indicate
    // the fix isn't working as expected and requires STOP-level
    // investigation.
    use iconnx::cuda::inference::testing::count_fused_patterns;
    let pattern_counts = count_fused_patterns(&executor);

    eprintln!("\n=== Cycle 3 fused pattern counts ===");
    eprintln!("  DivRsqrt:          {:>4}", pattern_counts.div_rsqrt);
    eprintln!("  AddMulAdd:         {:>4}", pattern_counts.add_mul_add);
    eprintln!("  Gelu:              {:>4}", pattern_counts.gelu);
    eprintln!("  MulSinPowMulAdd:   {:>4}", pattern_counts.mul_sin_pow_mul_add);
    eprintln!("  MulAdd:            {:>4}", pattern_counts.mul_add);
    eprintln!("  AddMul:            {:>4}", pattern_counts.add_mul);
    eprintln!("  SubMul:            {:>4}", pattern_counts.sub_mul);
    eprintln!("  DivMul:            {:>4}", pattern_counts.div_mul);

    assert!(
        pattern_counts.add_mul_add >= 40,
        "Cycle 3 AddMulAdd detection: expected at least 40 instances on \
         Kokoro seq_len=5, got {}. Leadline's chain analysis found 51 \
         real instances; the walker's single-use requirement can \
         legitimately exclude some. A count of 30-40 indicates the fix \
         is working but has additional exclusions worth investigating. \
         A count below 30 usually means the fix isn't landing — check \
         that is_static is recognizing Constant nodes.",
        pattern_counts.add_mul_add,
    );
```

#### Step 6: Build

```bash
cargo build --features cuda --tests
```

Expected: clean. The new imports (`count_fused_patterns`) should resolve via the existing `iconnx::cuda::inference::testing::*` path.

#### Step 7: Run the non-ignored test suite for regressions

```bash
cargo test --features cuda 2>&1 | grep -E "^test result" | tail -30
```

Expected: every test binary reports `0 failed`. The Kokoro test itself is `#[ignore]` and won't run by default; this is a regression check on everything else.

#### Step 8: Symlink the Kokoro model

```bash
ls -la kokoro-v1.0.onnx 2>&1 || \
    ln -sf /home/ryano/workspace/ryanoneill/claudio/kokoro-v1.0.onnx kokoro-v1.0.onnx
ls -la kokoro-v1.0.onnx
```

Confirm you see a symlink pointing at the real model file.

#### Step 9: Run the ignored Kokoro test with `--nocapture`

```bash
cargo test --features cuda --test kokoro_gpu_fusion_test -- --ignored --nocapture 2>&1 | tee /tmp/cycle3-kokoro-output.txt | tail -100
```

**Possible outcomes:**

1. **PASS** — the new assertion succeeds. `pattern_counts.add_mul_add >= 40`. Continue to Step 10.

2. **FAIL at `>= 40`** with count in `[30, 40)`. This means the fix is firing but more chains are excluded than expected. STOP and report BLOCKED with the exact count and the full output of the diagnostic printout. The investigation path involves counting multi-use intermediates or instances that would be subsumed by a future AddMulAddLeakyRelu — those are out of scope for Cycle 3. The decision is either to lower the threshold (if the exclusions are legitimate) or to escalate.

3. **FAIL at `>= 40`** with count below 30. This means the fix isn't working. STOP and report BLOCKED. Likely causes: `is_static` closure not reading `output_to_node` correctly, visibility change not applied, or a typo in the `Constant` op_type check.

4. **FAIL with a different error** — the full Kokoro test might fail for some other reason (e.g., a pre-existing Cycle 2 assertion regressing). STOP and report the exact failure.

Save the full output at `/tmp/cycle3-kokoro-output.txt` for Task 3's results doc (the `tee` command already does this).

#### Step 10: Clean up the symlink

```bash
rm -f kokoro-v1.0.onnx
git status
```

Expected: only `src/cuda/inference/mod.rs`, `src/cuda/inference/testing.rs`, and `tests/kokoro_gpu_fusion_test.rs` modified.

#### Step 11: Clippy

```bash
cargo clippy --features cuda -- -D warnings
```

Expected: clean.

#### Step 12: Commit (signed)

```bash
git branch --show-current
git add src/cuda/inference/mod.rs src/cuda/inference/testing.rs tests/kokoro_gpu_fusion_test.rs
git commit -S -m "$(cat <<'EOF'
test(kokoro): assert Cycle 3 AddMulAdd detection hit count

Adds count_fused_patterns + PatternCount to the testing facade so
integration tests can assert how many of each fused-pattern variant
the walker registered. Uses them to extend
fusion_fires_on_full_kokoro_at_seq_len_5 with a new Cycle 3
assertion: pattern_counts.add_mul_add >= 40.

Threshold rationale: Leadline's chain analysis found 51 real
Add->Mul->Add instances in Kokoro. The walker may find fewer due
to single-use exclusions and future pattern subsumption (22
AddMulAddLeakyRelu instances). The 40 threshold absorbs these
legitimate exclusions while catching a gross fix regression.

Also widens `fused_patterns` field visibility from private to
pub(crate) so the testing module can read it — same discipline
as Cycle 2's `nodes` field change. No new public API surface
outside the testing module.

The diagnostic printout shows all 8 fused-pattern variant counts,
providing soft regression signal for Cycle 1's DivRsqrt (65),
Gelu (12), MulSinPowMulAdd (48), and the Cycle 1.5a fusions.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

## Self-Review Checklist (Task 2)

- [ ] `fused_patterns` field is `pub(crate)` in `mod.rs`
- [ ] `PatternCount` struct added at the end of `testing.rs` with all 8 variant counters
- [ ] `count_fused_patterns` helper added with the full match
- [ ] Kokoro test has the new Cycle 3 assertion block with diagnostic printout + `>= 40` assertion
- [ ] `cargo build --features cuda` clean
- [ ] `cargo test --features cuda` non-ignored tests all pass
- [ ] Kokoro ignored test PASSES with the new assertion
- [ ] Full Kokoro output captured at `/tmp/cycle3-kokoro-output.txt`
- [ ] Symlink cleaned up
- [ ] `cargo clippy --features cuda -- -D warnings` clean
- [ ] Branch still on cycle3 branch (not detached)
- [ ] Commit signed

---

## Phase C: Cycle completion

### Task 3: Write the Cycle 3 results document

**Files:**
- Create: `docs/performance/2026-04-11-cycle3-addmuladd-detection-gap-results.md`

This task is pure documentation. No code changes.

#### Step 1: Extract the AddMulAdd count from Task 2's output

Open `/tmp/cycle3-kokoro-output.txt` and find the "Cycle 3 fused pattern counts" block. Record:
- The AddMulAdd count (the one we assert on)
- The other 7 variant counts (for the full snapshot)
- Any other interesting observations

#### Step 2: Write the results document

Create `docs/performance/2026-04-11-cycle3-addmuladd-detection-gap-results.md` using this template. Fill in all `<FILL IN>` placeholders with actual values from the Task 2 run:

```markdown
# Cycle 3 Results: AddMulAdd Detection Gap Fix

**Date:** 2026-04-11
**Spec:** [docs/superpowers/specs/2026-04-11-addmuladd-detection-gap-design.md](../superpowers/specs/2026-04-11-addmuladd-detection-gap-design.md)
**Plan:** [docs/superpowers/plans/2026-04-11-addmuladd-detection-gap.md](../superpowers/plans/2026-04-11-addmuladd-detection-gap.md)
**Branch:** `cycle3/addmuladd-detection-gap` (3 signed commits)

## Headline

The existing AddMulAdd fusion walker — already shipped in Cycle 1 — was finding **zero** matches on Kokoro despite Leadline's chain analysis identifying 51 real `Add → Mul → Add` instances in the graph. Cycle 3 fixes the walker's "is this a static value?" check to recognize Constant-node outputs, not just initializers. The walker now fires on **<FILL IN>** instances per inference (expected range: 40-51).

## Pattern counts on live Kokoro seq_len=5

From the diagnostic printout in `fusion_fires_on_full_kokoro_at_seq_len_5`:

```
<FILL IN: paste the "Cycle 3 fused pattern counts" block verbatim>
```

**AddMulAdd went from 0 to <FILL IN>** — eliminating 2 × <FILL IN> = <FILL IN * 2> kernel launches per inference (each AddMulAdd fusion replaces 3 ops with 1, saving 2 launches per instance).

## How

Root cause: the walker's semantic-intent check at `src/cuda/inference/fusion/detection.rs` required both `b` and `c` (the Mul's other operand and the second Add's other operand) to be in the `weights` HashMap — i.e., initializers registered via `add_initializer`. Modern ONNX exports (including Kokoro) emit affine transform scales and biases as `Constant` Float32 nodes rather than baking them into initializers, so the walker was rejecting every real match.

Fix: add an inline `is_static` closure that recognizes both initializers and Constant-node outputs, and replace the AddMulAdd walker's two `weights.contains_key()` calls with `is_static()` calls:

```rust
let is_static = |name: &str| -> bool {
    weights.contains_key(name)
        || output_to_node
            .get(name)
            .map(|n| n.op_type == "Constant")
            .unwrap_or(false)
};
```

Same class of gap as Cycle 2's Int64 precomputation fix: both discovered that modern ONNX graphs emit Constants where the walker expected initializers.

## What shipped

Three signed commits on `cycle3/addmuladd-detection-gap`:

1. `<FILL IN: sha>` — `fix(fusion): recognize Constant nodes in AddMulAdd static-value check`
2. `<FILL IN: sha>` — `test(kokoro): assert Cycle 3 AddMulAdd detection hit count`
3. `<FILL IN: sha>` — `docs(perf): Cycle 3 results — AddMulAdd detection gap fixed`

### Structural changes

- **`src/cuda/inference/fusion/detection.rs`** — added `is_static` closure near the top of `detect_fused_patterns`; replaced the two `weights.contains_key()` calls in the AddMulAdd walker with `is_static()` calls. 12 lines net.
- **`src/cuda/inference/mod.rs`** — widened `fused_patterns` field visibility from private to `pub(crate)` for testing module access. 1 line.
- **`src/cuda/inference/testing.rs`** — added `PatternCount` struct and `count_fused_patterns` helper for integration test visibility into detection results. ~70 lines.
- **`tests/fusion_detection_test.rs`** — added `float32_constant_attrs` helper and 4 new unit tests (hit from initializer, hit from Constant, reject when b is computed, reject when c is computed).
- **`tests/kokoro_gpu_fusion_test.rs`** — extended `fusion_fires_on_full_kokoro_at_seq_len_5` with Cycle 3 assertion block and diagnostic printout.

## Limitations

- **Cast-of-Constant chains not recognized.** If a graph has `Constant → Cast → Mul.b`, the walker still rejects the fusion because `b` is produced by a `Cast` node, not a `Constant`. A broader shape/value inference pass (Approach C from Cycle 2's brainstorming) would be needed.
- **Multi-use intermediate outputs.** The walker requires each intermediate (Add1's output, Mul's output) to be single-use. Some Kokoro chains may have the Add1 output feed multiple downstream ops, which legitimately excludes them from fusion.
- **AddMulAddLeakyRelu subsumption.** Leadline identified 22 `Add → Mul → Add → LeakyRelu` chains that share a prefix with plain AddMulAdd. Cycle 3 now matches those 22 as plain AddMulAdd — a future AddMulAddLeakyRelu walker would need to run BEFORE AddMulAdd (longer patterns first) to reclaim them.

## Validation gate scorecard

| Gate | Result |
|---|---|
| `cargo test --features cuda` (full suite) | ✅ 0 failed |
| `cargo clippy --features cuda -- -D warnings` | ✅ Clean |
| 4 new unit tests in fusion_detection_test.rs | ✅ All pass |
| `tests/kokoro_gpu_fusion_test.rs::fusion_fires_on_full_kokoro_at_seq_len_5` | ✅ PASS; add_mul_add count = <FILL IN> |
| `src/cuda/inference/fusion/detection.rs` growth | 12 lines (within spec's ≤ 12) |
| `src/cuda/inference/testing.rs` growth | ~70 lines (within spec's ~60-80) |
| All commits signed | ✅ 3/3 signed |
| Branch | `cycle3/addmuladd-detection-gap` |

## Follow-ups

From Leadline's priority list, remaining targets:

1. **AddMulAddLeakyRelu pattern** (22 instances) — new fusion pattern, 66 launches saved. Must run before AddMulAdd in walker order.
2. **cuDNN LayerNorm replacement** (31 calls, ~4 ms GPU time) — separate cycle.
3. **Approach C — broader shape/value inference** — would close the remaining Reshape 51/61 gap from Cycle 2 AND let AddMulAdd match Cast-of-Constant chains.
4. **Other shape-only ops** (Shape, Expand, Gather-on-constants, ConstantOfShape) — same Cycle 2 pattern, smaller wins.
5. **Cycles 1.5b/c/d profiling infrastructure** — `OnnxModel::tensor_shapes()`, per-inference pool metrics, per-node shapes in `NodeExecutionLog`.

Leadline-bench should re-run the comparison on this updated main to see the effect of 2 × <FILL IN> more launch eliminations.
```

#### Step 3: Commit (signed)

```bash
git branch --show-current
git add docs/performance/2026-04-11-cycle3-addmuladd-detection-gap-results.md
git commit -S -m "$(cat <<'EOF'
docs(perf): Cycle 3 results -- AddMulAdd detection gap fixed

Documents the Cycle 3 outcome: the existing AddMulAdd walker
(shipped in Cycle 1) was finding zero matches on Kokoro despite
Leadline's chain analysis identifying 51 real instances. Cycle 3
fixes the walker's "is this static?" check to recognize Constant
node outputs, not just initializers — same class of gap as Cycle 2
fixed for Int64 precomputation.

Result on live Kokoro seq_len=5: AddMulAdd fires on <N> instances
per inference (eliminating 2*<N> kernel launches).

Key insight for future cycles: modern ONNX exports emit affine
transform parameters as Constant nodes, not initializers. Any
fusion walker that checks "weights.contains_key" should also check
"is this a Constant node output." The `is_static` closure added
in Cycle 3 is the single source of truth for this predicate and
can be reused by future walkers.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

## Self-Review Checklist (Task 3)

- [ ] Results doc created at the right path
- [ ] All `<FILL IN>` placeholders replaced with real values from `/tmp/cycle3-kokoro-output.txt`
- [ ] Three commit SHAs filled in (from Tasks 1, 2, 3)
- [ ] Commit signed
- [ ] Branch still on cycle3 branch

---

## Final Validation Gates

Before marking Cycle 3 complete, confirm every gate:

1. [ ] `cargo test --features cuda` — all existing tests plus 4 new unit tests pass
2. [ ] `cargo test` (no-cuda) — pre-existing `validation_test.rs` failure unchanged; no new failures
3. [ ] `cargo clippy --features cuda -- -D warnings` — clean
4. [ ] `cargo fmt --check` on modified files — clean
5. [ ] Existing Cycles 1-2 fusion detection tests (5 total) — all still pass
6. [ ] Existing Cycle 2 precompute unit tests (9 total) — all still pass
7. [ ] `tests/kokoro_gpu_fusion_test.rs` with `--ignored --nocapture` — passes, `pattern_counts.add_mul_add >= 40`, diagnostic block printed
8. [ ] `src/cuda/inference/fusion/detection.rs` grew by ≤ 12 lines
9. [ ] `src/cuda/inference/testing.rs` grew by ~70 lines
10. [ ] All 3 commits signed
11. [ ] Branch still on `cycle3/addmuladd-detection-gap` (not detached)
12. [ ] Results doc includes actual Kokoro AddMulAdd count

---

## Review Checklist (self-review at cycle end)

- Did the AddMulAdd count actually land at 40+ on Kokoro? If lower, was the investigation conclusive about why?
- Did any other fused-pattern variant count shift unexpectedly? If DivRsqrt dropped from 65 or GELU dropped from 12, the `is_static` change somehow affected them — investigate.
- Is the `is_static` closure cleanly reusable if a future walker needs the same predicate?
- Did we remember to capture the real count in the results doc, not the template placeholder?
