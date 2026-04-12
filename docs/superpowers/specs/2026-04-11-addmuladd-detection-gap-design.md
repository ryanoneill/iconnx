# Cycle 3: AddMulAdd Detection Gap Fix — Design

**Date:** 2026-04-11
**Author:** Claude (brainstormed with Ryan O'Neill, reviewed by Leadline Claude)
**Status:** Design, pending implementation plan
**Supersedes/extends:** [Cycle 2 results](../../performance/2026-04-11-cycle2-zero-cost-shape-ops-results.md)

---

## Context

Leadline's chain analysis on the Kokoro graph identified **51 `Add → Mul → Add` affine-transform instances** in the model, each a candidate for the existing `FusedPattern::AddMulAdd` fusion that Cycle 1 already shipped. Yet the walker finds **zero matches** on the live Kokoro run. The chain pattern exists, the kernel exists, the fused dispatch exists — and somehow the walker is rejecting all 51 instances.

Investigation of `src/cuda/inference/fusion/detection.rs` (the Cycle 1 Phase A extraction of the walker) pinpoints the root cause at lines 161–166:

```rust
// Only fuse if b and c are weights (not computed values)
// This ensures the affine transform pattern (gamma, beta are weights)
if !weights.contains_key(&b_input) || !weights.contains_key(&c_input) {
    continue;
}
```

This check requires both `b` and `c` to be in the `weights` HashMap — i.e., **initializers** registered via `add_initializer`. Modern ONNX exports (including Kokoro) typically emit affine transform scales and biases as `Constant` Float32 nodes rather than baking them into initializers. The walker's check doesn't recognize Constant-node outputs, so every match fails.

This is the **same class of gap** Cycle 2 fixed for Int64 static values in the shape-op precomputation path. The fix is analogous: extend the "is this static?" definition to include `Constant`-node outputs, not just initializers.

## Goals

1. **Close the AddMulAdd detection gap on Kokoro.** The walker should fire on the ~51 real instances in the graph, allowing a safety margin down to 40 to absorb legitimate exclusions (multi-use intermediates, patterns subsumed by a larger variant like AddMulAddLeakyRelu).

2. **Extend the "is static?" check** via an inline `is_static` closure in `detect_fused_patterns` that recognizes either (a) initializers in the `weights` HashMap or (b) outputs of `Constant` nodes (looked up via the already-available `output_to_node` map).

3. **Preserve the semantic intent** of the original check. The fused kernel (`gpu_fused_add_mul_add`) is numerically correct for any operand values, but the walker's check was a deliberate semantic guard to match only "static affine transform" patterns — not arbitrary `Add → Mul → Add` triples. Cycle 3 broadens the definition of "static" but keeps the guard.

4. **Preserve bit-identical numerical output** on Kokoro.

5. **Add a testing-facade helper** `count_fused_patterns(executor) -> PatternCount` so the Kokoro integration test can assert the hit count by variant. Emits a diagnostic printout of all fused-pattern counts for future regression signal.

6. **The `is_static` closure becomes the single definition** of "static value" for fusion-pattern detection. If a future walker needs the same predicate, it reuses this closure. If the definition needs to broaden again (e.g., adding Cast-of-Constant support in a future cycle), the change happens in exactly one place.

## Non-Goals

1. **Dropping the static-value check entirely** (the "Option 2" considered during brainstorming). Rejected — preserves semantic intent.
2. **Extending recognition to Cast-of-Constant, Unsqueeze-of-Constant, or other chains.** A broader shape/value inference pass (Approach C from Cycle 2's brainstorming) would address this. Out of scope for Cycle 3.
3. **Adding the `AddMulAddLeakyRelu` fusion pattern** (Leadline's 22 instances, separately prioritized). Separate future cycle. See "Walker ordering interaction" under Risks below.
4. **Changing the fused kernel** `gpu_fused_add_mul_add` or its dispatch arm. Kernels are untouched.
5. **Re-measuring or adjusting other fusion walkers** (DivRsqrt, Gelu, MulSinPowMulAdd, MulAdd, AddMul, SubMul, DivMul). Their logic is unchanged.
6. **Generalizing `static_i64_cache`** or Cycle 2's precomputation infrastructure for Float32 constants. Cycle 3 only needs to know "is this name a Constant node output?" — it doesn't need the *value*, just the op_type, which is already in `output_to_node`.

## Approach

**Approach A from brainstorming: inline `is_static` closure in `detect_fused_patterns`.** Selected over two alternatives:

- **Approach B (standalone helper function):** rejected — adds a function that takes 3 parameters, most of which come from `detect_fused_patterns`'s locals. The closure captures them implicitly.
- **Approach C (pre-computed `HashSet<&str>` of static output names):** rejected — adds an allocation and traversal used by only one walker. YAGNI for Cycle 3's scope.

The closure is the minimal, principled fix. It matches the existing code style, keeps state local to the function that uses it, and doesn't introduce premature abstraction. If a future cycle needs the predicate across many walkers, Approaches B or C can be promoted then.

## File Layout

All changes land in three existing files plus extensions to one existing test file. No new files.

### Modified files

- **`src/cuda/inference/fusion/detection.rs`** — add the `is_static` closure, change the two `weights.contains_key()` calls in the AddMulAdd walker to `is_static()`. Net: ~12 lines added (closure + inline doc comment + two replacements).
- **`src/cuda/inference/testing.rs`** — add `PatternCount` struct and `count_fused_patterns` helper. Net: ~60-80 lines added.
- **`src/cuda/inference/mod.rs`** — possibly widen `fused_patterns` field visibility from private to `pub(crate)` so the testing module can read it. Net: 1 line modified, OR zero if already `pub(crate)` (check in Task 1 Step 1).
- **`tests/fusion_detection_test.rs`** — extend with 3-4 new synthetic-graph tests for the AddMulAdd walker (hit from initializer, hit from Constant, reject when b is computed, optional mirror reject when c is computed). Net: ~120 lines added.
- **`tests/kokoro_gpu_fusion_test.rs`** — extend `fusion_fires_on_full_kokoro_at_seq_len_5` with a new Cycle 3 assertion block that calls `count_fused_patterns` and asserts `add_mul_add >= 40`. Includes a diagnostic printout of all variant counts. Net: ~25 lines added.

### Files NOT touched

- `src/cuda/inference/fusion/patterns.rs` — `FusedPattern::AddMulAdd` variant already exists from Cycle 1.
- `src/cuda/kernels/fused/**` — no kernel changes.
- `src/cuda/inference/gpu_event_timer.rs` — no changes.
- `run_with_profiling` and other dispatch paths in `src/cuda/inference/mod.rs` — no changes.
- Cycle 2's precomputation infrastructure (`precomputed_*` fields, `try_precompute_*` helpers) — no changes.

### Responsibility split

- **`is_static` closure:** single predicate for "is this tensor name a static value?" Local to `detect_fused_patterns`; captures `weights` and `output_to_node` by reference.
- **AddMulAdd walker:** uses `is_static` in place of the old `weights.contains_key()` check. Two one-line edits.
- **`count_fused_patterns` helper:** read-only snapshot of `executor.fused_patterns` grouped by variant. Test-only surface in the testing facade.
- **`PatternCount` struct:** opaque value type with per-variant counters.
- **Synthetic tests:** exercise the walker's hit/miss behavior in isolation on minimal graphs (initializer source, Constant source, computed source).
- **Kokoro integration test:** exercise the walker on the real model and verify ≥40 AddMulAdd matches.

## API Contract Details

### The `is_static` closure

Placed immediately after the `output_to_node` / `output_uses` setup at the top of `detect_fused_patterns`, before the first walker (DivRsqrt):

```rust
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
// initializers, so the walker was missing all 51 real instances on
// Kokoro.
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
```

The closure captures `weights` and `output_to_node` by reference. No new allocations.

### AddMulAdd walker change

The existing check at lines 161–166:

```rust
// Only fuse if b and c are weights (not computed values)
// This ensures the affine transform pattern (gamma, beta are weights)
if !weights.contains_key(&b_input) || !weights.contains_key(&c_input) {
    continue;
}
```

Replaced with:

```rust
// Only fuse if b and c are static (initializers or Constant node
// outputs). This preserves the affine-transform semantic — gamma/beta
// are baked into the model — while recognizing that modern ONNX
// exports emit these as Constant Float32 nodes rather than
// initializers.
if !is_static(&b_input) || !is_static(&c_input) {
    continue;
}
```

Two-line edit. No other walker is touched. Other walkers may reuse `is_static` in the future if needed, but none are modified in Cycle 3.

### `PatternCount` struct in `testing.rs`

```rust
/// Snapshot of how many of each `FusedPattern` variant the detection
/// walker registered. Used by integration tests to verify specific
/// pattern detection works end-to-end on real graphs.
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
```

All counts exposed even though Cycle 3 only asserts on `add_mul_add`. The other variants serve as soft regression signal — a future change that silently drops DivRsqrt from 65 to 60 would be visible in the test output (though not asserted).

### `count_fused_patterns` function in `testing.rs`

```rust
/// Count how many of each fused-pattern variant the detection walker
/// has registered on the given executor. Call AFTER a `run()` or
/// `run_with_profiling()` invocation has triggered `detect_fused_patterns`.
///
/// This is a test-only helper — production code should not need it.
pub fn count_fused_patterns(
    executor: &crate::cuda::inference::GpuGraphExecutor,
) -> PatternCount {
    let patterns = executor.fused_patterns.borrow();
    let mut count = PatternCount::default();
    for info in patterns.values() {
        match &info.pattern {
            crate::cuda::inference::fusion::FusedPattern::DivRsqrt { .. } => {
                count.div_rsqrt += 1;
            }
            crate::cuda::inference::fusion::FusedPattern::AddMulAdd { .. } => {
                count.add_mul_add += 1;
            }
            crate::cuda::inference::fusion::FusedPattern::Gelu { .. } => {
                count.gelu += 1;
            }
            crate::cuda::inference::fusion::FusedPattern::MulSinPowMulAdd { .. } => {
                count.mul_sin_pow_mul_add += 1;
            }
            crate::cuda::inference::fusion::FusedPattern::MulAdd { .. } => {
                count.mul_add += 1;
            }
            crate::cuda::inference::fusion::FusedPattern::AddMul { .. } => {
                count.add_mul += 1;
            }
            crate::cuda::inference::fusion::FusedPattern::SubMul { .. } => {
                count.sub_mul += 1;
            }
            crate::cuda::inference::fusion::FusedPattern::DivMul { .. } => {
                count.div_mul += 1;
            }
        }
    }
    count
}
```

**Visibility requirement:** `executor.fused_patterns` must be at least `pub(crate)` for the testing module to read it. If it's currently private, widen it in Task 1 Step 1. The testing module lives in the same `cuda::inference` crate-internal subtree as `mod.rs`, so `pub(crate)` is sufficient.

## Testing Strategy

### Unit tests — extend `tests/fusion_detection_test.rs`

Three (or four, with the mirror) new synthetic-graph tests. All use the existing `detect_fused_patterns_for_tests` facade from Cycle 1.

**Helper** — a new `float32_constant_attrs(value: f32)` helper for building Constant Float32 attributes, analogous to the existing `int64_constant_attrs` from Cycle 2's testing code.

**Test 1: `add_mul_add_detects_with_initializer_bc`** (regression guard)
Build a 3-op chain `Add(x, a) → Mul(t0, b) → Add(t1, c)` with `b` and `c` as initializers in the `weights` HashMap. Assert the walker registers an `AddMulAdd` pattern keyed on `add1`, with field values `x, a, b, c, y`. Regression guard for the existing behavior.

**Test 2: `add_mul_add_detects_with_constant_bc`** (new behavior, the core Cycle 3 test)
Same 3-op chain, but `b` and `c` are outputs of Constant Float32 nodes. The `weights` map is empty. Assert the walker registers an `AddMulAdd` pattern. Would fail on the pre-Cycle-3 code (empty weights → `weights.contains_key` returns false → pattern rejected). Passes after the `is_static` fix.

**Test 3: `add_mul_add_rejects_when_b_is_computed`** (semantic guard)
Build a chain where `b` is the output of a non-Constant op (e.g., `Shape`) and `c` is an initializer. Assert the walker does NOT register an `AddMulAdd` pattern. Guards against accidentally dropping the semantic check or extending it too broadly.

**Test 4 (optional mirror): `add_mul_add_rejects_when_c_is_computed`**
Same as Test 3 but with `c` coming from a non-Constant op and `b` as an initializer. For symmetry. ~15 additional lines.

### Integration test — extend `tests/kokoro_gpu_fusion_test.rs`

Append a new assertion block to `fusion_fires_on_full_kokoro_at_seq_len_5`, after all existing Cycle 2 assertions, before the final closing brace:

```rust
// === Cycle 3: AddMulAdd detection gap fix ===
//
// Pre-Cycle-3, the AddMulAdd walker required b and c to be
// initializers (weights.contains_key) and found 0 matches on
// Kokoro, despite Leadline's chain analysis identifying 51 real
// `Add -> Mul -> Add` instances in the graph. Cycle 3 extends
// the check to also accept Constant-node outputs.
//
// Expected hit count: close to 51, but potentially lower because:
//   - Some chains may have multi-use intermediate outputs
//     (the walker requires single-use)
//   - 22 instances may be subsumed by a future AddMulAddLeakyRelu
//     variant which would compete for the same head Add (though
//     that variant is not yet implemented as of Cycle 3)
//
// The assertion threshold is 40, comfortably below 51 to absorb
// these exclusions. A count between 30 and 40 would indicate the
// walker is mostly working but has additional exclusions worth
// investigating (future cycle). A count below 30 would indicate
// the fix isn't working as expected and should STOP for
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

### What is NOT tested

- **Performance microbenchmarks.** Cycles 1.5a and 2 established that the GPU event timer has a per-call floor that makes timing deltas flaky at small scales. The hit-count assertion is the primary signal, analogous to Cycle 2's hit-rate checks.
- **Unit tests for `count_fused_patterns` in isolation.** The function walks a HashMap and matches variants — mechanical. The integration test exercises it on the real Kokoro graph, which is more valuable.
- **Tight assertions on other fused-pattern variants' counts.** The diagnostic printout gives soft signal but doesn't assert. A future cycle could add tight assertions on DivRsqrt (65), GELU (12), MulSinPowMulAdd (48) if flakiness becomes a concern.

### TDD discipline

The unit tests are written red-then-green:

1. Write Tests 1, 2, 3, (4).
2. Run — Test 1 passes (initializer case is existing behavior), Test 2 FAILS (Constant case is Cycle 3's new behavior), Tests 3 and 4 pass (semantic guard is preserved by any reasonable fix).
3. Add the `is_static` closure and replace the two `weights.contains_key` calls in `detection.rs`.
4. Run — all four tests pass.
5. Extend the Kokoro test with the new assertion block.
6. Run the Kokoro test on the real model. Verify the count meets the threshold.

## Error Handling

**No new error paths.** The `is_static` closure returns a `bool`; the walker's existing `continue` path handles rejection. No `CudaError` variants change. No `Result` types change. The only failure mode is "precomputation walker rejects a pattern," which is the existing behavior.

The `count_fused_patterns` helper takes `.borrow()` on a `RefCell`. If called concurrently with `.borrow_mut()` it panics — but in practice the walker is called inline inside `run_with_profiling` and completes before the function returns, so borrow lifetimes never overlap in normal usage. Documented in the helper's doc comment.

## Risks and Mitigations

**Risk 1: `is_static` too permissive.** Could match patterns that shouldn't be fused.
**Mitigation:** Kokoro is inference-only; any correctly-shaped `(x + a) * b + c` triple is safe to fuse. The bit-identical numerical test on the full Kokoro model catches any divergence. If numerical tests pass, fusion is semantically correct.

**Risk 2: `is_static` too restrictive — misses Cast-of-Constant chains.**
**Mitigation:** Explicitly documented as out-of-scope. The 40-count threshold has ~20% headroom below Leadline's 51 count, which absorbs this and the other known exclusions. Task 4 of the plan includes diagnostic instrumentation to count the remaining cases if the threshold fails.

**Risk 3: `executor.fused_patterns` visibility.**
**Mitigation:** One-line widening from private to `pub(crate)` if needed, same discipline as Cycle 2's `nodes` field change.

**Risk 4: Walker ordering interaction with the future AddMulAddLeakyRelu pattern.**
**Mitigation:** This is a **Cycle 4 problem**. When AddMulAddLeakyRelu lands, its walker must run BEFORE AddMulAdd (longer patterns first, same rule Cycle 1 Phase C established). Cycle 3 does not need to preemptively handle it — the conflict only materializes when Cycle 4's walker is added.

**Risk 5: `count_fused_patterns` panic on re-entrant borrow.**
**Mitigation:** Tests call it after `run_with_profiling` returns, when no `borrow_mut` is outstanding. Documented in the helper's doc comment.

**Risk 6: The 40-count threshold is too tight.** If Kokoro's real matches land at 35-39 due to exclusions Leadline didn't account for, the test fails and the investigation path determines the right threshold.
**Mitigation:** Task 4 of the plan includes a diagnostic instrumentation phase that counts what the walker is matching AND what it's rejecting, so the investigator has data rather than speculation.

**Risk 7: Signed-commit discipline from CLAUDE.md.**
**Mitigation:** Same discipline as Cycles 1-2. The implementer stops and asks if signing fails. No `--no-gpg-sign`.

## Validation Gates

Cycle 3 is complete when all of these are true:

1. `cargo test --features cuda` — all existing tests plus 3-4 new unit tests in `fusion_detection_test.rs` pass.
2. `cargo test` (no-cuda) — pre-existing `validation_test.rs` failure unchanged; no new failures.
3. `cargo clippy --features cuda -- -D warnings` — clean.
4. `cargo fmt --check` on modified files — clean.
5. Existing Cycles 1-2 fusion detection tests — all still pass.
6. Existing Cycle 2 precompute unit tests (9 total) — all still pass.
7. `tests/kokoro_gpu_fusion_test.rs::fusion_fires_on_full_kokoro_at_seq_len_5` with `--ignored --nocapture` — passes, with `pattern_counts.add_mul_add >= 40` asserted successfully and the `=== Cycle 3 fused pattern counts ===` diagnostic block appearing in the output.
8. `src/cuda/inference/fusion/detection.rs` grew by ≤ 12 lines net.
9. `src/cuda/inference/testing.rs` grew by ~60-80 lines (the `PatternCount` struct and `count_fused_patterns` helper).
10. All commits signed per CLAUDE.md.
11. The branch is not in detached HEAD state.
12. The actual Kokoro AddMulAdd count is captured in the final commit message and/or results doc so Leadline can update their analysis.

## Out of scope (for future cycles)

- **`AddMulAddLeakyRelu` pattern** (22 instances per Leadline). Separate fusion variant; requires ordering carefully with the existing AddMulAdd walker.
- **Broader shape/value inference** covering Cast-of-Constant and similar chains. Approach C from Cycle 2's brainstorming.
- **cuDNN LayerNorm replacement** (31 calls, ~4 ms GPU time). From Cycle 1.5a's backlog.
- **Cycles 1.5b/c/d profiling infrastructure** — `OnnxModel::tensor_shapes()`, per-inference pool metrics, per-node shapes in `NodeExecutionLog`.
- **Other precomputation targets** — Shape, Expand, Gather-on-constants, ConstantOfShape.
