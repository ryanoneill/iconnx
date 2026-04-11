# Cycle 2: Zero-Cost Unsqueeze / Reshape / Squeeze — Design

**Date:** 2026-04-11
**Author:** Claude (brainstormed with Ryan O'Neill, reviewed by Leadline Claude)
**Status:** Design, pending implementation plan
**Supersedes/extends:** [Cycle 1.5a results](../../performance/2026-04-11-cycle1.5a-gpu-side-timing-results.md)

---

## Context

Cycle 1.5a added GPU-side kernel timing via `GpuEventTimer` and discovered that `Unsqueeze` shows 4.39 ms of GPU time across 192 calls on the live Kokoro run, `Reshape` shows 11.19 ms across 61 calls, and `Squeeze` is smaller. These are supposed to be metadata-only operators — they change a tensor's shape vector without touching data — so any measurable GPU time is a bug, not a characteristic.

Investigation of the dispatch code in `src/cuda/inference/mod.rs` confirmed that the kernels themselves are already metadata-only (`inputs[0].reshape(new_shape)` is a pure CPU `Arc` clone on the `GpuTensor`). The measured GPU time comes from `get_or_fetch_i64` at lines 2033, 2091, and 2135, which falls through to `tensor.to_host_i64(&self.ctx)` — a D2H transfer that synchronizes the CUDA stream. Any pending kernels at the time of the sync get attributed to the shape-op's GPU interval, producing the observed overhead.

iconnx already has a targeted precomputation pattern for `Slice`: `try_precompute_slice_params` in `add_node` pulls Slice's starts/ends/axes/steps out of `static_i64_cache` at graph-build time, storing the results on the `ExecutionNode` so the dispatch arm can skip the D2H entirely. The cache is populated by `add_initializer` (for Int64 weights) and by the runtime Constant dispatch at line 4230 (for Constant Int64 nodes).

This cycle extends the same pattern to `Unsqueeze`, `Reshape`, and `Squeeze`, with one critical enabler: Constant Int64 caching moves to `add_node` time, so `try_precompute_*` helpers can find Constant-produced values during graph construction.

## Goals

1. **Eliminate D2H-sync overhead** in the `Unsqueeze`, `Reshape`, and `Squeeze` dispatch paths for shape/axes inputs that are statically resolvable (from initializers or from `Constant` nodes added earlier in topological order).

2. **Move Constant-Int64-node caching from runtime to graph-build time.** Currently Constant nodes populate `static_i64_cache` during dispatch (line 4230); this spec moves that population into `add_node` so downstream `try_precompute_*` helpers can find the values. The existing runtime-dispatch caching stays as defense-in-depth.

3. **Extend the existing `precomputed_slice` pattern** with three new `Option<Vec<i64>>` fields on `ExecutionNode`:
   - `precomputed_reshape_shape: Option<Vec<i64>>` — raw shape values (before `0`/`-1` resolution)
   - `precomputed_unsqueeze_axes: Option<Vec<i64>>`
   - `precomputed_squeeze_axes: Option<Vec<i64>>`

4. **Dispatch arms check the precomputed field first** and skip `get_or_fetch_i64` entirely on a hit. Runtime-computed axes (from `Shape → Cast → Gather` chains) continue to D2H — this is a known limitation of the targeted approach and is documented explicitly.

5. **Measurable improvement on the Kokoro live run:** `gpu_op_times["Unsqueeze"]`, `gpu_op_times["Reshape"]`, and `gpu_op_times["Squeeze"]` each drop to under 2 ms total in the integration test's assertion. Pre-cycle baseline: 4.39 ms / 11.19 ms / < 1 ms respectively.

6. **Preserve bit-identical numerical output on Kokoro.** A unit test compares pre-computed-path output to D2H-path output element-wise; the integration test verifies the Kokoro numerical path hasn't regressed.

## Non-Goals

1. **Symbolic shape inference or constant folding** over arbitrary op chains. That's Approach C from brainstorming — a much larger future cycle.

2. **Changes to the existing `precomputed_slice` field** or `try_precompute_slice_params` logic. Those stay as-is; new code is additive.

3. **Other shape-only ops** (`Shape`, `Expand`, `Gather`-on-constants, `ConstantOfShape`). Each has the same D2H pattern but a separate cycle for each.

4. **Any change** to `GpuTensor::reshape`, the memory pool, `GpuEventTimer`, `ProfileData`, or any fused kernel.

5. **Generalizing `static_i64_cache`** to cover Float32 or Int32 constants. Int64 remains the only cached dtype — it's the only one used for shape/axes parameters.

6. **Runtime-computed axes paths** (e.g., `Shape(X) → Cast → Gather → Unsqueeze.axes`). Precomputation can't cover these at graph-build time; they continue to D2H. The integration test's 2 ms threshold absorbs some residual D2H from these cases.

## Approach

**Approach A from brainstorming: extend `precomputed_slice` with new per-op helpers and a graph-build-time Constant caching pass.** This was chosen over two alternatives:

- **Approach B (lazy first-run precomputation)** — decouples from `add_node` order but adds runtime state machinery. Current architecture already assumes topological order at add_node time (fusion detection depends on it), so the robustness benefit is theoretical. Rejected.

- **Approach C (full graph optimizer pass)** — handles arbitrary chains and is the ORT-style long-term solution. Too large for a single cycle. Explicitly deferred.

Approach A is the right size for a focused targeted-fix cycle: it matches the existing pattern exactly (same field shape, same helper shape, same call-site shape as `precomputed_slice`), handles the common case (Constant-sourced axes), and leaves the runtime-computed-chain case as a documented follow-up. The fallback to `get_or_fetch_i64` at runtime preserves correctness for any case precomputation misses.

## File Layout

All changes land in a single file: `src/cuda/inference/mod.rs`. No new modules, no kernel changes.

### New test files

```
tests/operators/unsqueeze_precompute_test.rs   (~120 lines)
tests/operators/reshape_precompute_test.rs     (~140 lines)
tests/operators/squeeze_precompute_test.rs     (~100 lines)
```

Each file has three tests:
1. `precompute_hit_from_constant` — verifies precomputation works against a Constant-sourced axes/shape input.
2. `precompute_miss_when_input_is_computed` — verifies precomputation correctly returns `None` for runtime-computed inputs.
3. `precompute_matches_runtime_d2h_bitwise` — end-to-end behavioral equivalence between precomputed and D2H dispatch paths.

### Modified files

- `src/cuda/inference/mod.rs`:
  - `ExecutionNode` — three new `Option<Vec<i64>>` fields (~6 lines)
  - `add_node` — Constant-Int64 caching block (~15 lines) and three calls to new `try_precompute_*` helpers (~9 lines)
  - Three new `try_precompute_*` helpers (~30 lines total)
  - `execute_gpu_operator_with_precomputed` — signature extension (see Section 3 for Option A vs Option B)
  - Three dispatch arms (`Unsqueeze`, `Reshape`, `Squeeze`) — each gains a `let Some(precomp) = ...` branch before the existing fallback chain (~6 lines per arm)
  - Call sites that invoke `execute_gpu_operator_with_precomputed` — 5 sites, each threads the new fields through (~5 lines per site)
  - **Net growth: ≤ 150 lines.**

- `tests/operators/mod.rs`: three new `mod …;` declarations in alphabetical position.

- `tests/kokoro_gpu_fusion_test.rs`: one new assertion block after the existing top-10 printout in `fusion_fires_on_full_kokoro_at_seq_len_5`, asserting each of the three ops drops below 2 ms GPU time.

### Responsibility split

- **`ExecutionNode` fields:** static metadata populated once at `add_node`, read-only during dispatch.
- **`try_precompute_*` helpers:** pure queries against `static_i64_cache`, no side effects.
- **Dispatch arms:** hot-path check `if let Some(precomp) = …` before falling through to `get_or_fetch_i64`.
- **Constant-at-add_node caching:** single-responsibility mutation of `static_i64_cache` at graph-build time.

## API Contract Details

### `ExecutionNode` field initialization

All three new fields default to `None` for non-shape-op nodes. Only Reshape/Unsqueeze/Squeeze attempt precomputation. Inside `add_node`, after the existing `precomputed_slice` branch:

```rust
let precomputed_reshape_shape = if op_type == "Reshape" {
    self.try_precompute_reshape_shape(&inputs)
} else {
    None
};
let precomputed_unsqueeze_axes = if op_type == "Unsqueeze" {
    self.try_precompute_unsqueeze_axes(&inputs)
} else {
    None
};
let precomputed_squeeze_axes = if op_type == "Squeeze" {
    self.try_precompute_squeeze_axes(&inputs)
} else {
    None
};
```

Each helper returns `None` if `inputs.len() < 2`, if `inputs[1]` is empty, or if `inputs[1]` is not in `static_i64_cache`. The three helpers are almost identical one-liners; if the implementer wants to unify them into a single `try_precompute_i64_from_input_at_index(inputs, 1)` helper, that's a pure code-quality call with no behavioral impact.

### Constant-Int64 caching at `add_node` time

Before the op-type-specific branches, `add_node` checks whether the incoming node is a `Constant` with an `Int64` value attribute, and if so populates `static_i64_cache` using the output name:

```rust
if op_type == "Constant" {
    if let Some(value_tensor) = attributes.get_tensor("value") {
        if let crate::tensor::Tensor::Int64(_) = value_tensor {
            let data = value_tensor.as_slice_i64();
            if let Some(output_name) = outputs.first() {
                self.static_i64_cache
                    .borrow_mut()
                    .insert(output_name.to_string(), data.to_vec());
            }
        }
    }
}
```

This is the critical enabler for the precomputation. Without it, the static cache would be empty at `add_node` time (no nodes have dispatched yet), so `try_precompute_*` would always return `None` for Constant-sourced inputs.

The existing runtime caching at line 4230 stays in place as a harmless defense-in-depth: it replaces the already-cached value with the same value on each run, costing one `HashMap::insert` call per Constant node per inference. Removing it would regress the "Constant added out-of-order" edge case that the runtime path currently handles. An optional follow-up tidy is to add an early-return at line 4230 if the key is already present; the implementer's call, not blocking.

### `execute_gpu_operator_with_precomputed` signature extension

The implementer has two choices:

**Option A (straightforward extension):**
```rust
fn execute_gpu_operator_with_precomputed(
    &self,
    op_type: &str,
    inputs: &[&GpuTensor],
    input_names: &[String],
    attributes: &NodeAttributes,
    precomputed_slice: Option<&PrecomputedSlice>,
    precomputed_reshape_shape: Option<&Vec<i64>>,
    precomputed_unsqueeze_axes: Option<&Vec<i64>>,
    precomputed_squeeze_axes: Option<&Vec<i64>>,
) -> Result<GpuTensor, CudaError>
```

**Option B (bundled struct):**
```rust
#[derive(Default)]
struct NodePrecomputed<'a> {
    slice: Option<&'a PrecomputedSlice>,
    reshape_shape: Option<&'a Vec<i64>>,
    unsqueeze_axes: Option<&'a Vec<i64>>,
    squeeze_axes: Option<&'a Vec<i64>>,
}

fn execute_gpu_operator_with_precomputed(
    &self,
    op_type: &str,
    inputs: &[&GpuTensor],
    input_names: &[String],
    attributes: &NodeAttributes,
    precomputed: NodePrecomputed<'_>,
) -> Result<GpuTensor, CudaError>
```

Option B is cleaner long-term (one parameter instead of four); Option A is a smaller diff. Both produce identical external behavior. Leadline flagged that Option A's parameter count is getting unwieldy — use that as a tiebreaker in favor of Option B if the implementer feels the signature is cluttered. Neither is required.

### Dispatch arm changes

**Unsqueeze** (current lines 2125-2171). The axes-extraction block gains a precomputed branch:

```rust
let axes: Vec<i64> = if let Some(precomp) = precomputed_unsqueeze_axes {
    // Cache hit from try_precompute_unsqueeze_axes — no D2H.
    precomp.clone()
} else if let Some(axes_attr) = attributes.get_ints("axes") {
    axes_attr.to_vec()
} else if inputs.len() > 1 {
    let axes_name = input_names.get(1).map(|s| s.as_str()).unwrap_or("");
    self.get_or_fetch_i64(axes_name, inputs[1])?
} else {
    return Err(CudaError::Kernel(
        "Unsqueeze requires 'axes' attribute or second input tensor".into(),
    ));
};
```

**Squeeze** (current lines 2082-2124): analogous — the `precomputed_squeeze_axes` check goes first, the rest of the chain is unchanged.

**Reshape** (current lines 2022-2081): The `precomputed_reshape_shape` check goes first, and on a hit it's used as the `shape_data`. The downstream `0`/`-1` resolution and `new_shape` computation runs unchanged because it depends on the runtime input shape.

The `.clone()` on each precomputed value is unavoidable because the downstream resolution code mutates `axes` / `shape_data`. The clone is cheap — these vectors are typically a handful of i64s.

### Key invariant

**Precomputed values are equivalent to D2H values.** When `try_precompute_*` returns `Some(v)`, `v` must be the same `Vec<i64>` that `get_or_fetch_i64` would return at runtime. This is trivially true because both read from `static_i64_cache` when the key is present, and the cache only holds values from initializers (constant by definition) and Constant nodes (constant by definition).

The `precompute_matches_runtime_d2h_bitwise` unit test in each new test file verifies this empirically by running two graphs end-to-end and comparing element-wise outputs.

## Error Handling

**No new error paths.** Precomputation helpers return `Option<Vec<i64>>` — a `None` falls through to the existing code path, which already handles every error case:

- **Cache miss** → `None` → existing `get_or_fetch_i64` path.
- **Empty `inputs[1]`** → helper returns `None` → existing attribute/error path.
- **Constant with non-Int64 value** → the add-node-time caching block matches on `Tensor::Int64` and skips otherwise; no error.

Existing error messages in the dispatch arms stay exactly as they are. They fire only when precomputation AND attribute lookup both fail, same as today.

## Testing Strategy

### Unit tests

Each new test file contains three TDD-driven tests. They're written red-then-green against the new helpers and fields:

1. **`precompute_hit_from_constant`** — register a Constant Int64 first, then the target op. Assert the `ExecutionNode`'s precomputed field is `Some(expected_vec)`.
2. **`precompute_miss_when_input_is_computed`** — register a non-Constant op (e.g., Shape) first, then the target op consuming Shape's output. Assert the precomputed field is `None`.
3. **`precompute_matches_runtime_d2h_bitwise`** — build two executors: one where precomputation naturally hits, one where the axes/shape input comes through a name deliberately absent from the static cache (simulating a miss). Run both with the same inputs and compare outputs element-wise. Must be bit-identical.

For test 3, the implementer avoids a `#[cfg(test)]` test-only accessor on `static_i64_cache` by constructing two executors with different graph structures that exercise the two dispatch paths while computing the same numerical function. This keeps the public API clean.

### Integration test

Extend `tests/kokoro_gpu_fusion_test.rs::fusion_fires_on_full_kokoro_at_seq_len_5` with a new assertion block **after** the existing top-10 printout:

```rust
// === Cycle 2: zero-cost shape-only ops ===
//
// After precomputing Unsqueeze/Reshape/Squeeze axes at graph-build
// time, these ops should no longer incur per-call D2H syncs from
// get_or_fetch_i64. GPU event time should drop close to zero; a
// hard threshold of 2 ms per op accommodates noise and any residual
// D2H from runtime-computed axes paths (acknowledged gap).
const MAX_GPU_US_PER_OP_AFTER_PRECOMPUTE: u64 = 2_000;

for op in ["Unsqueeze", "Reshape", "Squeeze"] {
    if let Some(&(gpu_us, count)) = profile.gpu_op_times.get(op) {
        assert!(
            gpu_us < MAX_GPU_US_PER_OP_AFTER_PRECOMPUTE,
            "{op} GPU time ({gpu_us} us across {count} calls) exceeds \
             Cycle 2's {MAX_GPU_US_PER_OP_AFTER_PRECOMPUTE} us threshold. \
             This usually means Kokoro has these nodes whose axes come \
             from runtime-computed chains (Shape → Cast → Gather, etc.) \
             and therefore can't be pre-computed at graph-build time. \
             This is a known limitation of Cycle 2's targeted fix; a \
             broader shape-inference pass (Approach C) would be needed \
             to eliminate it.",
        );
    }
}
```

Baseline (Cycle 1.5a measurement, Kokoro seq_len=5):
- Unsqueeze: 4.39 ms / 192 calls = ~23 us/call
- Reshape: 11.19 ms / 61 calls = ~183 us/call
- Squeeze: < 1 ms

Target: each < 2 ms. The 2 ms threshold is generous — a ~4.5–5.5× reduction on Unsqueeze and ~5.5× on Reshape. If precomputation works for most Kokoro instances, actual values should be well below 1 ms.

### What is NOT tested

- **Performance microbenchmarks in unit tests.** GPU event timing has too much variance at microsecond scales to assert timing deltas without flakes. Timing lives only in the Kokoro integration test where 192 calls average the noise.
- **`static_i64_cache` eviction.** No eviction logic changes in this cycle.
- **Runtime-computed axes chains.** No test is written that would pass in the "unfixed" state of that case.

## Risks and Mitigations

**Risk 1: `add_node` order dependency.** Precomputation at `add_node` time requires Constant nodes before their consumers. iconnx's `OnnxParser` processes nodes in file order, which must be topologically valid per the ONNX spec. Existing fusion detection and `precomputed_slice` already assume this. Phase A's first unit test explicitly verifies the Constant-before-consumer ordering.

**Risk 2: Future refactoring silently breaks precomputation.** A reviewer might remove the add-node-time Constant caching block as "duplicated" with the runtime caching. The Kokoro integration test's 2 ms threshold is the regression signal — it'll fail immediately with a clear diagnostic pointing at the mechanism.

**Risk 3: Kokoro's shape inputs come mostly from runtime-computed chains.** If precomputation rarely hits on Kokoro, the threshold assertion fails and captures the real savings distribution. The failure is the diagnostic; the follow-up would escalate to Approach C.

**Risk 4: Signature clutter from four `Option<&...>` parameters.** Leadline flagged this in Section 2. Mitigation is Option B (`NodePrecomputed<'_>` struct) — implementer's call, no behavior impact.

**Risk 5: Double-cache of Constant values** (once at add_node, once at runtime dispatch). Correct but wastes one HashMap insert per Constant per run. Tidy via an early-return at line 4230 is optional, not required.

## Validation Gates

Cycle 2 is complete when all of these are true:

1. `cargo test --features cuda` — all existing tests plus 9 new unit tests (3 files × 3 tests) pass.
2. `cargo test` (no-cuda) — pre-existing `validation_test.rs` failure unchanged; no new failures.
3. `cargo clippy --features cuda -- -D warnings` — clean.
4. `cargo fmt --check` on all new and modified files — clean.
5. Existing fusion detection tests (5 in `fusion_detection_test.rs`) — all pass.
6. Existing GpuEventTimer tests (4 in `gpu_event_timer_test.rs`) — all pass.
7. `tests/kokoro_gpu_fusion_test.rs::fusion_fires_on_full_kokoro_at_seq_len_5` (run with `--ignored --nocapture`) — passes, with the new assertion showing each of Unsqueeze/Reshape/Squeeze under 2 ms GPU time.
8. `src/cuda/inference/mod.rs` grows by ≤ 150 lines net from the start of this cycle.
9. All commits signed per CLAUDE.md. If signing fails, the implementer stops and asks the user.
10. The top-10 launch-overhead table from the post-fix Kokoro run is captured in the final commit message or results doc so Leadline can update analysis.

## Out of scope (for future cycles)

- **Other shape-only ops**: `Shape`, `Expand`, `Gather`-on-constants, `ConstantOfShape`. Same D2H pattern, deferred to per-op or batched future cycles.
- **Runtime-computed axes chains**: `Shape → Cast → Gather → Unsqueeze.axes`. Requires symbolic shape inference (Approach C). Separate large cycle.
- **cuDNN LayerNorm replacement**: Leadline's #3 priority from the Cycle 1.5a results doc. Separate cycle.
- **AddMulAdd detection gap fix**: Cycle 1's documented follow-up. Separate cycle.
- **Approach C — full graph optimizer pass**: covers all of the above in one framework. Largest scope; most transformative; separate major project.
