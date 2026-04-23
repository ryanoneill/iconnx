# Phase 4 Graph Optimizer II: shape-extraction folding + memory planner

**Status:** amended 2026-04-22 after real-graph instrumentation revealed Phase 4 Commit 2's original predicate was too narrow. Original spec finalized 2026-04-19; see §Amendment history for the change record.
**Scope label:** Phase 4 (per the Phase 3 outcomes decomposition).
**Landing:** one PR with three signed commits on branch `ir-phase4` (refactor, B, A — no internal squashing across the three).
**Baseline:** Kokoro ratio median 3.954 (iconnx 110.6 ms / ORT 28.6 ms) on current machine + driver state. Phase 3's stacked-gate target was retired; Phase 4 sets its own frames per below.

## Motivation

Phase 3 landed four feat(ir) commits but left two threads open:

1. **The stacked-gate verdict for Phase 3 Commits #2 and #3 was "unresolved."** Direct Kokoro benefit was sub-noise because **51 of 61 Kokoro Reshape nodes** have runtime-computed shape inputs via `Shape → Cast → Gather` chains that Phase 3's precompute pass could not resolve. Phase 3 explicitly declared: "those Reshapes are the explicit Phase 4 unlock target." Phase 4 delivers the unlock.

2. **Phase 2's spec declared a Phase 4 memory planner.** `ExecutionPlan::tensor_last_use` was added in Phase 2 with the note "Phase 4 will extend it with static lifetime analysis and a pre-allocated arena." Phase 3's `tensor_shapes` field provided the size information the planner needs. Phase 4 completes the plumbing by consuming both fields to produce a static memory plan.

B (shape-aware folding) and A (memory planner) are coupled but separately-landable: B enables A to classify more tensors as arena-eligible; A is useful on its own even without B's unlocks, because it improves dispatch-time bookkeeping (no per-op HashMap lookup). Landing B first cleans up the shape-resolution story; A then converts the known shapes into an allocation plan.

## Scope

Three signed commits on branch `ir-phase4`:

| # | Commit | Files | Gate |
|---|---|---|---|
| 1 | `refactor(ir): extract fold loop into folding_core` | new `src/ir/passes/folding_core.rs`; `src/ir/passes/constant_folding.rs` (becomes a thin caller); `src/ir/passes/mod.rs` (re-exports) | Pure refactor. All existing `constant_folding` tests still pass. Zero behavioral change; zero new warnings. |
| 2 | `feat(ir): shape_extraction_folding pass` (B) | new `src/ir/passes/shape_extraction_folding.rs`; `src/ir/lowering.rs` (pipeline wiring); `src/ir/passes/mod.rs` (re-export); `src/operators/gather.rs` rank-0-indices correctness fix; `tests/kokoro_gpu_fusion_test.rs` loader update to pass concrete input shapes via `add_input` | **Deterministic gate (reframed 2026-04-22 second refinement):** `fold_counts.total == 124` when Kokoro test loader passes concrete input shapes via `add_input("tokens", vec![Some(1), Some(5)])`. Mechanism validated under concrete-shape calling contract; symbolic-dim-input unlock deferred to Phase 5+. See §Gating protocol B. Three-attempt counter stays at 1/3 — this is a reframe, not a new attempt. |
| 3 | `feat(ir): static memory planner + arena allocation path` (A) | new `src/ir/passes/memory_planner.rs`; `src/ir/plan.rs` (adds `memory_plan` field + `arena_offset` on `PlannedOp`); `src/ir/lowering.rs` (pipeline wiring); `src/cuda/executor/*` (arena allocation at run start, per-op branching); `src/ir/passes/mod.rs` (re-export) | **Generic-infrastructure frame.** Acceptance by correctness + classification coverage + cross-model follow-up. See §Gating protocol. |

**Out of scope (tracked separately):**
- `FusedPattern::GeneralChain` scalar-constants codegen extension (tracked from C PR).
- `src/cuda/kernels/mod.rs` refactor into per-category sub-modules (tracked from C PR — 1151 lines, pre-existing violation).
- Legacy `detect_fusion` deletion (candidate once Phase 4 memory planner eliminates any remaining reliance on its dynamic_candidates path).
- `GpuGraphExecutor` strangler shim deletion (when nothing external reaches through).
- Whisper benchmark integration (parallel leadline effort, not iconnx code).
- Cross-inference arena caching (see §A — per-inference allocation is Phase 4's choice; cross-inference caching is a clean follow-up if profiling shows allocation overhead).

## Gating protocol

Two frames in Phase 4, each deliberate:

### B — structural gate (not ratio-based)

Direct Kokoro ratio benefit for B is estimated well below the 0.05 ratio floor. Gating on a ratio measurement we already predict will fail is ceremonial, not a check. Phase 3's stacked-gate commitment was "Phase 4 resolves whether the Shape-rooted chain unlock mechanism works on Kokoro" — a structural question, noise-independent.

**Denominator revised.** Real-graph instrumentation (2026-04-22) measured Kokoro's 73 Shape nodes. Every downstream tensor inherits `None` on at least one dim via `seq_len` propagation from the `tokens` graph input, so 0 Shape nodes have fully-concrete input shapes — the original all-or-nothing predicate would never fire. The actual structural ceiling:

- **53 Shape ops** feed from a tensor with at least one concrete dim AND at least one `None` dim, and are consumed by extraction ops (`Gather` / `Slice` / optionally chained through `Cast`) that extract specific indices from the Shape vector. A predicate that folds *chains terminating at extraction ops whose extracted indices hit only concrete dims* can reach these.
- **20 Shape ops** feed from a tensor whose shape inferencer returned the empty-`Vec` "fully unknown" sentinel. Not reachable by this mechanism — would require shape-inference improvements upstream.

The 53-case denominator is the revised target. The 20 empty-sentinel cases are explicit non-goals for Phase 4 B.

**Structural gate (reframed 2026-04-22 second refinement — deterministic primary, no proxy secondary).**

Attempt #1 measurement (2026-04-22, implementer escalation) found that 0 of Kokoro's 124 extraction chains fire under ONNX-symbolic calling contract (loader passes `input_shapes()` with `seq_len` as `None`). The 53-case denominator from the first 2026-04-22 amendment was counted on "tensors with some concrete dim" without filtering for "and the extraction targets the concrete dim" — an instrumentation error. Empirically, 100% of Kokoro's Shape→Gather/Slice extractions target `batch` or `seq_len` (always dynamic) and 0% target `hidden_dim` (always static). The mechanism is architecturally inert on symbolic-dim inputs.

**What the pass is validated for (the gate closes).** When the caller passes **concrete dims** via `GpuGraphExecutor::add_input`, the mechanism fires on every extraction chain whose source has those concrete dims. Kokoro with `add_input("tokens", vec![Some(1), Some(5)])` — matching the concrete seq_len the test already uses for `kokoro_inputs(5)` — unlocks all 124 extraction chains. The gate is therefore deterministic, not statistical: either the mechanism fires on 124/124 under concrete-shape inputs or it doesn't.

**What the gate does NOT close.** Phase 3's stacked-gate verdict as originally phrased lumped together "mechanism works" and "mechanism unlocks Kokoro as-loaded-with-ONNX-symbolic-dims." Option A (this reframe) closes the first; it explicitly does NOT close the second. Unlocking on symbolic-dim inputs requires either symbolic-dim tracking (`Dyn("seq_len")` type expansion) or runtime specialization (plan recompile on concrete-shape discovery). Both are Phase 5+ work; see §Non-goals.

**Deterministic primary gate.** Measured via `testing::shape_extraction_folding_counts().total` after `kokoro_gpu_fusion_test::fusion_fires_on_full_kokoro_at_seq_len_5` runs its one `lower()` call. The test loader is updated to call `executor.add_input("tokens", vec![Some(1), Some(5)])` (matching the concrete seq_len of `kokoro_inputs(5)`); `add_input("style", ...)` and `add_input("speed", ...)` similarly pass their fully-concrete shapes. Gate: `fold_counts.total == 124` (exact match on the 124 extraction chains in Kokoro's graph). Deviations in either direction are failures:

- `total < 124`: the mechanism missed supported chains — bug in one of the three pattern helpers.
- `total > 124`: the mechanism over-matched — investigate whether Kokoro's graph grew new extraction chains or the pattern predicates accept chains they shouldn't.

Exact-match is safer than "≥ 124" because it surfaces BOTH regressions (mechanism stops firing on some chains) AND silent over-matching (a future iconnx op with Shape→Gather-like signature accidentally accepted).

**Per-pattern breakdown available via `ShapeExtractionFoldingCounts { shape_gather, shape_cast_gather, shape_slice, total }`** for diagnostic purposes (printed in the test's eprintln for easy inspection). Expected breakdown on Kokoro (from attempt #1's instrumentation): ~110 Gather, ~0 Cast→Gather, ~14 Slice. If the per-pattern breakdown materially shifts between runs (e.g., Gather count drops by 20), investigate via the breakdown even if `total == 124` — suggests one pattern broke while another over-matches to compensate.

**No secondary proxy gate.** The earlier 2026-04-22 amendment proposed `reshape_hits ≥ 48` as a sanity check, but that number assumed a fold-projection (38 unlocks) derived from the incorrect 53-case denominator. Under the deterministic reframe, `reshape_hits` is an informational measurement only — captured in the test's eprintln but not asserted. The primary gate is `fold_counts.total`.

**Three-attempt abandonment clause (unchanged, counter at 1/3).** If three pattern-implementation attempts fail to land the deterministic 124/124 gate, the Phase-3-style "try 3 times then abandon" discipline applies.

**This reframe is NOT a new attempt.** The first attempt's mechanism is correct (11 unit tests green, all three patterns work, rejection logic sound). The measurement premise was wrong — we were testing under the wrong calling contract. Running the same correct mechanism under the correct calling contract (concrete `add_input` shapes matching the test's actual inference inputs) is validation of attempt #1, not a new attempt. **Counter stays at 1/3.**

**What counts as an "attempt" (unchanged):** an attempt is a fresh predicate design OR a distinct new pattern variant added to the pass. Tweaking parameters on an in-progress pattern (including fixing a misaligned test-setup calling contract) is part of the SAME attempt.

**Mandatory correctness keeper (lands regardless of B's path).** During real-graph instrumentation the original `shape_aware_folding` predicate surfaced a latent bug: folding Shape on the empty-Vec unknown sentinel synthesized an `Int64[0]` tensor, which panicked downstream `Gather::forward` at `src/operators/gather.rs:144` (`index_axis(Axis(0), idx)` on a zero-length axis). The bugfix adds a guard + regression test and is orthogonal to B's predicate design. Commit SHA `8350f67` on branch `ir-phase4`. If B proceeds: included in B's squash. If B is abandoned: cherry-picked onto main as its own `fix(ops): gather rank-0-indices guard` commit.

### A — generic-infrastructure frame

Direct Kokoro ratio benefit for A is estimated at ~0.25 ms (per-op HashMap-lookup elimination) — sub-noise on this hardware. A's value is model-general: memory planners scale with graph size, and larger models (Whisper, Stable Diffusion, etc.) benefit more than Kokoro does.

**Landing criteria — all six must hold:**

1. **Correctness on Kokoro.** All Kokoro integration tests green with the planner active. Specifically `kokoro_gpu_fusion_test::fusion_fires_on_full_kokoro_at_seq_len_5` passes with no numerical regression vs pre-Phase-4 output (tolerance 1e-5).
2. **Pool fallback works.** All Kokoro tests green with `ICONNX_DISABLE_MEMORY_PLANNER=1` (runtime env var forces every tensor through the pool path regardless of `arena_offset`).
3. **Determinism.** Planner produces byte-identical `MemoryPlan` output on back-to-back `lower()` calls with the same inputs. Asserted in a snapshot test.
4. **Classification coverage.** On Kokoro, at least 70% of arena-eligible tensors are actually assigned to arena slots. Exposed via `MemoryPlan::classification_counts() → PlannerClassification { arena_eligible, arena_assigned, pool_classified }`. Predicate: `arena_assigned / arena_eligible ≥ 0.7`. Catches the "predicate has a silent bug that sends everything to pool" failure mode that binary pass/fail on correctness would not detect.
5. **Size assertion active.** The per-op runtime check (`assert_eq!(actual_tensor_bytes, arena_slot_bytes)`) is live in plain `--release` builds, not gated behind `debug-inference`. Single integer comparison per op; nanoseconds; trading nanoseconds for always-on corruption protection is principled (same class of bug as Phase 3's Gather undersizing would cause silent writes past slot boundaries). If post-landing measurement shows the check costs more than ~0.5 ms across the 2464-op Kokoro graph, re-scope to `debug-inference`; until then, always-on.
6. **Zero warnings, signed commit.** Standard iconnx hygiene.

**Follow-up acceptance criterion (not in-PR):** Whisper-Tiny executes through the planner with zero crashes and matches ORT numerically once leadline's ModelProfile refactor lands and provides a Whisper harness. Documented as a post-Phase-4 validation, not a gate.

### Combined Phase 4 measurement

Post-Phase-4 Kokoro ratio re-measurement is **informational, not a gate**. Recorded as the new Phase-4 baseline for any future work. Under the same stable-hardware protocol as Phase 3.

## Per-pass design

### Commit #1: `folding_core` extraction (refactor)

New module `src/ir/passes/folding_core.rs`:

```rust
/// Shared fold-loop primitive. Both `constant_folding` and
/// `shape_aware_folding` invoke it with their specific foldability
/// predicate.
pub fn fold_with_predicate(
    graph: OptimizableGraph,
    extra_foldable: impl Fn(&GraphNode, &HashMap<String, Tensor>) -> Option<Tensor>,
) -> OptimizableGraph;
```

**Contract.** Given a graph and an extension predicate, iteratively fold every node whose inputs are all in `graph.initializers` (or are in the intra-pass `folded` map) via `dispatch_cpu_forward`, OR whose extension predicate returns `Some(tensor)`. Continue until no more foldable nodes remain (fixpoint). Produce a new graph with folded outputs promoted to initializers and folded nodes removed.

`constant_folding` becomes:

```rust
pub fn constant_folding(graph: OptimizableGraph) -> OptimizableGraph {
    folding_core::fold_with_predicate(graph, |_node, _folded| None)
}
```

**No behavioral change.** Every existing `constant_folding` test passes unchanged.

### Commit #2: `shape_extraction_folding` (B)

```rust
// src/ir/passes/shape_extraction_folding.rs
pub fn shape_extraction_folding(
    graph: OptimizableGraph,
    tensor_shapes: &HashMap<String, Vec<Option<usize>>>,
) -> OptimizableGraph;
```

**Design rationale.** Real-graph instrumentation measured 0 Kokoro Shape nodes with fully-concrete input shapes (every downstream tensor inherits `seq_len = None` from the `tokens` graph input), but 53 Shape nodes with at least one concrete dim AND at least one `None` dim consumed by an extraction op that asks for specific indices. The predicate therefore targets *Shape-rooted chains terminating at an extraction op whose extracted indices are all concrete dims of the Shape input*, not the narrower "Shape when all-inputs-known" originally projected. See §Amendment history for the measurement that drove the rename + redesign.

**Supported extraction patterns (exactly these; anything else is scope creep per spec §Amendment history):**

1. **`Shape(x) → Gather(indices=I)`** where `I` is a literal initializer (`Int64[k]` tensor of `k` indices). Fires iff every index in `I` is a non-negative integer less than `rank(x)` AND `tensor_shapes[x][I[j]]` is `Some(_)` for every `j`.
2. **`Shape(x) → Cast(to=Int64) → Gather(indices=I)`** — identical eligibility to pattern 1. Cast is transparent; folds through to produce the same synthesized initializer type.
3. **`Shape(x) → Slice(start=s, end=e[, axis=0])`** where `s` and `e` are concrete indices into the Shape vector (and axis is 0 or absent — Shape's output is rank-1). Fires iff every position `[s, e)` has `tensor_shapes[x][pos]` as `Some(_)`.

For each firing pattern, the entire chain collapses to a single synthesized initializer carrying the extracted Int64 values. Intermediate op nodes are dropped; the extraction op's consumer now sees a static initializer. DCE runs after to clean orphans.

**Algorithm sketch.**

The predicate is structural: it needs to see the extraction op PLUS its Shape-rooted chain. `folding_core::fold_with_predicate`'s per-node closure receives only one node at a time, so the predicate for `shape_extraction_folding` inspects the consumer ops (not the Shape op itself):

```rust
pub fn shape_extraction_folding(
    graph: OptimizableGraph,
    tensor_shapes: &HashMap<String, Vec<Option<usize>>>,
) -> OptimizableGraph {
    // Pre-compute a producer-name → GraphNode index so the predicate
    // can walk back from an extraction op to its Shape source without
    // re-traversing the node list on every call.
    let producers = build_producer_map(&graph);

    folding_core::fold_with_predicate(graph, |node, folded| {
        match node.op_type.as_str() {
            "Gather" => try_fold_shape_gather(node, &graph, &producers, tensor_shapes),
            "Slice"  => try_fold_shape_slice(node, &graph, &producers, tensor_shapes),
            _ => None,
        }
    })
}

// try_fold_shape_gather: walks node.inputs[0]'s producer chain,
// skipping a single Cast(to=Int64) if present, terminating at Shape(x);
// reads node.inputs[1] as the indices initializer; checks every index
// j against tensor_shapes[x][j].is_some(); on success synthesizes the
// extracted Int64 tensor.
```

**Implementation notes.**

- The predicate must handle intermediate Cast ops transparently: a `Shape → Cast(Int64) → Gather` chain should fold exactly like `Shape → Gather`, because `Shape`'s native output is already Int64 and a Cast to Int64 is a no-op preserving values.
- The predicate does NOT fold Shape standalone. The `8350f67` bugfix (which added a guard against folding Shape on the unknown-sentinel case) stays — under the revised design, Shape is never folded by this pass; only the extraction-consuming chain is.
- The `folded` param in the fold-loop closure is still unused by this pass (it inherits from the fold-loop contract but the predicate consults `graph` + `tensor_shapes` + `producers` only). This is fine per `folding_core`'s documented contract.
- Per-pattern attribution counters (internal instrumentation, not public API) track how many times each pattern fires. Exposed via a `testing::shape_extraction_folding_counts()` helper for the Kokoro gate diagnostic. Used to distinguish "pattern X didn't fire" from "pattern X fired but wrongly" during Amber-tier investigation.

**Pipeline position.** Unchanged from the original design — inserted into `lower()` immediately after `shape_inference` + `tensor_shapes_from_graph`:

```rust
let graph = topo_sort(graph)?;
let graph = constant_folding(graph);
let graph = dead_code_elimination(graph);
let graph = shape_inference(graph);
let tensor_shapes = tensor_shapes_from_graph(&graph);
let graph = shape_extraction_folding(graph, &tensor_shapes);  // RENAMED
let graph = dead_code_elimination(graph);
let precomputed = precompute_params(&graph, &tensor_shapes);
let fusion_primary = elementwise_fusion_pass(&graph, &tensor_shapes);
// ... rest unchanged
```

**Tests (in `src/ir/passes/shape_extraction_folding.rs::tests`):**

Unit tests cover each supported pattern's success case, each pattern's rejection case, and the cross-pattern invariants:

- `gather_of_shape_with_all_concrete_indices_folds`: `Shape(x) → Gather([0, 2])` where `tensor_shapes[x] = [Some(1), None, Some(512)]`. Extracted indices 0 and 2 are both concrete. Post-fold: `initializers["out"] == [1, 512]`, chain removed.
- `gather_of_shape_with_none_at_extracted_index_does_not_fold`: same shape; `Gather([1])` where index 1 is `None`. Chain preserved.
- `cast_in_between_shape_and_gather_is_transparent`: `Shape → Cast(Int64) → Gather([0])`; Cast is optional no-op. Post-fold: chain removed, single initializer synthesized.
- `slice_of_shape_with_all_concrete_range_folds`: `Shape(x) → Slice(start=0, end=2)` where dims 0 and 1 are both `Some`. Post-fold: collapses to initializer `[x_shape[0], x_shape[1]]`.
- `slice_of_shape_with_none_in_range_does_not_fold`: similar but range hits a `None` dim. Chain preserved.
- `standalone_shape_without_extraction_consumer_does_not_fold`: a Shape op whose output feeds no supported extraction pattern stays. Explicit test that the revised design NEVER folds Shape alone.
- `gather_on_non_shape_input_does_not_fold`: Gather whose input[0] isn't a Shape output (e.g., an initializer Int64 tensor) — must be left untouched because the fold rationale only applies to Shape-rooted chains. Guard against over-matching.
- `out_of_bounds_gather_index_does_not_fold`: indices that exceed `rank(x)` must not panic and must not fold. Guard against undefined behavior on malformed inputs.
- `shape_extraction_folding_is_idempotent`: two consecutive calls produce identical graphs.

Plus the existing `8350f67` regression test: `gather_on_empty_int64_initializer_does_not_panic` (added as part of the surfaced bugfix; stays regardless).

**Integration test updates in `tests/kokoro_gpu_fusion_test.rs`:**

Two changes required under the 2026-04-22 second refinement.

**(1) Loader passes concrete input shapes via `add_input`.** The test's existing loader helper uses `model.input_shapes()` which preserves ONNX symbolic `dim_param` as `None`. The reframed mechanism fires only when the caller supplies concrete dims, so the test loader replaces the `input_shapes()` loop with explicit concrete shapes matching its inference inputs:

```rust
// Before (ONNX-symbolic calling contract — mechanism inert on all 124 chains):
//   for (name, shape) in model.input_shapes() {
//       executor.add_input(name, shape);
//   }
//
// After (concrete calling contract — mechanism fires on all 124 chains):
executor.add_input("tokens".to_string(), vec![Some(1), Some(5)]);   // seq_len=5 matches kokoro_inputs(5)
executor.add_input("style".to_string(), vec![Some(1), Some(256)]);
executor.add_input("speed".to_string(), vec![Some(1)]);
```

The concrete values come from the test's own `kokoro_inputs(5)` tensor shapes — no guessing, just surfacing what the test already knows at `run()` time. No iconnx API change; `GpuGraphExecutor::add_input(name: String, shape: Vec<Option<usize>>)` already accepts concrete dims (Phase 3 C PR landed this).

**(2) Deterministic primary gate.** Replace any `reshape_hits`-based assertion with exact-match on `fold_counts.total`:

```rust
// Reset counters before the lower() call that triggers the pass.
iconnx::cuda::inference::testing::reset_shape_extraction_folding_counters();

// ... setup_kokoro_gpu_executor + kokoro_inputs(5) + run_with_profiling ...

// ---- Deterministic primary gate: fold_counts.total == 124 ----
let fold_counts = iconnx::cuda::inference::testing::shape_extraction_folding_counts();
eprintln!(
    "shape_extraction_folding counts — Gather: {}, Cast→Gather: {}, Slice: {}, total: {}",
    fold_counts.shape_gather,
    fold_counts.shape_cast_gather,
    fold_counts.shape_slice,
    fold_counts.total,
);
assert_eq!(
    fold_counts.total, 124,
    "Phase 4 B deterministic gate failed: {} extraction chains folded \
     (expected exactly 124 under concrete-shape calling contract). \
     Per-pattern breakdown: Gather={}, Cast→Gather={}, Slice={}. \
     Deviation < 124 → mechanism missed supported chains; deviation > 124 \
     → mechanism over-matched. See spec §Gating protocol B.",
    fold_counts.total,
    fold_counts.shape_gather,
    fold_counts.shape_cast_gather,
    fold_counts.shape_slice,
);

// Informational only — reshape_hits captures a subset of unlocks that
// terminate at Reshape.shape inputs. Not a gate signal under the
// 2026-04-22 second refinement; primary gate is fold_counts.total.
eprintln!(
    "Informational: reshape_hits = {}/{} (pre-Phase-4 baseline 10/61; \
     not asserted — see spec §Gating protocol B).",
    precompute_stats.reshape_hits,
    precompute_stats.reshape_total,
);
```

Exact-match (`assert_eq!(..., 124)`) surfaces BOTH regression (mechanism stops firing on some chains) AND silent over-matching (future iconnx op with Shape→Gather-like signature accidentally accepted) as failures. The per-pattern breakdown in the panic message lets a diagnostic run identify which pattern drifted.

### Commit #3: `memory_planner` (A)

**Input data.** `ExecutionPlan` fields populated by earlier passes: `tensor_shapes: HashMap<String, Vec<Option<usize>>>` and `tensor_last_use: HashMap<String, usize>` (last op-index at which each tensor is consumed). Plus `ops: Vec<PlannedOp>` providing the topological ordering and per-op first-use (op-index that produces each output).

**Eligibility predicate (semantic — single function).**

```rust
// src/ir/passes/memory_planner.rs
pub(crate) fn is_planner_eligible(
    tensor_name: &str,
    tensor_shapes: &HashMap<String, Vec<Option<usize>>>,
    graph_inputs: &[String],
    graph_outputs: &[String],
    dtype_bytes: usize,
    element_count: usize,
) -> bool {
    // Not a graph input (inputs are written by the input-transfer
    // path, not by any op's output — arena-backing would have no
    // clean reconciliation with the input buffer at run() time).
    if graph_inputs.iter().any(|i| i == tensor_name) { return false; }

    // Not a graph output (runtime may need to hand ownership back
    // to the caller past the arena's lifetime).
    if graph_outputs.iter().any(|o| o == tensor_name) { return false; }

    // All dims fully known.
    let Some(shape) = tensor_shapes.get(tensor_name) else { return false };
    if shape.iter().any(|d| d.is_none()) { return false; }

    // Size >= 4 KiB (matches pool's size-class threshold).
    if dtype_bytes * element_count < 4096 { return false; }

    true
}
```

**Structural filter (separate from the semantic predicate).** Tensors produced by `PlannedOp::kind == Skip` ops (the targets of fusion head-writes) are structurally ineligible — the fused kernel writes directly into the head's slot, so the skipped op's output is never allocated at runtime. The planner's enumeration applies this filter alongside the semantic predicate:

```rust
let eligible_tensors: Vec<_> = all_tensors
    .into_iter()
    .filter(|t| is_planner_eligible(
        &t.name, &tensor_shapes, &graph_inputs, &graph_outputs,
        t.dtype_bytes, t.element_count,
    ))
    .filter(|t| !is_fusion_skipped(&t.producer_op_idx, &fusion_annotations))
    .collect();
```

Two concerns (semantic eligibility, structural dead-slot avoidance), two filters, no signature bloat on the predicate. `is_fusion_skipped` is a short internal helper in `memory_planner.rs` that inspects the fusion annotations for each tensor's producer op.

**Allocation algorithm.** Greedy linear scan with size-class slots. Deterministic by construction:

```
1. Collect eligible tensors with (name, dtype, size_class_bytes, first_use_op_idx, last_use_op_idx).
2. Sort by (first_use_op_idx ASC, then tensor_name ASC for deterministic tiebreak).
3. Maintain a Vec<Slot> (existing slots), each with {size_class_bytes, dtype, current_tenant_last_use}.
4. For each tensor in the sorted order:
   a. Find the first existing slot where dtype matches,
      size_class_bytes matches (power-of-2 rounding ≥ 4096 per pool convention;
      exact match for < 4096),
      and current_tenant_last_use < tensor's first_use_op_idx.
   b. If found: reassign the slot to this tensor.
   c. Otherwise: allocate a new slot, append to Vec<Slot>.
5. Compute arena_bytes = sum(slot.size_class_bytes for slot in slots).
6. Emit MemoryPlan { arena_bytes, slot_assignments: HashMap<tensor_name, slot_idx> }
   plus per-PlannedOp stamps of arena_offset.
```

**Determinism contract.** Given the same `(graph, tensor_shapes, tensor_last_use, graph_outputs)`, the planner produces byte-identical `MemoryPlan`. This is not an accident; it falls out of stable sort and deterministic slot search. Required so that: (a) snapshot tests of planner output work; (b) baselines captured by leadline-bench are reproducible across runs; (c) bisecting planner-vs-runtime bugs is tractable — same plan means any behavioral difference is a runtime bug, not a planner bug.

**`ExecutionPlan` additions:**

```rust
// src/ir/plan.rs
pub struct PlannedOp {
    // ... existing fields ...
    /// Arena slot offset (bytes) for this op's primary output tensor.
    /// `Some(offset)` → arena-backed allocation at `arena_base + offset`.
    /// `None` → fall back to the GpuMemoryPool's runtime get_tensor_* path.
    pub arena_offset: Option<usize>,
}

pub struct ExecutionPlan {
    // ... existing fields ...
    pub memory_plan: MemoryPlan,
}

pub struct MemoryPlan {
    pub arena_bytes: usize,
    pub slot_count: usize,
    slot_assignments: HashMap<String, usize>,  // tensor_name -> slot_idx
    slot_offsets_bytes: Vec<usize>,            // slot_idx -> byte offset in arena
    /// Classification counts computed at plan construction time so the
    /// accessor is parameter-less.
    classification: PlannerClassification,
}

impl MemoryPlan {
    pub fn arena_offset(&self, tensor_name: &str) -> Option<usize> {
        let &slot_idx = self.slot_assignments.get(tensor_name)?;
        Some(self.slot_offsets_bytes[slot_idx])
    }

    pub fn classification_counts(&self) -> &PlannerClassification {
        &self.classification
    }
}

pub struct PlannerClassification {
    /// Tensors passing `is_planner_eligible` AND the fusion-skipped filter.
    pub arena_eligible: usize,
    /// Subset of `arena_eligible` actually assigned a slot. Equal to
    /// `arena_eligible` unless the slot-assignment algorithm declines
    /// an eligible tensor (shouldn't happen under the current greedy
    /// algorithm; guarded by assertion in the planner).
    pub arena_assigned: usize,
    /// Tensors that did NOT pass `is_planner_eligible` or were filtered
    /// by the fusion-skipped filter. Take the pool path at runtime.
    pub pool_classified: usize,
}
```

**Pool surface additions.** The runtime arena allocation requires one new method on `GpuMemoryPool`:

```rust
// src/cuda/memory_pool.rs
impl GpuMemoryPool {
    /// Allocate a single contiguous byte buffer on the device for the
    /// memory planner's arena. Returns an RAII handle; dropping it
    /// returns the allocation to the pool as a standard pooled slice
    /// (bucketed by size-class). Phase 4 allocates one arena per
    /// `run()`; cross-inference caching is a separate follow-up.
    pub fn allocate_arena(&mut self, bytes: usize) -> Result<ArenaBuffer, CudaError>;
}

pub struct ArenaBuffer {
    // Holds a DeviceSlice<'_, u8> whose lifetime ties to the
    // ArenaBuffer itself (not 'static — the buffer lives exactly as
    // long as the surrounding run(), and dropping it returns the
    // allocation to the pool). Concrete lifetime form (explicit
    // lifetime param vs. Arc<DeviceSlice<'static>> vs. other) is an
    // implementation detail to be settled in the plan against the
    // actual garboard DeviceSlice surface; spec requires only the
    // RAII + slice_at(offset, bytes_typed) semantics.
}
```

No changes to pool's existing get/return API; `allocate_arena` is purely additive.

**Runtime integration:**

```rust
// src/cuda/executor/mod.rs (sketch)
pub fn run(&mut self, plan: &ExecutionPlan, inputs: HashMap<String, Tensor>) -> Result<...> {
    let arena = if disable_memory_planner() {
        None  // env var ICONNX_DISABLE_MEMORY_PLANNER=1 checked once per run
    } else {
        Some(self.pool.allocate_arena(plan.memory_plan.arena_bytes)?)
    };

    for op in plan.ops.iter() {
        let output_tensor = match (op.arena_offset, &arena) {
            (Some(offset), Some(arena_buf)) => {
                let slot = arena_buf.slice_at(offset, slot_bytes_for(op));
                // Always-on size assertion (nanoseconds, catches silent corruption):
                assert_eq!(
                    actual_output_bytes(op, plan), slot_bytes_for(op),
                    "memory_planner slot size mismatch for tensor {:?}: \
                     actual={} slot={}",
                    op.output_name, actual_output_bytes(op, plan), slot_bytes_for(op),
                );
                GpuTensor::from_slice(slot)
            }
            _ => self.pool.get_tensor_f32(...)?,  // pool fallback
        };
        // ... dispatch op with output_tensor ...
    }
    // At run end, arena is returned to pool (or freed) via drop.
}

fn disable_memory_planner() -> bool {
    std::env::var("ICONNX_DISABLE_MEMORY_PLANNER")
        .map(|v| v == "1")
        .unwrap_or(false)
}
```

**Three refinements baked in:**

1. **No runtime HashMap lookup.** `arena_offset: Option<usize>` stamped onto each `PlannedOp` at plan time. Runtime branches on `Option` — one cache line, no hash. The `MemoryPlan::arena_offset(tensor_name)` accessor exists for introspection/tests, not for runtime hot path.
2. **Always-on size assertion.** `assert_eq!(actual_tensor_bytes, arena_slot_bytes)` at op-execute time, live in plain release builds. Cost: one integer comparison per op; nanoseconds. Catches the class of silent corruption that Phase 3's Gather undersizing bug would have caused if a memory planner had been present. If post-landing profiling shows this costs > 0.5 ms across Kokoro's 2464-op graph, demote to `debug-inference`-gated; until then, always-on.
3. **Bisection escape hatch.** `ICONNX_DISABLE_MEMORY_PLANNER=1` env var forces every tensor through the pool path. Checked once per `run()` at the top. Enables bisecting planner bugs vs other regressions without a custom build.

**Pipeline position.** Inserted into `lower()` immediately before plan finalization:

```rust
// src/ir/lowering.rs (Phase 4 final)
let graph = topo_sort(graph)?;
let graph = constant_folding(graph);
let graph = dead_code_elimination(graph);
let graph = shape_inference(graph);
let tensor_shapes = tensor_shapes_from_graph(&graph);
let graph = shape_aware_folding(graph, &tensor_shapes);  // Phase 4 B
let graph = dead_code_elimination(graph);
let precomputed = precompute_params(&graph, &tensor_shapes);
let fusion = elementwise_fusion_pass(&graph, &tensor_shapes);
let legacy_fusion = detect_fusion(&graph);
let fusion = merge_fusion_annotations(fusion, legacy_fusion);
let tensor_last_use = compute_tensor_last_use(&graph);
let memory_plan = memory_planner(&graph, &tensor_shapes, &tensor_last_use);  // Phase 4 A
let planned_ops = build_planned_ops(&graph, &precomputed, &fusion, &memory_plan);
// ... finalize ExecutionPlan ...
```

**Tests (in `src/ir/passes/memory_planner.rs::tests`):**

- `is_planner_eligible_all_dims_known_yields_true`
- `is_planner_eligible_any_none_dim_yields_false`
- `is_planner_eligible_graph_input_yields_false`
- `is_planner_eligible_graph_output_yields_false`
- `is_planner_eligible_below_4kb_yields_false`
- `fusion_skipped_tensor_filtered_out_even_when_otherwise_eligible`: tensor whose producer op has `PlannedOp::kind == Skip`. Assert it is NOT in `arena_eligible` count.
- `slot_assignment_reuses_slot_for_disjoint_lifetimes`: two eligible f32 tensors of same size where `last_use(a) < first_use(b)`. Expect single slot.
- `slot_assignment_creates_new_slot_for_overlapping_lifetimes`: two eligible tensors with overlapping lifetimes. Expect two slots.
- `slot_assignment_preserves_size_class_separation`: one f32 (8192 bytes) and one i64 (8192 bytes) with disjoint lifetimes. Expect two slots (different dtypes can't share).
- `determinism_snapshot`: run `memory_planner` twice on the same input; assert byte-identical `MemoryPlan` output (via `Debug`-format comparison or a serialization roundtrip).
- `classification_counts_reports_eligibility_and_assignment`: build a synthetic graph with known counts of eligible / ineligible / assigned tensors; assert `classification_counts()` matches.

**Integration tests:**

- `kokoro_gpu_fusion_test` must pass with planner active (arena path in use). No numerical regression vs pre-Phase-4 output at 1e-5.
- Same test with `ICONNX_DISABLE_MEMORY_PLANNER=1` set: still passes.
- `kokoro_arena_coverage_test` (new, `#[ignore]`d on default run, runnable via `cargo test -- --ignored`): loads Kokoro, lowers, asserts `classification_counts().arena_assigned as f64 / arena_eligible as f64 >= 0.7`.

## Non-goals (expanded)

- **Interval-graph-coloring allocation strategy.** Greedy linear scan gets most of the memory-efficiency win on Kokoro-scale workloads (weights dominate peak footprint, not activations). Interval-graph-coloring is a future upgrade path if larger models (Stable Diffusion, batched Whisper) show memory pressure.
- **Arena for partially-dynamic tensors.** The `is_planner_eligible` predicate explicitly excludes any tensor with a `None` dim. The pool handles them.
- **Arena for tensors < 4096 bytes.** Matches pool's size-class threshold; pool handles small allocations.
- **Arena for graph-output tensors.** Ownership semantics are simpler if graph outputs always come from the pool (caller may retain them past the arena's lifetime). The exclusion is explicit in `is_planner_eligible`.
- **Cross-inference arena caching.** Each `run()` currently allocates its own arena. If profiling shows allocation overhead on high-throughput concurrent inference workloads, cache the arena buffer in the pool as a follow-up. Not Phase 4 scope; flagged here so it doesn't become an implicit assumption that bites later.
- **Multiple arenas per execution.** One arena per `run()`. No per-subgraph or per-stage arenas. Simpler reasoning.
- **A second constant_folding invocation post shape_extraction_folding.** The `folding_core` fixpoint loop handles the Shape-rooted chain internally in one pass.
- **Ratio-based gate for B.** See §Gating protocol: B's direct benefit is sub-noise by construction; gating on a measurement we predict will fail is ceremonial. Structural gate instead.
- **Symbolic-dim unlock under ONNX-symbolic calling contract (Phase 5+).** The 2026-04-22 second refinement explicitly deferred this. The mechanism fires only when the caller provides concrete shapes via `add_input` (matching Phase 3's C PR API). Two candidate mechanisms for a future phase: (a) extend `tensor_shapes` to distinguish `Dyn("seq_len")` from fully unknown; (b) runtime specialization (recompile plan on first inference using concrete shapes). Either deserves its own phase — both involve type-system or cache-invalidation design changes out of Phase 4 scope. Phase 3's original stacked-gate verdict on Kokoro-as-loaded-symbolic is formally DEFERRED (not closed) to whichever mechanism lands.
- **Whisper integration.** Separate leadline effort.

## Testing strategy

| Layer | B (`shape_extraction_folding`) | A (`memory_planner`) |
|---|---|---|
| Unit | 11 tests in `shape_extraction_folding.rs::tests` — Shape→Gather (5: all-concrete, None-at-extracted-index, non-Shape-input, out-of-bounds, negative-index-wrap), Shape→Cast→Gather (2: transparency, non-Int64-cast-doesn't-fold), Shape→Slice (2: all-concrete-range, None-in-range-doesn't-fold), standalone-Shape-doesn't-fold, idempotency. Plus 1 regression test in `src/operators/gather.rs::tests` for the rank-0-indices guard (ported from `8350f67`). | 9 tests in `memory_planner.rs::tests` — 4 eligibility predicate cases, 3 slot-assignment cases, determinism snapshot, classification counts |
| Integration | `kokoro_gpu_fusion_test::fusion_fires_on_full_kokoro_at_seq_len_5` — loader updated to pass concrete shapes via `add_input`; deterministic gate `fold_counts.total == 124` with per-pattern breakdown in the panic diagnostic. `reshape_hits` printed informationally (not asserted). | Same integration test passes with arena active (no numerical regression at 1e-5); passes with `ICONNX_DISABLE_MEMORY_PLANNER=1` |
| Coverage | N/A | new `kokoro_arena_coverage_test`: `arena_assigned / arena_eligible ≥ 0.7` |
| Cross-model | N/A in-PR | Post-Phase-4 follow-up: Whisper-Tiny executes through planner with numerical match to ORT (once leadline's harness lands) |
| Regression | Pre-Phase-4 tests (443 on `cuda`, 453 on `cuda,debug-inference`) all still pass unchanged | Same |

## Open questions

None remaining. All architectural choices made in brainstorming dialogue with leadline on 2026-04-19 + amendment dialogue on 2026-04-22.

## Amendment history

### 2026-04-22 (third refinement) — Commit 2 continues attempt #1: complete infer_reshape's Phase 3 Known limitation

**Trigger.** Second-refinement reframe (concrete-shape `add_input` loader) measured `fold_counts.total = 1/124` on Kokoro — not 124 as projected. Rejection-reason instrumentation showed 100 of 110 candidate Shape→Gather chains reject via `REJECT_NONE_DIM`: the extracted index (batch or seq_len) shows as `None` in `tensor_shapes`, EVEN THOUGH the caller seeded concrete `[Some(1), Some(5)]` for `tokens`.

**Root cause.** `src/ir/passes/shape_inference.rs:220-227`: `infer_reshape` returns `Shape::new()` unconditionally ("Phase 3 spec §Commit #3 Known limitation"). In Kokoro's BERT path, `tokens → Gather(embed_weight) → Reshape → Gemm → Add(bias)`, the first Reshape zeros out shape propagation. Downstream `infer_gemm` sees one empty-rank input and returns `Shape::new()` too. Downstream `infer_add` broadcasts `empty + bias[1,1,768]` → `[None, None, Some(768)]`. Every intermediate from the BERT body onward has batch+seq_len dropped. Extraction chains landing on those intermediates then reject — not because Kokoro "extracts dynamic dims" (the second-refinement framing) but because `infer_reshape`'s stub made them look dynamic upstream.

**Corrected diagnosis.** The 2026-04-22 second refinement was also incomplete. Attempt #1's "0/124 folds" and the 53-case structural ceiling were both downstream symptoms of the infer_reshape stub, not architectural facts about Kokoro. Completing infer_reshape unblocks the reframe.

**Scope (Option A2 per leadline, bounded):**

1. Extend `infer_reshape` to resolve its shape input when it's an initializer, reusing the same pattern `src/ir/passes/precompute.rs::precompute_reshape_shape` already uses (lookup in `graph.initializers`, read int64 values, resolve `0` placeholders against data shape, resolve single `-1` against data element count). Signature cascade: `infer_reshape` gains `&HashMap<String, Tensor>`; `infer_node_output_shapes` and `tensor_shapes_from_graph` plumb `&graph.initializers` through.
2. Re-run the Kokoro gate. If `fold_counts.total == 124`, attempt #1 converges. If partial, escalate before expanding scope.
3. **Pre-authorized:** fixing at most ONE additional op inferencer in this same attempt if infer_reshape unblocks most but not all chains. Three-or-more op fixes trigger a new spec amendment.
4. Update `infer_reshape`'s in-code comment: "Phase 3 Known limitation — completed in Phase 4 B 2026-04-22" replaces the "constant_folding often resolves Reshape before this pass runs anyway" rationale (empirically false for Kokoro's runtime-shape-sourced Reshapes).

**Attempt-count classification.** Still attempt #1. The three-attempt clause targets *mechanism pivots* (different predicate designs). Completing a Phase-3-acknowledged Known limitation in an upstream dependency isn't a mechanism pivot — it's finishing the work `shape_extraction_folding` legitimately assumed was already done. Counter stays at 1/3.

### 2026-04-22 (second refinement) — Commit 2 contract reframe: concrete-shape calling contract, deterministic gate

**Trigger.** Implementer executed attempt #1 of the redesigned `shape_extraction_folding` pass (three patterns: Gather, Cast→Gather, Slice; all 11 unit tests green) and ran the structural gate. Measured `fold_counts.total = 0` of the projected ≥ 38 — deep Red. Instrumentation showed all 124 extraction-chain candidates on Kokoro target `batch` or `seq_len` (both `None` under ONNX-symbolic calling contract); 0 target the `Some(768)` hidden_dim. The 53-case denominator from the first 2026-04-22 amendment was a counting error (tensors with SOME concrete dim, without filtering for "and the extraction targets it").

**Diagnosis.** The mechanism is correct. The calling contract was wrong. The test loader used `model.input_shapes()` which preserves ONNX symbolic `dim_param` as `None` for `seq_len` — so every downstream tensor's shape is partially dynamic in exactly the places extractions target. If the loader instead passes concrete `add_input("tokens", vec![Some(1), Some(5)])` matching the test's actual `kokoro_inputs(5)` tensor, every extraction chain has concrete dims to fold against.

**Reframe chosen by leadline (Option A):**

1. **Test loader updated** to pass concrete input shapes via `add_input`. No iconnx API change — `GpuGraphExecutor::add_input(name, Vec<Option<usize>>)` was added in Phase 3 C PR and already accepts concrete dims.
2. **Gate becomes deterministic:** `fold_counts.total == 124` exactly (both shortfall and overshoot fail). No statistical `≥ 38` — either the mechanism works on all 124 Kokoro chains or it doesn't.
3. **Secondary `reshape_hits` proxy removed.** Pre-Phase-4 baseline was 10/61; the 48-threshold was derived from the incorrect 38-unlock projection. Under the reframe, `reshape_hits` is printed informationally but not asserted.
4. **Three-attempt counter stays at 1/3.** The mechanism is unchanged; only the test's shape-seeding changed. Per the "attempt" disambiguation above, fixing a misaligned calling contract is part of the SAME attempt, not a new one.
5. **The `8350f67` bugfix continues to land regardless.**

**What this reframe explicitly closes (scope contract):**

- The Shape→extraction chain folding mechanism is validated end-to-end: on Kokoro with concrete-shape `add_input`, all 124 chains fold deterministically (110 Gather + 0 Cast→Gather + 14 Slice per attempt #1's instrumentation).
- Callers who cache plans per concrete input shape (Whisper's fixed `[1, 80, 3000]` encoder, Kokoro per-seq-len plans, etc.) get the optimization.

**What this reframe does NOT close (deferred to Phase 5+):**

- Unlocking Kokoro (or any ONNX-symbolic-dim model) when the caller loads shapes via `model.input_shapes()` and keeps `seq_len` as `None`. Two candidate mechanisms, both Phase 5+ in scope:
  - **Symbolic dim tracking.** Extend `tensor_shapes` from `Vec<Option<usize>>` to a richer type distinguishing `Dyn("seq_len")` from `Unknown`. Gather on a `Dyn` dim synthesizes a symbolic initializer that downstream dim-arithmetic ops can simplify.
  - **Runtime specialization.** Recompile the plan on first inference using the actual runtime shape; cache per-shape plans. Trades plan cache size for unlock coverage.
- Phase 3's stacked-gate verdict on Kokoro-as-loaded-with-ONNX-symbolic-dims formally DEFERRED (not closed) to whichever Phase 5+ mechanism lands.

**Rationale for reframe vs. abandon.** Abandoning the mechanism (Option B in the dispatch decision) would throw away correct code — 11 unit tests pass, all three patterns work, rejection logic is sound. Option C (new symbolic-dim mechanism) is genuinely too large for Phase 4. Option A — document what the mechanism actually delivers and defer the rest — is the honest landing.

### 2026-04-22 — Commit 2 redesign: shape_aware_folding → shape_extraction_folding

**Trigger.** Implementer wired the original `shape_aware_folding` pass into `lower()` and ran the structural gate against Kokoro. Hit rate landed at 10/61 — identical to Phase 3's baseline. Zero unlocks. Red tier on the original 33/61 gate. Implementer escalated per Phase 1 learning rather than weakening the gate.

**Diagnosis (real-graph instrumentation).** Kokoro has 73 Shape nodes at lowering time. 0 have fully-concrete input shapes: `OnnxParser::input_shapes()` preserves the ONNX-declared `dim_param` symbolic `seq_len` as `None` on the `tokens` input, and that `None` propagates through every downstream tensor. The original all-or-nothing predicate (fold Shape(x) iff every dim of x is `Some(_)`) was architecturally unable to fire on Kokoro's graph. Of the 73 Shape nodes: 20 have fully-unknown inputs (empty-sentinel, not reachable); **53 have partially-known inputs feeding into extraction ops (`Gather` / `Slice`, optionally through `Cast`) that ask for specific indices** — this 53-case set is what the mechanism can actually unlock.

**Scope guardrails chosen by leadline to prevent indefinite pattern bolt-on:**

1. Pass renamed to `shape_extraction_folding` — semantics shifted from "fold Shape when all-static" to "fold Shape-rooted chains terminating at extraction ops whose extracted indices hit static dims."
2. Pattern support limited to the three variants measured in the 53 partial cases: `Shape → Gather`, `Shape → Cast → Gather`, `Shape → Slice`. Additional patterns are explicit spec amendments, not ad-hoc extensions.
3. Structural gate denominator changed from 61 (total Reshape chains) to 53 (reachable partial cases). Thresholds calibrated to match Phase 3's original projection ratio: Green ≥ 72% = 38/53 unlocks, Amber 50–72% = 27–37/53, Red < 50% = < 27/53. Expressed against the `reshape_hits` metric: Green ≥ 48/61, Amber 37–47/61, Red < 37/61.
4. Three-attempt abandonment clause: if three pattern-implementation attempts fail to reach Green, B is abandoned, the `8350f67` bugfix is cherry-picked onto main as a standalone `fix(ops)` commit, Commit 3 (A) proceeds as the phase-only deliverable, and Phase 3's stacked-gate verdict is formally deferred to a future phase with a different mechanism (candidates: runtime seq_len specialization, upstream shape_inference improvements to eliminate the empty-sentinel category).
5. `8350f67` correctness bugfix lands regardless of B's path — orthogonal to the predicate design, fixes a real bug (rank-0 unknown sentinel → `Int64[0]` tensor panicking downstream Gather).

**Rationale for amending rather than retrying with a tweaked threshold.** The original predicate is architecturally wrong for Kokoro, not just numerically off. Lowering the gate to match the 10/61 floor would land a pass that does nothing, preserving the appearance of progress while leaving Phase 3's stacked-gate verdict permanently open. The amended design targets the structural form the data actually shows (extraction chains on partially-dynamic tensors) and sets a gate that requires the mechanism to demonstrably unlock the cases it's designed for.

### 2026-04-19 — Original spec

Initial finalization after brainstorming. Three commits on `ir-phase4`: refactor `folding_core`, `shape_aware_folding` (B) with all-or-nothing predicate and 33/61 gate, `memory_planner` (A) unchanged from amendment.
