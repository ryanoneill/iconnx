# Phase 4 Graph Optimizer II: shape-aware folding + memory planner

**Status:** finalized 2026-04-19 after brainstorming dialogue with leadline.
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
| 2 | `feat(ir): shape_aware_folding pass` (B) | new `src/ir/passes/shape_aware_folding.rs`; `src/ir/lowering.rs` (pipeline wiring); `src/ir/passes/mod.rs` (re-export) | **Structural gate.** Kokoro Reshape precompute hit rate (via `count_precomputed_shape_ops`) moves from 10/61 (pre-B) to ≥ 33/61 (10 existing + 23 projected unlocks), with a 3-tier protocol for the investigation zone — see §Gating protocol. |
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

Direct Kokoro ratio benefit for B is estimated at ~0.7 ms / 110 ms ≈ 0.006 ratio points — below the 0.05 floor. Gating on a ratio measurement we already predict will fail is ceremonial, not a check. Phase 3's stacked-gate commitment was not "B must measurably improve the Kokoro ratio" but "Phase 4 resolves whether the Shape→Cast→Gather unlock mechanism works" — a structural question, noise-independent.

**Structural gate.** Measured via `count_precomputed_shape_ops::reshape_hits` in `tests/kokoro_gpu_fusion_test.rs`. Three tiers:

- **Green (land without investigation):** hit rate ≥ 33/61 — matches or exceeds Phase 3's projection (10 existing + 23 unlocks).
- **Amber (investigate before landing):** 28/61 ≤ hit rate < 33/61. List which Reshape chains didn't unlock and why. Proceed if the gap is "projection was optimistic — the mechanism works on every chain it can reach." Block if the gap is "the unlock mechanism has a bug — some reachable chains didn't fold." The diagnostic script: for each of the 51 runtime-shape-chain Reshapes, walk back through `shape_inference`'s output; count how many have a terminating `Shape(x)` where `tensor_shapes[x]` is fully `Some(_)` (these SHOULD unlock). Compare to actual hit count post-B. Discrepancy > 2 = bug.
- **Red (block):** hit rate < 28/61. Mechanism clearly broken.

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

### Commit #2: `shape_aware_folding` (B)

```rust
// src/ir/passes/shape_aware_folding.rs
pub fn shape_aware_folding(
    graph: OptimizableGraph,
    tensor_shapes: &HashMap<String, Vec<Option<usize>>>,
) -> OptimizableGraph;
```

**Algorithm.**

```rust
pub fn shape_aware_folding(graph, tensor_shapes) -> OptimizableGraph {
    folding_core::fold_with_predicate(graph, |node, _folded| {
        if node.op_type != "Shape" {
            return None;
        }
        let input_name = node.inputs.first()?;
        let inferred = tensor_shapes.get(input_name)?;
        let fully_known: Vec<i64> = inferred
            .iter()
            .map(|d| d.map(|n| n as i64))
            .collect::<Option<Vec<_>>>()?;  // None if any dim is None
        Some(Tensor::from_vec_i64(fully_known, vec![inferred.len()]))
    })
}
```

**Pipeline position.** Inserted into `lower()` immediately after `shape_inference` + `tensor_shapes_from_graph`:

```rust
// src/ir/lowering.rs (revised)
let graph = topo_sort(graph)?;
let graph = constant_folding(graph);
let graph = dead_code_elimination(graph);
let graph = shape_inference(graph);
let tensor_shapes = tensor_shapes_from_graph(&graph);
let graph = shape_aware_folding(graph, &tensor_shapes);  // NEW
let graph = dead_code_elimination(graph);
let precomputed = precompute_params(&graph, &tensor_shapes);
let fusion_primary = elementwise_fusion_pass(&graph, &tensor_shapes);
// ... rest unchanged
```

**Why Shape-only predicate.** Shape is the sole op whose correctness is determined by its input's shape rather than value. Once `Shape(x)` folds into an initializer, downstream Cast/Gather/etc. become all-inputs-initializer and fold via `folding_core`'s existing dispatch in the same pass. No second const-folding pass is needed; the fixpoint loop handles the chain.

**Tests (in `src/ir/passes/shape_aware_folding.rs::tests`):**

- `shape_of_known_shape_tensor_folds`: graph with a runtime node producing a tensor `x` whose `tensor_shapes[x] = [Some(2), Some(3), Some(4)]`. Insert `Shape(x) → out`. Assert post-pass: `graph.initializers["out"] == [2, 3, 4]`, node removed.
- `shape_of_partially_dynamic_tensor_does_not_fold`: same setup but `tensor_shapes[x] = [None, Some(3), Some(4)]`. Assert post-pass: Shape node still present, no initializer added.
- `shape_cast_gather_chain_folds_to_single_initializer`: chain `Shape(x) → Cast(to=Int64) → Gather(indices=[0]) → out`. Post-pass: `out` initializer present with the expected gathered element.
- `shape_of_dynamic_graph_input_does_not_fold`: `x` is a graph input with shape `[None, Some(512)]`. Assert Shape node stays.
- `shape_aware_folding_is_idempotent`: two consecutive calls produce identical graphs.

**Integration test update in `tests/kokoro_gpu_fusion_test.rs`:**

```rust
// Pre-Phase-4 baseline was 10/61. Phase 4 B projects 23 additional
// unlocks via Shape-chain resolution, for a target of ≥ 33/61.
assert!(
    precompute_stats.reshape_hits >= 33,
    "Phase 4 B Reshape precompute hit rate regressed: got {}/{} hits \
     (expected at least 33 after Shape-chain unlock). Phase 3's \
     baseline was 10; shape_aware_folding should unlock ~23 more. \
     If in [28, 33), investigate which Reshape chains didn't unlock \
     and why — see spec §Gating protocol 'Amber' tier.",
    precompute_stats.reshape_hits,
    precompute_stats.reshape_total,
);
```

### Commit #3: `memory_planner` (A)

**Input data.** `ExecutionPlan` fields populated by earlier passes: `tensor_shapes: HashMap<String, Vec<Option<usize>>>` and `tensor_last_use: HashMap<String, usize>` (last op-index at which each tensor is consumed). Plus `ops: Vec<PlannedOp>` providing the topological ordering and per-op first-use (op-index that produces each output).

**Eligibility predicate (single function).**

```rust
// src/ir/passes/memory_planner.rs
pub(crate) fn is_planner_eligible(
    tensor_name: &str,
    tensor_shapes: &HashMap<String, Vec<Option<usize>>>,
    graph_outputs: &[String],
    dtype_bytes: usize,
    element_count: usize,
) -> bool {
    // All dims fully known.
    let Some(shape) = tensor_shapes.get(tensor_name) else { return false };
    if shape.iter().any(|d| d.is_none()) { return false; }

    // Not a graph output (runtime may need to hand ownership back).
    if graph_outputs.iter().any(|o| o == tensor_name) { return false; }

    // Size >= 4 KiB (matches pool's size-class threshold).
    if dtype_bytes * element_count < 4096 { return false; }

    true
}
```

Single predicate, one function. Adding conditions later (e.g., dtype restrictions) happens in one place.

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
}

impl MemoryPlan {
    pub fn arena_offset(&self, tensor_name: &str) -> Option<usize> {
        let &slot_idx = self.slot_assignments.get(tensor_name)?;
        Some(self.slot_offsets_bytes[slot_idx])
    }

    pub fn classification_counts(&self, /* eligibility input */) -> PlannerClassification;
}

pub struct PlannerClassification {
    pub arena_eligible: usize,
    pub arena_assigned: usize,
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
    // DeviceSlice<'static, u8> held by RAII; slice_at(...) returns
    // typed sub-slices into the arena for per-op allocation.
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
- `is_planner_eligible_graph_output_yields_false`
- `is_planner_eligible_below_4kb_yields_false`
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
- **A second constant_folding invocation post shape_aware_folding.** The `folding_core` fixpoint loop handles the Shape-rooted chain internally in one pass.
- **Ratio-based gate for B.** See §Gating protocol: B's direct benefit is sub-noise by construction; gating on a measurement we predict will fail is ceremonial. Structural gate instead.
- **Whisper integration.** Separate leadline effort.

## Testing strategy

| Layer | B (`shape_aware_folding`) | A (`memory_planner`) |
|---|---|---|
| Unit | 5 tests in `shape_aware_folding.rs::tests` — fold-when-known, no-fold-when-dynamic, chain-fold, dynamic-graph-input-no-fold, idempotency | 9 tests in `memory_planner.rs::tests` — 4 eligibility predicate cases, 3 slot-assignment cases, determinism snapshot, classification counts |
| Integration | `kokoro_gpu_fusion_test::fusion_fires_on_full_kokoro_at_seq_len_5` hit-rate assertion upgraded to `≥ 33` with Amber-tier investigation protocol | Same integration test passes with arena active (no numerical regression at 1e-5); passes with `ICONNX_DISABLE_MEMORY_PLANNER=1` |
| Coverage | N/A | new `kokoro_arena_coverage_test`: `arena_assigned / arena_eligible ≥ 0.7` |
| Cross-model | N/A in-PR | Post-Phase-4 follow-up: Whisper-Tiny executes through planner with numerical match to ORT (once leadline's harness lands) |
| Regression | Pre-Phase-4 tests (443 on `cuda`, 453 on `cuda,debug-inference`) all still pass unchanged | Same |

## Open questions

None remaining. All architectural choices made in brainstorming dialogue with leadline on 2026-04-19.
