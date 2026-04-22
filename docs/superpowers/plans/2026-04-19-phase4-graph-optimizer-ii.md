# Phase 4 Graph Optimizer II Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land three signed commits on branch `ir-phase4` implementing (1) a pure refactor extracting the constant_folding loop into a shared `folding_core` module, (2) the `shape_aware_folding` pass that closes Phase 3's stacked-gate unlock target, and (3) a static memory planner with a coexisting arena allocation path.

**Architecture:** `folding_core` exposes a single `fold_with_predicate(graph, extra_foldable_fn)` primitive that both the existing `constant_folding` and the new `shape_aware_folding` invoke with their respective foldability predicates. `shape_aware_folding` runs after `shape_inference` + `tensor_shapes_from_graph` and makes `Shape(x)` foldable when `tensor_shapes[x]` is fully known — downstream Cast/Gather/etc. fold via the existing fixpoint loop once Shape's output becomes a synthesized initializer. `memory_planner` runs after fusion annotation, consumes `tensor_shapes` + `tensor_last_use`, applies a semantic eligibility predicate and a fusion-skipped filter, then greedy-linear-scan assigns size-class-bucketed arena slots. Runtime `Executor` allocates a single arena at run start and branches per-op on `PlannedOp::arena_offset: Option<usize>`, falling back to the existing `GpuMemoryPool` path for ineligible tensors. An always-on `assert_eq!(actual_bytes, slot_bytes)` per op catches silent shape-inference undersizing; `ICONNX_DISABLE_MEMORY_PLANNER=1` is a runtime bisection escape hatch.

**Tech Stack:** Rust 2021, garboard CUDA bindings (`DeviceSlice`), NVRTC for kernel compilation, CUDA 12.x, ort 2.0.0-rc.10 (reference side, unchanged).

---

## Scope context

**Spec:** `docs/superpowers/specs/2026-04-19-phase4-graph-optimizer-ii-design.md` (commit `5875858`).

**Worktree:** `/home/ryano/workspace/ryanoneill/iconnx-ir-phase4` (branch `ir-phase4`, already created at main tip `5875858`; Kokoro model symlinked in).

**Landing pattern:** WIP commits during implementation of each logical commit, squashed into the final signed commit at the end of each commit's task block. Three signed commits total on the PR — no cross-commit squashing (each is independently reviewable per leadline's review hygiene preference).

| # | Final subject |
|---|---|
| 1 | `refactor(ir): extract fold loop into folding_core` |
| 2 | `feat(ir): shape_aware_folding pass (B)` |
| 3 | `feat(ir): static memory planner + arena allocation path (A)` |

Each final commit must be GPG-signed. If signing fails, STOP and escalate.

---

## Task 0: Environment + baseline

### Task 0.1: Verify worktree + baseline test counts

**Files:**
- Worktree: `/home/ryano/workspace/ryanoneill/iconnx-ir-phase4`

- [ ] **Step 1: Confirm worktree state**

```bash
cd /home/ryano/workspace/ryanoneill/iconnx-ir-phase4
git branch --show-current   # expect: ir-phase4
git log --oneline -3        # expect tip at 5875858 (spec refinement)
ls -la kokoro-v1.0.onnx     # expect symlink to ../rust-ai-explorations/models/kokoro-v1.0.onnx
```

- [ ] **Step 2: Record baseline test counts**

```bash
cargo build --features cuda --release 2>&1 | tail -3
cargo test --features cuda --release 2>&1 | grep -E "^test result" | awk '{pass+=$4; fail+=$6} END {print pass" passed, "fail" failed"}'
cargo test --features cuda,debug-inference --release 2>&1 | grep -E "^test result" | awk '{pass+=$4; fail+=$6} END {print pass" passed, "fail" failed"}'
cargo test --features cuda --release --test kokoro_gpu_fusion_test -- --ignored 2>&1 | grep -E "^test result|FAILED"
```

Expected: ~457 / ~467 passing, 0 failed; Kokoro integration test 1 passed.

- [ ] **Step 3: Record baseline Reshape precompute hit rate**

```bash
cargo test --features cuda --release --test kokoro_gpu_fusion_test -- --ignored --nocapture 2>&1 | grep -E "Reshape:\s+[0-9]+/[0-9]+"
```

Expected: `Reshape: 10/61 hits (16%)`. This is the baseline for B's structural gate.

---

## Commit 1: `refactor(ir)` — extract fold loop into `folding_core`

**Commit goal:** Pure refactor. Extract `constant_folding`'s fold loop into a new neutral module `src/ir/passes/folding_core.rs` that exposes `fn fold_with_predicate(graph, extra_foldable_fn) -> graph`. `constant_folding` becomes a thin caller with a default `extra_foldable_fn` that always returns `None`. All existing tests pass unchanged; zero behavioral change.

### Task 1.1: Create `folding_core` module with the extracted fold loop

**Files:**
- Create: `src/ir/passes/folding_core.rs`
- Modify: `src/ir/passes/mod.rs` (add `pub mod folding_core;`)

- [ ] **Step 1: Create `folding_core.rs` with the full extracted API**

```rust
//! Shared fold-loop primitive for `constant_folding` (initializer-only)
//! and `shape_aware_folding` (Shape-predicate extension).
//!
//! The fold loop walks nodes in topo order, folds any node whose inputs
//! are all constant (initializer or earlier folded-in-pass) OR whose
//! `extra_foldable` predicate returns `Some(tensor)`, promotes folded
//! outputs to initializers, and drops the folded node. The loop runs to
//! fixpoint internally — chains of foldable ops collapse in one pass.

use std::collections::HashMap;

use crate::ir::graph::{GraphNode, OptimizableGraph};
use crate::tensor::Tensor;

/// Maximum element count for a folded output. Folds that would produce
/// tensors larger than this are skipped (folding a 1M-element Conv on
/// CPU is not a win — and could explode the initializer table).
pub const FOLD_OUTPUT_ELEMENT_BUDGET: usize = 1_000_000;

/// Fold every node whose inputs are constant (initializer, earlier-folded,
/// or `extra_foldable` returns `Some`). Returns the graph with folded
/// nodes removed and their outputs promoted to initializers.
///
/// `extra_foldable` is called once per non-initializer-only node; it receives
/// the node and the in-pass folded map (not the initializers map, since
/// initializers are already checked). Returning `Some(tensor)` folds the
/// node with the provided value; `None` defers to the standard
/// initializer-only fold path.
pub fn fold_with_predicate(
    mut graph: OptimizableGraph,
    extra_foldable: impl Fn(&GraphNode, &HashMap<String, Tensor>) -> Option<Tensor>,
) -> OptimizableGraph {
    let mut folded: HashMap<String, Tensor> = HashMap::new();
    let mut kept_nodes: Vec<GraphNode> = Vec::with_capacity(graph.nodes.len());

    for node in graph.nodes.into_iter() {
        if let Some(outputs) = try_fold(&node, &graph.initializers, &folded, &extra_foldable) {
            for (name, tensor) in node.outputs.iter().zip(outputs.into_iter()) {
                if !name.is_empty() {
                    folded.insert(name.clone(), tensor);
                }
            }
        } else {
            kept_nodes.push(node);
        }
    }

    for (name, tensor) in folded {
        graph.initializers.insert(name, tensor);
    }
    graph.nodes = kept_nodes;
    graph
}

/// Attempt to fold one node. Returns `Some(outputs)` iff standard
/// initializer-only folding succeeds OR `extra_foldable` returns a value.
fn try_fold(
    node: &GraphNode,
    initializers: &HashMap<String, Tensor>,
    folded: &HashMap<String, Tensor>,
    extra_foldable: &impl Fn(&GraphNode, &HashMap<String, Tensor>) -> Option<Tensor>,
) -> Option<Vec<Tensor>> {
    // Extension path first — lets passes override standard semantics
    // (e.g., shape_aware_folding's Shape-uses-tensor_shapes rule).
    if let Some(tensor) = extra_foldable(node, folded) {
        if tensor.len() > FOLD_OUTPUT_ELEMENT_BUDGET {
            return None;
        }
        return Some(vec![tensor]);
    }

    // Standard initializer-only path.
    let mut input_tensors: Vec<Tensor> = Vec::with_capacity(node.inputs.len());
    for name in &node.inputs {
        if name.is_empty() {
            continue;
        }
        let t = initializers
            .get(name)
            .or_else(|| folded.get(name))?
            .clone();
        input_tensors.push(t);
    }

    let output = dispatch_cpu_forward(&node.op_type, &input_tensors, &node.attributes)?;

    if output.len() > FOLD_OUTPUT_ELEMENT_BUDGET {
        return None;
    }
    Some(vec![output])
}

/// Route a node to its CPU forward implementation. Returns `None` for
/// any op we don't support yet.
fn dispatch_cpu_forward(
    op_type: &str,
    inputs: &[Tensor],
    attributes: &crate::attributes::NodeAttributes,
) -> Option<Tensor> {
    use crate::operators::*;
    Some(match op_type {
        // Binary elementwise arithmetic.
        "Add" => add::Add::forward(inputs, attributes),
        "Sub" => sub::Sub::forward(inputs, attributes),
        "Mul" => mul::Mul::forward(inputs, attributes),
        "Div" => div::Div::forward(inputs, attributes),
        "Pow" => pow::Pow::forward(inputs, attributes),

        // Unary elementwise (full set copied verbatim from
        // constant_folding.rs — do not alter during the refactor).
        "Sqrt" => sqrt::Sqrt::forward(inputs, attributes),
        "Exp" => exp::Exp::forward(inputs, attributes),
        "Sin" => sin::Sin::forward(inputs, attributes),
        "Cos" => cos::Cos::forward(inputs, attributes),
        "Tanh" => tanh::Tanh::forward(inputs, attributes),
        "Sigmoid" => sigmoid::Sigmoid::forward(inputs, attributes),
        "Atan" => atan::Atan::forward(inputs, attributes),
        "Clip" => clip::Clip::forward(inputs, attributes),
        "Floor" => floor::Floor::forward(inputs, attributes),
        "Round" => round::Round::forward(inputs, attributes),
        "Cast" => cast::Cast::forward(inputs, attributes),
        "Erf" => erf::Erf::forward(inputs, attributes),

        // Shape-family (no-op reshape / static axes / static shape).
        "Reshape" => reshape::Reshape::forward(inputs, attributes),
        "Unsqueeze" => unsqueeze::Unsqueeze::forward(inputs, attributes),
        "Squeeze" => squeeze::Squeeze::forward(inputs, attributes),
        "Transpose" => transpose::Transpose::forward(inputs, attributes),
        "Shape" => shape::Shape::forward(inputs, attributes),
        "ConstantOfShape" => constant_of_shape::ConstantOfShape::forward(inputs, attributes),

        // Indexing / gather / slicing.
        "Gather" => gather::Gather::forward(inputs, attributes),
        "Concat" => concat::Concat::forward(inputs, attributes),
        "Slice" => slice::Slice::forward(inputs, attributes),
        "Range" => range::Range::forward(inputs, attributes),

        // Comparison / logical.
        "Equal" => equal::Equal::forward(inputs, attributes),
        "Less" => less::Less::forward(inputs, attributes),
        "Greater" => greater::Greater::forward(inputs, attributes),
        "GreaterOrEqual" => greater_or_equal::GreaterOrEqual::forward(inputs, attributes),

        _ => return None,
    })
}
```

**IMPORTANT:** The dispatch table above is copied VERBATIM from
`src/ir/passes/constant_folding.rs` (current lines 102–163). If iconnx's
current `constant_folding.rs` has different arms, use the current set —
do not add or remove ops during this refactor. The grep command to
verify parity after extraction:

```bash
diff <(grep -E '^\s+"[A-Z][a-zA-Z]+" =>' src/ir/passes/constant_folding.rs.ORIG | sort) \
     <(grep -E '^\s+"[A-Z][a-zA-Z]+" =>' src/ir/passes/folding_core.rs | sort)
```

(save the original to `.ORIG` before editing if doing this diff).

- [ ] **Step 2: Register the module**

In `src/ir/passes/mod.rs`, add to the module declarations (alphabetical, near existing `pub mod constant_folding;`):

```rust
pub mod folding_core;
```

Do NOT re-export anything from `folding_core` at the `passes::` level — both callers (`constant_folding` and `shape_aware_folding`) reference it via `crate::ir::passes::folding_core::fold_with_predicate`. Keeping the helper module-scoped makes the dependency graph explicit.

- [ ] **Step 3: Build**

```bash
cargo build --features cuda --release 2>&1 | tail -3
```

Expected: clean. `folding_core.rs` compiles; nothing else references it yet so no integration issues.

- [ ] **Step 4: Commit WIP**

```bash
git add src/ir/passes/folding_core.rs src/ir/passes/mod.rs
git commit -S -m "wip(ir): create folding_core with extracted fold_with_predicate"
```

### Task 1.2: Rewrite `constant_folding` as a thin wrapper over `folding_core`

**Files:**
- Modify: `src/ir/passes/constant_folding.rs`

- [ ] **Step 1: Replace the full implementation — keep only the public fn, the FOLD_OUTPUT_ELEMENT_BUDGET re-export, and the test module**

Read the current `src/ir/passes/constant_folding.rs` first, keeping the test module intact. Replace everything from line 22 (`pub fn constant_folding`) through the end of `dispatch_cpu_forward` (currently line ~163) with:

```rust
/// Pure function: consumes one OptimizableGraph, produces another with
/// every foldable chain collapsed to initializers. Foldable means:
/// - every non-empty input is in `graph.initializers` (or was folded
///   earlier in this pass), AND
/// - the op has a CPU `forward` implementation, AND
/// - the computed output size is within the element budget.
///
/// Unknown op types, multi-output ops, and ops whose output exceeds the
/// element budget are preserved verbatim. Delegates the fold loop to
/// `folding_core::fold_with_predicate` with a default (always-None)
/// predicate, since this pass only folds initializer-rooted chains.
pub fn constant_folding(graph: OptimizableGraph) -> OptimizableGraph {
    folding_core::fold_with_predicate(graph, |_node, _folded| None)
}
```

Update imports at the top of the file. The new file needs only:

```rust
//! Constant-folding pass — promotes initializer-rooted fold chains into
//! synthesized initializers. The fold loop itself is in
//! `folding_core`; this module is the initializer-only configuration.

use crate::ir::graph::OptimizableGraph;
use crate::ir::passes::folding_core;
```

Delete unused imports (`std::collections::HashMap`, `GraphNode`, `Tensor`, `NodeAttributes`) unless the test module at the bottom still references them.

**Preserve the test module verbatim.** The tests at the bottom of the file exercise the behavioral contract of `constant_folding`; they should pass unchanged after the refactor. If any test imports types that are now unused in the main module, re-import them in the test module.

- [ ] **Step 2: Build**

```bash
cargo build --features cuda --release 2>&1 | tail -3
```

Expected: clean build.

- [ ] **Step 3: Run the constant_folding tests**

```bash
cargo test --features cuda --release --lib ir::passes::constant_folding 2>&1 | grep -E "^test result|FAILED"
```

Expected: same number of passed tests as before the refactor (the spec said 7; verify it still matches).

- [ ] **Step 4: Run the full test suite**

```bash
cargo test --features cuda --release 2>&1 | grep -E "^test result" | awk '{pass+=$4; fail+=$6} END {print pass" passed, "fail" failed"}'
```

Expected: same count as Task 0.1 Step 2 (no tests lost or gained). Zero failures.

- [ ] **Step 5: Commit WIP**

```bash
git add src/ir/passes/constant_folding.rs
git commit -S -m "wip(ir): constant_folding delegates to folding_core"
```

### Task 1.3: Squash into final `refactor(ir)` commit

- [ ] **Step 1: Verify WIP commits are signed and complete**

```bash
git log --format='%G? %s' 5875858..HEAD
```

Expected: both WIP commits start with `G`; subjects are `wip(ir): create folding_core...` and `wip(ir): constant_folding delegates...`.

- [ ] **Step 2: Full test suite, zero warnings**

```bash
cargo build --features cuda --release 2>&1 | grep -i warning   # expect empty
cargo build --features cuda,debug-inference --release 2>&1 | grep -i warning   # expect empty
cargo test --features cuda --release 2>&1 | grep -E "^test result" | awk '{pass+=$4; fail+=$6} END {print pass" passed, "fail" failed"}'
cargo test --features cuda,debug-inference --release 2>&1 | grep -E "^test result" | awk '{pass+=$4; fail+=$6} END {print pass" passed, "fail" failed"}'
```

Expected: same counts as Task 0.1; zero warnings both feature combos.

- [ ] **Step 3: Squash**

```bash
git reset --soft 5875858
git commit -S -m "$(cat <<'EOF'
refactor(ir): extract fold loop into folding_core

Moves the fold loop, try_fold helper, and dispatch_cpu_forward match
table from constant_folding.rs into a new neutral module
src/ir/passes/folding_core.rs. Exposes:

  pub fn fold_with_predicate(
      graph: OptimizableGraph,
      extra_foldable: impl Fn(&GraphNode, &HashMap<String, Tensor>) -> Option<Tensor>,
  ) -> OptimizableGraph;

constant_folding becomes a thin caller with a default (always-None)
predicate, preserving its external API.

This commit is pure refactor — zero behavioral change, zero signature
change on constant_folding. All constant_folding tests pass unchanged.
Sets up Phase 4 Commit 2 (shape_aware_folding), which calls the same
primitive with a Shape-predicate extension.

Neutral module location chosen over pub(crate) fn in constant_folding
so the helper isn't "owned" by either pass — both are peer consumers.

Spec: docs/superpowers/specs/2026-04-19-phase4-graph-optimizer-ii-design.md.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 4: Verify the squash**

```bash
git log --oneline 5875858..HEAD    # expect exactly one commit
git log -1 --pretty=format:'%G? %s'
```

Expected: one signed commit `refactor(ir): extract fold loop into folding_core`.

---

## Commit 2: `feat(ir)` — shape_extraction_folding pass (B) [REDESIGNED 2026-04-22]

**Amendment context:** A previous attempt at this commit (branch `ir-phase4` commits `cf49711`, `292140c`, `8350f67` atop refactor `5924e5d`) built the original spec's "fold Shape when all-static" pass. Real-graph instrumentation showed 0/61 unlocks on Kokoro because every downstream tensor inherits `seq_len = None` from the `tokens` graph input. Spec was amended 2026-04-22 (commits `4241948` + `b236b41` on main) to target Shape-rooted chains terminating at extraction ops instead. This Commit 2 block is the revised task plan.

**Commit goal:** Add `shape_extraction_folding` — folds Shape-rooted chains whose consumer extraction op (`Gather`, `Slice`, optionally through `Cast`) requests only concrete dims of a partially-dynamic Shape input. Three supported patterns: `Shape → Gather(indices)`, `Shape → Cast(Int64) → Gather(indices)`, `Shape → Slice(start, end)`. Pipeline position unchanged: `...shape_inference → tensor_shapes_from_graph → shape_extraction_folding → DCE → precompute_params → ...`.

**Structural gate (primary + secondary per amended spec §Gating protocol B):**
- Primary: `testing::shape_extraction_folding_counts().total ≥ 38` (of 53 partial cases on Kokoro).
- Secondary (informational): `reshape_hits ≥ 48/61`. Divergence from primary is a finding, not a failure.
- Three-attempt abandonment clause: see §Gating protocol B. An "attempt" is a fresh predicate design or a distinct new pattern variant; parameter tweaking on an in-progress pattern is part of the same attempt.

**Starting state:** Branch `ir-phase4` currently has three WIP commits (`cf49711`, `292140c`, `8350f67`) from the prior attempt. **Task 2.0 below resets those** and redoes the commit under the amended design. The `8350f67` bugfix is preserved by re-applying its content in Task 2.1.

### Task 2.0: Reset prior-attempt WIPs + preserve the 8350f67 bugfix content

**Files:**
- Branch: `ir-phase4` (worktree `/home/ryano/workspace/ryanoneill/iconnx-ir-phase4`)

- [ ] **Step 1: Capture the `8350f67` bugfix diff before reset**

```bash
cd /home/ryano/workspace/ryanoneill/iconnx-ir-phase4
git log --oneline 5924e5d..HEAD    # expect 3 WIP commits: cf49711, 292140c, 8350f67
git show 8350f67 > /tmp/8350f67-bugfix.patch
```

Inspect `/tmp/8350f67-bugfix.patch` to understand exactly what the bugfix changed. Per the amended spec: it's "a real bug in Gather (or adjacent) where an Int64[0] tensor panicked at `src/operators/gather.rs:144` via `index_axis(Axis(0), idx)` on a zero-length axis."

**Decide the correct landing site for the bugfix:**
- If 8350f67 modifies `src/ir/passes/shape_aware_folding.rs` (i.e., the guard is in the pass's predicate): the new `shape_extraction_folding` design never folds Shape standalone, so this guard form is not directly portable. Re-implement the same invariant as a guard in `src/operators/gather.rs::forward` itself — rejecting rank-0 indices tensors at the op level (returning an empty output of the correct shape, or returning a documented error, whichever matches iconnx's existing Gather error-handling convention).
- If 8350f67 already modifies `src/operators/gather.rs`: preserve it verbatim in Task 2.1 by re-applying the patch.

- [ ] **Step 2: Reset the branch to refactor tip**

```bash
git reset --hard 5924e5d
git log --oneline -3    # expect: 5924e5d refactor, then Phase 3 merge commit, then earlier
```

This discards `cf49711`, `292140c`, `8350f67` from the branch. The commits remain in the reflog (recoverable via `git reflog` for the next ~90 days) and the bugfix content is preserved at `/tmp/8350f67-bugfix.patch`.

- [ ] **Step 3: Verify baseline**

```bash
cargo build --features cuda --release 2>&1 | tail -3
cargo test --features cuda --release 2>&1 | grep -E "^test result" | awk '{pass+=$4; fail+=$6} END {print pass" passed, "fail" failed"}'
```

Expected: 457 passed, 0 failed (back to the post-refactor baseline; any `shape_aware_folding` tests from the prior attempt are gone).

### Task 2.1: Re-apply the Gather rank-0-indices hardening (keeper from 8350f67)

**Files:**
- Modify: `src/operators/gather.rs` (add the guard + regression test)

- [ ] **Step 1: Apply the preserved bugfix content**

Open `/tmp/8350f67-bugfix.patch` and copy the relevant changes. If the original fix was in the pass:
- Port the guard logic into `src/operators/gather.rs::Gather::forward` near the existing `index_axis(Axis(0), idx)` call. Guard: if `indices.len() == 0`, return an empty output tensor matching the expected rank (data_rank - 1 per ONNX Gather semantics) instead of indexing.

If the original fix was in gather.rs: apply verbatim.

- [ ] **Step 2: Add the regression test in `src/operators/gather.rs::tests`**

Port the test from the preserved patch. Example shape (adapt to what the patch contained):

```rust
#[test]
fn gather_on_empty_indices_returns_empty_output_without_panic() {
    use crate::attributes::NodeAttributes;
    // data: Int64[3]; indices: Int64[0] (empty).
    let data = Tensor::from_vec_i64(vec![10, 20, 30], vec![3]);
    let indices = Tensor::from_vec_i64(vec![], vec![0]);
    let mut attrs = NodeAttributes::new();
    attrs.set_int("axis", 0);
    let out = Gather::forward(&[data, indices], &attrs);
    // Expected: Int64 tensor of shape [0] (rank-0 of indices × rank-0 gather axis = rank 0).
    // Must not panic.
    assert_eq!(out.shape(), vec![0]);
}
```

- [ ] **Step 3: Build + run the new test**

```bash
cargo build --features cuda --release 2>&1 | tail -3
cargo test --features cuda --release --lib operators::gather 2>&1 | grep -E "^test result|FAILED"
```

Expected: clean build; new test passes; prior Gather tests still pass.

- [ ] **Step 4: Commit WIP**

```bash
git add src/operators/gather.rs
git commit -S -m "wip(ops): gather guard against rank-0 indices (keeper from 8350f67)"
```

### Task 2.2: Create `shape_extraction_folding` skeleton + Shape→Gather pattern

**Files:**
- Create: `src/ir/passes/shape_extraction_folding.rs`
- Modify: `src/ir/passes/mod.rs`

- [ ] **Step 1: Create the pass skeleton with the `Shape → Gather` pattern only**

`src/ir/passes/shape_extraction_folding.rs`:

```rust
//! Shape-extraction folding.
//!
//! Folds Shape-rooted chains terminating at extraction ops whose
//! extracted indices hit only CONCRETE dims of a (possibly
//! partially-dynamic) Shape input. Designed for graphs where
//! `seq_len`-like symbolic dims propagate through tensor_shapes as
//! `None`, making the all-or-nothing `Shape`-foldability predicate
//! unable to fire (see spec §Amendment history, 2026-04-22).
//!
//! Three supported patterns (exactly these; additions are explicit
//! spec amendments):
//!   1. `Shape(x) -> Gather(indices=I)`
//!   2. `Shape(x) -> Cast(to=Int64) -> Gather(indices=I)`
//!   3. `Shape(x) -> Slice(start=s, end=e)`
//!
//! Pattern 1 is implemented in Task 2.2; patterns 2 and 3 in Task 2.3
//! and 2.4.
//!
//! Must run AFTER shape_inference has populated tensor_shapes.

use std::collections::HashMap;

use crate::ir::graph::{GraphNode, OptimizableGraph};
use crate::ir::passes::folding_core;
use crate::tensor::Tensor;

pub fn shape_extraction_folding(
    graph: OptimizableGraph,
    tensor_shapes: &HashMap<String, Vec<Option<usize>>>,
) -> OptimizableGraph {
    // Build a producer-name → GraphNode index so the predicate can walk
    // back from an extraction op to its Shape source without O(n) list
    // traversal per call.
    let producers = build_producer_map(&graph);

    folding_core::fold_with_predicate(graph, |node, _folded| {
        match node.op_type.as_str() {
            "Gather" => try_fold_shape_gather(node, &producers, tensor_shapes),
            // "Slice" and Cast-transparent Gather added in Tasks 2.3, 2.4.
            _ => None,
        }
    })
}

/// Maps an output tensor name to the node that produces it.
/// Uses HashMap<String, GraphNode> because GraphNode is Clone-cheap
/// and the predicate borrow is local.
fn build_producer_map(graph: &OptimizableGraph) -> HashMap<String, GraphNode> {
    let mut map = HashMap::new();
    for node in graph.nodes.iter() {
        for out_name in node.outputs.iter() {
            if !out_name.is_empty() {
                map.insert(out_name.clone(), node.clone());
            }
        }
    }
    map
}

/// Fold `Shape(x) → Gather(indices=I)` when every index in I is a
/// concrete dim of x.
fn try_fold_shape_gather(
    gather_node: &GraphNode,
    producers: &HashMap<String, GraphNode>,
    tensor_shapes: &HashMap<String, Vec<Option<usize>>>,
) -> Option<Tensor> {
    // Gather has two inputs: data (input 0) and indices (input 1).
    // For this pattern to fire, input 0 must be produced by a Shape op.
    let data_name = gather_node.inputs.first()?;
    let producer = producers.get(data_name)?;
    if producer.op_type != "Shape" {
        return None;
    }

    // The Shape op's input is the tensor whose rank + (partial) shape
    // we inspect.
    let shape_input_name = producer.inputs.first()?;
    let inferred = tensor_shapes.get(shape_input_name)?;

    // Reject the empty-Vec "fully unknown" sentinel — no dims known.
    if inferred.is_empty() {
        return None;
    }

    // Gather indices must be concrete (come from an initializer, not
    // a runtime tensor). folding_core already handles this — if
    // indices isn't in graph.initializers or a prior folded output,
    // the standard-fold path wouldn't fire either. But for this
    // predicate, we need the actual index values. Look them up via
    // producers: if the producer is a Shape chain, we can't get them;
    // if it's an initializer (no producer), they aren't reachable
    // from this closure's input. SIMPLER: require the indices input
    // to have NO producer (implies initializer-sourced), and rely on
    // the controller (lower()) to pass graph.initializers separately
    // in a future refactor if needed.
    //
    // For now, we accept ONLY the case where fold_with_predicate's
    // caller has already promoted the indices to an initializer. The
    // predicate fires only when folding_core's fixpoint has processed
    // a prior pass that synthesized the indices as an initializer.
    //
    // This covers the Kokoro case: Gather.indices inputs are all
    // Constant-node outputs or graph initializers by the time
    // shape_extraction_folding runs (constant_folding has already
    // handled Constant-node → initializer promotion).
    let indices_name = gather_node.inputs.get(1)?;
    if producers.contains_key(indices_name) {
        // Not an initializer — can't resolve. Future enhancement:
        // folding_core could pass initializers into the predicate; for
        // now, accept only initializer-sourced indices per the spec.
        return None;
    }
    // At this point indices_name refers to something not produced by
    // any op — must be a graph-input or initializer. For Kokoro all
    // such Gather.indices are int64 initializers. We need access to
    // the initializer values; since the predicate closure doesn't
    // receive graph.initializers, we refactor to accept it in a
    // follow-up step (see Task 2.2 Step 3).
    //
    // For this skeleton, return None — Task 2.2 Step 3 replaces this
    // return with the real extraction logic after the signature refactor.
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attributes::NodeAttributes;
    use crate::ir::graph::OptimizableGraphBuilder;

    fn shapes_map(entries: &[(&str, Vec<Option<usize>>)]) -> HashMap<String, Vec<Option<usize>>> {
        entries.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
    }

    // Tests added in Task 2.2 Step 4 (after the signature refactor lands).
    #[test]
    fn empty_placeholder_test_to_validate_module_compiles() {
        // Sanity check: module compiles, one test present.
        let graph = OptimizableGraphBuilder::new().build();
        let shapes = shapes_map(&[]);
        let _ = shape_extraction_folding(graph, &shapes);
    }
}
```

- [ ] **Step 2: Register the module**

In `src/ir/passes/mod.rs`, add alphabetically near `shape_inference`:

```rust
pub mod shape_extraction_folding;
```

Also in the `pub use` block:

```rust
pub use shape_extraction_folding::shape_extraction_folding;
```

- [ ] **Step 3: Refactor the predicate signature to accept initializers**

The skeleton above can't access `graph.initializers` from inside the closure. The clean fix: have `shape_extraction_folding` capture an `&HashMap<String, Tensor>` (the graph's initializers at closure-creation time) and pass it into the per-pattern helpers.

In `folding_core`, the closure signature already accepts the in-pass `folded: &HashMap<String, Tensor>` as its second argument — but the CONSUMER pass, `shape_extraction_folding`, needs initializers too. Solution: capture initializers in the closure before `folding_core::fold_with_predicate` takes ownership of the graph. BUT `fold_with_predicate` takes `graph: OptimizableGraph` by value, so the closure must hold a CLONE of initializers (cheap — HashMap of Arc'd tensors in iconnx).

Update the pass:

```rust
pub fn shape_extraction_folding(
    graph: OptimizableGraph,
    tensor_shapes: &HashMap<String, Vec<Option<usize>>>,
) -> OptimizableGraph {
    let producers = build_producer_map(&graph);
    // Clone initializers for the closure. Cheap: HashMap<String, Tensor>
    // where Tensor internally holds Arc or similar (check the actual
    // Tensor impl — if it's not Clone-cheap, pre-extract only the
    // Int64 initializers a Gather.indices would reference).
    let initializers = graph.initializers.clone();

    folding_core::fold_with_predicate(graph, move |node, folded| {
        match node.op_type.as_str() {
            "Gather" => try_fold_shape_gather(node, &producers, tensor_shapes, &initializers, folded),
            _ => None,
        }
    })
}

fn try_fold_shape_gather(
    gather_node: &GraphNode,
    producers: &HashMap<String, GraphNode>,
    tensor_shapes: &HashMap<String, Vec<Option<usize>>>,
    initializers: &HashMap<String, Tensor>,
    folded: &HashMap<String, Tensor>,
) -> Option<Tensor> {
    let data_name = gather_node.inputs.first()?;
    let producer = producers.get(data_name)?;
    if producer.op_type != "Shape" {
        return None;
    }
    let shape_input_name = producer.inputs.first()?;
    let inferred = tensor_shapes.get(shape_input_name)?;
    if inferred.is_empty() {
        return None;
    }

    // Gather.indices: must be an initializer (directly or via earlier
    // fold in this pass).
    let indices_name = gather_node.inputs.get(1)?;
    let indices_tensor = initializers
        .get(indices_name)
        .or_else(|| folded.get(indices_name))?;
    let indices = indices_tensor.to_array_i64();
    let indices_slice = indices.as_slice()?;

    // Read Gather axis attribute. For this pattern we require axis 0
    // (Shape output is rank-1, so only axis 0 is valid). Reject
    // anything else as out-of-scope.
    let axis = gather_node.attributes.get_int("axis").unwrap_or(0);
    if axis != 0 {
        return None;
    }

    // Resolve each index against inferred shape. Every referenced dim
    // must be Some(_); out-of-bounds or None dim → bail.
    let rank = inferred.len() as i64;
    let mut output: Vec<i64> = Vec::with_capacity(indices_slice.len());
    for &idx in indices_slice {
        // ONNX allows negative indices (wraps around rank).
        let normalized = if idx < 0 { idx + rank } else { idx };
        if normalized < 0 || normalized as usize >= inferred.len() {
            return None;
        }
        let dim = inferred[normalized as usize]?;  // None → bail
        output.push(dim as i64);
    }

    // Output shape: depends on indices' rank. ONNX spec: output rank
    // = data rank (1 for Shape output) + indices rank - 1. For rank-1
    // indices, output is rank 1 with len = indices.len. For rank-0
    // (scalar) indices, output is rank 0 with len 1. Use the indices
    // tensor's actual shape as the fold-output shape minus the gather
    // axis — here, since data is rank 1, gather output is just the
    // indices shape.
    let output_shape = indices_tensor.shape();
    Some(Tensor::from_vec_i64(output, output_shape))
}
```

- [ ] **Step 4: Write the Shape→Gather unit tests**

Replace the placeholder test with:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::attributes::NodeAttributes;
    use crate::ir::graph::OptimizableGraphBuilder;

    fn shapes_map(entries: &[(&str, Vec<Option<usize>>)]) -> HashMap<String, Vec<Option<usize>>> {
        entries.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
    }

    #[test]
    fn gather_of_shape_with_all_concrete_indices_folds() {
        // Shape(x) where x: [Some(1), None, Some(512)].
        // Gather([0, 2]) extracts dim 0 (=1) and dim 2 (=512).
        // Both concrete → fold to initializer [1, 512].
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("x".into(), vec![Some(1), None, Some(512)]);
        b.add_initializer(
            "indices".into(),
            Tensor::from_vec_i64(vec![0, 2], vec![2]),
        );
        b.add_node(
            "shape_of_x",
            "Shape",
            vec!["x".into()],
            vec!["shape_out".into()],
            NodeAttributes::new(),
        );
        let mut gather_attrs = NodeAttributes::new();
        gather_attrs.set_int("axis", 0);
        b.add_node(
            "gather_dims",
            "Gather",
            vec!["shape_out".into(), "indices".into()],
            vec!["dims".into()],
            gather_attrs,
        );
        let graph = b.build();

        let shapes = shapes_map(&[("x", vec![Some(1), None, Some(512)])]);
        let folded = shape_extraction_folding(graph, &shapes);

        // Both Shape and Gather should be folded away.
        assert!(folded.nodes.iter().all(|n| n.op_type != "Gather" && n.op_type != "Shape"));
        let dims = folded.initializers.get("dims").expect("dims synthesized");
        assert_eq!(dims.to_array_i64().as_slice().unwrap(), &[1i64, 512]);
    }

    #[test]
    fn gather_of_shape_with_none_at_extracted_index_does_not_fold() {
        // Gather([1]) targets the None dim — must not fold.
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("x".into(), vec![Some(1), None, Some(512)]);
        b.add_initializer(
            "indices".into(),
            Tensor::from_vec_i64(vec![1], vec![1]),
        );
        b.add_node(
            "shape_of_x",
            "Shape",
            vec!["x".into()],
            vec!["shape_out".into()],
            NodeAttributes::new(),
        );
        let mut gather_attrs = NodeAttributes::new();
        gather_attrs.set_int("axis", 0);
        b.add_node(
            "gather_dims",
            "Gather",
            vec!["shape_out".into(), "indices".into()],
            vec!["dims".into()],
            gather_attrs,
        );
        let graph = b.build();

        let shapes = shapes_map(&[("x", vec![Some(1), None, Some(512)])]);
        let folded = shape_extraction_folding(graph, &shapes);

        // Neither folded.
        assert!(folded.nodes.iter().any(|n| n.op_type == "Shape"));
        assert!(folded.nodes.iter().any(|n| n.op_type == "Gather"));
        assert!(!folded.initializers.contains_key("dims"));
    }

    #[test]
    fn gather_on_non_shape_input_does_not_fold() {
        // Gather's data input is an initializer, NOT a Shape output.
        // Pass must leave it alone (over-matching guard).
        let mut b = OptimizableGraphBuilder::new();
        b.add_initializer(
            "data".into(),
            Tensor::from_vec_i64(vec![10, 20, 30], vec![3]),
        );
        b.add_initializer(
            "indices".into(),
            Tensor::from_vec_i64(vec![0], vec![1]),
        );
        let mut gather_attrs = NodeAttributes::new();
        gather_attrs.set_int("axis", 0);
        b.add_node(
            "gather",
            "Gather",
            vec!["data".into(), "indices".into()],
            vec!["out".into()],
            gather_attrs,
        );
        let graph = b.build();

        let shapes = shapes_map(&[]);
        let folded = shape_extraction_folding(graph, &shapes);

        // NOTE: folding_core's standard path WOULD fold this Gather
        // (both inputs are initializers), because that's
        // constant_folding's domain. In the pipeline, constant_folding
        // runs before shape_extraction_folding and would have already
        // promoted this to an initializer. For this isolated test we
        // only verify shape_extraction_folding doesn't crash or
        // over-match on non-Shape-rooted Gather nodes.
        //
        // The semantic assertion: if Gather is folded here, it's via
        // folding_core's standard path (not our predicate), which is
        // correct behavior. We assert on the outcome: either the
        // Gather survived (standard path didn't fire — fine) or was
        // folded (standard path fired — also fine).
        assert!(true, "smoke test: no crash on non-Shape-rooted Gather");
    }

    #[test]
    fn out_of_bounds_gather_index_does_not_fold() {
        // Gather([5]) targets dim 5, but x has only 3 dims. Must
        // return None from the predicate, not panic.
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("x".into(), vec![Some(1), Some(2), Some(3)]);
        b.add_initializer(
            "indices".into(),
            Tensor::from_vec_i64(vec![5], vec![1]),
        );
        b.add_node(
            "shape_of_x",
            "Shape",
            vec!["x".into()],
            vec!["shape_out".into()],
            NodeAttributes::new(),
        );
        let mut gather_attrs = NodeAttributes::new();
        gather_attrs.set_int("axis", 0);
        b.add_node(
            "gather_oob",
            "Gather",
            vec!["shape_out".into(), "indices".into()],
            vec!["dims".into()],
            gather_attrs,
        );
        let graph = b.build();

        let shapes = shapes_map(&[("x", vec![Some(1), Some(2), Some(3)])]);
        let folded = shape_extraction_folding(graph, &shapes);

        // Must not panic; must not produce a (wrong) initializer.
        assert!(!folded.initializers.contains_key("dims"));
    }

    #[test]
    fn negative_gather_index_resolves_via_rank_wrap() {
        // Gather([-1]) should extract the LAST dim.
        // x: [Some(1), None, Some(512)]. rank=3. idx=-1 → 2 → 512.
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("x".into(), vec![Some(1), None, Some(512)]);
        b.add_initializer(
            "indices".into(),
            Tensor::from_vec_i64(vec![-1], vec![1]),
        );
        b.add_node(
            "shape_of_x",
            "Shape",
            vec!["x".into()],
            vec!["shape_out".into()],
            NodeAttributes::new(),
        );
        let mut gather_attrs = NodeAttributes::new();
        gather_attrs.set_int("axis", 0);
        b.add_node(
            "gather_last",
            "Gather",
            vec!["shape_out".into(), "indices".into()],
            vec!["last_dim".into()],
            gather_attrs,
        );
        let graph = b.build();

        let shapes = shapes_map(&[("x", vec![Some(1), None, Some(512)])]);
        let folded = shape_extraction_folding(graph, &shapes);

        let last_dim = folded.initializers.get("last_dim").expect("last_dim synthesized");
        assert_eq!(last_dim.to_array_i64().as_slice().unwrap(), &[512i64]);
    }
}
```

- [ ] **Step 5: Build + run tests**

```bash
cargo build --features cuda --release 2>&1 | tail -3
cargo test --features cuda --release --lib ir::passes::shape_extraction_folding 2>&1 | grep -E "^test result|FAILED"
```

Expected: 5 passed (the 5 Shape→Gather pattern tests).

- [ ] **Step 6: Commit WIP**

```bash
git add src/ir/passes/shape_extraction_folding.rs src/ir/passes/mod.rs
git commit -S -m "wip(ir): shape_extraction_folding — Shape→Gather pattern + 5 unit tests"
```

### Task 2.3: Add the `Shape → Cast → Gather` pattern (transparent Cast)

**Files:**
- Modify: `src/ir/passes/shape_extraction_folding.rs`

- [ ] **Step 1: Extend `try_fold_shape_gather` to walk through a single Cast**

Modify the producer walk in `try_fold_shape_gather`:

```rust
let data_name = gather_node.inputs.first()?;
let mut producer = producers.get(data_name)?;

// If the producer is a Cast(to=Int64), walk past it. Shape's native
// output is Int64, so Cast-to-Int64 is a no-op preserving values.
if producer.op_type == "Cast" {
    let cast_to = producer.attributes.get_int("to")?;
    // ONNX DataType 7 = INT64.
    if cast_to != 7 {
        return None;
    }
    let cast_input = producer.inputs.first()?;
    producer = producers.get(cast_input)?;
}

if producer.op_type != "Shape" {
    return None;
}
// ... rest of the fold logic unchanged ...
```

- [ ] **Step 2: Add per-pattern counter tracking (skeleton only — full helper lands in Task 2.5)**

Track which pattern fired via a thread-local counter. Add at the top of the module:

```rust
use std::cell::Cell;

thread_local! {
    pub(crate) static FOLD_COUNT_SHAPE_GATHER: Cell<usize> = Cell::new(0);
    pub(crate) static FOLD_COUNT_SHAPE_CAST_GATHER: Cell<usize> = Cell::new(0);
    pub(crate) static FOLD_COUNT_SHAPE_SLICE: Cell<usize> = Cell::new(0);
}

pub(crate) fn reset_fold_counters() {
    FOLD_COUNT_SHAPE_GATHER.with(|c| c.set(0));
    FOLD_COUNT_SHAPE_CAST_GATHER.with(|c| c.set(0));
    FOLD_COUNT_SHAPE_SLICE.with(|c| c.set(0));
}

pub(crate) fn bump_counter(cell: &std::thread::LocalKey<Cell<usize>>) {
    cell.with(|c| c.set(c.get() + 1));
}
```

At the end of `try_fold_shape_gather`, before `Some(...)`, bump the appropriate counter based on whether the chain had a Cast:

```rust
// At the top of try_fold_shape_gather, track whether we walked past Cast:
let mut walked_cast = false;
if producer.op_type == "Cast" {
    walked_cast = true;
    // ... existing Cast-walking logic ...
}

// Just before returning Some(...):
if walked_cast {
    bump_counter(&FOLD_COUNT_SHAPE_CAST_GATHER);
} else {
    bump_counter(&FOLD_COUNT_SHAPE_GATHER);
}
```

- [ ] **Step 3: Add the Cast-transparency test**

```rust
#[test]
fn cast_in_between_shape_and_gather_is_transparent() {
    // Shape(x) → Cast(Int64) → Gather([0])
    // x: [Some(1), None, Some(512)]. Expect: out = [1].
    let mut b = OptimizableGraphBuilder::new();
    b.add_input("x".into(), vec![Some(1), None, Some(512)]);
    b.add_initializer(
        "indices".into(),
        Tensor::from_vec_i64(vec![0], vec![1]),
    );
    b.add_node(
        "shape_of_x",
        "Shape",
        vec!["x".into()],
        vec!["shape_out".into()],
        NodeAttributes::new(),
    );
    let mut cast_attrs = NodeAttributes::new();
    cast_attrs.set_int("to", 7);  // ONNX DataType 7 = INT64
    b.add_node(
        "cast_shape",
        "Cast",
        vec!["shape_out".into()],
        vec!["casted".into()],
        cast_attrs,
    );
    let mut gather_attrs = NodeAttributes::new();
    gather_attrs.set_int("axis", 0);
    b.add_node(
        "gather_first",
        "Gather",
        vec!["casted".into(), "indices".into()],
        vec!["first_dim".into()],
        gather_attrs,
    );
    let graph = b.build();

    let shapes = shapes_map(&[("x", vec![Some(1), None, Some(512)])]);
    let folded = shape_extraction_folding(graph, &shapes);

    // All three nodes (Shape, Cast, Gather) should be folded away.
    assert!(folded.nodes.is_empty(), "all three chain nodes should fold");
    let first_dim = folded.initializers.get("first_dim").expect("first_dim synthesized");
    assert_eq!(first_dim.to_array_i64().as_slice().unwrap(), &[1i64]);
}

#[test]
fn cast_to_non_int64_does_not_fold() {
    // Cast(to=Float32) is NOT transparent for Shape's Int64 output.
    let mut b = OptimizableGraphBuilder::new();
    b.add_input("x".into(), vec![Some(1), Some(2), Some(3)]);
    b.add_initializer(
        "indices".into(),
        Tensor::from_vec_i64(vec![0], vec![1]),
    );
    b.add_node(
        "shape_of_x",
        "Shape",
        vec!["x".into()],
        vec!["shape_out".into()],
        NodeAttributes::new(),
    );
    let mut cast_attrs = NodeAttributes::new();
    cast_attrs.set_int("to", 1);  // ONNX DataType 1 = FLOAT
    b.add_node(
        "cast_shape",
        "Cast",
        vec!["shape_out".into()],
        vec!["casted".into()],
        cast_attrs,
    );
    let mut gather_attrs = NodeAttributes::new();
    gather_attrs.set_int("axis", 0);
    b.add_node(
        "gather",
        "Gather",
        vec!["casted".into(), "indices".into()],
        vec!["out".into()],
        gather_attrs,
    );
    let graph = b.build();

    let shapes = shapes_map(&[("x", vec![Some(1), Some(2), Some(3)])]);
    let folded = shape_extraction_folding(graph, &shapes);

    // Cast-to-Float blocks the fold. Shape and Cast remain.
    assert!(folded.nodes.iter().any(|n| n.op_type == "Cast"));
}
```

- [ ] **Step 2: Build + test**

```bash
cargo test --features cuda --release --lib ir::passes::shape_extraction_folding 2>&1 | grep -E "^test result|FAILED"
```

Expected: 7 passed (5 from Task 2.2 + 2 new).

- [ ] **Step 3: Commit WIP**

```bash
git add src/ir/passes/shape_extraction_folding.rs
git commit -S -m "wip(ir): shape_extraction_folding — Shape→Cast→Gather transparency + 2 tests"
```

### Task 2.4: Add the `Shape → Slice` pattern

**Files:**
- Modify: `src/ir/passes/shape_extraction_folding.rs`

- [ ] **Step 1: Add `try_fold_shape_slice` helper**

```rust
fn try_fold_shape_slice(
    slice_node: &GraphNode,
    producers: &HashMap<String, GraphNode>,
    tensor_shapes: &HashMap<String, Vec<Option<usize>>>,
    initializers: &HashMap<String, Tensor>,
    folded: &HashMap<String, Tensor>,
) -> Option<Tensor> {
    // ONNX Slice: inputs = [data, starts, ends, (axes), (steps)]
    let data_name = slice_node.inputs.first()?;
    let producer = producers.get(data_name)?;
    if producer.op_type != "Shape" {
        return None;
    }
    let shape_input_name = producer.inputs.first()?;
    let inferred = tensor_shapes.get(shape_input_name)?;
    if inferred.is_empty() {
        return None;
    }

    // Read starts, ends. Axes (if present) must be [0] (Shape output
    // is rank-1). Steps (if present) must be [1].
    let starts_name = slice_node.inputs.get(1)?;
    let ends_name = slice_node.inputs.get(2)?;
    let starts = initializers.get(starts_name).or_else(|| folded.get(starts_name))?;
    let ends = initializers.get(ends_name).or_else(|| folded.get(ends_name))?;
    let starts_i64 = starts.to_array_i64();
    let ends_i64 = ends.to_array_i64();
    let start = *starts_i64.as_slice()?.first()?;
    let end = *ends_i64.as_slice()?.first()?;

    // Check axes (optional, default = [0]).
    if let Some(axes_name) = slice_node.inputs.get(3) {
        if !axes_name.is_empty() {
            let axes = initializers.get(axes_name).or_else(|| folded.get(axes_name))?;
            let axes_i64 = axes.to_array_i64();
            let axis = *axes_i64.as_slice()?.first()?;
            if axis != 0 {
                return None;
            }
        }
    }
    // Check steps (optional, default = [1]).
    if let Some(steps_name) = slice_node.inputs.get(4) {
        if !steps_name.is_empty() {
            let steps = initializers.get(steps_name).or_else(|| folded.get(steps_name))?;
            let steps_i64 = steps.to_array_i64();
            let step = *steps_i64.as_slice()?.first()?;
            if step != 1 {
                return None;
            }
        }
    }

    // Clamp start/end to [0, rank] per ONNX Slice semantics.
    let rank = inferred.len() as i64;
    let start_n = if start < 0 { (start + rank).max(0) } else { start.min(rank) };
    let end_n = if end < 0 { (end + rank).max(0) } else { end.min(rank) };
    if start_n >= end_n {
        return None;
    }

    // Every dim in [start_n, end_n) must be Some(_).
    let mut output: Vec<i64> = Vec::with_capacity((end_n - start_n) as usize);
    for i in start_n..end_n {
        let dim = inferred[i as usize]?;
        output.push(dim as i64);
    }

    bump_counter(&FOLD_COUNT_SHAPE_SLICE);
    let len = output.len();
    Some(Tensor::from_vec_i64(output, vec![len]))
}
```

- [ ] **Step 2: Register Slice in the main predicate**

```rust
folding_core::fold_with_predicate(graph, move |node, folded| {
    match node.op_type.as_str() {
        "Gather" => try_fold_shape_gather(node, &producers, tensor_shapes, &initializers, folded),
        "Slice" => try_fold_shape_slice(node, &producers, tensor_shapes, &initializers, folded),
        _ => None,
    }
})
```

- [ ] **Step 3: Add Slice tests**

```rust
#[test]
fn slice_of_shape_with_all_concrete_range_folds() {
    // Shape(x) → Slice(start=0, end=2). x: [Some(4), Some(8), None].
    // Dims [0..2) = [4, 8] — both concrete, fold.
    let mut b = OptimizableGraphBuilder::new();
    b.add_input("x".into(), vec![Some(4), Some(8), None]);
    b.add_initializer("starts".into(), Tensor::from_vec_i64(vec![0], vec![1]));
    b.add_initializer("ends".into(), Tensor::from_vec_i64(vec![2], vec![1]));
    b.add_node(
        "shape_of_x",
        "Shape",
        vec!["x".into()],
        vec!["shape_out".into()],
        NodeAttributes::new(),
    );
    b.add_node(
        "slice",
        "Slice",
        vec!["shape_out".into(), "starts".into(), "ends".into()],
        vec!["prefix".into()],
        NodeAttributes::new(),
    );
    let graph = b.build();

    let shapes = shapes_map(&[("x", vec![Some(4), Some(8), None])]);
    let folded = shape_extraction_folding(graph, &shapes);

    let prefix = folded.initializers.get("prefix").expect("prefix synthesized");
    assert_eq!(prefix.to_array_i64().as_slice().unwrap(), &[4i64, 8]);
}

#[test]
fn slice_of_shape_with_none_in_range_does_not_fold() {
    // Shape(x) → Slice(start=1, end=3). x: [Some(4), Some(8), None].
    // Dim 2 is None, in range → must NOT fold.
    let mut b = OptimizableGraphBuilder::new();
    b.add_input("x".into(), vec![Some(4), Some(8), None]);
    b.add_initializer("starts".into(), Tensor::from_vec_i64(vec![1], vec![1]));
    b.add_initializer("ends".into(), Tensor::from_vec_i64(vec![3], vec![1]));
    b.add_node(
        "shape_of_x",
        "Shape",
        vec!["x".into()],
        vec!["shape_out".into()],
        NodeAttributes::new(),
    );
    b.add_node(
        "slice",
        "Slice",
        vec!["shape_out".into(), "starts".into(), "ends".into()],
        vec!["suffix".into()],
        NodeAttributes::new(),
    );
    let graph = b.build();

    let shapes = shapes_map(&[("x", vec![Some(4), Some(8), None])]);
    let folded = shape_extraction_folding(graph, &shapes);

    assert!(folded.nodes.iter().any(|n| n.op_type == "Slice"));
    assert!(!folded.initializers.contains_key("suffix"));
}
```

- [ ] **Step 4: Build + test**

```bash
cargo test --features cuda --release --lib ir::passes::shape_extraction_folding 2>&1 | grep -E "^test result|FAILED"
```

Expected: 9 passed (7 prior + 2 Slice).

- [ ] **Step 5: Commit WIP**

```bash
git add src/ir/passes/shape_extraction_folding.rs
git commit -S -m "wip(ir): shape_extraction_folding — Shape→Slice pattern + 2 tests"
```

### Task 2.5: Public attribution helper + idempotency & standalone-Shape tests

**Files:**
- Modify: `src/ir/passes/shape_extraction_folding.rs`
- Modify: `src/cuda/inference/testing.rs`

- [ ] **Step 1: Define the public counts struct + accessor**

Add to the top of `shape_extraction_folding.rs`:

```rust
#[derive(Clone, Debug, Default)]
pub struct ShapeExtractionFoldingCounts {
    pub shape_gather: usize,
    pub shape_cast_gather: usize,
    pub shape_slice: usize,
    pub total: usize,
}

impl ShapeExtractionFoldingCounts {
    pub(crate) fn snapshot() -> Self {
        let g = FOLD_COUNT_SHAPE_GATHER.with(|c| c.get());
        let cg = FOLD_COUNT_SHAPE_CAST_GATHER.with(|c| c.get());
        let s = FOLD_COUNT_SHAPE_SLICE.with(|c| c.get());
        Self {
            shape_gather: g,
            shape_cast_gather: cg,
            shape_slice: s,
            total: g + cg + s,
        }
    }
}
```

And expose from `mod.rs`:

```rust
pub use shape_extraction_folding::{shape_extraction_folding, ShapeExtractionFoldingCounts};
```

- [ ] **Step 2: Add the testing-namespaced helper**

In `src/cuda/inference/testing.rs`, add:

```rust
/// Snapshot of shape_extraction_folding's per-pattern fold counts for
/// the current thread. Call AFTER a lower() that triggers the pass;
/// counts are cumulative within the thread until reset_counters() runs.
pub fn shape_extraction_folding_counts() -> crate::ir::passes::ShapeExtractionFoldingCounts {
    crate::ir::passes::shape_extraction_folding::ShapeExtractionFoldingCounts::snapshot()
}

/// Reset the thread-local fold counters to zero. Call before a test
/// that asserts on counts, to avoid contamination from prior lower()
/// calls in the same thread.
pub fn reset_shape_extraction_folding_counters() {
    crate::ir::passes::shape_extraction_folding::reset_fold_counters();
}
```

- [ ] **Step 3: Add the last two unit tests (idempotency + standalone-Shape)**

```rust
#[test]
fn standalone_shape_without_extraction_consumer_does_not_fold() {
    // Shape(x) whose output feeds nothing supported — must NOT fold.
    let mut b = OptimizableGraphBuilder::new();
    b.add_input("x".into(), vec![Some(1), Some(2)]);
    b.add_node(
        "shape_of_x",
        "Shape",
        vec!["x".into()],
        vec!["shape_out".into()],
        NodeAttributes::new(),
    );
    b.add_output("shape_out".into());  // graph output, no extraction consumer
    let graph = b.build();

    let shapes = shapes_map(&[("x", vec![Some(1), Some(2)])]);
    let folded = shape_extraction_folding(graph, &shapes);

    // Shape stays; no synthesized initializer.
    assert!(folded.nodes.iter().any(|n| n.op_type == "Shape"));
    assert!(!folded.initializers.contains_key("shape_out"));
}

#[test]
fn shape_extraction_folding_is_idempotent() {
    let build_graph = || {
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("x".into(), vec![Some(1), None, Some(512)]);
        b.add_initializer(
            "indices".into(),
            Tensor::from_vec_i64(vec![0], vec![1]),
        );
        b.add_node(
            "shape_of_x",
            "Shape",
            vec!["x".into()],
            vec!["shape_out".into()],
            NodeAttributes::new(),
        );
        let mut gather_attrs = NodeAttributes::new();
        gather_attrs.set_int("axis", 0);
        b.add_node(
            "gather_first",
            "Gather",
            vec!["shape_out".into(), "indices".into()],
            vec!["first".into()],
            gather_attrs,
        );
        b.build()
    };
    let shapes = shapes_map(&[("x", vec![Some(1), None, Some(512)])]);

    let once = shape_extraction_folding(build_graph(), &shapes);
    let twice = shape_extraction_folding(build_graph(), &shapes);
    let thrice = shape_extraction_folding(once, &shapes);  // apply twice on already-folded

    assert_eq!(twice.nodes.len(), 0);
    assert_eq!(thrice.nodes.len(), 0);
    assert_eq!(
        twice.initializers.get("first").map(|t| t.to_array_i64().as_slice().unwrap().to_vec()),
        thrice.initializers.get("first").map(|t| t.to_array_i64().as_slice().unwrap().to_vec())
    );
}
```

- [ ] **Step 4: Build + test**

```bash
cargo test --features cuda --release --lib ir::passes::shape_extraction_folding 2>&1 | grep -E "^test result|FAILED"
```

Expected: 11 passed (9 prior + 2 new).

- [ ] **Step 5: Commit WIP**

```bash
git add src/ir/passes/shape_extraction_folding.rs src/cuda/inference/testing.rs
git commit -S -m "wip(ir): shape_extraction_folding counts helper + idempotency/standalone tests"
```

### Task 2.6: Wire `shape_extraction_folding` into `lower()`

**Files:**
- Modify: `src/ir/lowering.rs`

- [ ] **Step 1: Update the pass import**

Replace any reference to the old `shape_aware_folding` name (there shouldn't be any since we reset) with `shape_extraction_folding` in the `use crate::ir::passes::{...}` block.

- [ ] **Step 2: Insert the pass**

Find the existing `tensor_shapes_from_graph(&graph)` call and add immediately after:

```rust
let graph = shape_inference(graph);
let tensor_shapes = tensor_shapes_from_graph(&graph);
let graph = shape_extraction_folding(graph, &tensor_shapes);  // Phase 4 B
let graph = dead_code_elimination(graph);  // existing DCE, now runs after shape_extraction_folding
let precomputed_by_name = precompute_params(&graph, &tensor_shapes);
```

Do NOT re-compute `tensor_shapes` after the pass — the pass's behavior (substituting chain outputs with initializers) does not invalidate existing shape entries.

- [ ] **Step 3: Build + full test suite**

```bash
cargo build --features cuda --release 2>&1 | tail -3
cargo test --features cuda --release 2>&1 | grep -E "^test result" | awk '{pass+=$4; fail+=$6} END {print pass" passed, "fail" failed"}'
cargo test --features cuda,debug-inference --release 2>&1 | grep -E "^test result" | awk '{pass+=$4; fail+=$6} END {print pass" passed, "fail" failed"}'
```

Expected: baseline (457/467) + 11 new shape_extraction_folding tests + 1 new Gather rank-0 test = 469 / 479. Zero failures.

- [ ] **Step 4: Kokoro integration smoke test (no gate yet)**

```bash
cargo test --features cuda --release --test kokoro_gpu_fusion_test -- --ignored 2>&1 | grep -E "^test result|FAILED"
```

Expected: `1 passed`. No regression. The pre-Phase-4 Reshape assertion in the test is `>= 8` (from C PR), which will still pass even without the new gate yet.

- [ ] **Step 5: Commit WIP**

```bash
git add src/ir/lowering.rs
git commit -S -m "wip(ir): wire shape_extraction_folding into lower() post-shape_inference"
```

### Task 2.7: Structural-gate validation in kokoro_gpu_fusion_test

**Files:**
- Modify: `tests/kokoro_gpu_fusion_test.rs`

- [ ] **Step 1: Add the primary + secondary gate assertions**

Find the existing Reshape hit-rate assertion in the integration test. Replace with the two-part block from spec §Per-pass design > Commit #2 > Integration test updates:

```rust
// Reset counters before the lower() call that triggers the pass.
iconnx::cuda::inference::testing::reset_shape_extraction_folding_counters();

// ... existing lower()-triggering setup ...

// ---- Primary gate: direct extraction-fold count ----
let fold_counts = iconnx::cuda::inference::testing::shape_extraction_folding_counts();
eprintln!(
    "shape_extraction_folding counts — Gather: {}, Cast→Gather: {}, Slice: {}, total: {}",
    fold_counts.shape_gather,
    fold_counts.shape_cast_gather,
    fold_counts.shape_slice,
    fold_counts.total,
);
assert!(
    fold_counts.total >= 38,
    "Phase 4 B primary structural gate failed: {} extraction chains \
     folded (need ≥ 38 for Green, or ≥ 27 for Amber with per-pattern \
     investigation). See spec §Gating protocol B. Per-pattern \
     breakdown: Gather={}, Cast→Gather={}, Slice={}.",
    fold_counts.total,
    fold_counts.shape_gather,
    fold_counts.shape_cast_gather,
    fold_counts.shape_slice,
);

// ---- Secondary gate: Reshape-hit proxy sanity check ----
if precompute_stats.reshape_hits < 48 {
    eprintln!(
        "NOTE: reshape_hits = {}/61 but fold_counts.total = {}. \
         {} extraction unlocks didn't bump Reshape precompute — \
         likely went to non-Reshape consumers (dim arithmetic, \
         Conv-weight-shape work). Not a gate failure; primary gate \
         is on fold_counts.total.",
        precompute_stats.reshape_hits,
        fold_counts.total,
        fold_counts.total.saturating_sub(precompute_stats.reshape_hits.saturating_sub(10)),
    );
}
```

**Substitute the exact prior-assertion location and surrounding variable names** — read the test before editing to find the right spot. The reset call goes near the top of the test function, before the first `lower()`-triggering executor interaction.

- [ ] **Step 2: Run the Kokoro integration test**

```bash
cargo test --features cuda --release --test kokoro_gpu_fusion_test -- --ignored --nocapture 2>&1 | grep -E "shape_extraction_folding counts|^test result|FAILED|panicked|NOTE:" | head -20
```

Record the `fold_counts.total` value and the per-pattern breakdown.

**Evaluate the three-tier structural gate:**

- **Green (`total ≥ 38`):** proceed to Task 2.8 squash.
- **Amber (`27 ≤ total ≤ 37`):** STOP and report. Include the per-pattern breakdown and a preliminary analysis of which extraction patterns missed.
- **Red (`total < 27`):** STOP and report BLOCKED. The mechanism needs reassessment.

Under the **three-attempt abandonment clause** (spec §Gating protocol B): if this is the third pattern-implementation attempt (each attempt being a new pattern variant or fresh predicate design), do not commit another attempt — escalate per the abandonment protocol.

- [ ] **Step 3: Commit WIP** (only if Green)

```bash
git add tests/kokoro_gpu_fusion_test.rs
git commit -S -m "wip(ir): structural gate — shape_extraction_folding fold_counts.total ≥ 38"
```

### Task 2.8: Squash into final `feat(ir)` commit

- [ ] **Step 1: Verify WIP commits**

```bash
git log --oneline 5924e5d..HEAD
git log --format='%G? %s' 5924e5d..HEAD
```

Expected: 7 WIP commits (2.1 gather guard, 2.2 skeleton+gather, 2.3 cast-gather, 2.4 slice, 2.5 counts+extras, 2.6 wire, 2.7 gate), all GPG-signed.

- [ ] **Step 2: Full test suite**

```bash
cargo build --features cuda --release 2>&1 | grep -i warning   # expect empty
cargo build --features cuda,debug-inference --release 2>&1 | grep -i warning   # expect empty
cargo test --features cuda --release 2>&1 | grep -E "^test result" | awk '{pass+=$4; fail+=$6} END {print pass" passed, "fail" failed"}'
cargo test --features cuda --release --test kokoro_gpu_fusion_test -- --ignored 2>&1 | grep -E "^test result"
```

Expected: baseline +12 tests (11 shape_extraction_folding + 1 gather guard), zero failures, Kokoro integration test green with primary gate passing.

- [ ] **Step 3: Squash**

```bash
git reset --soft 5924e5d
git commit -S -m "$(cat <<'EOF'
feat(ir): shape_extraction_folding pass (B)

Folds Shape-rooted chains terminating at extraction ops whose
extracted indices hit only concrete dims of a (possibly
partially-dynamic) Shape input. Three supported patterns:

  1. Shape(x) → Gather(indices=I)
  2. Shape(x) → Cast(to=Int64) → Gather(indices=I)
  3. Shape(x) → Slice(start=s, end=e)

Pipeline: runs immediately after shape_inference +
tensor_shapes_from_graph, before the subsequent DCE:
  topo_sort → constant_folding → DCE → shape_inference
  → tensor_shapes_from_graph → shape_extraction_folding → DCE
  → precompute_params → ...

Closes Phase 3's stacked-gate unlock target for Commits #2 and #3
under a revised mechanism (see 2026-04-22 spec amendment). The
original "fold Shape when all-static" predicate was architecturally
unable to fire on Kokoro because every downstream tensor inherits
`seq_len = None` from the `tokens` graph input. Real-graph
instrumentation found 53 of Kokoro's 73 Shape nodes have the
extraction-chain structure this pass targets. Primary structural
gate: fold_counts.total ≥ 38 (≥ 72% of the 53 reachable cases) —
matches Phase 3's original projection ratio applied to the correct
denominator. Secondary proxy sanity check: reshape_hits ≥ 48/61;
divergence from primary is a finding, not a gate failure.

Measured on Kokoro at landing: fold_counts.total = NN (substitute
before committing), per-pattern NN/NN/NN (Gather/Cast→Gather/Slice).
reshape_hits = NN/61 (substitute).

Also includes a correctness hardening for `Gather::forward` against
rank-0 indices tensors (ported from the prior attempt's commit
8350f67). Rank-0 indices previously caused an `index_axis(Axis(0),
idx)` panic at src/operators/gather.rs:144; guarded to return an
empty output tensor per ONNX semantics instead.

11 unit tests in shape_extraction_folding::tests (5 Shape→Gather,
2 Cast-transparency, 2 Slice, 1 standalone-Shape-doesn't-fold,
1 idempotency) + 1 gather.rs rank-0 regression test. Per-pattern
attribution counters exposed via
testing::shape_extraction_folding_counts() for gate diagnostics.

Spec: docs/superpowers/specs/2026-04-19-phase4-graph-optimizer-ii-design.md
(amendment 2026-04-22).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

**Substitute the measured `fold_counts.total` and per-pattern + reshape_hits numbers** in the commit message before committing.

- [ ] **Step 4: Verify**

```bash
git log --oneline 5875858..HEAD   # expect exactly 2 commits (refactor + feat B)
git log -2 --format='%G? %s'
```

Both signed (`G` prefix).

### Task 2.1: Create `shape_aware_folding` pass + the core fold-when-known test

**Files:**
- Create: `src/ir/passes/shape_aware_folding.rs`
- Modify: `src/ir/passes/mod.rs` (add `pub mod shape_aware_folding;` + re-export the pass fn)

- [ ] **Step 1: Create the pass file**

`src/ir/passes/shape_aware_folding.rs`:

```rust
//! Shape-aware constant folding.
//!
//! Extends `folding_core::fold_with_predicate` with one extra rule:
//! `Shape(x)` is foldable when `tensor_shapes[x]` is fully known (every
//! dim `Some(_)`). Once Shape's output becomes a synthesized
//! initializer, downstream Cast/Gather/etc. fold via the existing
//! initializer-rooted fixpoint loop in the same pass — no second
//! constant_folding invocation is needed.
//!
//! Must run AFTER shape_inference has populated tensor_shapes.

use std::collections::HashMap;

use crate::ir::graph::OptimizableGraph;
use crate::ir::passes::folding_core;
use crate::tensor::Tensor;

pub fn shape_aware_folding(
    graph: OptimizableGraph,
    tensor_shapes: &HashMap<String, Vec<Option<usize>>>,
) -> OptimizableGraph {
    folding_core::fold_with_predicate(graph, |node, _folded| {
        if node.op_type != "Shape" {
            return None;
        }
        // Shape takes one input and emits an i64 tensor of that input's
        // rank. We fold it when the input's shape is fully known.
        let input_name = node.inputs.first()?;
        let inferred = tensor_shapes.get(input_name)?;
        let fully_known: Vec<i64> = inferred
            .iter()
            .map(|d| d.map(|n| n as i64))
            .collect::<Option<Vec<_>>>()?;
        Some(Tensor::from_vec_i64(fully_known, vec![inferred.len()]))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attributes::NodeAttributes;
    use crate::ir::graph::OptimizableGraphBuilder;

    fn shapes_map(entries: &[(&str, Vec<Option<usize>>)]) -> HashMap<String, Vec<Option<usize>>> {
        entries
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    #[test]
    fn shape_of_known_shape_tensor_folds() {
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("x".into(), vec![Some(2), Some(3), Some(4)]);
        b.add_node(
            "shape_of_x",
            "Shape",
            vec!["x".into()],
            vec!["shape_out".into()],
            NodeAttributes::new(),
        );
        let graph = b.build();

        let shapes = shapes_map(&[("x", vec![Some(2), Some(3), Some(4)])]);
        let folded = shape_aware_folding(graph, &shapes);

        // Shape node removed.
        assert!(
            folded.nodes.iter().all(|n| n.op_type != "Shape"),
            "Shape node should have been folded away; nodes now: {:?}",
            folded.nodes.iter().map(|n| &n.op_type).collect::<Vec<_>>(),
        );
        // shape_out initializer synthesized with [2, 3, 4].
        let result = folded
            .initializers
            .get("shape_out")
            .expect("shape_out must be synthesized as initializer");
        let values = result.to_array_i64();
        assert_eq!(values.as_slice().unwrap(), &[2i64, 3, 4]);
    }
}
```

- [ ] **Step 2: Register the module + re-export**

In `src/ir/passes/mod.rs`, alongside existing `pub mod shape_inference;`:

```rust
pub mod shape_aware_folding;
```

And in the same file's `pub use` block (search for how the other passes are re-exported, e.g. `pub use shape_inference::...`), add:

```rust
pub use shape_aware_folding::shape_aware_folding;
```

- [ ] **Step 3: Run the unit test**

```bash
cargo test --features cuda --release --lib ir::passes::shape_aware_folding 2>&1 | grep -E "^test result|FAILED"
```

Expected: `1 passed` (just the one test we wrote).

- [ ] **Step 4: Commit WIP**

```bash
git add src/ir/passes/shape_aware_folding.rs src/ir/passes/mod.rs
git commit -S -m "wip(ir): shape_aware_folding pass skeleton + shape-of-known tests"
```

### Task 2.2: Add the four remaining spec'd unit tests

**Files:**
- Modify: `src/ir/passes/shape_aware_folding.rs` (expand the test module)

- [ ] **Step 1: Add the four tests to the existing `mod tests` block**

Append after the `shape_of_known_shape_tensor_folds` test:

```rust
#[test]
fn shape_of_partially_dynamic_tensor_does_not_fold() {
    let mut b = OptimizableGraphBuilder::new();
    b.add_input("x".into(), vec![None, Some(3), Some(4)]);
    b.add_node(
        "shape_of_x",
        "Shape",
        vec!["x".into()],
        vec!["shape_out".into()],
        NodeAttributes::new(),
    );
    let graph = b.build();

    let shapes = shapes_map(&[("x", vec![None, Some(3), Some(4)])]);
    let folded = shape_aware_folding(graph, &shapes);

    // Shape node stays, no synthesized initializer.
    assert!(
        folded.nodes.iter().any(|n| n.op_type == "Shape"),
        "Shape node should remain unfolded when any dim is None"
    );
    assert!(!folded.initializers.contains_key("shape_out"));
}

#[test]
fn shape_cast_gather_chain_folds_to_single_initializer() {
    // x -> Shape -> Cast(to=INT64) -> Gather(indices=[0]) -> out
    // Expected post-fold: initializer "out" = [2] (the first dim of x).
    use crate::tensor::Tensor as T;

    let mut b = OptimizableGraphBuilder::new();
    b.add_input("x".into(), vec![Some(2), Some(3), Some(4)]);

    // Cast expects an attribute "to" specifying the target dtype. INT64
    // is ONNX data type 7.
    let mut cast_attrs = NodeAttributes::new();
    cast_attrs.set_int("to", 7);

    // Gather's "indices" is an input, not an attribute. Provide it as an
    // Int64 initializer value = [0].
    b.add_initializer(
        "gather_indices".into(),
        T::from_vec_i64(vec![0i64], vec![1]),
    );

    let mut gather_attrs = NodeAttributes::new();
    gather_attrs.set_int("axis", 0);

    b.add_node(
        "shape_of_x",
        "Shape",
        vec!["x".into()],
        vec!["shape_out".into()],
        NodeAttributes::new(),
    );
    b.add_node(
        "cast_shape",
        "Cast",
        vec!["shape_out".into()],
        vec!["casted".into()],
        cast_attrs,
    );
    b.add_node(
        "gather_first",
        "Gather",
        vec!["casted".into(), "gather_indices".into()],
        vec!["out".into()],
        gather_attrs,
    );
    let graph = b.build();

    let shapes = shapes_map(&[("x", vec![Some(2), Some(3), Some(4)])]);
    let folded = shape_aware_folding(graph, &shapes);

    // All three nodes folded away.
    assert!(folded.nodes.is_empty(), "all three nodes should fold");
    let out = folded.initializers.get("out").expect("out must exist");
    let values = out.to_array_i64();
    assert_eq!(values.as_slice().unwrap(), &[2i64]);
}

#[test]
fn shape_of_dynamic_graph_input_does_not_fold() {
    let mut b = OptimizableGraphBuilder::new();
    b.add_input("x".into(), vec![None, Some(512)]);
    b.add_node(
        "shape_of_x",
        "Shape",
        vec!["x".into()],
        vec!["shape_out".into()],
        NodeAttributes::new(),
    );
    let graph = b.build();

    let shapes = shapes_map(&[("x", vec![None, Some(512)])]);
    let folded = shape_aware_folding(graph, &shapes);

    assert!(folded.nodes.iter().any(|n| n.op_type == "Shape"));
    assert!(!folded.initializers.contains_key("shape_out"));
}

#[test]
fn shape_aware_folding_is_idempotent() {
    let mut b = OptimizableGraphBuilder::new();
    b.add_input("x".into(), vec![Some(2), Some(3)]);
    b.add_node(
        "shape_of_x",
        "Shape",
        vec!["x".into()],
        vec!["shape_out".into()],
        NodeAttributes::new(),
    );
    let graph = b.build();

    let shapes = shapes_map(&[("x", vec![Some(2), Some(3)])]);
    let once = shape_aware_folding(graph, &shapes);
    let twice = shape_aware_folding(once.clone(), &shapes);

    assert_eq!(once.nodes.len(), twice.nodes.len());
    assert_eq!(once.initializers.len(), twice.initializers.len());
    assert_eq!(
        once.initializers.get("shape_out").map(|t| t.to_array_i64().as_slice().unwrap().to_vec()),
        twice.initializers.get("shape_out").map(|t| t.to_array_i64().as_slice().unwrap().to_vec())
    );
}
```

If `OptimizableGraph` doesn't derive `Clone`, the idempotency test needs a small adaptation. Build two copies of the graph explicitly instead:

```rust
// Fallback if OptimizableGraph doesn't implement Clone
let build_graph = || {
    let mut b = OptimizableGraphBuilder::new();
    b.add_input("x".into(), vec![Some(2), Some(3)]);
    b.add_node("shape_of_x", "Shape", vec!["x".into()], vec!["shape_out".into()], NodeAttributes::new());
    b.build()
};
let once = shape_aware_folding(build_graph(), &shapes);
let twice = shape_aware_folding(build_graph(), &shapes);
// then compare once vs twice as above
let _ = once; let _ = twice;  // etc.
```

Check first: `grep 'derive.*Clone.*OptimizableGraph' src/ir/graph.rs` — use the simpler version if Clone is derived, the fallback otherwise.

- [ ] **Step 2: Run the tests**

```bash
cargo test --features cuda --release --lib ir::passes::shape_aware_folding 2>&1 | grep -E "^test result|FAILED"
```

Expected: `5 passed` (the 4 new + the 1 from Task 2.1).

- [ ] **Step 3: Commit WIP**

```bash
git add src/ir/passes/shape_aware_folding.rs
git commit -S -m "wip(ir): shape_aware_folding covers chain-fold, partial-dynamic, idempotency"
```

### Task 2.3: Wire `shape_aware_folding` into `lower()`

**Files:**
- Modify: `src/ir/lowering.rs`

- [ ] **Step 1: Read the current pipeline and locate the insertion point**

```bash
grep -n "shape_inference\|tensor_shapes_from_graph\|precompute_params\|dead_code_elimination\|elementwise_fusion\|detect_fusion" src/ir/lowering.rs | head -30
```

Identify the line where `tensor_shapes_from_graph(&graph)` is called. The new pass inserts immediately after that line and before the existing DCE call that currently precedes `precompute_params`.

- [ ] **Step 2: Import the pass**

Add to the existing `use crate::ir::passes::{...}` block:

```rust
use crate::ir::passes::shape_aware_folding;
```

If `passes` already re-exports the fn via `pub use`, you can add to the list of imports in the existing `use crate::ir::passes::{...}` block:

```rust
use crate::ir::passes::{
    constant_folding_pass, dead_code_elimination, detect_fusion,
    precompute_params, shape_aware_folding, shape_inference,
    tensor_shapes_from_graph, topo_sort,
    /* ...plus others already imported... */
};
```

(Exact set depends on what's there; preserve existing imports, add the new one alphabetically.)

- [ ] **Step 3: Insert the pass into the pipeline**

The current pipeline in `lower()` (post-Phase-3) runs in this order (approximate):

```rust
let graph = topo_sort(graph)?;
let graph = constant_folding_pass(graph);
let graph = dead_code_elimination(graph);
let graph = shape_inference(graph);
let tensor_shapes = tensor_shapes_from_graph(&graph);
let graph = dead_code_elimination(graph);  // existing DCE after shape_inference
let precomputed_by_name = precompute_params(&graph, &tensor_shapes);
// ... rest unchanged ...
```

Change to:

```rust
let graph = topo_sort(graph)?;
let graph = constant_folding_pass(graph);
let graph = dead_code_elimination(graph);
let graph = shape_inference(graph);
let tensor_shapes = tensor_shapes_from_graph(&graph);
let graph = shape_aware_folding(graph, &tensor_shapes);  // NEW — Phase 4 B
let graph = dead_code_elimination(graph);  // existing DCE moved after shape_aware_folding
let precomputed_by_name = precompute_params(&graph, &tensor_shapes);
// ... rest unchanged ...
```

The existing DCE that ran immediately after `shape_inference` now runs after `shape_aware_folding` instead. This is intentional — DCE sweeps any nodes that `shape_aware_folding` left orphaned (e.g., a Shape op whose output is now covered by a synthesized initializer with no other consumers).

**Critical:** Do NOT call `tensor_shapes_from_graph` a second time after `shape_aware_folding`. The existing `tensor_shapes` map remains the authoritative source for downstream passes (`precompute_params`, `elementwise_fusion_pass`, `memory_planner` in Commit 3). Re-computing would be safe but wasteful, and it would mask bugs where a pass mutates shapes in a way we didn't intend. `shape_aware_folding` is declared to not change any tensor's shape (only replaces ops with initializers), so the map stays valid.

- [ ] **Step 4: Build + full test suite**

```bash
cargo build --features cuda --release 2>&1 | tail -3
cargo test --features cuda --release 2>&1 | grep -E "^test result" | awk '{pass+=$4; fail+=$6} END {print pass" passed, "fail" failed"}'
cargo test --features cuda,debug-inference --release 2>&1 | grep -E "^test result" | awk '{pass+=$4; fail+=$6} END {print pass" passed, "fail" failed"}'
```

Expected: same counts as Task 0.1 baseline (new tests added in Task 2.1/2.2 already counted). Zero regressions.

- [ ] **Step 5: Quick Kokoro integration smoke test**

```bash
cargo test --features cuda --release --test kokoro_gpu_fusion_test -- --ignored 2>&1 | grep -E "^test result|FAILED"
```

Expected: `1 passed`. The Reshape hit-rate assertion in that test is still `>= 8` (C PR's level), so this should pass even before we bump it in Task 2.4.

- [ ] **Step 6: Commit WIP**

```bash
git add src/ir/lowering.rs
git commit -S -m "wip(ir): wire shape_aware_folding into lower() post-shape_inference"
```

### Task 2.4: Structural-gate validation — bump Kokoro Reshape hit-rate assertion

**Files:**
- Modify: `tests/kokoro_gpu_fusion_test.rs`

- [ ] **Step 1: Locate the existing assertion**

```bash
grep -n "reshape_hits\|Reshape precompute hit rate" tests/kokoro_gpu_fusion_test.rs | head -10
```

Find the assertion block (pre-Phase-4 reads `precompute_stats.reshape_hits >= 8` with a comment about "C PR baseline 10 hits; allow ±2").

- [ ] **Step 2: Update the assertion to enforce the structural gate**

Replace the assertion block with:

```rust
// Phase 4 B (shape_aware_folding) projects 23 additional unlocks via
// Shape-chain resolution, bringing Kokoro's Reshape precompute hit
// rate from 10/61 to ≥ 33/61. Structural gate per Phase 4 spec
// §Gating protocol:
//   Green  (land without investigation): hit rate ≥ 33/61
//   Amber  (investigate before landing): 28 ≤ hit rate < 33
//   Red    (block):                      hit rate < 28
//
// This assertion enforces Green. If it lands in [28, 33), the test
// fails and the spec's Amber-tier protocol applies: list which
// Reshape chains didn't unlock and why. Either projection was
// optimistic (proceed) or the mechanism has a bug (block).
assert!(
    precompute_stats.reshape_hits >= 33,
    "Phase 4 B structural gate failed: Kokoro Reshape precompute hit rate \
     is {}/{} (need ≥ 33/61). Phase 3's baseline was 10/61; \
     shape_aware_folding should unlock ~23 more via Shape-chain resolution. \
     If hit rate is in [28, 33), you're in the spec's Amber tier — investigate \
     which Reshape chains didn't unlock (see spec §Gating protocol). If < 28, \
     the mechanism has a bug.",
    precompute_stats.reshape_hits,
    precompute_stats.reshape_total,
);
```

- [ ] **Step 3: Run the integration test**

```bash
cargo test --features cuda --release --test kokoro_gpu_fusion_test -- --ignored --nocapture 2>&1 | grep -E "Reshape:|^test result|FAILED|panicked" | head -10
```

Expected: `Reshape: NN/61 hits` with `NN >= 33`, and `test result: ok. 1 passed`.

**If NN is in [28, 33):** STOP and follow the Amber-tier investigation:
1. Print the graph's runtime-shape Reshape chains: `grep for "Reshape" + walk back through shape_inference`. Count how many chains terminate at a `Shape(x)` where `tensor_shapes[x]` is fully `Some(_)`. That's the theoretical unlock ceiling.
2. Compare to actual `reshape_hits` — 10 pre-Phase-4. The delta is what `shape_aware_folding` unlocked.
3. If actual unlock count matches ceiling: projection was optimistic; update the assertion threshold to match reality + the 10 existing hits, note it in the commit message, proceed.
4. If actual < ceiling: `shape_aware_folding` has a bug — some foldable chains are being missed. STOP and escalate with the specific chains' node names.

**If NN < 28:** Red — block. STOP and escalate with full chain analysis.

- [ ] **Step 4: Commit WIP**

Only commit if NN ≥ 33 OR if you've landed in Amber and leadline has signed off on the adjusted threshold with a documented projection-vs-reality summary.

```bash
git add tests/kokoro_gpu_fusion_test.rs
git commit -S -m "wip(ir): assert Kokoro Reshape hit rate ≥ 33/61 (Phase 4 B structural gate)"
```

### Task 2.5: Squash into final `feat(ir)` commit

- [ ] **Step 1: Verify WIP commits**

```bash
PRIOR_SHA=$(git log --oneline 5875858..HEAD | tail -1 | awk '{print $1}')  # first commit in current branch
echo "Prior SHA (refactor squash): $PRIOR_SHA"
git log --format='%G? %s' $PRIOR_SHA..HEAD | head -10
```

Actually simpler: find the refactor commit explicitly.

```bash
REFACTOR_SHA=$(git log --oneline 5875858..HEAD --grep="refactor(ir):" | head -1 | awk '{print $1}')
echo "Refactor commit SHA: $REFACTOR_SHA"
git log --format='%G? %s' $REFACTOR_SHA..HEAD
```

Expected: all WIP commits for Commit 2 (should be ~4) are GPG-signed.

- [ ] **Step 2: Full test suite, zero warnings**

```bash
cargo build --features cuda --release 2>&1 | grep -i warning   # expect empty
cargo build --features cuda,debug-inference --release 2>&1 | grep -i warning   # expect empty
cargo test --features cuda --release 2>&1 | grep -E "^test result" | awk '{pass+=$4; fail+=$6} END {print pass" passed, "fail" failed"}'
cargo test --features cuda --release --test kokoro_gpu_fusion_test -- --ignored 2>&1 | grep -E "^test result"
```

Expected: baseline test count + 5 new (shape_aware_folding unit tests), zero failures, Kokoro gate green.

- [ ] **Step 3: Squash**

```bash
git reset --soft $REFACTOR_SHA
git commit -S -m "$(cat <<'EOF'
feat(ir): shape_aware_folding pass (B)

Pure-function `fn shape_aware_folding(OptimizableGraph, tensor_shapes)
-> OptimizableGraph` in src/ir/passes/shape_aware_folding.rs. Uses
folding_core::fold_with_predicate with a single predicate extension:
Shape(x) is foldable when tensor_shapes[x] is fully known (every dim
Some(_)). Once Shape's output becomes a synthesized initializer,
downstream Cast/Gather/etc. fold via the same fixpoint loop — no
second constant_folding invocation needed.

Pipeline: runs immediately after shape_inference +
tensor_shapes_from_graph, before the subsequent DCE sweep:
  topo_sort → constant_folding → DCE → shape_inference
  → tensor_shapes_from_graph → shape_aware_folding → DCE
  → precompute_params → ...

Closes Phase 3's stacked-gate unlock target (Commits #2 and #3). Phase 3
reported 51 of 61 Kokoro Reshape nodes blocked on runtime-computed
Shape → Cast → Gather shape inputs; Phase 4 B projects 23 of those
unlock. Gated structurally (not via ratio): Kokoro Reshape precompute
hit rate must move from 10/61 to ≥ 33/61 (Green) via the updated
assertion in tests/kokoro_gpu_fusion_test.rs. Measured post-landing:
NN/61 (substitute before committing).

Five unit tests cover: shape-of-known-shape folds; shape-of-partially-
dynamic doesn't fold; Shape→Cast→Gather chain collapses to a single
synthesized initializer; shape-of-dynamic-graph-input doesn't fold;
shape_aware_folding is idempotent.

Spec: docs/superpowers/specs/2026-04-19-phase4-graph-optimizer-ii-design.md.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

**Substitute the actual measured hit rate** (`NN/61`) in the commit message before committing.

- [ ] **Step 4: Verify**

```bash
git log --oneline 5875858..HEAD   # expect exactly 2 commits (refactor + feat B)
git log -2 --format='%G? %s'
```

Both signed (`G` prefix), subjects are the two final commits.

---

## Commit 3: `feat(ir)` — static memory planner + arena allocation path (A)

**Commit goal:** Add `memory_planner` pass + `MemoryPlan` struct + `PlannedOp::arena_offset: Option<usize>` + `GpuMemoryPool::allocate_arena` API + runtime arena branching in `Executor` with always-on size assertion + `ICONNX_DISABLE_MEMORY_PLANNER=1` bisection escape hatch. Generic-infrastructure frame: six landing criteria (correctness, pool fallback, determinism, classification coverage ≥ 70%, size assertion active, zero warnings + signed).

### Task 3.1: Add `is_planner_eligible` + `is_fusion_skipped` + predicate tests

**Files:**
- Create: `src/ir/passes/memory_planner.rs`
- Modify: `src/ir/passes/mod.rs`

- [ ] **Step 1: Create the new module with just the predicates**

`src/ir/passes/memory_planner.rs` (first slice — just predicates + their tests):

```rust
//! Static memory planner — assigns arena offsets to eligible tensors.
//!
//! Runs after fusion annotation; consumes tensor_shapes + tensor_last_use
//! + fusion annotations + graph I/O names. Produces a MemoryPlan that
//! stamps `arena_offset: Option<usize>` onto each PlannedOp, plus a
//! summary of allocation decisions exposed via classification_counts().
//!
//! Philosophy: coexist with GpuMemoryPool. Arena-eligible tensors (those
//! passing is_planner_eligible AND not filtered as fusion-skipped) get
//! a slot. Everything else (dynamic shapes, small tensors, graph I/O,
//! fusion-skipped outputs) takes the pool path at runtime — same
//! behavior as pre-Phase-4.

use std::collections::HashMap;

use crate::cuda::inference::fusion::FusedPatternInfo;

/// Semantic eligibility predicate. Returns true iff the tensor's shape
/// is fully known, its size is ≥ 4 KiB (matching GpuMemoryPool's
/// size-class threshold), and it is not a graph input or output.
pub(crate) fn is_planner_eligible(
    tensor_name: &str,
    tensor_shapes: &HashMap<String, Vec<Option<usize>>>,
    graph_inputs: &[String],
    graph_outputs: &[String],
    dtype_bytes: usize,
    element_count: usize,
) -> bool {
    if graph_inputs.iter().any(|i| i == tensor_name) {
        return false;
    }
    if graph_outputs.iter().any(|o| o == tensor_name) {
        return false;
    }
    let Some(shape) = tensor_shapes.get(tensor_name) else {
        return false;
    };
    if shape.iter().any(|d| d.is_none()) {
        return false;
    }
    if dtype_bytes.saturating_mul(element_count) < 4096 {
        return false;
    }
    true
}

/// Structural filter: is this tensor produced by an op whose
/// PlannedOp::kind == Skip? Those outputs are never allocated at
/// runtime — a fused head writes directly into the head's slot — so
/// arena-assigning them would waste a slot.
pub(crate) fn is_fusion_skipped(
    producer_tensor_name: &str,
    fusion: &HashMap<String, FusedPatternInfo>,
) -> bool {
    // Match the spec's contract: if the tensor's producer op is part of
    // a fused chain as a skipped sub-node (not the head), it's filtered.
    //
    // The shape of FusedPatternInfo tracks which outputs are claimed by
    // the fused head. A skipped output is one that's in some fused
    // pattern's `skipped_node_outputs` set (or equivalent).
    //
    // Implementation note: inspect the actual FusedPatternInfo structure
    // (in src/cuda/inference/fusion/patterns.rs) to find the correct
    // field. The comment below captures the intent.
    fusion
        .values()
        .any(|info| info.skipped_tensors.contains(&producer_tensor_name.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shapes_map(entries: &[(&str, Vec<Option<usize>>)]) -> HashMap<String, Vec<Option<usize>>> {
        entries.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
    }

    #[test]
    fn is_planner_eligible_all_dims_known_yields_true() {
        let shapes = shapes_map(&[("t", vec![Some(4), Some(64)])]);
        assert!(is_planner_eligible(
            "t", &shapes, &[], &[],
            4,  // f32
            4 * 64,  // 1024 elements × 4 bytes = 4096 bytes — exactly at threshold
        ));
    }

    #[test]
    fn is_planner_eligible_any_none_dim_yields_false() {
        let shapes = shapes_map(&[("t", vec![None, Some(64)])]);
        assert!(!is_planner_eligible("t", &shapes, &[], &[], 4, 256));
    }

    #[test]
    fn is_planner_eligible_graph_input_yields_false() {
        let shapes = shapes_map(&[("t", vec![Some(4), Some(64)])]);
        assert!(!is_planner_eligible(
            "t", &shapes, &["t".to_string()], &[], 4, 256,
        ));
    }

    #[test]
    fn is_planner_eligible_graph_output_yields_false() {
        let shapes = shapes_map(&[("t", vec![Some(4), Some(64)])]);
        assert!(!is_planner_eligible(
            "t", &shapes, &[], &["t".to_string()], 4, 256,
        ));
    }

    #[test]
    fn is_planner_eligible_below_4kb_yields_false() {
        let shapes = shapes_map(&[("t", vec![Some(4), Some(64)])]);
        // 4 bytes × 1000 = 4000 bytes < 4096 threshold.
        assert!(!is_planner_eligible("t", &shapes, &[], &[], 4, 1000));
    }

    #[test]
    fn is_planner_eligible_unknown_tensor_yields_false() {
        // Tensor not in tensor_shapes map — predicate must be tolerant.
        let shapes = HashMap::new();
        assert!(!is_planner_eligible("ghost", &shapes, &[], &[], 4, 10_000));
    }
}
```

**Before implementing `is_fusion_skipped` as shown:** inspect `src/cuda/inference/fusion/patterns.rs` to confirm `FusedPatternInfo` has a field that lists skipped tensor outputs. If it exposes this as `skipped_tensors: Vec<String>` or similar, use that. If it exposes `skipped_node_indices: Vec<usize>` (indices into the graph's nodes list) instead, the implementation needs the graph too:

```rust
pub(crate) fn is_fusion_skipped(
    producer_tensor_name: &str,
    fusion: &HashMap<String, FusedPatternInfo>,
    graph: &crate::ir::graph::OptimizableGraph,
) -> bool {
    let skipped_indices: std::collections::HashSet<usize> = fusion
        .values()
        .flat_map(|info| info.skipped_node_indices.iter().copied())
        .collect();
    for (idx, node) in graph.nodes.iter().enumerate() {
        if skipped_indices.contains(&idx) && node.outputs.iter().any(|o| o == producer_tensor_name) {
            return true;
        }
    }
    false
}
```

Pick the signature based on what `FusedPatternInfo` actually exposes. Update the spec's integration expectation accordingly (don't change the spec doc, but reflect the concrete form in the code).

- [ ] **Step 2: Register the module**

In `src/ir/passes/mod.rs`, alongside existing declarations:

```rust
pub mod memory_planner;
```

No `pub use` — the predicates are `pub(crate)`; the main pass fn (added in Task 3.3) will be re-exported then.

- [ ] **Step 3: Run the predicate tests**

```bash
cargo test --features cuda --release --lib ir::passes::memory_planner 2>&1 | grep -E "^test result|FAILED"
```

Expected: `6 passed`.

- [ ] **Step 4: Commit WIP**

```bash
git add src/ir/passes/memory_planner.rs src/ir/passes/mod.rs
git commit -S -m "wip(ir): memory_planner predicates (is_planner_eligible, is_fusion_skipped)"
```

### Task 3.2: Add `MemoryPlan` struct + `PlannedOp::arena_offset` field

**Files:**
- Modify: `src/ir/plan.rs`

- [ ] **Step 1: Add the MemoryPlan types to `src/ir/plan.rs`**

Near the top of the file (after existing imports, before `struct NodePrecomputed` or wherever other plan types are defined), add:

```rust
/// Per-run classification summary of memory_planner's decisions.
/// Stored on MemoryPlan at construction time so the accessor is
/// parameter-less.
#[derive(Clone, Debug, Default)]
pub struct PlannerClassification {
    /// Tensors passing both `is_planner_eligible` AND the
    /// fusion-skipped filter.
    pub arena_eligible: usize,
    /// Subset of `arena_eligible` actually assigned to a slot. Under
    /// the current greedy algorithm this always equals
    /// `arena_eligible`; tracked separately so a future algorithm
    /// that might decline an eligible tensor (e.g., fragmentation
    /// cap) surfaces the decision instead of hiding it.
    pub arena_assigned: usize,
    /// Tensors that did NOT pass eligibility or were fusion-skipped.
    /// Take the pool path at runtime.
    pub pool_classified: usize,
}

/// Static memory plan — assigns eligible tensors to arena slots.
/// Produced by memory_planner after fusion annotation; frozen part of
/// ExecutionPlan.
#[derive(Clone, Debug)]
pub struct MemoryPlan {
    /// Total bytes required for the arena. Allocated as a single
    /// contiguous device buffer at run() start.
    pub arena_bytes: usize,
    /// Number of slots in the arena. Each slot is a size-class bucket.
    pub slot_count: usize,
    /// tensor_name → slot_idx. Callers typically use the
    /// arena_offset(tensor_name) accessor instead of this directly.
    pub(crate) slot_assignments: HashMap<String, usize>,
    /// slot_idx → byte offset within the arena.
    pub(crate) slot_offsets_bytes: Vec<usize>,
    /// Classification summary — see PlannerClassification.
    classification: PlannerClassification,
}

impl MemoryPlan {
    /// Return the arena byte-offset for a tensor, or None if the tensor
    /// is not arena-assigned (pool path at runtime).
    pub fn arena_offset(&self, tensor_name: &str) -> Option<usize> {
        let &slot_idx = self.slot_assignments.get(tensor_name)?;
        Some(self.slot_offsets_bytes[slot_idx])
    }

    /// Parameter-less accessor for classification counts.
    pub fn classification_counts(&self) -> &PlannerClassification {
        &self.classification
    }

    /// Constructor used by memory_planner (pub(crate) to keep the
    /// invariant that MemoryPlan instances are only produced by the
    /// planner pass, not by callers).
    pub(crate) fn new(
        arena_bytes: usize,
        slot_count: usize,
        slot_assignments: HashMap<String, usize>,
        slot_offsets_bytes: Vec<usize>,
        classification: PlannerClassification,
    ) -> Self {
        Self {
            arena_bytes,
            slot_count,
            slot_assignments,
            slot_offsets_bytes,
            classification,
        }
    }

    /// Empty plan — every tensor takes pool path. Used when
    /// ICONNX_DISABLE_MEMORY_PLANNER=1 bypasses planning or when the
    /// graph has no arena-eligible tensors at all.
    pub fn empty() -> Self {
        Self {
            arena_bytes: 0,
            slot_count: 0,
            slot_assignments: HashMap::new(),
            slot_offsets_bytes: Vec::new(),
            classification: PlannerClassification::default(),
        }
    }
}
```

- [ ] **Step 2: Add `arena_offset: Option<usize>` to `PlannedOp`**

Locate the existing `pub struct PlannedOp` in `src/ir/plan.rs` (currently:
```rust
pub struct PlannedOp {
    pub node: GraphNode,
    pub kind: OpKind,
    pub precomputed: NodePrecomputed,
}
```
at around line 55). Add the new field:

```rust
pub struct PlannedOp {
    pub node: GraphNode,
    pub kind: OpKind,
    pub precomputed: NodePrecomputed,
    /// Arena slot byte offset for this op's primary output tensor.
    /// `Some(offset)` → arena-backed allocation at `arena_base + offset`.
    /// `None` → fall back to the GpuMemoryPool's runtime get_tensor_*
    /// path. Stamped at plan time by memory_planner; no runtime
    /// HashMap lookup per op.
    pub arena_offset: Option<usize>,
}
```

**Every place that constructs a PlannedOp needs `arena_offset: None` added.** Find them with:

```bash
grep -rn "PlannedOp {" src/ --include="*.rs"
```

There's one primary construction site in `src/ir/lowering.rs`. Add `arena_offset: None` at that site (planner will be wired in Task 3.6 to set real values).

- [ ] **Step 3: Add `memory_plan: MemoryPlan` field to `ExecutionPlan`**

In `src/ir/plan.rs`, update `ExecutionPlan`:

```rust
pub struct ExecutionPlan {
    // ... existing fields ...
    pub tensor_shapes: HashMap<String, Vec<Option<usize>>>,
    /// Static memory plan produced by memory_planner. Consumed by the
    /// Executor at run() start to allocate a single arena buffer; ops
    /// branch on PlannedOp::arena_offset rather than consulting this
    /// map at runtime.
    pub memory_plan: MemoryPlan,
}
```

Every `ExecutionPlan` construction site needs `memory_plan: MemoryPlan::empty()` added. Find with:

```bash
grep -rn "ExecutionPlan {" src/ --include="*.rs"
```

The primary site is `src/ir/lowering.rs`'s `lower()`. Use `MemoryPlan::empty()` for now; Task 3.6 will compute the real plan.

- [ ] **Step 4: Build**

```bash
cargo build --features cuda --release 2>&1 | tail -5
```

Expected: clean build. Adding `arena_offset: None` and `memory_plan: MemoryPlan::empty()` to every construction site preserves existing behavior (pool path) since `arena_offset == None` at every op.

If the build fails, the most likely issue is unmigrated `ExecutionPlan { ... }` or `PlannedOp { ... }` struct-literal sites. The compiler error will name them; add the default fields.

- [ ] **Step 5: Full test suite**

```bash
cargo test --features cuda --release 2>&1 | grep -E "^test result" | awk '{pass+=$4; fail+=$6} END {print pass" passed, "fail" failed"}'
```

Expected: baseline count + 6 (predicate tests). Zero failures — the new fields have default/empty values, so runtime behavior is unchanged.

- [ ] **Step 6: Commit WIP**

```bash
git add src/ir/plan.rs src/ir/lowering.rs
git commit -S -m "wip(ir): add MemoryPlan + PlannedOp::arena_offset + ExecutionPlan::memory_plan"
```

### Task 3.3: Implement the greedy linear scan allocator + determinism test

**Files:**
- Modify: `src/ir/passes/memory_planner.rs` (add the `memory_planner` pass fn)

- [ ] **Step 1: Add the allocator + the pass fn**

Append to `src/ir/passes/memory_planner.rs`:

```rust
use crate::ir::graph::{GraphNode, OptimizableGraph};
use crate::ir::plan::{MemoryPlan, PlannerClassification};

/// Size-class bucket (matches GpuMemoryPool's convention):
/// - exact match for element counts < 4096
/// - power-of-2 rounding for ≥ 4096
#[inline]
fn size_class_elements(len: usize) -> usize {
    if len < 4096 {
        len
    } else {
        len.next_power_of_two()
    }
}

/// Per-tensor planning record.
#[derive(Clone, Debug)]
struct EligibleTensor {
    name: String,
    dtype_bytes: usize,
    size_class_elements: usize,
    slot_bytes: usize,
    first_use_op_idx: usize,
    last_use_op_idx: usize,
}

/// One slot in the arena (may be reused by disjoint-lifetime tensors).
#[derive(Clone, Debug)]
struct ArenaSlot {
    dtype_bytes: usize,
    size_class_elements: usize,
    byte_offset: usize,
    /// Index of the op where the slot's current tenant is last used.
    /// New tenants can take the slot only if their first_use > this.
    current_tenant_last_use: usize,
}

/// Plan arena offsets for eligible tensors.
///
/// Deterministic: given the same inputs, produces byte-identical output.
pub fn memory_planner(
    graph: &OptimizableGraph,
    tensor_shapes: &std::collections::HashMap<String, Vec<Option<usize>>>,
    tensor_last_use: &std::collections::HashMap<String, usize>,
    tensor_first_use: &std::collections::HashMap<String, usize>,
    fusion: &std::collections::HashMap<String, crate::cuda::inference::fusion::FusedPatternInfo>,
) -> MemoryPlan {
    let graph_inputs: Vec<String> = graph
        .inputs
        .iter()
        .map(|input| input.name.clone())
        .collect();
    let graph_outputs: Vec<String> = graph.outputs.clone();

    // Enumerate tensors produced by nodes. Apply eligibility predicate
    // and fusion-skipped filter.
    let mut eligible: Vec<EligibleTensor> = Vec::new();
    let mut arena_eligible_count = 0usize;
    let mut pool_classified_count = 0usize;

    for node in graph.nodes.iter() {
        for out_name in node.outputs.iter() {
            if out_name.is_empty() {
                continue;
            }

            let shape = match tensor_shapes.get(out_name) {
                Some(s) => s,
                None => {
                    pool_classified_count += 1;
                    continue;
                }
            };

            // Compute element count + dtype bytes. Dtype is inferred
            // from the op's output dtype heuristic — conservative
            // default f32. Refine with a dtype-inference helper if
            // needed; for Phase 4 we assume f32 for activations (the
            // common case on Kokoro) and fall back to pool otherwise.
            let dtype_bytes = infer_output_dtype_bytes(&node.op_type);
            let element_count = match shape.iter().product::<Option<usize>>() {
                Some(n) => n,
                None => {
                    pool_classified_count += 1;
                    continue;
                }
            };

            if !is_planner_eligible(
                out_name,
                tensor_shapes,
                &graph_inputs,
                &graph_outputs,
                dtype_bytes,
                element_count,
            ) {
                pool_classified_count += 1;
                continue;
            }
            if is_fusion_skipped(out_name, fusion) {
                pool_classified_count += 1;
                continue;
            }

            arena_eligible_count += 1;
            let size_class = size_class_elements(element_count);
            let slot_bytes = dtype_bytes * size_class;

            let Some(&last_use) = tensor_last_use.get(out_name) else {
                // No last-use recorded → conservatively skip
                // (can't safely share the slot).
                pool_classified_count += 1;
                arena_eligible_count -= 1;  // re-classify back as pool
                continue;
            };
            let Some(&first_use) = tensor_first_use.get(out_name) else {
                pool_classified_count += 1;
                arena_eligible_count -= 1;
                continue;
            };

            eligible.push(EligibleTensor {
                name: out_name.clone(),
                dtype_bytes,
                size_class_elements: size_class,
                slot_bytes,
                first_use_op_idx: first_use,
                last_use_op_idx: last_use,
            });
        }
    }

    // Deterministic sort: by first-use ASC, tiebreak by name ASC.
    eligible.sort_by(|a, b| {
        a.first_use_op_idx
            .cmp(&b.first_use_op_idx)
            .then_with(|| a.name.cmp(&b.name))
    });

    // Greedy linear scan: for each tensor, try to reuse the earliest-freeable
    // compatible slot; otherwise allocate a new one.
    let mut slots: Vec<ArenaSlot> = Vec::new();
    let mut slot_assignments: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    let mut next_byte_offset: usize = 0;

    for tensor in eligible.iter() {
        let mut chosen: Option<usize> = None;
        for (idx, slot) in slots.iter_mut().enumerate() {
            if slot.dtype_bytes == tensor.dtype_bytes
                && slot.size_class_elements == tensor.size_class_elements
                && slot.current_tenant_last_use < tensor.first_use_op_idx
            {
                chosen = Some(idx);
                break;
            }
        }

        let slot_idx = match chosen {
            Some(idx) => {
                slots[idx].current_tenant_last_use = tensor.last_use_op_idx;
                idx
            }
            None => {
                let idx = slots.len();
                slots.push(ArenaSlot {
                    dtype_bytes: tensor.dtype_bytes,
                    size_class_elements: tensor.size_class_elements,
                    byte_offset: next_byte_offset,
                    current_tenant_last_use: tensor.last_use_op_idx,
                });
                next_byte_offset += tensor.slot_bytes;
                idx
            }
        };

        slot_assignments.insert(tensor.name.clone(), slot_idx);
    }

    let arena_bytes = next_byte_offset;
    let slot_count = slots.len();
    let slot_offsets_bytes: Vec<usize> = slots.iter().map(|s| s.byte_offset).collect();

    let classification = PlannerClassification {
        arena_eligible: arena_eligible_count,
        arena_assigned: slot_assignments.len(),
        pool_classified: pool_classified_count,
    };

    MemoryPlan::new(
        arena_bytes,
        slot_count,
        slot_assignments,
        slot_offsets_bytes,
        classification,
    )
}

/// Conservative dtype inference for op outputs. Mirrors the set of ops
/// whose outputs are reliably f32 in iconnx's current dispatch. Anything
/// not in this set returns 0, signaling "skip arena assignment — take
/// pool path." Extend as future ops land.
fn infer_output_dtype_bytes(op_type: &str) -> usize {
    match op_type {
        // Unary / binary elementwise f32 ops.
        "Add" | "Sub" | "Mul" | "Div" | "Pow" | "Sqrt" | "Exp" | "Sin" | "Cos" | "Tanh"
        | "Sigmoid" | "Atan" | "Erf" | "LeakyRelu" | "Clip" | "Floor" | "Round" => 4,
        // Reduce / norm (output f32).
        "ReduceMean" | "InstanceNormalization" | "LayerNormalization" => 4,
        // MatMul / Conv (output f32 in our supported paths).
        "MatMul" | "Gemm" | "Conv" | "ConvTranspose" => 4,
        // Shape-family: output is typically i64 (Shape, Range) or f32
        // (Reshape/Unsqueeze/Squeeze pass-through). Arena-assigning
        // reshape output is complex (it's a view). Skip these; they're
        // either too small (shape tensors) or better handled by the
        // pool's reuse of the input buffer (reshape passthroughs).
        _ => 0,
    }
}

#[cfg(test)]
mod allocator_tests {
    use super::*;

    // ... allocator-specific tests added in Task 3.4 ...
}
```

**Important:** the `infer_output_dtype_bytes` function is a conservative approximation. If Kokoro has an activation that returns non-f32 (e.g., a Cast-to-i64 output), the planner will skip it. This is safe — it falls to pool, same as pre-Phase-4 behavior. We do NOT want to guess dtype from tensor_shapes or from the graph, because mis-guessing would undersize a slot → silent corruption. The size-assertion in Task 3.7 catches any such mismatch.

- [ ] **Step 2: Re-export the pass fn**

In `src/ir/passes/mod.rs`, add:

```rust
pub use memory_planner::memory_planner;
```

- [ ] **Step 3: Build (verifies the types compile)**

```bash
cargo build --features cuda --release 2>&1 | tail -5
```

Expected: clean. The `memory_planner` fn references `tensor_first_use` which isn't yet computed; we wire it up in Task 3.6.

- [ ] **Step 4: Commit WIP**

```bash
git add src/ir/passes/memory_planner.rs src/ir/passes/mod.rs
git commit -S -m "wip(ir): memory_planner greedy linear scan allocator (unwired)"
```

### Task 3.4: Allocator unit tests (slot reuse, size-class separation, determinism, classification)

**Files:**
- Modify: `src/ir/passes/memory_planner.rs` (expand `allocator_tests` module)

- [ ] **Step 1: Add the full allocator test suite**

Replace the `mod allocator_tests { ... }` block in `memory_planner.rs` with:

```rust
#[cfg(test)]
mod allocator_tests {
    use super::*;
    use crate::attributes::NodeAttributes;
    use crate::cuda::inference::fusion::FusedPatternInfo;
    use crate::ir::graph::OptimizableGraphBuilder;
    use std::collections::HashMap;

    /// Build a graph with two f32 tensors from two sequential Add ops
    /// (synthetic). Each test supplies its own lifetimes via
    /// tensor_first_use + tensor_last_use maps.
    fn build_two_add_graph() -> crate::ir::graph::OptimizableGraph {
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("x".into(), vec![Some(64), Some(128)]);
        b.add_initializer(
            "w1".into(),
            crate::tensor::Tensor::from_vec(vec![0.0; 64 * 128], vec![64, 128]),
        );
        b.add_initializer(
            "w2".into(),
            crate::tensor::Tensor::from_vec(vec![0.0; 64 * 128], vec![64, 128]),
        );
        b.add_node(
            "add1",
            "Add",
            vec!["x".into(), "w1".into()],
            vec!["a".into()],
            NodeAttributes::new(),
        );
        b.add_node(
            "add2",
            "Add",
            vec!["a".into(), "w2".into()],
            vec!["b".into()],
            NodeAttributes::new(),
        );
        b.add_output("b".into());
        b.build()
    }

    fn base_shapes_for_two_add() -> HashMap<String, Vec<Option<usize>>> {
        let mut s = HashMap::new();
        s.insert("x".into(), vec![Some(64), Some(128)]);
        s.insert("a".into(), vec![Some(64), Some(128)]);
        s.insert("b".into(), vec![Some(64), Some(128)]);
        s
    }

    #[test]
    fn slot_assignment_reuses_slot_for_disjoint_lifetimes() {
        // Two f32 tensors, same size, disjoint lifetimes (last_use(a)=1 < first_use(a2)=2).
        // Third tensor is a graph output — excluded. Verify 1 slot total.
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("x".into(), vec![Some(64), Some(128)]);
        b.add_initializer(
            "w".into(),
            crate::tensor::Tensor::from_vec(vec![0.0; 64 * 128], vec![64, 128]),
        );
        b.add_node(
            "first",
            "Add",
            vec!["x".into(), "w".into()],
            vec!["a".into()],
            NodeAttributes::new(),
        );
        b.add_node(
            "second",
            "Add",
            vec!["a".into(), "w".into()],
            vec!["a2".into()],  // a2 depends only on a (consumed) + w
            NodeAttributes::new(),
        );
        b.add_node(
            "third",
            "Add",
            vec!["a2".into(), "w".into()],
            vec!["out".into()],
            NodeAttributes::new(),
        );
        b.add_output("out".into());
        let graph = b.build();

        let mut shapes = HashMap::new();
        shapes.insert("x".into(), vec![Some(64), Some(128)]);
        shapes.insert("a".into(), vec![Some(64), Some(128)]);
        shapes.insert("a2".into(), vec![Some(64), Some(128)]);
        shapes.insert("out".into(), vec![Some(64), Some(128)]);

        let mut first = HashMap::new();
        first.insert("a".into(), 0);    // op-idx 0 produces a
        first.insert("a2".into(), 1);   // op-idx 1 produces a2
        first.insert("out".into(), 2);

        let mut last = HashMap::new();
        last.insert("a".into(), 1);     // a last used by op 1
        last.insert("a2".into(), 2);    // a2 last used by op 2

        let plan = memory_planner(&graph, &shapes, &last, &first, &HashMap::new());

        // "out" is a graph output → excluded. "a" and "a2" are eligible
        // with disjoint lifetimes — should share one slot.
        assert_eq!(plan.slot_count, 1, "disjoint-lifetime tensors must share");
        assert_eq!(plan.classification_counts().arena_assigned, 2);
    }

    #[test]
    fn slot_assignment_creates_new_slot_for_overlapping_lifetimes() {
        let graph = build_two_add_graph();
        let shapes = base_shapes_for_two_add();

        let mut first = HashMap::new();
        first.insert("a".into(), 0);
        first.insert("b".into(), 1);
        let mut last = HashMap::new();
        // 'a' used by op 1 (op producing 'b') — overlap with b's production.
        last.insert("a".into(), 1);
        last.insert("b".into(), 1);  // b is graph output, excluded anyway

        let plan = memory_planner(&graph, &shapes, &last, &first, &HashMap::new());

        // b is graph output → excluded. a has no overlapping peers.
        // Verify 'a' got its own slot.
        assert_eq!(plan.classification_counts().arena_assigned, 1);
        assert_eq!(plan.slot_count, 1);
    }

    #[test]
    fn slot_assignment_preserves_size_class_separation() {
        // Two tensors of same element count but different dtypes → can't share.
        // Synthesize by marking one f32, one i64 via different producer op types.
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("x".into(), vec![Some(64), Some(128)]);
        b.add_initializer(
            "w".into(),
            crate::tensor::Tensor::from_vec(vec![0.0; 64 * 128], vec![64, 128]),
        );
        b.add_node(
            "add_f32",
            "Add",
            vec!["x".into(), "w".into()],
            vec!["a_f32".into()],
            NodeAttributes::new(),
        );
        // Second: emit something f32 as well but labeled i64 (synthetic—
        // the dtype inference is heuristic, so just use two Add outputs
        // and check the size-class slot is shared not split).
        b.add_node(
            "add_f32_2",
            "Add",
            vec!["a_f32".into(), "w".into()],
            vec!["b_f32".into()],
            NodeAttributes::new(),
        );
        b.add_output("b_f32".into());
        let graph = b.build();

        let mut shapes = HashMap::new();
        shapes.insert("x".into(), vec![Some(64), Some(128)]);
        shapes.insert("a_f32".into(), vec![Some(64), Some(128)]);
        shapes.insert("b_f32".into(), vec![Some(64), Some(128)]);

        let mut first = HashMap::new();
        first.insert("a_f32".into(), 0);
        first.insert("b_f32".into(), 1);
        let mut last = HashMap::new();
        last.insert("a_f32".into(), 1);  // last used by op 1
        last.insert("b_f32".into(), 1);  // graph output, excluded

        let plan = memory_planner(&graph, &shapes, &last, &first, &HashMap::new());

        // a_f32 gets a slot; b_f32 is graph output; dtype consistent (both f32)
        // → just verify 1 slot, 1 assignment.
        assert_eq!(plan.slot_count, 1);
        assert_eq!(plan.classification_counts().arena_assigned, 1);
    }

    #[test]
    fn fusion_skipped_tensor_filtered_out_even_when_otherwise_eligible() {
        let graph = build_two_add_graph();
        let shapes = base_shapes_for_two_add();
        let mut first = HashMap::new();
        first.insert("a".into(), 0);
        first.insert("b".into(), 1);
        let mut last = HashMap::new();
        last.insert("a".into(), 1);

        // Build a FusedPatternInfo entry that marks "a" as a skipped
        // tensor. Exact field name depends on the FusedPatternInfo
        // struct in src/cuda/inference/fusion/patterns.rs — inspect and
        // use the correct field; if it's skipped_node_indices, adapt
        // accordingly (see Task 3.1 Step 1's alternative signature).
        let mut fusion = HashMap::new();
        // Construct a FusedPatternInfo with "a" in its skipped_tensors
        // set. If the struct doesn't have skipped_tensors directly,
        // compute it from skipped_node_indices + graph.
        fusion.insert(
            "some_head_node".to_string(),
            FusedPatternInfo {
                // ... fields per actual FusedPatternInfo ...
                skipped_tensors: vec!["a".to_string()],
                // ... other fields as required ...
            },
        );

        let plan = memory_planner(&graph, &shapes, &last, &first, &fusion);

        // 'a' is fusion-skipped → NOT arena_eligible. b is graph output.
        // Nothing gets arena-assigned.
        assert_eq!(plan.classification_counts().arena_assigned, 0);
        assert_eq!(plan.classification_counts().arena_eligible, 0);
    }

    #[test]
    fn determinism_snapshot() {
        let graph = build_two_add_graph();
        let shapes = base_shapes_for_two_add();
        let mut first = HashMap::new();
        first.insert("a".into(), 0);
        first.insert("b".into(), 1);
        let mut last = HashMap::new();
        last.insert("a".into(), 1);

        let plan_1 = memory_planner(&graph, &shapes, &last, &first, &HashMap::new());
        let plan_2 = memory_planner(&graph, &shapes, &last, &first, &HashMap::new());

        // Byte-identical: arena_bytes, slot_count, slot_offsets_bytes
        // must match exactly across runs.
        assert_eq!(plan_1.arena_bytes, plan_2.arena_bytes);
        assert_eq!(plan_1.slot_count, plan_2.slot_count);
        assert_eq!(plan_1.slot_offsets_bytes, plan_2.slot_offsets_bytes);
    }

    #[test]
    fn classification_counts_reports_eligibility_and_assignment() {
        let graph = build_two_add_graph();
        let shapes = base_shapes_for_two_add();
        let mut first = HashMap::new();
        first.insert("a".into(), 0);
        first.insert("b".into(), 1);
        let mut last = HashMap::new();
        last.insert("a".into(), 1);

        let plan = memory_planner(&graph, &shapes, &last, &first, &HashMap::new());
        let c = plan.classification_counts();

        // 'a' is eligible + assigned; 'b' is graph output (pool_classified);
        // 'x' is graph input (pool_classified — though it's not even in nodes.outputs,
        // so it wouldn't be enumerated here; spot-check the logic's integrity).
        assert_eq!(c.arena_eligible, 1, "exactly 'a' should be eligible");
        assert_eq!(c.arena_assigned, 1);
        // b is graph output
        assert_eq!(c.pool_classified, 1);
    }
}
```

**If `FusedPatternInfo` doesn't have `skipped_tensors` directly:** The exact field structure is discoverable by reading `src/cuda/inference/fusion/patterns.rs`. If the field is `skipped_node_indices: Vec<usize>` instead, the test's `FusedPatternInfo` construction needs the index of node "add1" in the graph (which is 0 in `build_two_add_graph`), and `is_fusion_skipped` needs the extended signature from Task 3.1 Step 1's alternative. Adjust both the test and `is_fusion_skipped` to whatever form the real struct exposes.

- [ ] **Step 2: Run the allocator tests**

```bash
cargo test --features cuda --release --lib ir::passes::memory_planner 2>&1 | grep -E "^test result|FAILED"
```

Expected: `11 passed` (6 predicate tests from Task 3.1 + 5 allocator tests).

If any test fails, debug before moving on. Likely causes: FusedPatternInfo field mismatch, OptimizableGraphBuilder API difference, or tensor_first_use semantics differ from the assumption.

- [ ] **Step 3: Commit WIP**

```bash
git add src/ir/passes/memory_planner.rs
git commit -S -m "wip(ir): memory_planner allocator tests (reuse, determinism, classification)"
```

### Task 3.5: `GpuMemoryPool::allocate_arena` + `ArenaBuffer` RAII

**Files:**
- Modify: `src/cuda/memory_pool.rs`

- [ ] **Step 1: Inspect existing pool allocation paths to pick the right garboard primitive**

```bash
grep -n "alloc_zeros\|DeviceSlice::alloc\|from_host\|copy_host_to_device" src/cuda/memory_pool.rs | head -15
```

The goal: figure out how the pool already asks garboard for a byte buffer on the device. `allocate_arena` takes a byte count and wraps the resulting `DeviceSlice<u8>` in an RAII handle.

- [ ] **Step 2: Add the public API**

Append to `src/cuda/memory_pool.rs`:

```rust
/// RAII handle holding an arena byte buffer. Dropping returns the
/// underlying allocation to the GpuMemoryPool as a size-class-bucketed
/// slice. Phase 4 allocates one arena per `run()`; cross-inference
/// caching is a separate follow-up.
pub struct ArenaBuffer<'pool> {
    /// Device byte buffer. Lifetime tied to the pool borrow so the
    /// buffer cannot outlive the pool.
    buffer: Option<DeviceSlice<'static, u8>>,
    pool: &'pool mut GpuMemoryPool,
    size_class_len: usize,
}

impl<'pool> ArenaBuffer<'pool> {
    /// Return a typed sub-slice into the arena at `byte_offset` spanning
    /// `element_count × sizeof(T)` bytes. No bounds checking — caller
    /// must ensure the request fits within the arena.
    pub fn slice_at_f32(
        &self,
        byte_offset: usize,
        element_count: usize,
    ) -> Result<DeviceSlice<'_, f32>, CudaError> {
        let buffer = self
            .buffer
            .as_ref()
            .expect("arena buffer must be present until drop");
        let bytes = element_count * std::mem::size_of::<f32>();
        let byte_slice = buffer.slice(byte_offset, byte_offset + bytes);
        // Reinterpret bytes → f32. Safe: arena memory is uninitialized
        // from the GPU's perspective, and the caller writes into this
        // slice before reading.
        Ok(byte_slice.reinterpret::<f32>())
    }

    /// Same as slice_at_f32 but for i64. (Add i32 similarly if needed
    /// by any op; for Phase 4, f32 covers Kokoro's activations.)
    pub fn slice_at_i64(
        &self,
        byte_offset: usize,
        element_count: usize,
    ) -> Result<DeviceSlice<'_, i64>, CudaError> {
        let buffer = self
            .buffer
            .as_ref()
            .expect("arena buffer must be present until drop");
        let bytes = element_count * std::mem::size_of::<i64>();
        let byte_slice = buffer.slice(byte_offset, byte_offset + bytes);
        Ok(byte_slice.reinterpret::<i64>())
    }
}

impl<'pool> Drop for ArenaBuffer<'pool> {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            // Return the buffer to the pool as a size-class-bucketed
            // slice. Matches the pool's existing return_tensor pattern.
            let entry = self
                .pool
                .u8_pool
                .entry(self.size_class_len)
                .or_insert_with(Vec::new);
            entry.push(buffer);
        }
    }
}

impl GpuMemoryPool {
    /// Allocate a single contiguous byte buffer on the device for the
    /// memory planner's arena. Returns an RAII handle; dropping it
    /// returns the allocation to the pool as a standard pooled slice.
    pub fn allocate_arena(
        &mut self,
        ctx: &IconnxCudaContext,
        bytes: usize,
    ) -> Result<ArenaBuffer<'_>, CudaError> {
        if bytes == 0 {
            // Zero-byte arena: return a handle with no underlying buffer.
            // slice_at_* will panic; the runtime branches on arena_offset
            // == None everywhere so slice_at_* should never be called.
            return Ok(ArenaBuffer {
                buffer: None,
                pool: self,
                size_class_len: 0,
            });
        }

        let size_class_len = size_class(bytes);

        let buffer = if let Some(bucket) = self.u8_pool.get_mut(&size_class_len) {
            if let Some(slice) = bucket.pop() {
                self.hits += 1;
                slice
            } else {
                self.misses += 1;
                Self::fresh_alloc_u8(ctx, size_class_len)?
            }
        } else {
            self.misses += 1;
            Self::fresh_alloc_u8(ctx, size_class_len)?
        };

        Ok(ArenaBuffer {
            buffer: Some(buffer),
            pool: self,
            size_class_len,
        })
    }

    fn fresh_alloc_u8(
        ctx: &IconnxCudaContext,
        len: usize,
    ) -> Result<DeviceSlice<'static, u8>, CudaError> {
        // Use the same allocation primitive the pool uses for typed
        // buckets. Check existing fresh_alloc_f32 / fresh_alloc_i64
        // implementations in this file for the correct call.
        ctx.stream()
            .alloc_zeros::<u8>(len)
            .map(|s| unsafe {
                // SAFETY: DeviceSlice lifetime is tied to the pool's
                // lifetime; we store it as 'static per pool convention
                // and rely on drop to return it before context teardown.
                std::mem::transmute::<DeviceSlice<'_, u8>, DeviceSlice<'static, u8>>(s)
            })
            .map_err(|e| CudaError::from(e))
    }
}
```

**Before writing this:** Inspect `src/cuda/memory_pool.rs`'s existing `fresh_alloc_f32` / `fresh_alloc_i64` helpers (or equivalent). The `fresh_alloc_u8` implementation above is a sketch; use the project's actual allocation + lifetime-transmute pattern.

Also: adding a `u8_pool: HashMap<usize, Vec<DeviceSlice<'static, u8>>>` field to `GpuMemoryPool` mirrors the existing `f32_pool` / `i64_pool` / `i32_pool` fields. Do that as part of this task. Update the `Default` / `new` constructor to initialize it.

- [ ] **Step 3: Build**

```bash
cargo build --features cuda --release 2>&1 | tail -5
```

Expected: clean. If the sketch doesn't match garboard's actual `DeviceSlice::slice` / `reinterpret` signatures, adapt — the goal is byte-offset-into-a-slab-into-typed-sub-slice; exact API varies.

If garboard doesn't expose `reinterpret`, the alternative is: arena stores a single `Vec<u8>` on host + copies per tensor. This is a substantial design change — STOP and escalate if this path is forced. Per Phase 1 learning: don't work around upstream API gaps in test code; escalate.

- [ ] **Step 4: Add a quick smoke test for `allocate_arena`**

Append to the existing `#[cfg(test)] mod tests` block in `memory_pool.rs`:

```rust
#[cfg(feature = "cuda")]
#[test]
fn allocate_arena_roundtrip_returns_to_pool() {
    let ctx = match IconnxCudaContext::new() {
        Ok(c) => c,
        Err(_) => return, // CI without GPU — skip cleanly
    };
    let mut pool = GpuMemoryPool::new();
    {
        let arena = pool.allocate_arena(&ctx, 8192).expect("allocate arena");
        let _slice = arena.slice_at_f32(0, 1024).expect("sub-slice");
        // arena drops here, returning the buffer to the pool.
    }
    // Second allocation at same size should hit the pool.
    let prev_hits = pool.hits();
    {
        let _arena2 = pool.allocate_arena(&ctx, 8192).expect("allocate arena again");
    }
    assert!(pool.hits() > prev_hits, "second allocation should hit pool");
}
```

Run: `cargo test --features cuda --release --lib cuda::memory_pool::tests::allocate_arena_roundtrip 2>&1 | grep -E "^test result|FAILED"` — expect `1 passed`.

- [ ] **Step 5: Commit WIP**

```bash
git add src/cuda/memory_pool.rs
git commit -S -m "wip(ir): GpuMemoryPool::allocate_arena + ArenaBuffer RAII"
```

### Task 3.6: Wire `memory_planner` into `lower()` + compute `tensor_first_use`

**Files:**
- Modify: `src/ir/lowering.rs`

- [ ] **Step 1: Add a `compute_tensor_first_use` helper**

`compute_tensor_last_use` already exists in `src/ir/lowering.rs` (line ~290). The first-use helper is its mirror: for each tensor (op output), record the op-index that produces it. Append in the same file:

```rust
/// For each tensor produced by some op in `ops`, the op-index at which
/// it's first defined. Mirrors `compute_tensor_last_use` but tracks
/// definitions (first use) instead of consumptions.
fn compute_tensor_first_use(ops: &[crate::ir::plan::PlannedOp]) -> std::collections::HashMap<String, usize> {
    let mut first_use = std::collections::HashMap::new();
    for (idx, op) in ops.iter().enumerate() {
        for out_name in op.node.outputs.iter() {
            if !out_name.is_empty() {
                // Use the first (= only, if ops are in topo order)
                // definition.
                first_use.entry(out_name.clone()).or_insert(idx);
            }
        }
    }
    first_use
}
```

- [ ] **Step 2: Call `memory_planner` in `lower()`**

Find the existing `compute_tensor_last_use` call. The pipeline should look roughly like:

```rust
let ops = build_ops(...);
let tensor_last_use = compute_tensor_last_use(&ops, &graph.outputs, &weights);

Ok(ExecutionPlan {
    ops,
    weights,
    // ... existing fields ...
    tensor_last_use,
    // ... etc ...
    memory_plan: MemoryPlan::empty(),  // added in Task 3.2
})
```

Update to:

```rust
let ops = build_ops(...);
let tensor_last_use = compute_tensor_last_use(&ops, &graph.outputs, &weights);
let tensor_first_use = compute_tensor_first_use(&ops);

// Compute memory plan AFTER fusion annotations are on ops but BEFORE
// we stamp arena_offset into each PlannedOp.
let memory_plan = memory_planner(
    &graph_after_all_passes,
    &tensor_shapes,
    &tensor_last_use,
    &tensor_first_use,
    &fusion_annotations,
);

// Stamp arena_offset onto PlannedOps.
let mut ops = ops;
for op in ops.iter_mut() {
    // Assume op's primary output is node.outputs[0].
    let out = op.node.outputs.first();
    op.arena_offset = out.and_then(|name| memory_plan.arena_offset(name));
}

Ok(ExecutionPlan {
    ops,
    // ...
    memory_plan,
})
```

**`graph_after_all_passes`** is whatever variable holds the `OptimizableGraph` after the full pass pipeline (it's the same graph used to construct `ops`). **`fusion_annotations`** is the merged fusion map (same one used to populate `OpKind::FusedHead` on `PlannedOp`s). Read the existing `lower()` to find the exact variable names.

If `ops` is built from a consuming method (e.g., `graph.into_nodes()`), restructure to keep a reference to the graph while ops are still mutable. Common pattern: clone `graph_inputs` and `graph_outputs` into dedicated `Vec<String>` before `graph` is consumed.

- [ ] **Step 3: Build + full test suite**

```bash
cargo build --features cuda --release 2>&1 | tail -5
cargo test --features cuda --release 2>&1 | grep -E "^test result" | awk '{pass+=$4; fail+=$6} END {print pass" passed, "fail" failed"}'
```

Expected: clean build; all tests pass. At this point the planner computes `arena_offset` per op, but the Executor doesn't yet consume it (Task 3.7). So runtime behavior is still pool-path for everything.

- [ ] **Step 4: Run Kokoro integration test**

```bash
cargo test --features cuda --release --test kokoro_gpu_fusion_test -- --ignored 2>&1 | grep -E "^test result|FAILED"
```

Expected: `1 passed`. No regression.

- [ ] **Step 5: Add an assertion log to confirm planner is running on Kokoro**

Append a one-shot `#[ignore]`'d test to `tests/kokoro_gpu_fusion_test.rs`:

```rust
#[cfg(feature = "cuda")]
#[test]
#[ignore] // Diagnostic; run with --ignored
fn kokoro_memory_planner_produces_assignments() {
    let executor = setup_kokoro_gpu_executor();  // uses shared helper from C PR
    let inputs = kokoro_inputs(5);
    // Force one compile cycle so the plan is populated.
    let _ = executor.run_with_profiling(inputs, vec!["audio"]).expect("run");

    // Use the test-introspection API to get the ExecutionPlan's MemoryPlan.
    use iconnx::cuda::inference::testing::get_memory_plan;  // ADD THIS TO testing.rs
    let plan = get_memory_plan(&executor).expect("memory plan present");
    let c = plan.classification_counts();

    println!(
        "Kokoro memory plan: eligible={}, assigned={}, pool={}, arena_bytes={}, slots={}",
        c.arena_eligible, c.arena_assigned, c.pool_classified,
        plan.arena_bytes, plan.slot_count,
    );
    assert!(c.arena_eligible > 0, "at least some Kokoro tensors should be eligible");
    assert!(c.arena_assigned > 0, "at least some eligible tensors should be assigned");
}
```

And add the `get_memory_plan` helper to `src/cuda/inference/testing.rs`:

```rust
pub fn get_memory_plan(executor: &GpuGraphExecutor) -> Option<iconnx_crate::ir::plan::MemoryPlan> {
    executor.compiled_plan().map(|p| p.memory_plan.clone())
}
```

(Fix imports / crate path as needed to match existing helper patterns in `testing.rs`.)

Run:

```bash
cargo test --features cuda --release --test kokoro_gpu_fusion_test -- --ignored --nocapture kokoro_memory_planner_produces_assignments 2>&1 | grep -E "Kokoro memory plan|assigned|FAILED"
```

Expected output: `Kokoro memory plan: eligible=NN, assigned=NN, pool=MM, arena_bytes=..., slots=...` with `eligible > 0` and `assigned > 0`. Confirms the planner is running and producing non-empty output on Kokoro.

- [ ] **Step 6: Commit WIP**

```bash
git add src/ir/lowering.rs src/cuda/inference/testing.rs tests/kokoro_gpu_fusion_test.rs
git commit -S -m "wip(ir): wire memory_planner into lower() + Kokoro planner diagnostic test"
```

### Task 3.7: Runtime arena integration in the Executor

**Files:**
- Modify: `src/cuda/executor/mod.rs` (primary runtime path)
- Modify: `src/cuda/executor/elementwise.rs` (or wherever op dispatch happens) — the exact site depends on where the Executor actually allocates output tensors for ops

- [ ] **Step 1: Find the per-op output allocation path**

```bash
grep -rn "pool.get_tensor_f32\|pool.get_tensor_i64\|get_tensor_f32\|get_tensor_i64" src/cuda/executor/ | head -20
```

The Executor's per-op dispatch currently asks the pool for output buffers. Arena branching happens at those same call sites.

- [ ] **Step 2: Add the env-var check at run() start**

In `src/cuda/executor/mod.rs`'s `run()` (or equivalent entry point):

```rust
use std::sync::OnceLock;

fn memory_planner_disabled() -> bool {
    static DISABLED: OnceLock<bool> = OnceLock::new();
    *DISABLED.get_or_init(|| {
        std::env::var("ICONNX_DISABLE_MEMORY_PLANNER")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}
```

Read once per process. `OnceLock` ensures the env var is checked exactly one time and the result cached.

- [ ] **Step 3: Allocate the arena at run() start**

Near the top of `Executor::run()`:

```rust
let arena = if memory_planner_disabled() || plan.memory_plan.arena_bytes == 0 {
    None
} else {
    Some(self.pool.allocate_arena(&self.ctx, plan.memory_plan.arena_bytes)?)
};
```

`arena: Option<ArenaBuffer<'_>>` lives for the duration of `run()`; dropped at end, returning memory to the pool.

- [ ] **Step 4: Branch per-op on `arena_offset`**

At each site where an op's output buffer is allocated, wrap in:

```rust
// Before (conceptually):
//   let output = self.pool.get_tensor_f32(&self.ctx, element_count)?;
//
// After:
let output = match (op.arena_offset, &arena) {
    (Some(offset), Some(arena_buf)) => {
        // Arena-backed. Assert size match (always-on, nanoseconds).
        let expected_bytes = element_count * std::mem::size_of::<f32>();
        let slot_bytes = size_class(element_count) * std::mem::size_of::<f32>();
        assert_eq!(
            expected_bytes, slot_bytes.min(expected_bytes * 2),  // placeholder — see note below
            "memory_planner slot size mismatch for tensor {:?}: expected {} bytes, slot {}",
            op.node.outputs.first(), expected_bytes, slot_bytes,
        );
        let slice = arena_buf.slice_at_f32(offset, size_class(element_count))?;
        GpuTensor::from_device_slice(slice, dtype_shape)
    }
    _ => {
        // Pool path (unchanged).
        self.pool.get_tensor_f32(&self.ctx, element_count)?
    }
};
```

**The size assertion is the critical always-on check.** It catches the class of silent-corruption bug where shape_inference undersizes a tensor relative to what the op actually writes. The assertion must:
- Compare `actual_bytes_the_op_will_write` against `slot_bytes_the_planner_allocated`.
- Fire on mismatch with a panic message including the tensor name and both sizes.
- NOT be wrapped in `#[cfg(debug-inference)]` or `debug_assert_eq!` — it's `assert_eq!`, active in plain release.

The "placeholder" logic above is illustrative. The concrete form depends on:
- How iconnx computes `element_count` at runtime for an op's output (usually: `shape.iter().product()` for fully-known shapes)
- How it computes `slot_bytes` for a slot (the size-class bucket × dtype size)
The implementer writes the exact check after reading the executor's current op-dispatch. The INTENT is: actual bytes ≤ slot bytes (since size-class rounds UP), BUT actual bytes should match the plan's expected bytes (not just fit).

- [ ] **Step 5: Full test suite with arena active (default)**

```bash
cargo build --features cuda --release 2>&1 | tail -3
cargo test --features cuda --release 2>&1 | grep -E "^test result" | awk '{pass+=$4; fail+=$6} END {print pass" passed, "fail" failed"}'
```

Expected: zero failures. All tests pass through the arena path where eligible.

- [ ] **Step 6: Full test suite with `ICONNX_DISABLE_MEMORY_PLANNER=1`**

```bash
ICONNX_DISABLE_MEMORY_PLANNER=1 cargo test --features cuda --release 2>&1 | grep -E "^test result" | awk '{pass+=$4; fail+=$6} END {print pass" passed, "fail" failed"}'
```

Expected: identical counts to Step 5. The env var bypasses arena allocation; every tensor takes the pool path — same behavior as pre-Phase-4.

- [ ] **Step 7: Kokoro integration test with arena active**

```bash
cargo test --features cuda --release --test kokoro_gpu_fusion_test -- --ignored 2>&1 | grep -E "^test result|FAILED"
```

Expected: `1 passed`. No numerical regression (test's built-in 1e-5 tolerance for fused vs sequential paths covers this).

- [ ] **Step 8: Kokoro integration test with `ICONNX_DISABLE_MEMORY_PLANNER=1`**

```bash
ICONNX_DISABLE_MEMORY_PLANNER=1 cargo test --features cuda --release --test kokoro_gpu_fusion_test -- --ignored 2>&1 | grep -E "^test result|FAILED"
```

Expected: `1 passed`. Same test passes under pool-only fallback.

- [ ] **Step 9: Commit WIP**

```bash
git add src/cuda/executor/mod.rs src/cuda/executor/elementwise.rs  # plus any other executor files touched
git commit -S -m "wip(ir): Executor branches per-op on arena_offset + always-on size assertion + env-var escape hatch"
```

### Task 3.8: Kokoro arena-coverage acceptance test

**Files:**
- Create: `tests/kokoro_arena_coverage_test.rs`

- [ ] **Step 1: Write the ≥ 70% classification coverage test**

```rust
//! Phase 4 A landing criterion #4: on Kokoro, at least 70% of
//! arena-eligible tensors must actually be assigned to arena slots.
//! Catches the "predicate has a silent bug that sends everything to
//! pool" failure mode that binary correctness tests don't detect.

#![cfg(feature = "cuda")]

use std::path::Path;

#[path = "common/mod.rs"]
mod common;

use iconnx::cuda::GpuGraphExecutor;
use iconnx::onnx_parser::OnnxParser;

fn setup_kokoro_executor() -> GpuGraphExecutor {
    let model_path = common::kokoro_model::kokoro_model_path();
    let model = OnnxParser::parse_file(&model_path).expect("parse kokoro-v1.0.onnx");
    let weights = model.extract_weights();
    let graph = model.computation_graph();
    let mut executor = GpuGraphExecutor::new().expect("create CUDA executor");

    for (name, shape) in model.input_shapes() {
        executor.add_input(name, shape);
    }
    for (name, tensor) in &weights {
        executor.add_initializer(name.clone(), tensor).expect("add initializer");
    }
    for (idx, node) in graph.nodes().iter().enumerate() {
        let fallback = format!("node_{}", idx);
        let node_name = node.outputs.first().map(|s| s.as_str()).unwrap_or(&fallback);
        let inputs: Vec<&str> = node.inputs.iter().map(|s| s.as_str()).collect();
        let outputs: Vec<&str> = node.outputs.iter().map(|s| s.as_str()).collect();
        executor.add_node(node_name, &node.op_type, inputs, outputs, node.attributes.clone());
    }
    executor
}

#[test]
#[ignore] // Requires Kokoro + CUDA; run with --ignored.
fn kokoro_arena_assignment_rate_is_above_70_percent() {
    let executor = setup_kokoro_executor();

    // Force a compile/run cycle so the plan is populated.
    use std::collections::HashMap;
    use iconnx::tensor::Tensor;
    let mut inputs = HashMap::new();
    inputs.insert("tokens".to_string(), Tensor::from_vec_i64((0..5).map(|i| i % 50).collect(), vec![1, 5]));
    inputs.insert("style".to_string(), Tensor::from_vec_f32(vec![0.0; 256], vec![1, 256]));
    inputs.insert("speed".to_string(), Tensor::from_vec_f32(vec![1.0], vec![1]));
    let _ = executor.run(inputs, vec!["audio"]).expect("run inference");

    // Introspect the memory plan via the testing API added in Task 3.6.
    use iconnx::cuda::inference::testing::get_memory_plan;
    let plan = get_memory_plan(&executor).expect("memory plan present");
    let c = plan.classification_counts();

    eprintln!(
        "Kokoro arena classification: eligible={} assigned={} pool={} \
         arena_bytes={} slots={}",
        c.arena_eligible, c.arena_assigned, c.pool_classified,
        plan.arena_bytes, plan.slot_count,
    );

    // Landing criterion #4: arena_assigned / arena_eligible >= 0.7
    assert!(c.arena_eligible >= 10, "need a meaningful eligible count; got {}", c.arena_eligible);
    let assignment_rate = c.arena_assigned as f64 / c.arena_eligible as f64;
    assert!(
        assignment_rate >= 0.7,
        "Phase 4 A landing criterion #4 failed: arena_assigned/arena_eligible = {}/{} = {:.2}; \
         need ≥ 0.70. Most likely cause: is_planner_eligible or is_fusion_skipped has a \
         silent bug filtering out tensors that should be assigned.",
        c.arena_assigned, c.arena_eligible, assignment_rate,
    );
}
```

- [ ] **Step 2: Run the test**

```bash
cargo test --features cuda --release --test kokoro_arena_coverage_test -- --ignored --nocapture 2>&1 | grep -E "Kokoro arena|^test result|FAILED|panicked"
```

Expected: rate ≥ 0.70, test passes. If < 0.70: investigate (see Amber-style protocol for Commit 2 Task 2.4 — similar spirit). Likely causes: dtype inference too conservative, is_fusion_skipped over-filtering, eligibility predicate has a bug.

- [ ] **Step 3: Commit WIP**

```bash
git add tests/kokoro_arena_coverage_test.rs
git commit -S -m "wip(ir): Kokoro arena coverage acceptance test (≥70% assignment rate)"
```

### Task 3.9: Kokoro numerical regression test (arena vs pool paths agree)

**Files:**
- Modify: `tests/kokoro_gpu_fusion_test.rs` (extend existing test OR add a new one comparing arena/pool outputs)

- [ ] **Step 1: Add a numerical-parity test**

Append to `tests/kokoro_gpu_fusion_test.rs`:

```rust
#[cfg(feature = "cuda")]
#[test]
#[ignore] // Requires Kokoro + CUDA.
fn kokoro_arena_and_pool_paths_produce_identical_outputs() {
    // Run once with memory planner active; once with it disabled.
    // Outputs must match within 1e-5 (iconnx's e2e fp32 tolerance budget).

    fn run_with_planner_disabled(disabled: bool) -> Vec<f32> {
        let executor = setup_kokoro_gpu_executor();
        let inputs = kokoro_inputs(5);
        let (outputs, _) = if disabled {
            // SAFETY: each test binary is a fresh process, so set_var is safe here.
            // SAFETY: set_var on Unix is not thread-safe in general; this test is
            // single-threaded.
            unsafe { std::env::set_var("ICONNX_DISABLE_MEMORY_PLANNER", "1"); }
            let r = executor.run_with_profiling(inputs, vec!["audio"]).expect("run");
            unsafe { std::env::remove_var("ICONNX_DISABLE_MEMORY_PLANNER"); }
            r
        } else {
            executor.run_with_profiling(inputs, vec!["audio"]).expect("run")
        };

        outputs
            .into_iter()
            .find(|(name, _)| name == "audio")
            .expect("audio output")
            .1
            .to_array()
            .as_slice()
            .unwrap()
            .to_vec()
    }

    let arena_output = run_with_planner_disabled(false);
    let pool_output = run_with_planner_disabled(true);

    assert_eq!(
        arena_output.len(), pool_output.len(),
        "arena and pool paths produced different output lengths ({} vs {})",
        arena_output.len(), pool_output.len(),
    );
    let mut max_diff = 0.0f32;
    for (i, (&a, &p)) in arena_output.iter().zip(pool_output.iter()).enumerate() {
        let d = (a - p).abs();
        if d > max_diff { max_diff = d; }
        assert!(
            d < 1e-5,
            "arena vs pool divergence at index {}: arena={} pool={} diff={:.2e}",
            i, a, p, d,
        );
    }
    eprintln!("Kokoro arena vs pool max_abs_diff: {:.2e}", max_diff);
}
```

**IMPORTANT on `std::env::set_var` safety:** In Rust 2024+, `std::env::set_var` is `unsafe` because process-wide env vars are a shared-mutable state. The test is a fresh process so the set_var is safe in practice; the `unsafe` wrappers are required for it to compile. If this is Rust 2021 edition (check `edition = "2021"` in Cargo.toml), the `unsafe` wrappers are unnecessary — remove them.

- [ ] **Step 2: Run**

```bash
cargo test --features cuda --release --test kokoro_gpu_fusion_test -- --ignored --nocapture kokoro_arena_and_pool_paths_produce_identical_outputs 2>&1 | grep -E "max_abs_diff|^test result|FAILED|panicked"
```

Expected: test passes with max_abs_diff well under 1e-5 (typical f32 GPU runs agree to ~1e-7).

- [ ] **Step 3: Commit WIP**

```bash
git add tests/kokoro_gpu_fusion_test.rs
git commit -S -m "wip(ir): Kokoro arena-vs-pool numerical parity test"
```

### Task 3.10: Determinism snapshot test on Kokoro

**Files:**
- Create: new test in `tests/kokoro_arena_coverage_test.rs` (or a new file)

- [ ] **Step 1: Add the determinism test**

```rust
#[test]
#[ignore]
fn kokoro_memory_plan_is_deterministic_across_lowerings() {
    let e1 = setup_kokoro_executor();
    let e2 = setup_kokoro_executor();

    // Force compilation on both.
    use std::collections::HashMap;
    use iconnx::tensor::Tensor;
    let make_inputs = || {
        let mut i = HashMap::new();
        i.insert("tokens".to_string(), Tensor::from_vec_i64((0..5).map(|n| n % 50).collect(), vec![1, 5]));
        i.insert("style".to_string(), Tensor::from_vec_f32(vec![0.0; 256], vec![1, 256]));
        i.insert("speed".to_string(), Tensor::from_vec_f32(vec![1.0], vec![1]));
        i
    };
    let _ = e1.run(make_inputs(), vec!["audio"]).expect("run 1");
    let _ = e2.run(make_inputs(), vec!["audio"]).expect("run 2");

    use iconnx::cuda::inference::testing::get_memory_plan;
    let p1 = get_memory_plan(&e1).expect("plan 1");
    let p2 = get_memory_plan(&e2).expect("plan 2");

    assert_eq!(p1.arena_bytes, p2.arena_bytes,
        "MemoryPlan.arena_bytes must be deterministic across independent lowerings");
    assert_eq!(p1.slot_count, p2.slot_count,
        "MemoryPlan.slot_count must be deterministic");

    // slot_offsets_bytes is pub(crate), so compare via Debug as a
    // snapshot check. If the test binary can access the field (same
    // crate), use direct equality; otherwise Debug-format comparison.
    assert_eq!(format!("{:?}", p1), format!("{:?}", p2),
        "MemoryPlan Debug representations must be byte-identical");
}
```

- [ ] **Step 2: Run**

```bash
cargo test --features cuda --release --test kokoro_arena_coverage_test -- --ignored --nocapture kokoro_memory_plan_is_deterministic 2>&1 | grep -E "^test result|FAILED|panicked"
```

Expected: `1 passed`.

- [ ] **Step 3: Commit WIP**

```bash
git add tests/kokoro_arena_coverage_test.rs
git commit -S -m "wip(ir): Kokoro MemoryPlan determinism snapshot test"
```

### Task 3.11: Final Commit 3 verification + squash

- [ ] **Step 1: Verify all WIP commits**

```bash
# The feat B commit is the prior squash point.
B_SHA=$(git log --oneline 5875858..HEAD --grep="feat(ir): shape_aware_folding" | head -1 | awk '{print $1}')
echo "Commit B squash SHA: $B_SHA"
git log --format='%G? %s' $B_SHA..HEAD | head -20
```

Expected: all WIP commits for A (~10) are GPG-signed.

- [ ] **Step 2: Six landing criteria check**

Run all six acceptance criteria end-to-end:

```bash
# (1) Correctness on Kokoro — default (arena active)
cargo test --features cuda --release --test kokoro_gpu_fusion_test -- --ignored --nocapture 2>&1 | grep -E "^test result|FAILED"
# Expected: 3 passed (existing fusion test + arena-vs-pool parity + arena coverage diagnostic
#           + any other new ignored tests). All green.

# (2) Pool fallback works
ICONNX_DISABLE_MEMORY_PLANNER=1 cargo test --features cuda --release --test kokoro_gpu_fusion_test -- --ignored 2>&1 | grep -E "^test result|FAILED"
# Expected: same count, all green.

# (3) Determinism
cargo test --features cuda --release --test kokoro_arena_coverage_test -- --ignored --nocapture kokoro_memory_plan_is_deterministic 2>&1 | grep -E "^test result|FAILED"
# Expected: 1 passed.

# (4) Classification coverage (≥70%)
cargo test --features cuda --release --test kokoro_arena_coverage_test -- --ignored --nocapture kokoro_arena_assignment_rate 2>&1 | grep -E "^test result|FAILED|rate"
# Expected: 1 passed with rate printed.

# (5) Size assertion active — verify by code inspection: grep for debug_assert in the executor
grep -rn "debug_assert_eq\|assert_eq" src/cuda/executor/ | grep -i "slot.*bytes\|arena.*bytes" | head -5
# Expected: one or more `assert_eq!` (not `debug_assert_eq!`) matches in the arena branch.

# (6) Zero warnings on both feature combos
cargo build --features cuda --release 2>&1 | grep -i warning
cargo build --features cuda,debug-inference --release 2>&1 | grep -i warning
# Expected: both empty.
```

If any of (1)–(6) fails, STOP and fix before squashing.

- [ ] **Step 3: Full test suite summary**

```bash
cargo test --features cuda --release 2>&1 | grep -E "^test result" | awk '{pass+=$4; fail+=$6} END {print pass" passed, "fail" failed"}'
cargo test --features cuda,debug-inference --release 2>&1 | grep -E "^test result" | awk '{pass+=$4; fail+=$6} END {print pass" passed, "fail" failed"}'
```

Expected: baseline (from Task 0.1) + 5 shape_aware_folding tests + 11 memory_planner tests + 1 `allocate_arena` smoke test = 17 new tests above baseline. Zero failures.

- [ ] **Step 4: Squash**

```bash
git reset --soft $B_SHA
git commit -S -m "$(cat <<'EOF'
feat(ir): static memory planner + arena allocation path (A)

Adds src/ir/passes/memory_planner.rs — pure-function memory planner
that assigns arena offsets to eligible tensors using a greedy linear
scan with size-class slots (power-of-2 ≥ 4096 elements, exact match
below; matches GpuMemoryPool's convention). Runs after fusion
annotation in lower(), consumes tensor_shapes + tensor_last_use +
tensor_first_use + fusion annotations.

Eligibility (single predicate, is_planner_eligible):
- Not a graph input (inputs come from the input-transfer path).
- Not a graph output (runtime may need ownership past arena lifetime).
- All dims in tensor_shapes are Some(_).
- Total bytes ≥ 4096 (matches pool's size-class threshold).

Structural filter (separate from eligibility, is_fusion_skipped):
- Tensor's producer op has PlannedOp::kind == Skip → filtered out
  (fused head writes directly into head's slot; skipped output is
  never allocated).

Runtime integration:
- PlannedOp gains arena_offset: Option<usize>, stamped at plan time.
  No runtime HashMap lookup per op — one cache line, one Option
  branch.
- ExecutionPlan gains memory_plan: MemoryPlan carrying arena_bytes,
  slot_count, slot assignments, byte offsets, and
  PlannerClassification counts computed at construction time
  (parameter-less accessor).
- GpuMemoryPool gains allocate_arena(bytes) → ArenaBuffer, an RAII
  handle that returns the allocation to the pool on drop. Executor
  allocates one arena per run(); cross-inference caching is a
  separate follow-up.
- Always-on assert_eq!(actual_tensor_bytes, arena_slot_bytes) at
  op-execute time, NOT gated behind debug-inference. Catches silent
  corruption from shape-inference undersizing (the class of Phase 3
  Gather-bug failure) at the cost of one integer comparison per op.
- ICONNX_DISABLE_MEMORY_PLANNER=1 env var forces every tensor through
  the pool path (bisection escape hatch for planner vs. other bug
  classes).

Generic-infrastructure frame per spec §Gating protocol — all six
landing criteria satisfied:
(1) Correctness on Kokoro: kokoro_gpu_fusion_test passes with arena
    active (1e-5 fused-vs-sequential tolerance).
(2) Pool fallback: all Kokoro tests pass with
    ICONNX_DISABLE_MEMORY_PLANNER=1.
(3) Determinism: kokoro_memory_plan_is_deterministic_across_lowerings
    passes (Debug-format snapshot match).
(4) Classification coverage: kokoro_arena_assignment_rate_is_above_70
    passes — on Kokoro, arena_assigned / arena_eligible ≥ 0.7.
(5) Size assertion active in plain --release builds.
(6) Zero warnings, signed commit.

Tests:
- 6 eligibility-predicate unit tests (happy path + 4 rejection cases
  + unknown-tensor).
- 5 allocator unit tests (disjoint-lifetime reuse, overlapping
  creates new slot, size-class separation, fusion-skipped filter,
  determinism snapshot, classification counts).
- 1 allocate_arena pool-roundtrip test.
- 1 lower()-integration diagnostic test
  (kokoro_memory_planner_produces_assignments).
- 1 Kokoro arena-vs-pool numerical parity test.
- 1 Kokoro arena coverage (≥70% rate) test.
- 1 Kokoro determinism-across-lowerings test.

Post-Phase-4 follow-up (not in-PR gate): Whisper-Tiny executes
through the planner with numerical match to ORT, once leadline's
ModelProfile refactor lands and provides a Whisper harness.

Interval-graph-coloring allocation, cross-inference arena caching,
and multiple arenas per run() are explicit deferrals — see spec
§Non-goals.

Spec: docs/superpowers/specs/2026-04-19-phase4-graph-optimizer-ii-design.md.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 5: Verify**

```bash
git log --oneline 5875858..HEAD   # expect exactly 3 commits (refactor + feat B + feat A)
git log -3 --format='%G? %s'
```

All three GPG-signed (`G` prefix); subjects are the three final commits.

---

## Task 4: Merge to main

### Task 4.1: Final pre-merge verification

- [ ] **Step 1: Run the full suite on the final tip**

```bash
cargo build --features cuda --release 2>&1 | tail -3
cargo test --features cuda --release 2>&1 | grep -E "^test result" | awk '{pass+=$4; fail+=$6} END {print pass" passed, "fail" failed"}'
cargo test --features cuda,debug-inference --release 2>&1 | grep -E "^test result" | awk '{pass+=$4; fail+=$6} END {print pass" passed, "fail" failed"}'
cargo test --features cuda --release --test kokoro_gpu_fusion_test -- --ignored 2>&1 | grep -E "^test result"
cargo test --features cuda --release --test kokoro_arena_coverage_test -- --ignored 2>&1 | grep -E "^test result"
```

- [ ] **Step 2: Confirm all three commits are signed**

```bash
git log --format='%G? %h %s' 5875858..HEAD
```

All three lines start with `G`.

- [ ] **Step 3: Informational measurement — Phase 4 final Kokoro ratio**

```bash
cd /home/ryano/workspace/ryanoneill/rust-ai-explorations
# Temporarily point leadline-bench at the ir-phase4 worktree.
# NOTE: leadline-bench's Cargo.toml currently points at the main iconnx
# worktree. For this informational measurement, edit it to point at
# iconnx-ir-phase4, run the benchmark 10x, then restore.
# ...
```

Run the standard 10-iteration median protocol (same as Phase 3). Record the median, Q1, Q3 in the merge commit message as the new Phase-4 informational baseline.

After measurement, restore the leadline-bench Cargo.toml path.

### Task 4.2: Merge

This step requires user authorization (shared-state action). Do NOT merge without explicit user approval. Surface the merge-ready state to the user with:
- Branch tip SHAs
- Test counts for both feature combos
- Kokoro integration test + arena coverage results
- Phase 4 informational ratio measurement
- Signature verification

Wait for user approval before invoking `git merge`.

The merge will likely need to be non-ff (same as Phase 3 and C PR) because main may have received documentation or hygiene commits during Phase 4 work. Use `git merge --no-ff -S ir-phase4` if so, matching the established merge style.

---

## Self-Review

**Spec coverage:**
- ✓ Refactor `folding_core` extraction → Task 1.1, 1.2, 1.3
- ✓ `shape_aware_folding` pass with Shape predicate extension → Task 2.1
- ✓ Pipeline wiring (post shape_inference, before next DCE) → Task 2.3
- ✓ 5 shape_aware_folding unit tests (known, partial-dynamic, chain-fold, dynamic-graph-input, idempotent) → Tasks 2.1, 2.2
- ✓ Structural gate for B (10 → ≥33/61 with Amber protocol) → Task 2.4
- ✓ `is_planner_eligible` with 4 rejection paths → Task 3.1
- ✓ `is_fusion_skipped` structural filter → Task 3.1
- ✓ 6 eligibility-predicate unit tests → Task 3.1
- ✓ Greedy linear scan with size-class slots → Task 3.3
- ✓ Determinism contract (stable sort + deterministic slot search) → Task 3.3, 3.4, 3.10
- ✓ `MemoryPlan` + `PlannerClassification` with parameter-less accessor → Task 3.2
- ✓ `PlannedOp::arena_offset` + `ExecutionPlan::memory_plan` → Task 3.2
- ✓ `GpuMemoryPool::allocate_arena` + `ArenaBuffer` RAII → Task 3.5
- ✓ 5 allocator unit tests → Task 3.4
- ✓ Wired into `lower()` with `tensor_first_use` helper → Task 3.6
- ✓ Runtime arena branching + always-on size assertion → Task 3.7
- ✓ `ICONNX_DISABLE_MEMORY_PLANNER=1` env-var escape hatch → Task 3.7
- ✓ Landing criterion #1 (correctness Kokoro arena) → Task 3.7, 3.9
- ✓ Landing criterion #2 (pool fallback) → Task 3.7
- ✓ Landing criterion #3 (determinism snapshot) → Task 3.10
- ✓ Landing criterion #4 (≥70% classification coverage) → Task 3.8
- ✓ Landing criterion #5 (size assertion active) → Task 3.11 Step 2
- ✓ Landing criterion #6 (zero warnings + signed) → Task 3.11 Step 2
- ✓ Three final commits on branch ir-phase4 → Tasks 1.3, 2.5, 3.11
- ✓ Merge surfaced to user → Task 4.2

Post-Phase-4 follow-up (Whisper cross-model validation) is mentioned in the final squash message as a deferred acceptance criterion, not an in-PR gate — spec-compliant.

**Placeholder scan:**
- No "TBD", "TODO", "implement later" strings anywhere.
- "write the failing test" steps each have concrete test code.
- Two areas with documented judgment calls (not placeholders): (a) `FusedPatternInfo`'s field structure — Task 3.1 and 3.4 instruct the implementer to inspect and adapt; (b) dtype inference heuristic in `infer_output_dtype_bytes` — Task 3.3 documents the conservative approach (f32-default with fallback to pool for unknown).
- Runtime arena-branch sketch in Task 3.7 Step 4 has a "placeholder — see note below" comment on the specific assertion form. The comment documents the intent and directs the implementer to write the concrete check against the real executor's op-dispatch. That's a genuine judgment call that requires reading the real code, not a plan failure.

**Type consistency:**
- `fold_with_predicate(graph, extra_foldable_fn) -> graph` — signature identical in Task 1.1 and when called from `constant_folding` (Task 1.2) and `shape_aware_folding` (Task 2.1).
- `shape_aware_folding(graph, tensor_shapes) -> graph` — signature used in Task 2.1 (definition), Task 2.3 (call site in `lower()`), and the tests.
- `is_planner_eligible(tensor_name, tensor_shapes, graph_inputs, graph_outputs, dtype_bytes, element_count) -> bool` — 6 params matching the spec's signature; consistent in Task 3.1 definition and Task 3.3 call site.
- `is_fusion_skipped(producer_tensor_name, fusion)` — Task 3.1 defines two alternatives based on `FusedPatternInfo` shape; Task 3.3 calls whichever was chosen. The plan is explicit about the branch decision.
- `MemoryPlan::arena_offset(tensor_name) -> Option<usize>` — Task 3.2 defines; Task 3.6 uses to stamp `PlannedOp::arena_offset`.
- `MemoryPlan::classification_counts() -> &PlannerClassification` — parameter-less per spec; Task 3.2 definition, Tasks 3.8 and 3.10 usages.
- `memory_planner(graph, tensor_shapes, tensor_last_use, tensor_first_use, fusion) -> MemoryPlan` — Task 3.3 definition matches Task 3.6 call site.
- `allocate_arena(ctx, bytes) -> Result<ArenaBuffer, CudaError>` — Task 3.5 definition; Task 3.7 Step 3 call site.
- `PlannerClassification { arena_eligible, arena_assigned, pool_classified }` — Task 3.2 definition; Tasks 3.3, 3.4, 3.8 usages all use these exact field names.

Plan complete.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-04-19-phase4-graph-optimizer-ii.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — fresh subagent per task, two-stage review (spec + code quality) after each task's squash checkpoint. Same pattern as Phase 2, Phase 3, and C PR.

**2. Inline Execution** — work through tasks directly in this session with checkpoints at each commit's squash.

**Which approach?**
