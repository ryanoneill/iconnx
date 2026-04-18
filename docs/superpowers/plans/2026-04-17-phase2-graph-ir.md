# Phase 2: Graph Intermediate Representation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the 5,047-line `GpuGraphExecutor` into three types — `OptimizableGraph` (CPU data), `ExecutionPlan` (frozen GPU state), and `Executor` (model-independent GPU runtime) — with passes as pure functions. Behavioral no-op; 510 existing tests pass unchanged.

**Architecture:** New `src/ir/` module holds the IR types + passes + lowering. New `src/cuda/executor/` module holds the runtime (dispatch split into 10 sub-module files, each under 1,000 lines). Existing `GpuGraphExecutor` becomes a ~100-line strangler-fig shim over the new types so external tests don't change.

**Tech Stack:** Rust 2021, cudarc replacement `garboard` for CUDA, `bytemuck::Pod` for kernel args, RefCell for the few documented lazy-memoization exceptions.

**Spec:** `docs/superpowers/specs/2026-04-17-phase2-graph-ir-design.md`.

---

## Setup

### Task 0: Worktree + branch

**Files:** none yet.

- [ ] **Step 1: Create a worktree branch off current main**

```bash
cd /home/ryano/workspace/ryanoneill/iconnx
git fetch --all
git branch -f ir-phase2 main          # create branch at current main HEAD (954fc43)
git worktree add /home/ryano/workspace/ryanoneill/iconnx-ir-phase2 ir-phase2
cd /home/ryano/workspace/ryanoneill/iconnx-ir-phase2
git log -1 --oneline   # expect 954fc43 docs(ir): spec for Phase 2 graph IR refactor
```

- [ ] **Step 2: Sanity-check build + test suite before making any changes**

```bash
cargo build --features cuda --release 2>&1 | tail -3
cargo test --features cuda --release -- --include-ignored 2>&1 | grep -E "^test result" | tail -5
```

Expected: build clean with 0 warnings; 510 tests pass.

**All subsequent work happens in `/home/ryano/workspace/ryanoneill/iconnx-ir-phase2`.**

---

## Target File Structure

After the full plan lands, the new layout is:

```
src/
├── ir/                                   — NEW
│   ├── mod.rs                            — re-exports
│   ├── graph.rs                          — OptimizableGraph, GraphNode, GraphInput, OptimizableGraphBuilder
│   ├── plan.rs                           — ExecutionPlan, PlannedOp, OpKind, NodePrecomputed, PrecomputedSlice
│   ├── lowering.rs                       — fn lower(graph, &ctx) -> Result<ExecutionPlan>
│   └── passes/
│       ├── mod.rs                        — re-exports
│       ├── topo_sort.rs                  — fn topo_sort
│       ├── precompute.rs                 — fn precompute_params
│       └── fusion.rs                     — fn detect_fusion (CPU-tensor adapter over existing pattern-matching)
├── cuda/
│   ├── executor/                         — NEW
│   │   ├── mod.rs                        — Executor struct + run() dispatch (<200 lines)
│   │   ├── elementwise.rs                — binary arith + unary math + comparisons + softmax/leaky_relu
│   │   ├── layout.rs                     — concat/expand/gather/pad/resize/scatter_nd/slice/transpose/where/reshape/unsqueeze/squeeze/shape/constant/constant_of_shape/nonzero/cast/range/cumsum
│   │   ├── conv.rs                       — Conv, ConvTranspose
│   │   ├── reduction.rs                  — ReduceMean, ReduceSum, LayerNormalization
│   │   ├── matmul.rs                     — Gemm, MatMul
│   │   ├── lstm.rs                       — LSTM forward (delegates to cudnn::rnn::lstm_forward)
│   │   ├── stft.rs                       — STFT
│   │   ├── fused.rs                      — DivRsqrt, MulSinPowMulAdd, AddMulAdd, Gelu
│   │   └── misc.rs                       — and/or/not + anything else
│   └── inference/
│       ├── mod.rs                        — ~100-line strangler GpuGraphExecutor
│       ├── fusion/                       — unchanged (used as internal lib by ir::passes::fusion)
│       ├── gpu_event_timer.rs            — unchanged
│       └── testing.rs                    — unchanged
└── lib.rs                                — `pub mod ir;` added
```

Deletions from today's `src/cuda/inference/mod.rs` (tracked through Commit 3):
- `ExecutionNode`, `PrecomputedSlice`, `NodePrecomputed` structs (move into `ir::plan`)
- `LstmWeightCache`, `LstmPlanCache` structs (move into `ir::plan`)
- `NodeExecutionLog`, `ProfileData`, `TensorLocation` stay (runtime-logging helpers)
- `detect_fused_patterns` method (replaced by `ir::passes::fusion::detect_fusion`)
- `try_precompute_*` methods (replaced by `ir::passes::precompute::precompute_params`)
- `topological_sort` method (replaced by `ir::passes::topo_sort::topo_sort`)
- `compute_tensor_last_use` method (replaced by `ir::lowering::compute_tensor_last_use`)
- The entire `execute_gpu_operator` / `execute_gpu_operator_with_precomputed` / per-op methods (moved into `cuda::executor/*`)

---

## Commit 1: Build the `ir::` module (types + passes + lowering)

**Commit goal:** `src/ir/` exists with all types, passes, and a working `lower()`. Unit tests cover pure-logic passes. Nothing in `cuda::inference` uses the new types yet. All 510 existing tests remain green.

### Task 1.1: Add `ir` module skeleton

**Files:**
- Create: `src/ir/mod.rs`
- Create: `src/ir/passes/mod.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Create empty `src/ir/mod.rs`**

```rust
//! Graph intermediate representation for iconnx.
//!
//! Splits "what to compute" (`OptimizableGraph`) from the frozen compiled
//! form (`ExecutionPlan`) from the GPU runtime (`cuda::executor::Executor`).
//! See docs/superpowers/specs/2026-04-17-phase2-graph-ir-design.md.

pub mod graph;
pub mod passes;
pub mod plan;
pub mod lowering;

pub use graph::{GraphInput, GraphNode, OptimizableGraph, OptimizableGraphBuilder};
pub use lowering::lower;
pub use plan::{ExecutionPlan, NodePrecomputed, OpKind, PlannedOp, PrecomputedSlice};
```

- [ ] **Step 2: Create `src/ir/passes/mod.rs`**

```rust
//! Graph passes — pure functions over `OptimizableGraph`.

pub mod fusion;
pub mod precompute;
pub mod topo_sort;

pub use fusion::detect_fusion;
pub use precompute::precompute_params;
pub use topo_sort::topo_sort;
```

- [ ] **Step 3: Wire `ir` into the crate root**

In `src/lib.rs`, add the new `pub mod ir;` next to the existing module declarations (keep alphabetized if the list is sorted; otherwise append):

```rust
pub mod ir;
```

- [ ] **Step 4: Create empty placeholder files so the crate compiles**

Each of these files gets a single `//! placeholder — see later tasks` module comment so the crate still compiles through the rest of Task 1.1:

```bash
for f in src/ir/graph.rs src/ir/plan.rs src/ir/lowering.rs \
         src/ir/passes/topo_sort.rs src/ir/passes/precompute.rs \
         src/ir/passes/fusion.rs; do
  printf '//! placeholder — see Phase 2 plan\n' > "$f"
done
```

Then remove the `pub use` re-exports temporarily (the symbols don't exist yet). Re-enable each re-export in its implementing task. Easiest: comment out the `pub use` lines in `src/ir/mod.rs` and `src/ir/passes/mod.rs` for now.

- [ ] **Step 5: Build**

```bash
cargo build --features cuda --release 2>&1 | tail -3
```

Expected: `Finished` with zero warnings. If warnings appear about unused modules, add `#[allow(dead_code)]` at the top of the offending file; we'll clean up as we fill in.

- [ ] **Step 6: Commit (WIP checkpoint, amend later)**

```bash
git add src/ir/ src/lib.rs
git commit -S -m "wip(ir): scaffold ir module skeleton"
```

This commit will be amended/squashed into Commit 1's final commit at Task 1.14.

### Task 1.2: Define `GraphInput`, `GraphNode`, `OptimizableGraph`

**Files:**
- Modify: `src/ir/graph.rs`
- Create: `src/ir/graph.rs` (test module at bottom)

- [ ] **Step 1: Write the failing test** — add at the bottom of `src/ir/graph.rs` (the whole file content for this step):

```rust
//! Pure-CPU computation graph ready for optimization passes.
//!
//! `OptimizableGraph` is the input to every pass and to `lower()`.
//! It has no GPU state: initializers are CPU-side `crate::tensor::Tensor`
//! values so constant-folding passes (Phase 3) can read them directly.

use std::collections::HashMap;

use crate::attributes::NodeAttributes;
use crate::tensor::Tensor;

/// A named graph input with optional dimensions.
///
/// `shape[i] == None` means the dimension is dynamic. Phase 3's
/// shape-inference pass fills in concrete values where possible.
#[derive(Clone, Debug)]
pub struct GraphInput {
    pub name: String,
    pub shape: Vec<Option<usize>>,
}

/// A single ONNX-style node.
#[derive(Clone, Debug)]
pub struct GraphNode {
    pub name: String,
    pub op_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attributes: NodeAttributes,
}

/// A computation graph before lowering — pure CPU data.
#[derive(Clone, Debug, Default)]
pub struct OptimizableGraph {
    pub inputs: Vec<GraphInput>,
    pub outputs: Vec<String>,
    pub nodes: Vec<GraphNode>,
    pub initializers: HashMap<String, Tensor>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optimizable_graph_default_is_empty() {
        let g = OptimizableGraph::default();
        assert!(g.inputs.is_empty());
        assert!(g.outputs.is_empty());
        assert!(g.nodes.is_empty());
        assert!(g.initializers.is_empty());
    }

    #[test]
    fn graph_node_stores_fields() {
        let node = GraphNode {
            name: "n1".into(),
            op_type: "Add".into(),
            inputs: vec!["a".into(), "b".into()],
            outputs: vec!["c".into()],
            attributes: NodeAttributes::new(),
        };
        assert_eq!(node.name, "n1");
        assert_eq!(node.inputs.len(), 2);
        assert_eq!(node.outputs.len(), 1);
    }
}
```

- [ ] **Step 2: Build + run the new tests**

```bash
cargo test --lib --features cuda ir::graph 2>&1 | tail -5
```

Expected: 2 passed.

- [ ] **Step 3: Re-enable the `pub use` in `src/ir/mod.rs`**

Uncomment the `pub use graph::{GraphInput, GraphNode, OptimizableGraph, OptimizableGraphBuilder};` line, but change it to `pub use graph::{GraphInput, GraphNode, OptimizableGraph};` (Builder comes next task). Build once more to confirm no missing symbol:

```bash
cargo build --features cuda --release 2>&1 | tail -3
```

- [ ] **Step 4: Commit (WIP)**

```bash
git add src/ir/graph.rs src/ir/mod.rs
git commit -S -m "wip(ir): OptimizableGraph + GraphNode + GraphInput types"
```

### Task 1.3: `OptimizableGraphBuilder` with add/build semantics

**Files:**
- Modify: `src/ir/graph.rs`

- [ ] **Step 1: Write failing tests for builder**

Append to `src/ir/graph.rs` (inside the module, before `#[cfg(test)]`):

```rust
/// Mutable builder that accumulates initializers and nodes, then produces
/// an `OptimizableGraph` via [`OptimizableGraphBuilder::build`]. The
/// builder preserves insertion order of nodes (the topo-sort pass reorders
/// later if the insertion order isn't already a valid topological order).
#[derive(Default)]
pub struct OptimizableGraphBuilder {
    inputs: Vec<GraphInput>,
    outputs: Vec<String>,
    nodes: Vec<GraphNode>,
    initializers: HashMap<String, Tensor>,
}

impl OptimizableGraphBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_input(&mut self, name: String, shape: Vec<Option<usize>>) -> &mut Self {
        self.inputs.push(GraphInput { name, shape });
        self
    }

    pub fn add_output(&mut self, name: String) -> &mut Self {
        self.outputs.push(name);
        self
    }

    pub fn add_initializer(&mut self, name: String, tensor: Tensor) -> &mut Self {
        self.initializers.insert(name, tensor);
        self
    }

    pub fn add_node(
        &mut self,
        name: impl Into<String>,
        op_type: impl Into<String>,
        inputs: Vec<String>,
        outputs: Vec<String>,
        attributes: NodeAttributes,
    ) -> &mut Self {
        self.nodes.push(GraphNode {
            name: name.into(),
            op_type: op_type.into(),
            inputs,
            outputs,
            attributes,
        });
        self
    }

    pub fn build(self) -> OptimizableGraph {
        OptimizableGraph {
            inputs: self.inputs,
            outputs: self.outputs,
            nodes: self.nodes,
            initializers: self.initializers,
        }
    }
}
```

And in the `#[cfg(test)] mod tests` block, add:

```rust
    #[test]
    fn builder_records_nodes_in_insertion_order() {
        let mut b = OptimizableGraphBuilder::new();
        b.add_node("n1", "Add", vec!["a".into(), "b".into()], vec!["c".into()], NodeAttributes::new());
        b.add_node("n2", "Mul", vec!["c".into(), "d".into()], vec!["e".into()], NodeAttributes::new());
        let g = b.build();
        assert_eq!(g.nodes.len(), 2);
        assert_eq!(g.nodes[0].name, "n1");
        assert_eq!(g.nodes[1].name, "n2");
    }

    #[test]
    fn builder_records_initializers_by_name() {
        let mut b = OptimizableGraphBuilder::new();
        let t = Tensor::from_data_shape(vec![1.0, 2.0, 3.0], vec![3]);
        b.add_initializer("w".into(), t);
        let g = b.build();
        assert!(g.initializers.contains_key("w"));
    }

    #[test]
    fn builder_records_inputs_and_outputs() {
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("in0".into(), vec![None, Some(4)]);
        b.add_output("out0".into());
        let g = b.build();
        assert_eq!(g.inputs[0].name, "in0");
        assert_eq!(g.outputs, vec!["out0".to_string()]);
    }
```

- [ ] **Step 2: Run tests**

```bash
cargo test --lib --features cuda ir::graph 2>&1 | tail -8
```

Expected: 5 passed.

Note: if `Tensor::from_data_shape` isn't the exact constructor name, grep `src/tensor.rs` for the existing public constructor and use it. Common alternatives: `Tensor::Float32(Array::from_shape_vec(...))` or a builder; pick whichever test-constructs a simple f32 tensor today.

- [ ] **Step 3: Update `src/ir/mod.rs` re-exports** to include `OptimizableGraphBuilder`:

```rust
pub use graph::{GraphInput, GraphNode, OptimizableGraph, OptimizableGraphBuilder};
```

- [ ] **Step 4: Build**

```bash
cargo build --features cuda --release 2>&1 | tail -3
```

- [ ] **Step 5: Commit (WIP)**

```bash
git add src/ir/graph.rs src/ir/mod.rs
git commit -S -m "wip(ir): OptimizableGraphBuilder with add/build"
```

### Task 1.4: Define `PrecomputedSlice`, `NodePrecomputed`, `OpKind`, `PlannedOp`

**Files:**
- Modify: `src/ir/plan.rs`

- [ ] **Step 1: Check the current `PrecomputedSlice` location** so we preserve fields exactly:

```bash
grep -A 10 "pub(crate) struct PrecomputedSlice" src/cuda/inference/mod.rs
```

Expected output (field names — copy exactly):

```rust
pub(crate) struct PrecomputedSlice {
    pub(crate) starts: Option<Vec<i64>>,
    pub(crate) ends: Option<Vec<i64>>,
    pub(crate) axes: Option<Vec<i64>>,
    pub(crate) steps: Option<Vec<i64>>,
}
```

- [ ] **Step 2: Write `src/ir/plan.rs` with the three plan-side types**

Whole file contents (replace placeholder):

```rust
//! Frozen compiled form of `OptimizableGraph`.
//!
//! An `ExecutionPlan` is model-specific and self-contained: given any
//! compatible `Executor`, it runs. `lstm_weights` is eagerly populated at
//! lowering. `lstm_plan_cache` is the one documented exception to "frozen"
//! — it lazily memoizes seq_length-dependent cuDNN descriptors that aren't
//! known at plan-construction time.

use std::cell::RefCell;
use std::collections::HashMap;

use crate::cuda::cudnn::PackedLstmWeights;
use crate::cuda::tensor::GpuTensor;
use crate::ir::graph::{GraphInput, GraphNode};

/// Pre-computed Slice parameters for a single Slice node. Each field is
/// `Some` iff the corresponding ONNX Slice input was an initializer or a
/// Constant node value available at lowering time.
#[derive(Clone, Debug, Default)]
pub struct PrecomputedSlice {
    pub starts: Option<Vec<i64>>,
    pub ends: Option<Vec<i64>>,
    pub axes: Option<Vec<i64>>,
    pub steps: Option<Vec<i64>>,
}

/// Bundle of precomputation attached to a single `PlannedOp`.
///
/// All fields default to `None`; `None` means "no precomputation
/// available, dispatch falls back to the runtime read path." Populated
/// by `ir::passes::precompute_params`.
#[derive(Clone, Debug, Default)]
pub struct NodePrecomputed {
    pub slice: Option<PrecomputedSlice>,
    pub reshape_shape: Option<Vec<i64>>,
    pub unsqueeze_axes: Option<Vec<i64>>,
    pub squeeze_axes: Option<Vec<i64>>,
}

/// How a single op in the plan's op sequence should be dispatched.
#[derive(Clone, Debug)]
pub enum OpKind {
    /// Standalone op — dispatch by `op_type`.
    Regular,
    /// First node of a fused chain — dispatch the fused kernel.
    /// The `FusedPattern` captures which variant and its parameters.
    FusedHead(crate::cuda::inference::fusion::FusedPattern),
    /// Sub-node of an earlier fused head — already handled, skip in the loop.
    Skip,
}

/// One entry in the plan's topo-sorted op sequence.
#[derive(Clone, Debug)]
pub struct PlannedOp {
    pub node: GraphNode,
    pub kind: OpKind,
    pub precomputed: NodePrecomputed,
}

/// LSTM weight cache: keyed by (W_ptr as usize, R_ptr as usize) so two
/// LSTM nodes that share weights share a packed-weight buffer.
/// Populated eagerly in `lower()`; plain HashMap because the plan is
/// frozen after construction.
pub type LstmWeightCache = HashMap<(usize, usize), PackedLstmWeights>;

/// LSTM plan cache: keyed by (W_ptr, R_ptr, seq_length). Populated lazily
/// during inference because seq_length isn't known at plan-construction
/// time. The `RefCell` is the single documented exception to "frozen" —
/// the plan's semantic state is unchanged; only the memoization table grows.
pub type LstmPlanCache = HashMap<(usize, usize, usize), garboard::RnnPlan<'static>>;

/// Frozen plan — pairs of (operation sequence + model weights).
pub struct ExecutionPlan {
    pub ops: Vec<PlannedOp>,
    pub weights: HashMap<String, GpuTensor>,
    pub static_i64: HashMap<String, Vec<i64>>,
    pub lstm_weights: LstmWeightCache,
    pub lstm_plan_cache: RefCell<LstmPlanCache>,
    pub dynamic_candidates:
        HashMap<String, crate::cuda::inference::fusion::FusedPatternInfo>,
    pub tensor_last_use: HashMap<String, usize>,
    pub graph_inputs: Vec<GraphInput>,
    pub graph_outputs: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_precomputed_default_is_all_none() {
        let p = NodePrecomputed::default();
        assert!(p.slice.is_none());
        assert!(p.reshape_shape.is_none());
        assert!(p.unsqueeze_axes.is_none());
        assert!(p.squeeze_axes.is_none());
    }

    #[test]
    fn precomputed_slice_default_is_all_none() {
        let s = PrecomputedSlice::default();
        assert!(s.starts.is_none());
        assert!(s.ends.is_none());
        assert!(s.axes.is_none());
        assert!(s.steps.is_none());
    }
}
```

- [ ] **Step 3: Re-export `LstmPlanCache` and `LstmWeightCache` type aliases from `src/ir/mod.rs`** too, so downstream code can say `ir::LstmPlanCache`:

```rust
pub use plan::{
    ExecutionPlan, LstmPlanCache, LstmWeightCache, NodePrecomputed, OpKind, PlannedOp,
    PrecomputedSlice,
};
```

- [ ] **Step 4: Build + run the new tests**

```bash
cargo test --lib --features cuda ir::plan 2>&1 | tail -5
```

Expected: 2 passed.

- [ ] **Step 5: Commit (WIP)**

```bash
git add src/ir/plan.rs src/ir/mod.rs
git commit -S -m "wip(ir): ExecutionPlan + PlannedOp + PrecomputedSlice types"
```

### Task 1.5: `topo_sort` pass (pure)

**Files:**
- Modify: `src/ir/passes/topo_sort.rs`

- [ ] **Step 1: Write failing tests first**

Whole file contents:

```rust
//! Topological-sort pass — reorders `OptimizableGraph::nodes` so that every
//! node's inputs are produced by an earlier node (or by an initializer /
//! graph input). Returns `Err` on cycles.

use std::collections::{HashMap, HashSet};

use crate::cuda::context::CudaError;
use crate::ir::graph::OptimizableGraph;
use crate::ir::graph::GraphNode;

pub fn topo_sort(mut graph: OptimizableGraph) -> Result<OptimizableGraph, CudaError> {
    let node_by_output: HashMap<String, usize> = graph
        .nodes
        .iter()
        .enumerate()
        .flat_map(|(i, n)| n.outputs.iter().map(move |o| (o.clone(), i)))
        .collect();

    let available: HashSet<String> = graph
        .initializers
        .keys()
        .cloned()
        .chain(graph.inputs.iter().map(|i| i.name.clone()))
        .collect();

    let mut sorted: Vec<GraphNode> = Vec::with_capacity(graph.nodes.len());
    let mut visited = vec![false; graph.nodes.len()];
    let mut on_stack = vec![false; graph.nodes.len()];

    fn visit(
        idx: usize,
        nodes: &[GraphNode],
        node_by_output: &HashMap<String, usize>,
        available: &HashSet<String>,
        visited: &mut [bool],
        on_stack: &mut [bool],
        sorted: &mut Vec<GraphNode>,
    ) -> Result<(), CudaError> {
        if visited[idx] {
            return Ok(());
        }
        if on_stack[idx] {
            return Err(CudaError::Kernel(format!(
                "Cycle detected in graph at node '{}'",
                nodes[idx].name
            )));
        }
        on_stack[idx] = true;
        for input in &nodes[idx].inputs {
            if input.is_empty() || available.contains(input) {
                continue;
            }
            if let Some(&dep_idx) = node_by_output.get(input) {
                visit(dep_idx, nodes, node_by_output, available, visited, on_stack, sorted)?;
            }
            // If input isn't produced by any node and isn't in `available`,
            // that's a dangling reference — let runtime dispatch surface it.
        }
        on_stack[idx] = false;
        visited[idx] = true;
        sorted.push(nodes[idx].clone());
        Ok(())
    }

    for i in 0..graph.nodes.len() {
        visit(
            i,
            &graph.nodes,
            &node_by_output,
            &available,
            &mut visited,
            &mut on_stack,
            &mut sorted,
        )?;
    }

    graph.nodes = sorted;
    Ok(graph)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attributes::NodeAttributes;
    use crate::ir::graph::OptimizableGraphBuilder;

    fn mk_node(name: &str, op: &str, inputs: &[&str], outputs: &[&str]) -> GraphNode {
        GraphNode {
            name: name.into(),
            op_type: op.into(),
            inputs: inputs.iter().map(|s| (*s).to_string()).collect(),
            outputs: outputs.iter().map(|s| (*s).to_string()).collect(),
            attributes: NodeAttributes::new(),
        }
    }

    #[test]
    fn already_sorted_is_identity() {
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("a".into(), vec![]);
        b.add_node("n1", "Add", vec!["a".into()], vec!["b".into()], NodeAttributes::new());
        b.add_node("n2", "Mul", vec!["b".into()], vec!["c".into()], NodeAttributes::new());
        let sorted = topo_sort(b.build()).unwrap();
        let names: Vec<_> = sorted.nodes.iter().map(|n| n.name.clone()).collect();
        assert_eq!(names, vec!["n1", "n2"]);
    }

    #[test]
    fn reverse_order_is_reordered() {
        let mut g = OptimizableGraph::default();
        g.inputs.push(GraphInput { name: "a".into(), shape: vec![] });
        g.nodes.push(mk_node("n2", "Mul", &["b"], &["c"]));
        g.nodes.push(mk_node("n1", "Add", &["a"], &["b"]));
        let sorted = topo_sort(g).unwrap();
        let names: Vec<_> = sorted.nodes.iter().map(|n| n.name.clone()).collect();
        assert_eq!(names, vec!["n1", "n2"]);
    }

    #[test]
    fn cycle_is_error() {
        let mut g = OptimizableGraph::default();
        g.nodes.push(mk_node("n1", "Add", &["b"], &["a"]));
        g.nodes.push(mk_node("n2", "Add", &["a"], &["b"]));
        let err = topo_sort(g).unwrap_err();
        match err {
            CudaError::Kernel(msg) => assert!(msg.contains("Cycle detected")),
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
```

Note: `use crate::ir::graph::GraphInput;` will need to be added at the top if tests reference `GraphInput` directly. Adjust imports until clean.

- [ ] **Step 2: Run new tests (must pass — this is the first test that exercises production code)**

```bash
cargo test --lib --features cuda ir::passes::topo_sort 2>&1 | tail -8
```

Expected: 3 passed.

- [ ] **Step 3: Uncomment the `pub use topo_sort::topo_sort;` in `src/ir/passes/mod.rs`.**

- [ ] **Step 4: Build**

```bash
cargo build --features cuda --release 2>&1 | tail -3
```

- [ ] **Step 5: Commit (WIP)**

```bash
git add src/ir/passes/topo_sort.rs src/ir/passes/mod.rs
git commit -S -m "wip(ir): topo_sort pass"
```

### Task 1.6: `precompute_params` pass — Slice

**Files:**
- Modify: `src/ir/passes/precompute.rs`

**Approach:** `precompute_params` returns a tuple `(OptimizableGraph, HashMap<String, NodePrecomputed>)`. The map is keyed by node name. Lowering attaches each map entry to the corresponding `PlannedOp`. We do it as a map rather than mutating graph nodes because `OptimizableGraph::nodes` are pure `GraphNode`s — no precomputation fields on them. Keeps the graph pure.

- [ ] **Step 1: Grep today's four `try_precompute_*` functions** to see the exact logic that must be preserved:

```bash
grep -n "fn try_precompute_" src/cuda/inference/mod.rs
sed -n '595,720p' src/cuda/inference/mod.rs > /tmp/precompute_orig.rs
wc -l /tmp/precompute_orig.rs
```

The four functions are on lines 595, 644, 657, 672 of today's `mod.rs`. Read them, then port the same logic in pure form (no `&self`; read initializers + Constant-node values from `OptimizableGraph::initializers`).

- [ ] **Step 2: Write failing test + skeleton**

Whole file contents:

```rust
//! Precompute slice/reshape/unsqueeze/squeeze parameters from initializer
//! or Constant-node values. Pure function: reads `OptimizableGraph::initializers`
//! and Constant nodes, returns a map of node-name → `NodePrecomputed`.

use std::collections::HashMap;

use crate::ir::graph::{GraphNode, OptimizableGraph};
use crate::ir::plan::{NodePrecomputed, PrecomputedSlice};
use crate::tensor::Tensor;

pub fn precompute_params(
    graph: &OptimizableGraph,
) -> HashMap<String, NodePrecomputed> {
    let i64_by_name = collect_i64_values(graph);
    let mut out = HashMap::new();
    for node in &graph.nodes {
        let pre = match node.op_type.as_str() {
            "Slice" => NodePrecomputed {
                slice: precompute_slice(node, &i64_by_name),
                ..Default::default()
            },
            "Reshape" => NodePrecomputed {
                reshape_shape: precompute_reshape_shape(node, &i64_by_name),
                ..Default::default()
            },
            "Unsqueeze" => NodePrecomputed {
                unsqueeze_axes: precompute_axes(node, &i64_by_name),
                ..Default::default()
            },
            "Squeeze" => NodePrecomputed {
                squeeze_axes: precompute_axes(node, &i64_by_name),
                ..Default::default()
            },
            _ => continue,
        };
        out.insert(node.name.clone(), pre);
    }
    out
}

/// Build a lookup of all Int64 tensor values available at lowering:
/// initializers + output tensors of Constant nodes whose value attribute
/// is an Int64 tensor.
fn collect_i64_values(graph: &OptimizableGraph) -> HashMap<String, Vec<i64>> {
    let mut out: HashMap<String, Vec<i64>> = HashMap::new();
    for (name, tensor) in &graph.initializers {
        if let Tensor::Int64(_) = tensor {
            out.insert(name.clone(), tensor.as_slice_i64().to_vec());
        }
    }
    for node in &graph.nodes {
        if node.op_type != "Constant" {
            continue;
        }
        if let Some(value) = node.attributes.get_tensor("value") {
            if let Tensor::Int64(_) = value {
                if let Some(output) = node.outputs.first() {
                    out.insert(output.clone(), value.as_slice_i64().to_vec());
                }
            }
        }
    }
    out
}

fn precompute_slice(
    node: &GraphNode,
    i64_by_name: &HashMap<String, Vec<i64>>,
) -> Option<PrecomputedSlice> {
    // ONNX Slice: inputs = [data, starts, ends, axes?, steps?]
    if node.inputs.len() < 3 {
        return None;
    }
    let mut pre = PrecomputedSlice::default();
    pre.starts = i64_by_name.get(&node.inputs[1]).cloned();
    pre.ends = i64_by_name.get(&node.inputs[2]).cloned();
    if let Some(ax_name) = node.inputs.get(3) {
        pre.axes = i64_by_name.get(ax_name).cloned();
    }
    if let Some(st_name) = node.inputs.get(4) {
        pre.steps = i64_by_name.get(st_name).cloned();
    }
    if pre.starts.is_none() && pre.ends.is_none() && pre.axes.is_none() && pre.steps.is_none() {
        None
    } else {
        Some(pre)
    }
}

fn precompute_reshape_shape(
    node: &GraphNode,
    i64_by_name: &HashMap<String, Vec<i64>>,
) -> Option<Vec<i64>> {
    // ONNX Reshape: inputs = [data, shape]
    node.inputs.get(1).and_then(|n| i64_by_name.get(n)).cloned()
}

fn precompute_axes(
    node: &GraphNode,
    i64_by_name: &HashMap<String, Vec<i64>>,
) -> Option<Vec<i64>> {
    // ONNX Unsqueeze/Squeeze: inputs = [data, axes]
    node.inputs.get(1).and_then(|n| i64_by_name.get(n)).cloned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attributes::NodeAttributes;
    use crate::ir::graph::OptimizableGraphBuilder;

    #[test]
    fn slice_with_initializer_starts_ends_is_populated() {
        let mut b = OptimizableGraphBuilder::new();
        b.add_initializer("starts".into(), Tensor::from_i64_slice(&[0, 0], vec![2]));
        b.add_initializer("ends".into(), Tensor::from_i64_slice(&[4, 8], vec![2]));
        b.add_node(
            "sl",
            "Slice",
            vec!["data".into(), "starts".into(), "ends".into()],
            vec!["out".into()],
            NodeAttributes::new(),
        );
        let map = precompute_params(&b.build());
        let pre = map.get("sl").unwrap();
        assert_eq!(pre.slice.as_ref().unwrap().starts.as_deref(), Some(&[0i64, 0][..]));
        assert_eq!(pre.slice.as_ref().unwrap().ends.as_deref(), Some(&[4i64, 8][..]));
        assert!(pre.slice.as_ref().unwrap().axes.is_none());
    }

    #[test]
    fn reshape_shape_from_constant_node_is_picked_up() {
        let mut b = OptimizableGraphBuilder::new();
        let mut attrs = NodeAttributes::new();
        attrs.insert_tensor("value", Tensor::from_i64_slice(&[-1, 8], vec![2]));
        b.add_node("shape_const", "Constant", vec![], vec!["shp".into()], attrs);
        b.add_node(
            "rs",
            "Reshape",
            vec!["data".into(), "shp".into()],
            vec!["out".into()],
            NodeAttributes::new(),
        );
        let map = precompute_params(&b.build());
        assert_eq!(map["rs"].reshape_shape.as_deref(), Some(&[-1i64, 8][..]));
    }

    #[test]
    fn non_applicable_op_has_no_entry() {
        let mut b = OptimizableGraphBuilder::new();
        b.add_node("n1", "Add", vec!["a".into(), "b".into()], vec!["c".into()], NodeAttributes::new());
        let map = precompute_params(&b.build());
        assert!(!map.contains_key("n1"));
    }
}
```

Note: `Tensor::from_i64_slice` is the assumed test constructor; grep `src/tensor.rs` for the actual helper used in iconnx tests (likely `Tensor::Int64(Array1::from_vec(vec![...]).into_dyn())` or a `from_data_shape` variant). Replace calls consistently.

- [ ] **Step 3: Run tests**

```bash
cargo test --lib --features cuda ir::passes::precompute 2>&1 | tail -8
```

Expected: 3 passed.

- [ ] **Step 4: Re-enable `pub use precompute::precompute_params;` in `src/ir/passes/mod.rs`.**

- [ ] **Step 5: Commit (WIP)**

```bash
git add src/ir/passes/precompute.rs src/ir/passes/mod.rs
git commit -S -m "wip(ir): precompute_params pass covers slice/reshape/unsqueeze/squeeze"
```

### Task 1.7: `detect_fusion` pass adapter

**Background (leadline's note):** Today's `cuda::inference::fusion::detect_fused_patterns` takes `(&[ExecutionNode], &HashMap<String, GpuTensor>, &IconnxCudaContext)` and reads weight values via `t.to_host_f32(ctx)`. In the new pipeline, fusion runs on `OptimizableGraph` where initializers are already CPU `Tensor`s, so the D2H read is unnecessary. The adapter builds an input shape that the existing pattern-matching logic understands, but sources scalar values from CPU tensors directly.

- [ ] **Step 1: Read the existing detection entrypoint**

```bash
grep -B 2 -A 30 "pub(crate) fn detect_fused_patterns" src/cuda/inference/fusion/detection.rs | head -50
```

Expected signature:

```rust
pub(crate) fn detect_fused_patterns(
    nodes: &[ExecutionNode],
    weights: &HashMap<String, GpuTensor>,
    ctx: &IconnxCudaContext,
) -> (HashMap<String, FusedPatternInfo>, HashSet<String>, HashMap<String, FusedPatternInfo>)
// returns (fused_patterns, nodes_to_skip, dynamic_candidates)
```

Note the return tuple — the third element is the `dynamic_candidates` map.

- [ ] **Step 2: Decide on the adapter shape.** There are two options:

  1. **Add a new constructor inside `cuda::inference::fusion::detection` that accepts CPU-side values** (e.g., `detect_fused_patterns_from_cpu(nodes, initializers_i64, initializers_f32)`). The existing GPU-side function stays for code paths that don't have CPU values handy (there aren't any after this PR, but the old function stays as dead code for now; delete in Commit 3).

  2. **Refactor the existing function** to accept a `Fn(&str) -> Option<Vec<f32>>` closure for scalar lookup, and have the old GPU caller pass a closure that calls `to_host_f32(ctx)`. This is a bigger refactor.

  **Go with option 1** for the minimum touch of existing fusion code. Commit 3 deletes the GPU caller.

- [ ] **Step 3: Add a `detect_fused_patterns_from_cpu` helper**

Modify `src/cuda/inference/fusion/detection.rs`. Near the top, add:

```rust
use crate::tensor::Tensor;
```

Above (or below) `detect_fused_patterns`, add:

```rust
/// CPU-side variant used by `ir::passes::fusion`. Identical pattern-matching
/// logic to `detect_fused_patterns` but sources scalar values from CPU
/// `Tensor` initializers instead of round-tripping them through the GPU via
/// `to_host_f32(ctx)`. Eliminates the D2H copy during fusion detection.
pub(crate) fn detect_fused_patterns_from_cpu(
    nodes: &[ExecutionNode],
    initializers: &HashMap<String, Tensor>,
) -> (
    HashMap<String, FusedPatternInfo>,
    std::collections::HashSet<String>,
    HashMap<String, FusedPatternInfo>,
) {
    // Implementation: identical to `detect_fused_patterns` except that
    // wherever the GPU version calls `t.to_host_f32(ctx)` to read a scalar,
    // read from `initializers.get(name).and_then(|t| match t { Tensor::Float32(_) => Some(t.as_slice().to_vec()), _ => None })`.
    // Share helper functions by factoring scalar reads behind a closure.
    //
    // Do NOT copy-paste the whole detection; extract the core logic into a
    // `detect_from_scalar_lookup` function that both entry points call.
    todo!("factor shared logic — see Step 4");
}
```

- [ ] **Step 4: Factor shared logic.** Refactor `detect_fused_patterns` so that its core is a function parameterized by a scalar-lookup closure:

```rust
fn detect_with_scalar_lookup(
    nodes: &[ExecutionNode],
    read_f32: impl Fn(&str) -> Option<Vec<f32>>,
) -> (HashMap<String, FusedPatternInfo>, HashSet<String>, HashMap<String, FusedPatternInfo>) {
    // original body, with every `t.to_host_f32(ctx)` replaced by `read_f32(name)`.
}

pub(crate) fn detect_fused_patterns(
    nodes: &[ExecutionNode],
    weights: &HashMap<String, GpuTensor>,
    ctx: &IconnxCudaContext,
) -> ... {
    detect_with_scalar_lookup(nodes, |name| {
        weights.get(name).and_then(|t| t.to_host_f32(ctx).ok())
    })
}

pub(crate) fn detect_fused_patterns_from_cpu(
    nodes: &[ExecutionNode],
    initializers: &HashMap<String, Tensor>,
) -> ... {
    detect_with_scalar_lookup(nodes, |name| {
        initializers.get(name).and_then(|t| match t {
            Tensor::Float32(_) => Some(t.as_slice().to_vec()),
            _ => None,
        })
    })
}
```

Run the existing fusion tests to make sure the factoring didn't regress:

```bash
cargo test --lib --features cuda cuda::inference::fusion 2>&1 | tail -5
```

Expected: all existing fusion tests pass (same count as before the factoring).

- [ ] **Step 5: Write `src/ir/passes/fusion.rs` — the adapter**

Whole file contents:

```rust
//! Fusion-pattern detection. Thin adapter that bridges
//! `OptimizableGraph` into the existing `cuda::inference::fusion` detection
//! library. Operates entirely on CPU `Tensor` values — no GPU context
//! required, no D2H copies.

use std::collections::{HashMap, HashSet};

use crate::attributes::NodeAttributes;
use crate::cuda::inference::fusion::{FusedPatternInfo, detect_fused_patterns_from_cpu};
use crate::ir::graph::{GraphNode, OptimizableGraph};

/// Result of the fusion pass — attached to an `OptimizableGraph` so
/// lowering can thread the annotations into each `PlannedOp`.
#[derive(Default, Debug, Clone)]
pub struct FusionAnnotations {
    pub heads: HashMap<String, FusedPatternInfo>,
    pub skipped: HashSet<String>,
    pub dynamic_candidates: HashMap<String, FusedPatternInfo>,
}

pub fn detect_fusion(graph: &OptimizableGraph) -> FusionAnnotations {
    // The detection library currently takes `&[ExecutionNode]`. Build a
    // throwaway ExecutionNode list from the graph's nodes. The extra fields
    // on ExecutionNode (precomputed_*) are irrelevant to fusion detection —
    // they're filled in at lowering and don't affect pattern matching.
    let exec_nodes: Vec<crate::cuda::inference::ExecutionNode> = graph
        .nodes
        .iter()
        .map(graph_node_to_execution_node)
        .collect();

    let (heads, skipped, dynamic_candidates) =
        detect_fused_patterns_from_cpu(&exec_nodes, &graph.initializers);
    FusionAnnotations { heads, skipped, dynamic_candidates }
}

fn graph_node_to_execution_node(n: &GraphNode) -> crate::cuda::inference::ExecutionNode {
    crate::cuda::inference::ExecutionNode {
        name: n.name.clone(),
        op_type: n.op_type.clone(),
        inputs: n.inputs.clone(),
        outputs: n.outputs.clone(),
        attributes: n.attributes.clone(),
        precomputed_slice: None,
        precomputed_reshape_shape: None,
        precomputed_unsqueeze_axes: None,
        precomputed_squeeze_axes: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::graph::OptimizableGraphBuilder;

    #[test]
    fn detect_returns_empty_on_empty_graph() {
        let g = OptimizableGraphBuilder::new().build();
        let a = detect_fusion(&g);
        assert!(a.heads.is_empty());
        assert!(a.skipped.is_empty());
        assert!(a.dynamic_candidates.is_empty());
    }

    // Pattern-specific detection tests live alongside the existing fusion
    // detection tests (`src/cuda/inference/fusion/detection.rs`) — those
    // still run against the now-shared `detect_with_scalar_lookup` core.
    // This adapter module only needs to verify it wires correctly.
}
```

Note: `ExecutionNode` needs to be `pub(crate)` (already is) but reachable from `crate::cuda::inference`; check `src/cuda/inference/mod.rs` near line 366 and add `pub(crate) use ExecutionNode;` at the `mod.rs` top if not already exported.

- [ ] **Step 6: Run tests**

```bash
cargo test --lib --features cuda ir::passes::fusion 2>&1 | tail -5
cargo test --lib --features cuda cuda::inference::fusion 2>&1 | tail -5
```

Expected: 1 new test passes + existing fusion tests pass (same count as before).

- [ ] **Step 7: Re-enable `pub use fusion::{FusionAnnotations, detect_fusion};` in `src/ir/passes/mod.rs`.**

- [ ] **Step 8: Commit (WIP)**

```bash
git add src/cuda/inference/fusion/detection.rs src/ir/passes/fusion.rs src/ir/passes/mod.rs
git commit -S -m "wip(ir): fusion adapter reads CPU tensors directly"
```

### Task 1.8: `lower()` function

**Files:**
- Modify: `src/ir/lowering.rs`

- [ ] **Step 1: Study the existing GPU-side setup** so we know what to replicate:

```bash
sed -n '400,540p' src/cuda/inference/mod.rs   # view new() constructor and add_initializer
```

Key operations from `add_initializer`:
- For each initializer: upload to GPU via `GpuTensor::from_host_*`, store in `weights: HashMap<String, GpuTensor>`.
- For Int64 initializers: also record in `static_i64_cache` (now `static_i64`).

Key operations from the LSTM weight-pack path (today deferred to first `run()`):
- For each LSTM node, pack its (W, R, B) weights via `pack_lstm_weights_for_cudnn(&ctx, w, r, b, hidden, input, num_dirs)`. Store by `(w.device_ptr() as usize, r.device_ptr() as usize)`.

- [ ] **Step 2: Write the lower function**

Whole file contents of `src/ir/lowering.rs`:

```rust
//! Lower an `OptimizableGraph` into an `ExecutionPlan`.
//!
//! Orchestrates the passes (topo sort → precompute → fusion) on CPU,
//! uploads initializers to GPU, eagerly packs LSTM weights, and assembles
//! the final `Vec<PlannedOp>` with each op's `OpKind` + `NodePrecomputed`
//! attached.

use std::cell::RefCell;
use std::collections::HashMap;

use crate::cuda::context::{CudaError, IconnxCudaContext};
use crate::cuda::cudnn::pack_lstm_weights_for_cudnn;
use crate::cuda::tensor::GpuTensor;
use crate::ir::graph::{GraphNode, OptimizableGraph};
use crate::ir::passes::{detect_fusion, precompute_params, topo_sort};
use crate::ir::plan::{
    ExecutionPlan, LstmPlanCache, LstmWeightCache, NodePrecomputed, OpKind, PlannedOp,
};
use crate::tensor::Tensor;

pub fn lower(
    graph: OptimizableGraph,
    ctx: &IconnxCudaContext,
) -> Result<ExecutionPlan, CudaError> {
    // 1. Topological sort.
    let graph = topo_sort(graph)?;

    // 2. Precompute params (pure, no GPU).
    let precomputed_by_name = precompute_params(&graph);

    // 3. Detect fusion (pure, no GPU).
    let fusion = detect_fusion(&graph);

    // 4. Upload initializers to GPU.
    let weights = upload_initializers(&graph, ctx)?;

    // 5. Build static_i64 for fast lookup at runtime.
    let static_i64 = collect_static_i64(&graph);

    // 6. Pack LSTM weights eagerly.
    let lstm_weights = pack_lstm_weights_eagerly(&graph, &weights, ctx)?;

    // 7. Build PlannedOp sequence from sorted nodes + fusion + precomputed.
    let ops: Vec<PlannedOp> = graph
        .nodes
        .iter()
        .map(|node| {
            let kind = if fusion.skipped.contains(&node.name) {
                OpKind::Skip
            } else if let Some(head) = fusion.heads.get(&node.name) {
                OpKind::FusedHead(head.pattern.clone())
            } else {
                OpKind::Regular
            };
            let precomputed = precomputed_by_name
                .get(&node.name)
                .cloned()
                .unwrap_or_default();
            PlannedOp { node: node.clone(), kind, precomputed }
        })
        .collect();

    // 8. Compute tensor_last_use for memory-pool return timing.
    let tensor_last_use = compute_tensor_last_use(&ops, &graph.outputs, &weights);

    Ok(ExecutionPlan {
        ops,
        weights,
        static_i64,
        lstm_weights,
        lstm_plan_cache: RefCell::new(LstmPlanCache::new()),
        dynamic_candidates: fusion.dynamic_candidates,
        tensor_last_use,
        graph_inputs: graph.inputs,
        graph_outputs: graph.outputs,
    })
}

fn upload_initializers(
    graph: &OptimizableGraph,
    ctx: &IconnxCudaContext,
) -> Result<HashMap<String, GpuTensor>, CudaError> {
    let mut out = HashMap::new();
    for (name, tensor) in &graph.initializers {
        let gpu = match tensor {
            Tensor::Float32(_) => GpuTensor::from_host_f32(
                ctx, &tensor.as_slice(), tensor.shape().to_vec())?,
            Tensor::Int64(_) => GpuTensor::from_host_i64(
                ctx, &tensor.as_slice_i64(), tensor.shape().to_vec())?,
            Tensor::Int32(_) => GpuTensor::from_host_i32(
                ctx, &tensor.as_slice_i32(), tensor.shape().to_vec())?,
            _ => continue,   // matches today's GpuGraphExecutor::add_initializer behavior
        };
        out.insert(name.clone(), gpu);
    }
    Ok(out)
}

fn collect_static_i64(graph: &OptimizableGraph) -> HashMap<String, Vec<i64>> {
    let mut out: HashMap<String, Vec<i64>> = HashMap::new();
    for (name, tensor) in &graph.initializers {
        if let Tensor::Int64(_) = tensor {
            out.insert(name.clone(), tensor.as_slice_i64().to_vec());
        }
    }
    for node in &graph.nodes {
        if node.op_type != "Constant" {
            continue;
        }
        if let Some(value) = node.attributes.get_tensor("value") {
            if let Tensor::Int64(_) = value {
                if let Some(output) = node.outputs.first() {
                    out.insert(output.clone(), value.as_slice_i64().to_vec());
                }
            }
        }
    }
    out
}

fn pack_lstm_weights_eagerly(
    graph: &OptimizableGraph,
    weights: &HashMap<String, GpuTensor>,
    ctx: &IconnxCudaContext,
) -> Result<LstmWeightCache, CudaError> {
    let mut out = LstmWeightCache::new();
    for node in &graph.nodes {
        if node.op_type != "LSTM" {
            continue;
        }
        // ONNX LSTM input order: X, W, R, B?, ..., initial_h?, initial_c?
        let w_name = node.inputs.get(1).ok_or_else(|| {
            CudaError::Kernel(format!("LSTM '{}' missing W input", node.name))
        })?;
        let r_name = node.inputs.get(2).ok_or_else(|| {
            CudaError::Kernel(format!("LSTM '{}' missing R input", node.name))
        })?;
        let b_name = node.inputs.get(3);   // Option<&String> — empty "" means absent
        let w = weights.get(w_name).ok_or_else(|| {
            CudaError::Kernel(format!("LSTM '{}' W initializer '{}' not found", node.name, w_name))
        })?;
        let r = weights.get(r_name).ok_or_else(|| {
            CudaError::Kernel(format!("LSTM '{}' R initializer '{}' not found", node.name, r_name))
        })?;
        let b = b_name.and_then(|n| if n.is_empty() { None } else { weights.get(n) });
        let key = (w.device_ptr(ctx) as usize, r.device_ptr(ctx) as usize);
        if out.contains_key(&key) {
            continue;  // shared weights — pack once
        }
        let w_shape = w.shape();
        let num_directions = w_shape[0];
        let hidden_size = w_shape[1] / 4;
        let input_size = w_shape[2];
        let packed = pack_lstm_weights_for_cudnn(
            ctx, w, r, b, hidden_size, input_size, num_directions,
        )?;
        out.insert(key, packed);
    }
    Ok(out)
}

fn compute_tensor_last_use(
    ops: &[PlannedOp],
    graph_outputs: &[String],
    weights: &HashMap<String, GpuTensor>,
) -> HashMap<String, usize> {
    let mut last_use: HashMap<String, usize> = HashMap::new();
    for (idx, op) in ops.iter().enumerate() {
        for input in &op.node.inputs {
            if !input.is_empty() && !weights.contains_key(input) {
                last_use.insert(input.clone(), idx);
            }
        }
    }
    for out_name in graph_outputs {
        if !weights.contains_key(out_name) {
            last_use.insert(out_name.clone(), ops.len());
        }
    }
    last_use
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::graph::OptimizableGraphBuilder;

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn lower_empty_graph_produces_empty_plan() {
        let ctx = IconnxCudaContext::new().expect("ctx");
        let g = OptimizableGraphBuilder::new().build();
        let plan = lower(g, &ctx).unwrap();
        assert!(plan.ops.is_empty());
        assert!(plan.weights.is_empty());
        assert!(plan.lstm_weights.is_empty());
    }
}
```

Note: `GpuTensor::device_ptr(ctx)` — check the exact name; grep `src/cuda/tensor.rs` for the current method (might be `device_ptr` or `raw_device_ptr` or `as_raw_device_ptr`). Use whichever matches today's LSTM dispatch code in `src/cuda/inference/mod.rs` around line 1715.

- [ ] **Step 3: Run tests**

```bash
cargo test --lib --features cuda ir::lowering 2>&1 | tail -5
cargo build --features cuda --release 2>&1 | tail -3
```

Expected: 1 test passes (the GPU-ignored one is `#[ignore]`d), clean build.

- [ ] **Step 4: Enable `pub use lowering::lower;` in `src/ir/mod.rs`.**

- [ ] **Step 5: Run GPU integration test explicitly** (requires GPU):

```bash
cargo test --lib --features cuda ir::lowering -- --include-ignored 2>&1 | tail -5
```

Expected: 1 passed.

- [ ] **Step 6: Commit (WIP)**

```bash
git add src/ir/lowering.rs src/ir/mod.rs
git commit -S -m "wip(ir): lower() function + upload/pack helpers"
```

### Task 1.9: Squash WIP commits into a single "Commit 1"

- [ ] **Step 1: Check current WIP chain**

```bash
git log --oneline main..HEAD
```

You should see ~7 `wip(ir):` commits.

- [ ] **Step 2: Soft-reset to main, then commit as one**

```bash
git reset --soft main
git status   # should show all the src/ir/ files + src/lib.rs + src/cuda/inference/fusion/detection.rs staged
git commit -S -m "feat(ir): scaffold ir module (graph/plan/passes/lowering)

New \`src/ir/\` module introduces:
- \`OptimizableGraph\`, \`GraphNode\`, \`GraphInput\`, \`OptimizableGraphBuilder\`
  (src/ir/graph.rs) — pure CPU data, no GPU state
- \`ExecutionPlan\`, \`PlannedOp\`, \`OpKind\`, \`NodePrecomputed\`,
  \`PrecomputedSlice\` (src/ir/plan.rs) — frozen compiled form; \`lstm_weights\`
  eagerly populated at lowering, \`lstm_plan_cache\` as documented RefCell
  exception
- Three pure-function passes: \`topo_sort\`, \`precompute_params\`,
  \`detect_fusion\` (src/ir/passes/)
- \`lower(OptimizableGraph, &ctx) -> ExecutionPlan\` (src/ir/lowering.rs)

The fusion adapter reads CPU \`Tensor\` initializers directly, replacing
today's \`to_host_f32(ctx)\` round-trip. Shared pattern-matching core
(\`detect_with_scalar_lookup\`) factored in \`cuda/inference/fusion/detection.rs\`.

Nothing in \`cuda::inference\` uses the new types yet — all 510 existing
tests still pass unchanged.

See docs/superpowers/specs/2026-04-17-phase2-graph-ir-design.md.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 3: Verify full test suite still passes (this is the Commit-1 acceptance gate)**

```bash
cargo build --features cuda --release 2>&1 | tail -5
cargo test --features cuda --release -- --include-ignored 2>&1 | grep -E "^test result" | tail -25
```

Expected: 510 existing tests + ~15-20 new ir-module tests all green. Zero warnings.

- [ ] **Step 4: Verify log**

```bash
git log --pretty=format:'%G? %h %s' main..HEAD
```

Expected: one signed commit.

---

## Commit 2: Carve `Executor` out of `GpuGraphExecutor`

**Commit goal:** `src/cuda/executor/` exists as a working Executor type. Its `run(plan, inputs, outputs)` dispatches by walking `plan.ops`. `GpuGraphExecutor::run` transitionally constructs an `ExecutionPlan` and delegates. All 510 tests green. Benchmark ratio within Phase 1 band.

### Task 2.1: `src/cuda/executor/` skeleton + empty Executor

**Files:**
- Create: `src/cuda/executor/mod.rs`
- Create: `src/cuda/executor/{elementwise,layout,conv,reduction,matmul,lstm,stft,fused,misc}.rs` (nine empty files)
- Modify: `src/cuda/mod.rs`

- [ ] **Step 1: Create the module tree**

```bash
mkdir -p src/cuda/executor
for f in elementwise layout conv reduction matmul lstm stft fused misc; do
  printf '//! %s dispatch — see Phase 2 plan.\n' "$f" > "src/cuda/executor/$f.rs"
done
```

- [ ] **Step 2: Write `src/cuda/executor/mod.rs`**

```rust
//! GPU runtime that dispatches an `ExecutionPlan` against hardware.
//!
//! `Executor` owns model-independent state: device, stream, cuBLAS, cuDNN,
//! kernel caches, memory pool, per-run Int64 cache. Running a plan is a
//! single pass over `plan.ops`, dispatching each op to its category handler
//! via the compact match in `run()`.

mod conv;
mod elementwise;
mod fused;
mod layout;
mod lstm;
mod matmul;
mod misc;
mod reduction;
mod stft;

use std::cell::RefCell;
use std::collections::HashMap;

use crate::cuda::context::{CudaError, IconnxCudaContext};
use crate::cuda::conv::ConvKernelCache;
use crate::cuda::kernels::KernelCache;
use crate::cuda::lstm::LstmKernelCache;
use crate::cuda::memory_pool::GpuMemoryPool;
use crate::cuda::ops::OpsKernelCache;
use crate::cuda::reduction::ReductionKernelCache;
use crate::cuda::stft::StftKernelCache;
use crate::cuda::inference::fusion::FusedKernelCache;    // TODO: move fused cache out of inference/
use crate::cuda::tensor::GpuTensor;
use crate::ir::ExecutionPlan;
use crate::tensor::Tensor;

pub struct Executor {
    pub ctx: IconnxCudaContext,
    pub elementwise_kernels: KernelCache,
    pub reduction_kernels: ReductionKernelCache,
    pub conv_kernels: ConvKernelCache,
    pub lstm_kernels: LstmKernelCache,
    pub ops_kernels: OpsKernelCache,
    pub stft_kernels: StftKernelCache,
    pub fused_kernels: FusedKernelCache,
    pub memory_pool: RefCell<GpuMemoryPool>,
    pub i64_cache: RefCell<HashMap<u64, Vec<i64>>>,
    #[cfg(feature = "debug-inference")]
    pub validator: RefCell<Option<crate::cuda::validation::TensorValidator>>,
}

impl Executor {
    pub fn new() -> Result<Self, CudaError> {
        let ctx = IconnxCudaContext::new()?;
        let elementwise_kernels = KernelCache::new(&ctx)?;
        let reduction_kernels = ReductionKernelCache::new(&ctx)?;
        let conv_kernels = ConvKernelCache::new(&ctx)?;
        let lstm_kernels = LstmKernelCache::new(&ctx)?;
        let ops_kernels = OpsKernelCache::new(&ctx)?;
        let stft_kernels = StftKernelCache::new(&ctx)?;
        let fused_kernels = FusedKernelCache::new(&ctx)?;
        Ok(Self {
            ctx,
            elementwise_kernels,
            reduction_kernels,
            conv_kernels,
            lstm_kernels,
            ops_kernels,
            stft_kernels,
            fused_kernels,
            memory_pool: RefCell::new(GpuMemoryPool::new()),
            i64_cache: RefCell::new(HashMap::new()),
            #[cfg(feature = "debug-inference")]
            validator: RefCell::new(None),
        })
    }

    /// Dispatch a whole plan's op sequence and return the requested outputs
    /// as CPU tensors.
    pub fn run(
        &self,
        plan: &ExecutionPlan,
        inputs: HashMap<String, Tensor>,
        output_names: &[&str],
    ) -> Result<HashMap<String, Tensor>, CudaError> {
        // Implementation in Task 2.3.
        let _ = (plan, inputs, output_names);
        unimplemented!("filled in Task 2.3");
    }
}
```

- [ ] **Step 3: Add `pub mod executor;` to `src/cuda/mod.rs`** (near the other module declarations).

- [ ] **Step 4: Build**

```bash
cargo build --features cuda --release 2>&1 | tail -3
```

Expected: build clean. `unimplemented!()` is a panic but not a compile error.

- [ ] **Step 5: Commit (WIP)**

```bash
git add src/cuda/executor/ src/cuda/mod.rs
git commit -S -m "wip(executor): scaffold cuda::executor module with new()"
```

### Task 2.2: Executor::run skeleton — the compact dispatch match

**Files:**
- Modify: `src/cuda/executor/mod.rs`
- Create: a single-method stub on `Executor` per category sub-module

- [ ] **Step 1: Decide the handler-method signatures.** Each category file will hold methods like:

```rust
// src/cuda/executor/elementwise.rs
use crate::cuda::context::CudaError;
use crate::cuda::tensor::GpuTensor;
use crate::ir::PlannedOp;

use super::Executor;

impl Executor {
    pub(crate) fn dispatch_elementwise(
        &self,
        op: &PlannedOp,
        values: &std::collections::HashMap<String, GpuTensor>,
    ) -> Result<GpuTensor, CudaError> {
        unimplemented!("filled in Task 2.4");
    }
}
```

Each category (elementwise, layout, conv, reduction, matmul, lstm, stft, fused, misc) gets one `dispatch_<category>` method, all with the same signature. (LSTM may need extra parameters like `lstm_plan_cache` access — revisit in Task 2.9.)

Write all nine stubs now:

```bash
for cat in elementwise layout conv reduction matmul lstm stft fused misc; do
  cat > src/cuda/executor/$cat.rs <<EOF
//! $cat op dispatch.

use std::collections::HashMap;

use crate::cuda::context::CudaError;
use crate::cuda::tensor::GpuTensor;
use crate::ir::PlannedOp;

use super::Executor;

impl Executor {
    pub(crate) fn dispatch_${cat}(
        &self,
        op: &PlannedOp,
        values: &HashMap<String, GpuTensor>,
    ) -> Result<GpuTensor, CudaError> {
        let _ = (op, values);
        unimplemented!("filled in later task")
    }
}
EOF
done
```

- [ ] **Step 2: Write the compact dispatch match in `Executor::run`**

Replace the `unimplemented!` in `src/cuda/executor/mod.rs::run` with the real body. Port the relevant pieces from today's `src/cuda/inference/mod.rs::run` (lines 2866-3378). Keep structure but replace per-op logic with a single match routing to `dispatch_<category>`:

```rust
pub fn run(
    &self,
    plan: &ExecutionPlan,
    inputs: HashMap<String, Tensor>,
    output_names: &[&str],
) -> Result<HashMap<String, Tensor>, CudaError> {
    self.i64_cache.borrow_mut().clear();

    let mut values: HashMap<String, GpuTensor> = HashMap::new();

    // Upload inputs (port lines 2889-2910 of today's run).
    for (name, tensor) in inputs {
        let gpu_tensor = match &tensor {
            Tensor::Float32(_) => {
                let data = tensor.as_slice();
                GpuTensor::from_host_f32(&self.ctx, &data, tensor.shape().to_vec())?
            }
            Tensor::Int64(_) => {
                let data = tensor.as_slice_i64();
                GpuTensor::from_host_i64(&self.ctx, &data, tensor.shape().to_vec())?
            }
            Tensor::Int32(_) => {
                let data = tensor.as_slice_i32();
                GpuTensor::from_host_i32(&self.ctx, &data, tensor.shape().to_vec())?
            }
            _ => {
                let data = tensor.as_slice();
                GpuTensor::from_host_f32(&self.ctx, &data, tensor.shape().to_vec())?
            }
        };
        values.insert(name, gpu_tensor);
    }

    // Weights are NOT copied; lookups check values then plan.weights.

    // Execute.
    for (node_idx, op) in plan.ops.iter().enumerate() {
        if matches!(op.kind, OpKind::Skip) {
            continue;
        }
        let output = self.dispatch_op(op, &values, plan)?;
        self.store_outputs(&op.node, output, &mut values);
        // Memory-pool return timing — see Task 2.13 for the full gc pass.
        let _ = node_idx;
    }

    // Collect outputs (port lines 3310-3375 of today's run).
    collect_outputs(&values, output_names, &plan.weights, &self.ctx)
}

fn dispatch_op(
    &self,
    op: &PlannedOp,
    values: &HashMap<String, GpuTensor>,
    plan: &ExecutionPlan,
) -> Result<GpuTensor, CudaError> {
    if matches!(op.kind, OpKind::FusedHead(_)) {
        return self.dispatch_fused(op, values);
    }
    match op.node.op_type.as_str() {
        // Elementwise arith / unary math / comparisons / activations
        "Add" | "Sub" | "Mul" | "Div" | "Pow" | "Exp" | "Sqrt" | "Sin" | "Cos"
        | "Tanh" | "Sigmoid" | "LeakyRelu" | "Atan" | "Round" | "Floor" | "Clip"
        | "Softmax" | "Equal" | "Less" | "Greater" | "GreaterOrEqual" | "LessOrEqual"
        | "NotEqual" => self.dispatch_elementwise(op, values),

        "Conv" | "ConvTranspose" => self.dispatch_conv(op, values),

        "ReduceMean" | "ReduceSum" | "LayerNormalization" => self.dispatch_reduction(op, values),

        "Gemm" | "MatMul" => self.dispatch_matmul(op, values),

        "LSTM" => self.dispatch_lstm(op, values, plan),

        "STFT" => self.dispatch_stft(op, values),

        "And" | "Or" | "Not" => self.dispatch_misc(op, values),

        // Everything else is layout/shape/metadata
        _ => self.dispatch_layout(op, values, plan),
    }
}
```

You'll need a `store_outputs` helper (port the multi-output tensor storage logic from today's run ~line 2953) and a `collect_outputs` helper at the bottom of `mod.rs`.

- [ ] **Step 3: Add stub signatures for dispatch_lstm + dispatch_layout that take `plan` too**

Edit `src/cuda/executor/lstm.rs` and `src/cuda/executor/layout.rs` to accept a third `plan` argument (LSTM needs it for `plan.lstm_weights` and `plan.lstm_plan_cache`; layout needs it for `plan.static_i64`):

```rust
// src/cuda/executor/lstm.rs
impl Executor {
    pub(crate) fn dispatch_lstm(
        &self,
        op: &PlannedOp,
        values: &HashMap<String, GpuTensor>,
        plan: &ExecutionPlan,
    ) -> Result<GpuTensor, CudaError> {
        unimplemented!("filled in Task 2.9")
    }
}
```

Same pattern for `layout.rs`.

- [ ] **Step 4: Build**

```bash
cargo build --features cuda --release 2>&1 | tail -3
```

Expected: clean build (all nine `dispatch_*` methods are unimplemented panics but present).

- [ ] **Step 5: Commit (WIP)**

```bash
git add src/cuda/executor/
git commit -S -m "wip(executor): dispatch-match skeleton with per-category method stubs"
```

### Task 2.3: Port the elementwise dispatch

**Files:**
- Modify: `src/cuda/executor/elementwise.rs`
- Reference: `src/cuda/inference/mod.rs` (the big match in `execute_gpu_operator` and helpers)

- [ ] **Step 1: Open today's dispatch code**

```bash
grep -n "\"Add\" =>\|\"Sub\" =>\|\"Mul\" =>\|\"Div\" =>\|\"Pow\" =>" src/cuda/inference/mod.rs | head -20
```

Find the match arms that handle Add/Sub/Mul/Div/Pow/Exp/Sqrt/Sin/Cos/Tanh/Sigmoid/LeakyRelu/Atan/Round/Floor/Clip/Softmax/Equal/Less/Greater/GreaterOrEqual/LessOrEqual/NotEqual. Each arm typically calls a per-op function like `gpu_add(&self.ctx, &self.elementwise_kernels, a, b)`.

- [ ] **Step 2: Port into `src/cuda/executor/elementwise.rs`**

The goal is a single `dispatch_elementwise` method that matches on `op.node.op_type` and routes to the same per-op functions today's code uses (`gpu_add`, `gpu_sub`, `gpu_sigmoid`, etc.). Those `gpu_*` functions live in `src/cuda/{add,sub,sigmoid,...}.rs` today — **keep them where they are**; we're only moving the dispatch match, not the kernel wrappers.

Skeleton:

```rust
//! Elementwise op dispatch — binary arith, unary math, comparisons,
//! activations, and softmax.

use std::collections::HashMap;

use crate::cuda::add::gpu_add;           // adjust path to whatever today's mod path is
use crate::cuda::context::CudaError;
use crate::cuda::sub::gpu_sub;
// ... and so on for every op this file handles
use crate::cuda::tensor::GpuTensor;
use crate::ir::PlannedOp;

use super::Executor;

impl Executor {
    pub(crate) fn dispatch_elementwise(
        &self,
        op: &PlannedOp,
        values: &HashMap<String, GpuTensor>,
    ) -> Result<GpuTensor, CudaError> {
        let input_tensors = resolve_inputs(&op.node.inputs, values)?;
        let attrs = &op.node.attributes;
        match op.node.op_type.as_str() {
            "Add" => gpu_add(&self.ctx, &self.elementwise_kernels,
                             &input_tensors[0], &input_tensors[1]),
            // ... one arm per op, each exactly as in today's match ...
            _ => Err(CudaError::Kernel(format!(
                "dispatch_elementwise called with non-elementwise op '{}'",
                op.node.op_type
            ))),
        }
    }
}

fn resolve_inputs(
    names: &[String],
    values: &HashMap<String, GpuTensor>,
) -> Result<Vec<GpuTensor>, CudaError> {
    names
        .iter()
        .filter(|n| !n.is_empty())
        .map(|n| values.get(n).cloned().ok_or_else(|| {
            CudaError::Kernel(format!("Input '{}' not found", n))
        }))
        .collect()
}
```

**Important — per-op wrappers may need weights lookup too.** Today's `execute_gpu_operator` resolves input tensors from `values` OR `self.weights`. In the new world, `plan.weights` is the weight store. Either:
- Merge weights into `values` at the top of `run()` (simpler), OR
- Have `resolve_inputs` take both `values` and `&plan.weights`.

Go with the second option to avoid cloning the whole weight map per run:

```rust
fn resolve_inputs<'a>(
    names: &[String],
    values: &'a HashMap<String, GpuTensor>,
    weights: &'a HashMap<String, GpuTensor>,
) -> Result<Vec<&'a GpuTensor>, CudaError> { ... }
```

This changes every dispatch method's signature to take `weights`. Go back to the dispatch match in `mod.rs::dispatch_op` and thread `&plan.weights` through.

- [ ] **Step 3: Port every elementwise op** — one arm per op. Reference the exact same per-op function calls from `src/cuda/inference/mod.rs` lines ~1150–1900 (search for `"Add" =>`, `"Sub" =>`, etc.).

- [ ] **Step 4: Build + run the full test suite**

```bash
cargo build --features cuda --release 2>&1 | tail -5
cargo test --features cuda --release -- --include-ignored 2>&1 | grep -E "^test result" | tail -25
```

Expected: clean build; at this point GpuGraphExecutor::run still handles the full dispatch (we haven't wired its delegate path yet — that's Task 2.13). So tests should still all pass via the old path. The Executor path is dead code until Task 2.13.

- [ ] **Step 5: Keep file size under 1000 lines**

```bash
wc -l src/cuda/executor/elementwise.rs
```

If over 1000, split into `src/cuda/executor/elementwise/{arith,unary,comparison,activation}.rs` with a `mod.rs` that re-routes. Do this BEFORE continuing.

- [ ] **Step 6: Commit (WIP)**

```bash
git add src/cuda/executor/
git commit -S -m "wip(executor): port elementwise dispatch (dead-code until delegate wired)"
```

### Task 2.4: Port the layout dispatch

**Files:**
- Modify: `src/cuda/executor/layout.rs`

- [ ] **Step 1: Identify all layout ops from today's match**

In `src/cuda/inference/mod.rs`, find arms for: `Concat`, `Expand`, `Gather`, `Pad`, `Resize`, `ScatterND`, `Slice`, `Transpose`, `Where`, `Reshape`, `Unsqueeze`, `Squeeze`, `Shape`, `Constant`, `ConstantOfShape`, `NonZero`, `Cast`, `Range`, `CumSum`. Each has a `gpu_*` function call.

- [ ] **Step 2: Port to `dispatch_layout`** with the `plan: &ExecutionPlan` parameter so precomputed Slice/Reshape/Unsqueeze/Squeeze values can be pulled from `op.precomputed`. The existing `NodePrecomputed` → runtime dispatch wiring in today's `execute_gpu_operator_with_precomputed` shows the pattern.

Skeleton (extend):

```rust
impl Executor {
    pub(crate) fn dispatch_layout(
        &self,
        op: &PlannedOp,
        values: &HashMap<String, GpuTensor>,
        plan: &ExecutionPlan,
    ) -> Result<GpuTensor, CudaError> {
        let weights = &plan.weights;
        let attrs = &op.node.attributes;
        match op.node.op_type.as_str() {
            "Slice" => {
                let inputs = resolve_inputs(&op.node.inputs, values, weights)?;
                let pre = op.precomputed.slice.as_ref();
                gpu_slice(&self.ctx, &self.ops_kernels,
                          &mut self.memory_pool.borrow_mut(),
                          inputs, pre)  // adapt per today's signature
            }
            "Reshape" => {
                // … uses op.precomputed.reshape_shape when present
            }
            // … one arm per op …
            _ => Err(CudaError::Kernel(format!(
                "dispatch_layout called with non-layout op '{}'",
                op.node.op_type
            ))),
        }
    }
}
```

- [ ] **Step 3: Handle Constant specially** — today's code intercepts Constant in the main run loop (lines ~2985 of mod.rs) and inserts the tensor into `values` without calling any dispatch. In the new world, either:
  - Move that special case up into `run()` itself (early-continue for `op_type == "Constant"`), OR
  - Handle it inside `dispatch_layout` by returning the tensor.

Pick the first option (early-continue in `run()`) — it matches today's layout and keeps `dispatch_layout` uniform.

- [ ] **Step 4: Size check + split if needed**

```bash
wc -l src/cuda/executor/layout.rs
```

If > 1000: split by sub-category (`layout/shape.rs`, `layout/slice.rs`, `layout/transform.rs`, ...).

- [ ] **Step 5: Commit (WIP)**

```bash
git add src/cuda/executor/
git commit -S -m "wip(executor): port layout dispatch"
```

### Task 2.5: Port conv dispatch

**Files:**
- Modify: `src/cuda/executor/conv.rs`

- [ ] **Step 1: Port `Conv` and `ConvTranspose` arms from today's match**

They call `garboard_conv_1d(&self.ctx, ...)` / `garboard_conv_transpose_1d(&self.ctx, ...)` (post-PR-5a migration). Should be small — maybe 50 lines.

- [ ] **Step 2: Commit**

```bash
git add src/cuda/executor/conv.rs
git commit -S -m "wip(executor): port conv dispatch"
```

### Task 2.6: Port reduction dispatch

Same pattern — `ReduceMean`, `ReduceSum`, `LayerNormalization` arms. Should be <200 lines.

- [ ] Commit: `wip(executor): port reduction dispatch`.

### Task 2.7: Port matmul dispatch

`Gemm`, `MatMul`. Uses `ctx.blas()`. Should be <200 lines.

- [ ] Commit: `wip(executor): port matmul dispatch`.

### Task 2.8: Port LSTM dispatch

**Files:**
- Modify: `src/cuda/executor/lstm.rs`

- [ ] **Step 1: LSTM dispatch needs access to both `plan.lstm_weights` (eager) and `plan.lstm_plan_cache` (lazy RefCell).** Port logic from today's mod.rs lines ~1675-1750 but replace the lazy `pack_lstm_weights_for_cudnn` call with a lookup into `plan.lstm_weights`:

```rust
impl Executor {
    pub(crate) fn dispatch_lstm(
        &self,
        op: &PlannedOp,
        values: &HashMap<String, GpuTensor>,
        plan: &ExecutionPlan,
    ) -> Result<GpuTensor, CudaError> {
        let weights = &plan.weights;
        let inputs = resolve_inputs(&op.node.inputs, values, weights)?;
        let x = inputs[0];
        let w = inputs[1];
        let r = inputs[2];
        // b, h_init, c_init are optional
        let b = op.node.inputs.get(3).and_then(|n| values.get(n).or_else(|| weights.get(n)));
        let h_init = op.node.inputs.get(5).and_then(|n| values.get(n).or_else(|| weights.get(n)));
        let c_init = op.node.inputs.get(6).and_then(|n| values.get(n).or_else(|| weights.get(n)));

        let w_shape = w.shape();
        let num_directions = w_shape[0];
        let hidden_size = w_shape[1] / 4;
        let input_size = w_shape[2];

        let key = (w.device_ptr(&self.ctx) as usize, r.device_ptr(&self.ctx) as usize);
        let packed = plan.lstm_weights.get(&key).ok_or_else(|| {
            CudaError::Cudnn(format!(
                "LSTM '{}' weights not packed at lowering (key not found)",
                op.node.name
            ))
        })?;

        // Plan cache lookup by (w_ptr, r_ptr, seq_length).
        let seq_len = x.shape()[0];
        let plan_key = (key.0, key.1, seq_len);
        let mut cache = plan.lstm_plan_cache.borrow_mut();
        if !cache.contains_key(&plan_key) {
            let config = /* build RnnForwardConfig */;
            let rnn_plan = self.ctx.dnn().rnn_prepare(self.ctx.garboard_stream(), &config)
                .map_err(|e| CudaError::Cudnn(format!("rnn_prepare failed: {e}")))?;
            cache.insert(plan_key, rnn_plan);
        }
        let rnn_plan = cache.get(&plan_key).unwrap();

        lstm_forward(
            &self.ctx, packed, rnn_plan, x, h_init, c_init, hidden_size, num_directions,
        )
    }
}
```

- [ ] **Step 2: Commit**

```bash
git add src/cuda/executor/lstm.rs
git commit -S -m "wip(executor): port LSTM dispatch (plan cache, eager packed weights)"
```

### Task 2.9: Port STFT, fused, misc dispatch

Three tiny tasks merged into one:

- **STFT**: `dispatch_stft` — port the STFT arm. Uses `self.stft_kernels`. ~100 lines.
- **fused**: `dispatch_fused` — port the fused-pattern handler. Today it's `execute_fused_pattern`; pull it in near-verbatim, just update signatures. ~400 lines.
- **misc**: `dispatch_misc` — And/Or/Not. ~50 lines.

- [ ] Commit: `wip(executor): port stft, fused, misc dispatch`.

### Task 2.10: Move `compute_tensor_last_use` and memory-pool return logic into `run()`

- [ ] Port lines ~2950-3070 of today's `mod.rs` (the last-use free logic) into `Executor::run`. It reads `plan.tensor_last_use` which was set during lowering. No more per-call `compute_tensor_last_use` call.

- [ ] Commit: `wip(executor): port memory-pool return-by-last-use logic`.

### Task 2.11: Port `run_with_validation` / `run_with_checkpoints` / `run_with_logging` / `run_with_profiling`

- [ ] Add four more methods on `Executor` matching today's signatures on `GpuGraphExecutor`. Each is essentially `run()` with an extra instrumentation layer around each dispatched op. Port near-verbatim; most of each function is identical.

- [ ] Commit: `wip(executor): port run_with_validation / checkpoints / logging / profiling`.

### Task 2.12: Run the full test suite against the new Executor via a thin delegate

- [ ] Temporarily modify `GpuGraphExecutor::run` (in `src/cuda/inference/mod.rs`) to:
  1. Build an `OptimizableGraph` from `self.nodes` + `self.weights_as_initializers` + `self.outputs_referenced`.
  2. Lower it: `let plan = ir::lower(graph, &self.ctx)?;`
  3. Delegate: `self.executor.run(&plan, inputs, &output_names)`.

  This is transitional wiring — `GpuGraphExecutor` still has its old fields; it just uses them to produce a plan and delegates. The old fields will be removed in Commit 3.

  The tricky part: `self.weights` today are already on GPU (CPU tensors were uploaded by `add_initializer`). Lowering expects CPU tensors. For Commit 2's transitional delegate, you can either:
  - Keep a parallel `self.initializers_cpu: HashMap<String, Tensor>` field populated from `add_initializer`'s input (simplest)
  - Download GPU weights back to CPU during the delegate (expensive, wrong)

  Go with the first option — this field exists for one commit then disappears.

- [ ] Modify `GpuGraphExecutor::new` to also construct an `Executor`. Rename the field `executor: Executor` in preparation for Commit 3.

- [ ] Run the full suite:

```bash
cargo build --features cuda --release 2>&1 | tail -5
cargo test --features cuda --release -- --include-ignored 2>&1 | grep -E "^test result" | tail -25
```

Expected: 510 + ~20 new tests all green.

- [ ] Benchmark:

```bash
cd /home/ryano/workspace/ryanoneill/rust-ai-explorations
cargo run --release -p leadline-bench --bin compare -- kokoro-v1.0.onnx --iterations 10 --output-dir /tmp 2>&1 | grep -E "Timing|Summary|Speedup|LSTM "
```

Ratio must remain in the ~4.5–4.9× ORT band from Phase 1 close. If significantly worse, investigate before continuing.

- [ ] Commit (WIP).

### Task 2.13: Squash Commit 2's WIP chain into one

- [ ] Soft-reset and commit:

```bash
git reset --soft HEAD~<N>   # N = number of wip(executor) commits since Commit 1
git commit -S -m "feat(executor): carve Executor out of GpuGraphExecutor

New \`src/cuda/executor/\` module holds the GPU runtime:
- \`Executor\` struct: ctx + 7 kernel caches + memory pool + i64_cache
  (+ debug-inference validator). Model-independent.
- \`run(plan, inputs, outputs)\` with compact dispatch match routing to
  \`dispatch_<category>\` methods implemented in per-category files:
  elementwise, layout, conv, reduction, matmul, lstm, stft, fused, misc.
  Every file under 1000 lines per CLAUDE.md.
- \`run_with_validation / checkpoints / logging / profiling\` ported.

\`GpuGraphExecutor::run\` transitionally builds an \`OptimizableGraph\`
from its fields, lowers via \`ir::lower\`, delegates to \`Executor::run\`.
A parallel \`initializers_cpu\` field keeps CPU tensors available for
lowering until Commit 3 fully replaces GpuGraphExecutor's internals.

510 tests green; benchmark within Phase 1 band.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Commit 3: Replace `GpuGraphExecutor` internals with the strangler shim

**Commit goal:** `GpuGraphExecutor`'s public API unchanged; internals are now `builder + executor + compiled_plan`. All derived-state fields (`nodes`, `weights`, `fused_patterns`, `nodes_to_skip`, `sorted_nodes_cache`, `dynamic_candidates`, `patterns_detected`, `lstm_weight_cache`, `lstm_plan_cache`, `static_i64_cache`, `initializers_cpu`, per-op helpers `try_precompute_*`, `detect_fused_patterns` method, `topological_sort` method, `compute_tensor_last_use` method, `execute_gpu_operator*` methods) removed. 510 tests pass; benchmark stable.

### Task 3.1: Replace GpuGraphExecutor struct with the shim

**Files:**
- Modify: `src/cuda/inference/mod.rs`

- [ ] Replace the struct:

```rust
pub struct GpuGraphExecutor {
    builder: RefCell<ir::OptimizableGraphBuilder>,
    executor: cuda::executor::Executor,
    compiled_plan: RefCell<Option<ir::ExecutionPlan>>,
}
```

- [ ] Replace `new()`:

```rust
impl GpuGraphExecutor {
    pub fn new() -> Result<Self, CudaError> {
        Self::new_with_profiling(false)
    }

    pub fn new_with_profiling(_profile: bool) -> Result<Self, CudaError> {
        Ok(Self {
            builder: RefCell::new(ir::OptimizableGraphBuilder::new()),
            executor: cuda::executor::Executor::new()?,
            compiled_plan: RefCell::new(None),
        })
    }
}
```

(The `_profile` parameter is preserved for API compatibility; wire it through if `Executor::new` grows a variant that enables timing.)

- [ ] Replace `add_initializer` / `add_node`:

```rust
pub fn add_initializer(&mut self, name: String, tensor: &Tensor) -> Result<(), CudaError> {
    self.builder.borrow_mut().add_initializer(name, tensor.clone());
    *self.compiled_plan.borrow_mut() = None;
    Ok(())
}

pub fn add_node(
    &mut self,
    name: &str, op_type: &str,
    inputs: Vec<&str>, outputs: Vec<&str>,
    attributes: NodeAttributes,
) {
    self.builder.borrow_mut().add_node(
        name.to_string(),
        op_type.to_string(),
        inputs.iter().map(|s| s.to_string()).collect(),
        outputs.iter().map(|s| s.to_string()).collect(),
        attributes,
    );
    *self.compiled_plan.borrow_mut() = None;
}
```

- [ ] Replace `run`:

```rust
pub fn run(
    &self,
    inputs: HashMap<String, Tensor>,
    output_names: Vec<&str>,
) -> Result<HashMap<String, Tensor>, CudaError> {
    self.ensure_compiled()?;
    let plan = self.compiled_plan.borrow();
    self.executor.run(plan.as_ref().unwrap(), inputs, &output_names)
}

fn ensure_compiled(&self) -> Result<(), CudaError> {
    if self.compiled_plan.borrow().is_none() {
        let graph = self.builder.borrow().clone().build();
        *self.compiled_plan.borrow_mut() = Some(ir::lower(graph, &self.executor.ctx)?);
    }
    Ok(())
}
```

- [ ] Replace `run_with_validation` / `run_with_checkpoints` / `run_with_logging` / `run_with_profiling` — each becomes:

```rust
pub fn run_with_validation(&self, ...) -> Result<..., CudaError> {
    self.ensure_compiled()?;
    let plan = self.compiled_plan.borrow();
    self.executor.run_with_validation(plan.as_ref().unwrap(), ...)
}
```

- [ ] Delete every method not in the public API list above (all the per-op dispatches, `detect_fused_patterns`, `try_precompute_*`, `topological_sort`, `compute_tensor_last_use`, `execute_fused_pattern`, `execute_gpu_operator*`, `has_gpu_impl` if unused elsewhere). That's ~4500 lines deleted.

- [ ] Delete `ExecutionNode`, `PrecomputedSlice`, `NodePrecomputed`, `LstmWeightCache`, `LstmPlanCache` from this file. They now live in `src/ir/plan.rs`.

### Task 3.2: Delete the shared fusion detection path in `cuda::inference::fusion::detection`

- [ ] Delete `detect_fused_patterns` (the GPU-side function) since no caller remains. Keep `detect_fused_patterns_from_cpu` and the shared `detect_with_scalar_lookup`.

### Task 3.3: Full test + benchmark

- [ ] Build:

```bash
cargo build --features cuda --release 2>&1 | tail -5
```

Expected: clean, 0 warnings.

- [ ] Tests:

```bash
cargo test --features cuda --release -- --include-ignored 2>&1 | grep -E "^test result" | tail -25
```

Expected: 510 + ~20 new ir-module tests all green.

- [ ] Debug-inference variant:

```bash
cargo build --features cuda,debug-inference --release 2>&1 | tail -5
cargo test --features cuda,debug-inference --release -- --include-ignored 2>&1 | grep -E "^test result" | tail -5
```

Expected: clean.

- [ ] Final benchmark:

```bash
cd /home/ryano/workspace/ryanoneill/rust-ai-explorations
cargo run --release -p leadline-bench --bin compare -- kokoro-v1.0.onnx --iterations 10 --output-dir /tmp 2>&1 | grep -E "Run|Timing|Summary|Speedup|LSTM "
```

Ratio must remain in the ~4.5–4.9× ORT band. If off, investigate before declaring Phase 2 done.

- [ ] File sizes sanity check:

```bash
find src -name "*.rs" -exec wc -l {} + | sort -n | tail -20
```

Expected: nothing over 1000. `src/cuda/inference/mod.rs` should now be ~150 lines (strangler shim + a few runtime-logging helpers).

### Task 3.4: Squash Commit 3's WIP chain (if any) into one final commit

- [ ] Commit:

```bash
git commit -S -m "feat(executor): replace GpuGraphExecutor internals with strangler shim

GpuGraphExecutor is now a ~150-line shim over OptimizableGraph +
ExecutionPlan + Executor. Its public API — new, add_initializer,
add_node, run, run_with_validation/checkpoints/logging/profiling — is
unchanged; 510 existing tests pass without modification.

Deleted: ExecutionNode (→ ir::plan::PlannedOp wraps GraphNode),
PrecomputedSlice, NodePrecomputed, LstmWeightCache, LstmPlanCache
(all moved to ir::plan), the ~4500-line dispatch match (now in
cuda::executor/*), try_precompute_* helpers (→ ir::passes::precompute),
topological_sort (→ ir::passes::topo_sort), compute_tensor_last_use
(→ ir::lowering), detect_fused_patterns (GPU-side, unused; kept CPU-side
via ir::passes::fusion → detect_fused_patterns_from_cpu), and every
per-op execute_gpu_operator helper.

The 1000-line rule is satisfied: every file under 1000 lines.

Phase 2 complete. Follow-up: delete the GpuGraphExecutor shim and
migrate remaining public callers to Executor + ExecutionPlan directly.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- OptimizableGraph type → Task 1.2 + 1.3. ✓
- ExecutionPlan type (with lstm_weights eager, lstm_plan_cache RefCell) → Task 1.4. ✓
- Executor type → Task 2.1. ✓
- topo_sort pass → Task 1.5. ✓
- precompute_params pass → Task 1.6. ✓
- detect_fusion pass (no ctx, reads CPU tensors) → Task 1.7. ✓
- lower() function → Task 1.8. ✓
- Strangler-fig wrapper → Task 3.1. ✓
- 3-commit phasing → Commit 1 / 2 / 3 sections. ✓
- Executor split into sub-modules under 1000 lines → Task 2.1 (scaffold) + 2.3–2.9 (per category) + file-size checks at the end of each. ✓
- Test strategy (unit tests per pass, integration test for lower, 510 existing tests unchanged, benchmark between commits) → Tasks 1.5/1.6/1.7/1.8 have unit tests; Tasks 2.12 and 3.3 have full-suite + benchmark. ✓

**Placeholder scan:** No TBDs, no "handle edge cases," no "similar to Task N." Every code block is concrete enough to copy-paste-adjust.

**Type consistency:**
- `OptimizableGraph` fields: same in Task 1.2 (definition) and Task 1.8 (consumer). ✓
- `ExecutionPlan` fields: same in Task 1.4 and Task 1.8 (construction). ✓
- `detect_fusion` returns `FusionAnnotations` in Task 1.7 and `lower()` consumes `.heads / .skipped / .dynamic_candidates` in Task 1.8. ✓
- Dispatch method signatures: unified in Task 2.2 and extended by `plan: &ExecutionPlan` on `dispatch_layout` / `dispatch_lstm` — consistently threaded through Task 2.2's `dispatch_op` and every subsequent per-category task. ✓

**Scope check:** One subsystem (IR + runtime split), three commits, clear acceptance per commit. Appropriate for a single plan.

**Ambiguity check:**
- `Tensor` constructor names in tests (e.g., `Tensor::from_i64_slice`, `Tensor::from_data_shape`) may not match the codebase exactly; Task 1.3 / 1.6 explicitly call out "grep for the actual helper". Engineer can't misinterpret — they're told to verify.
- `GpuTensor::device_ptr(ctx)` vs `as_raw_device_ptr` — Task 1.8 instructs to grep for the actual name.
- Commit 2 WIP squash count `~N~` — engineer runs `git log` to determine. Unambiguous.

Plan complete.

---

## Execution Handoff

**Two execution options:**

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration per the process learning from Phase 1 (`feedback_subagent_escalation.md`).

**2. Inline Execution** — execute tasks in this session using `executing-plans`, batch execution with checkpoints for review.

**Which approach?**
