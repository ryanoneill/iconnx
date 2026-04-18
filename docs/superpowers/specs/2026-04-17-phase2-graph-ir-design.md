# Phase 2: Graph Intermediate Representation

**Status:** Design approved 2026-04-17.
**Scope:** Introduce three types — `OptimizableGraph`, `ExecutionPlan`, `Executor` — and split the GPU inference path so "what to compute" is data and "how to compute it" is a runtime.
**Non-goals:** Any actual optimization. Phase 2 is a deliberately no-op transformation. Phase 3 adds the passes that do real work.

## Motivation

`GpuGraphExecutor` in `src/cuda/inference/mod.rs` is 5,047 lines and owns:

- Graph structure: `nodes`, `weights`, `static_i64_cache`
- Derived optimizations: `fused_patterns`, `nodes_to_skip`, `sorted_nodes_cache`, `dynamic_candidates`, `patterns_detected` — all behind `RefCell` because they're populated lazily on the first `run()` after `add_node` mutations
- GPU runtime state: seven kernel caches, memory pool, LSTM caches, i64 caches, context

Graph passes are methods on the executor that mutate internal `RefCell`s. Adding an optimization pass means writing more methods and invalidating more caches. Phase 3's five planned passes cannot be built on this foundation without making the mess deeper.

## Design

### Three types, three ownership domains

```
OptimizableGraph    →  CPU tensors, pure data, optimizable
ExecutionPlan       →  GPU weights, packed LSTM weights, operation sequence, frozen
Executor            →  Device, stream, kernel caches, memory pool; model-independent
```

`OptimizableGraph` holds CPU-side `Tensor` initializers so that Phase 3's constant-folding pass can read values on the host. `ExecutionPlan` is self-contained — hand it to any compatible `Executor` and it runs. `Executor` is the GPU runtime: stream, kernel caches, memory pool, cuBLAS handle, cuDNN handle. No model-specific state.

Plan construction (`lower(graph, &ctx) → plan`) is inherently a GPU operation: it uploads weights, packs LSTM weights, builds `PackedLstmWeights` instances. This is the "prepare for hardware" boundary, analogous to an ORT session's construction cost.

### Type definitions

```rust
// src/ir/graph.rs — pure CPU data, no GPU dependencies
pub struct OptimizableGraph {
    pub inputs:       Vec<GraphInput>,
    pub outputs:      Vec<String>,
    pub nodes:        Vec<GraphNode>,
    pub initializers: HashMap<String, Tensor>,
}

pub struct GraphNode {
    pub name: String,
    pub op_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attributes: NodeAttributes,
}

pub struct GraphInput {
    pub name: String,
    pub shape: Vec<Option<usize>>,   // None for dynamic dims; Phase 3 shape-inference fills
}
```

```rust
// src/ir/plan.rs — frozen model-specific state, self-contained
pub struct ExecutionPlan {
    pub ops:                  Vec<PlannedOp>,
    pub weights:              HashMap<String, GpuTensor>,
    pub static_i64:           HashMap<String, Vec<i64>>,
    /// LSTM weights packed eagerly at lowering. Every (W, R) pair referenced
    /// by an LSTM node is packed here once; lstm_forward looks up by
    /// (W_ptr, R_ptr). Fully determined by the plan — no RefCell.
    pub lstm_weights:         HashMap<(usize, usize), PackedLstmWeights>,
    /// Lazy memoization only. Keyed by (W_ptr, R_ptr, seq_length) where
    /// seq_length is input-dependent and cannot be known at lowering. The
    /// plan's semantic state is unchanged by population of this cache — it is
    /// pure memoization of seq_length-dependent cuDNN descriptors. Logically
    /// frozen; RefCell is the minimum additive mutability required.
    pub lstm_plan_cache:      RefCell<LstmPlanCache>,
    pub dynamic_candidates:   HashMap<String, FusedPatternInfo>,
    pub tensor_last_use:      HashMap<String, usize>,
    pub graph_inputs:         Vec<GraphInput>,
    pub graph_outputs:        Vec<String>,
}

pub struct PlannedOp {
    pub node:        GraphNode,
    pub kind:        OpKind,
    pub precomputed: NodePrecomputed,
}

pub enum OpKind {
    Regular,
    FusedHead(FusedPattern),   // execute the fused kernel, feeds nodes_to_skip logic
    Skip,                       // sub-node of an earlier fused head
}

pub struct NodePrecomputed {
    pub slice:            Option<PrecomputedSlice>,
    pub reshape_shape:    Option<Vec<i64>>,
    pub unsqueeze_axes:   Option<Vec<i64>>,
    pub squeeze_axes:     Option<Vec<i64>>,
}
```

```rust
// src/cuda/executor/mod.rs — model-independent GPU runtime
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
    pub i64_cache: RefCell<HashMap<u64, Vec<i64>>>,   // per-run, cleared each run
    #[cfg(feature = "debug-inference")]
    pub validator: RefCell<Option<TensorValidator>>,
}

impl Executor {
    pub fn new() -> Result<Self, CudaError>;
    pub fn run(
        &self,
        plan: &ExecutionPlan,
        inputs: HashMap<String, Tensor>,
        output_names: &[&str],
    ) -> Result<HashMap<String, Tensor>, CudaError>;

    // Variants preserved from today's GpuGraphExecutor:
    pub fn run_with_validation(...) -> ...;
    pub fn run_with_checkpoints(...) -> ...;
    pub fn run_with_logging(...) -> ...;
    pub fn run_with_profiling(...) -> ...;
}
```

### Dispositions of today's fields

| Today on `GpuGraphExecutor` | New home |
|---|---|
| `ctx` | `Executor::ctx` |
| `elementwise_kernels` / `reduction_kernels` / `conv_kernels` / `lstm_kernels` / `ops_kernels` / `stft_kernels` / `fused_kernels` | `Executor` — all kernel caches |
| `lstm_weight_cache` | `ExecutionPlan::lstm_weights` (eager at lowering, no RefCell) |
| `lstm_plan_cache` | `ExecutionPlan::lstm_plan_cache` (RefCell, documented exception) |
| `i64_cache` | `Executor::i64_cache` (per-run runtime state) |
| `static_i64_cache` | `ExecutionPlan::static_i64` (derived at lowering) |
| `weights` | `ExecutionPlan::weights` (uploaded at lowering) |
| `nodes` | Replaced by `ExecutionPlan::ops: Vec<PlannedOp>` |
| `sorted_nodes_cache` | Eliminated — topo-sort happens once in lowering, result is stored as `ops` order |
| `fused_patterns` | Embedded into `PlannedOp::kind == FusedHead(…)` |
| `nodes_to_skip` | Embedded into `PlannedOp::kind == Skip` |
| `dynamic_candidates` | `ExecutionPlan::dynamic_candidates` (plain field, derived at lowering) |
| `patterns_detected` | Eliminated — lowering runs all passes exactly once |
| `memory_pool` | `Executor::memory_pool` (runtime state) |
| `validator` | `Executor::validator` (runtime state, `debug-inference`-gated) |

### Graph passes — pure functions

Each pass consumes an `OptimizableGraph` and returns a new one, with `Result` for fallible passes:

```rust
// src/ir/passes/topo_sort.rs
pub fn topo_sort(g: OptimizableGraph) -> Result<OptimizableGraph, CudaError>;

// src/ir/passes/precompute.rs
pub fn precompute_params(g: OptimizableGraph) -> OptimizableGraph;

// src/ir/passes/fusion.rs
pub fn detect_fusion(g: OptimizableGraph) -> OptimizableGraph;
```

`detect_fusion` operates on `g.initializers` directly — CPU `Tensor` values — so it doesn't need a `IconnxCudaContext`. This is a purer signature than the current `fusion::detect_fused_patterns(&nodes, &weights, &ctx)` which read constant scalars via `to_host_f32(ctx)`.

`precompute_params` replaces today's four `try_precompute_*` helpers on `GpuGraphExecutor`. It reads initializer and Constant-node values from `g.initializers`, attaches per-node precomputation into a side-map, and returns the updated graph. The results land on `PlannedOp::precomputed` during lowering.

The existing `src/cuda/inference/fusion/{patterns, detection}` pattern-matching code is kept unchanged as an internal library — `ir::passes::fusion` is a thin adapter that calls into it. 865 lines of proven logic isn't worth moving for a Phase 2 refactor.

### Lowering

```rust
// src/ir/lowering.rs
pub fn lower(
    graph: OptimizableGraph,
    ctx: &IconnxCudaContext,
) -> Result<ExecutionPlan, CudaError> {
    // 1. Run passes in order: topo_sort → precompute_params → detect_fusion
    let g = topo_sort(graph)?;
    let g = precompute_params(g);
    let g = detect_fusion(g);

    // 2. Upload initializers to GPU as `weights: HashMap<String, GpuTensor>`.
    // 3. For each LSTM node, pack its (W, R, B) weights → lstm_weights map.
    // 4. Build Vec<PlannedOp> from g.nodes, attaching PlannedOp::kind from
    //    the fusion annotations and PlannedOp::precomputed from the
    //    precompute pass.
    // 5. Compute tensor_last_use from the op order.
    // 6. Collect static_i64 from initializers + Constant nodes.
    // 7. Derive dynamic_candidates (addmuladd chains with runtime b/c).
    // 8. Construct ExecutionPlan with lstm_plan_cache: RefCell::new(empty).
}
```

### Strangler-fig wrapper

`GpuGraphExecutor` stays as a thin public API shim:

```rust
// src/cuda/inference/mod.rs — shrunk to ~100 lines
pub struct GpuGraphExecutor {
    builder:       RefCell<OptimizableGraphBuilder>,
    executor:      Executor,
    compiled_plan: RefCell<Option<ExecutionPlan>>,
}

impl GpuGraphExecutor {
    pub fn new() -> Result<Self, CudaError> { /* builds Executor, empty builder */ }

    pub fn add_initializer(&mut self, name: String, tensor: &Tensor) -> Result<(), CudaError> {
        self.builder.borrow_mut().add_initializer(name, tensor)?;
        *self.compiled_plan.borrow_mut() = None;
        Ok(())
    }

    pub fn add_node(
        &mut self,
        name: &str, op_type: &str,
        inputs: Vec<&str>, outputs: Vec<&str>,
        attrs: NodeAttributes,
    ) {
        self.builder.borrow_mut().add_node(name, op_type, inputs, outputs, attrs);
        *self.compiled_plan.borrow_mut() = None;
    }

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
            let graph = self.builder.borrow().build()?;
            *self.compiled_plan.borrow_mut() = Some(lower(graph, &self.executor.ctx)?);
        }
        Ok(())
    }

    // run_with_validation / run_with_checkpoints / run_with_logging / run_with_profiling
    // each becomes ~10 lines: ensure_compiled() + delegate to Executor's equivalent.
}
```

`add_*` invalidates the compiled plan; `run` lazily compiles on first call. This matches today's "detect_fused_patterns caches on first run" semantics. No test sees a behavioral change.

### Module layout

```
src/
├── ir/
│   ├── mod.rs                     — re-exports
│   ├── graph.rs                   — OptimizableGraph, OptimizableGraphBuilder, GraphNode, GraphInput
│   ├── plan.rs                    — ExecutionPlan, PlannedOp, OpKind, NodePrecomputed
│   ├── lowering.rs                — fn lower
│   └── passes/
│       ├── mod.rs                 — re-exports
│       ├── topo_sort.rs
│       ├── precompute.rs
│       └── fusion.rs              — adapter onto cuda::inference::fusion
├── cuda/
│   ├── executor/
│   │   ├── mod.rs                 — Executor struct + run() with compact dispatch match (<200 lines)
│   │   ├── elementwise.rs         — add/sub/mul/div/pow/exp/sqrt/sin/cos/tanh/sigmoid/leaky_relu/round/floor/clip/softmax/atan and comparisons
│   │   ├── layout.rs              — concat, expand, gather, pad, resize, scatter_nd, slice, transpose, where, reshape, unsqueeze, squeeze, shape, constant, constant_of_shape, nonzero, cast, range, cumsum
│   │   ├── conv.rs                — Conv, ConvTranspose
│   │   ├── reduction.rs           — ReduceMean, ReduceSum, LayerNormalization
│   │   ├── matmul.rs              — Gemm, MatMul
│   │   ├── lstm.rs                — LSTM forward (delegates to cudnn::rnn::lstm_forward)
│   │   ├── stft.rs                — STFT
│   │   ├── fused.rs               — DivRsqrt, MulSinPowMulAdd, AddMulAdd, Gelu (fused kernel dispatch)
│   │   └── misc.rs                — remaining ops (and/or/not, equal/less/greater family if not in elementwise.rs)
│   └── inference/
│       ├── mod.rs                 — ~100-line GpuGraphExecutor strangler shim
│       ├── fusion/                — kept as-is (detection + patterns)
│       ├── gpu_event_timer.rs     — kept
│       └── testing.rs             — updated to build via OptimizableGraphBuilder
```

**1,000-line rule:** every file listed above must stay under 1,000 lines. The executor split is the lever for that — 3,000-3,500 lines of dispatch distributed across ~10 sub-modules averages ~350 lines each. If any sub-module grows past 800 during implementation, further split by op (e.g., `elementwise/arith.rs` + `elementwise/unary.rs`) before adding more. No exceptions.

### Phase 2 phasing

Phase 2 lands as three commits on a new branch `ir-phase2`. Between commits, all 510 tests stay green and the benchmark ratio stays within the Phase 1 landing band (~4.5–4.9× ORT).

**Commit 1 — `ir::` module with types only.**
`OptimizableGraph`, `ExecutionPlan`, `PlannedOp`, passes as free functions, `lower()` — all defined and unit-tested against synthetic graphs, but not wired into any live code path. 510 tests pass because nothing depends on the new types yet.

**Commit 2 — Carve `Executor` out of `GpuGraphExecutor`.**
Create `src/cuda/executor/` with the split sub-modules. Move dispatch logic from today's `cuda/inference/mod.rs::run` into the new files. At this commit `GpuGraphExecutor` still holds `nodes`/`weights`/etc. but its `run()` constructs a temporary `ExecutionPlan` and delegates. Some duplication permitted transitionally.

**Commit 3 — Replace `GpuGraphExecutor`'s internals with the strangler shim.**
Delete `nodes`, `weights`, `fused_patterns`, `sorted_nodes_cache`, etc. from `GpuGraphExecutor`. Switch it to `builder` / `compiled_plan` / `executor`. Run 510 tests + benchmark.

A follow-up PR after Phase 2 lands deletes the strangler shim once the public API has migrated to `Executor` + `ExecutionPlan` directly. That cleanup is not part of Phase 2.

## Test strategy

- **Existing 510 tests:** pass unchanged on the strangler API. Any failure is definitively a regression in the new types (the test didn't change).
- **New unit tests in `src/ir/`:**
  - `graph::OptimizableGraphBuilder` — `add_initializer` / `add_node` / `build` semantics.
  - `passes::topo_sort` — valid DAGs produce correct order; cycles return `Err`.
  - `passes::precompute` — hand-built graphs with initializer-sourced Slice/Reshape/Unsqueeze/Squeeze verify the precomputation matches today's `try_precompute_*` helpers.
  - `passes::fusion` — adapter preserves existing fusion detection behavior; tested against the same inputs as `cuda::inference::fusion::detection`'s existing tests.
  - `lowering::lower` — GPU-required (`#[ignore]`); verifies weights are uploaded, LSTM weights are eagerly packed, PlannedOps match expected order.
- **Benchmark:** `cargo run --release -p leadline-bench --bin compare -- kokoro-v1.0.onnx --iterations 10 --output-dir /tmp` after each commit. Ratio must stay in the ~4.5–4.9× band measured at Phase 1 close.

## What Phase 2 explicitly does not do

- No new optimizations — every pass produces the same plan today's executor produces.
- No rename of public symbols (`GpuGraphExecutor` stays; a follow-up deletes it).
- No changes to `graph_executor.rs` (the CPU executor) — it's out of scope.
- No graph-level shape inference. `GraphInput::shape` fields exist as `Vec<Option<usize>>` to give Phase 3's shape-inference pass a place to write into; Phase 2 leaves them as-is.
- No memory planner. `tensor_last_use` on `ExecutionPlan` is the same data today's `run()` computes; Phase 4 will extend it with static lifetime analysis and a pre-allocated arena.
- No splitting of the dispatch match's individual op handlers beyond the category split needed for the 1,000-line rule. Further per-op factoring is a separate refactor.

## Open questions

None remaining. Leadline approved Sections 1–5 on 2026-04-17, with the executor split and the `detect_fusion` ctx removal corrections captured above.
