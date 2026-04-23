//! Frozen compiled form of `OptimizableGraph`.
//!
//! An `ExecutionPlan` is model-specific and self-contained: given any
//! compatible `Executor`, it runs. `lstm_weights` is eagerly populated at
//! lowering. `lstm_plan_cache` is the one documented exception to "frozen"
//! — it lazily memoizes seq_length-dependent cuDNN descriptors that aren't
//! known at plan-construction time.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};

use crate::cuda::cudnn::PackedLstmWeights;
use crate::cuda::inference::fusion::{FusedPattern, FusedPatternInfo};
use crate::cuda::GpuTensor;
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
    FusedHead(FusedPattern),
    /// Sub-node of an earlier fused head — already handled, skip in the loop.
    Skip,
}

/// One entry in the plan's topo-sorted op sequence.
#[derive(Clone, Debug)]
pub struct PlannedOp {
    pub node: GraphNode,
    pub kind: OpKind,
    pub precomputed: NodePrecomputed,
    /// Arena slot byte offset for this op's primary output tensor.
    /// `Some(offset)` → arena-backed allocation at `arena_base + offset`
    /// (A2 runtime path, not yet wired).
    /// `None` → fall back to `GpuMemoryPool::get_tensor_*` at runtime.
    ///
    /// Stamped at plan time by `memory_planner`; no runtime HashMap
    /// lookup per op. Under Phase 4 A1 this field is populated but the
    /// runtime does not yet consume it — the dispatch-model refactor
    /// that bridges arena slices to `GpuTensor` is deferred to Phase 5
    /// A2. Having the interface contract in place now lets A2 wire the
    /// runtime without re-plumbing `lower()`.
    pub arena_offset: Option<usize>,
}

/// Per-run classification summary of `memory_planner`'s decisions.
/// Stored on [`MemoryPlan`] at construction time so the accessor is
/// parameter-less (see [`MemoryPlan::classification_counts`]).
#[derive(Clone, Debug, Default)]
pub struct PlannerClassification {
    /// Tensors passing both `is_planner_eligible` AND the
    /// fusion-skipped filter.
    pub arena_eligible: usize,
    /// Subset of `arena_eligible` actually assigned to an arena slot.
    /// Under the current greedy allocator this always equals
    /// `arena_eligible`; tracked separately so a future algorithm
    /// that might decline an eligible tensor (e.g., fragmentation
    /// cap) surfaces the decision rather than hiding it.
    pub arena_assigned: usize,
    /// Tensors that did NOT pass eligibility OR were fusion-skipped.
    /// Take the pool path at runtime.
    pub pool_classified: usize,
}

/// Static memory plan — assigns eligible tensors to arena slots.
/// Produced by `memory_planner` after fusion annotation; frozen part
/// of [`ExecutionPlan`].
///
/// Phase 4 A1 lands this as pure data. Runtime consumption (Executor
/// allocating an arena buffer, ops reading slot byte offsets) is
/// deferred to Phase 5 A2.
#[derive(Clone, Debug)]
pub struct MemoryPlan {
    /// Total bytes required for the arena. Allocated as a single
    /// contiguous device buffer at `run()` start (under A2; currently
    /// not consumed).
    pub arena_bytes: usize,
    /// Number of slots in the arena. Each slot is one size-class
    /// bucket that may host one or more tensors with disjoint
    /// lifetimes.
    pub slot_count: usize,
    /// `tensor_name → slot_idx`. Callers typically use the
    /// [`MemoryPlan::arena_offset`] accessor instead of this directly.
    /// `BTreeMap` so the `Debug` representation is deterministic
    /// across independent planner calls (required by the
    /// determinism-snapshot test).
    pub(crate) slot_assignments: BTreeMap<String, usize>,
    /// `slot_idx → byte offset within the arena`.
    pub(crate) slot_offsets_bytes: Vec<usize>,
    /// Classification summary — see [`PlannerClassification`].
    classification: PlannerClassification,
}

impl MemoryPlan {
    /// Return the arena byte-offset for a tensor, or `None` if the
    /// tensor is not arena-assigned (pool path at runtime).
    pub fn arena_offset(&self, tensor_name: &str) -> Option<usize> {
        let &slot_idx = self.slot_assignments.get(tensor_name)?;
        Some(self.slot_offsets_bytes[slot_idx])
    }

    /// Parameter-less accessor for classification counts.
    pub fn classification_counts(&self) -> &PlannerClassification {
        &self.classification
    }

    /// Constructor used by `memory_planner`. `pub(crate)` so
    /// `MemoryPlan` instances are only produced by the planner pass,
    /// not by arbitrary callers.
    pub(crate) fn new(
        arena_bytes: usize,
        slot_count: usize,
        slot_assignments: BTreeMap<String, usize>,
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

    /// Empty plan — every tensor takes the pool path. Used when
    /// `lower()` runs on a graph with no arena-eligible tensors (or
    /// before Task 3.6 wires the planner in place).
    pub fn empty() -> Self {
        Self {
            arena_bytes: 0,
            slot_count: 0,
            slot_assignments: BTreeMap::new(),
            slot_offsets_bytes: Vec::new(),
            classification: PlannerClassification::default(),
        }
    }
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
    pub dynamic_candidates: HashMap<String, FusedPatternInfo>,
    pub tensor_last_use: HashMap<String, usize>,
    pub graph_inputs: Vec<GraphInput>,
    pub graph_outputs: Vec<String>,
    /// Per-tensor inferred shapes keyed by tensor name. Populated by
    /// `shape_inference` pass; consumed by `precompute_params` (for
    /// Reshape `0`/`-1` resolution) and available for Phase 4's memory
    /// planner. `Vec<Option<usize>>` per tensor — `Some(n)` is a statically
    /// inferred dim, `None` is dynamic. Empty Vec means the tensor's shape
    /// is completely unknown.
    pub tensor_shapes: HashMap<String, Vec<Option<usize>>>,
    /// Static memory plan produced by `memory_planner`. Phase 4 A1
    /// populates this; the runtime arena path that consumes it lands
    /// in Phase 5 A2. Until then, every op takes the pool path
    /// regardless of `memory_plan.arena_offset(...)`.
    pub memory_plan: MemoryPlan,
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
