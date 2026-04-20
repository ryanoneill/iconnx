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
