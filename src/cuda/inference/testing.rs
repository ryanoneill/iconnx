//! Test-only bridge for exercising fusion detection and plan-level
//! precomputation state from integration tests without reaching into
//! the private internals of the Phase 2 executor.
//!
//! With the Phase 2 strangler shim, precompute/fusion state lives on a
//! lazily-constructed [`crate::ir::ExecutionPlan`] rather than on
//! mutable fields of [`crate::cuda::inference::GpuGraphExecutor`]. These
//! helpers trigger `compile()` and inspect the resulting plan so tests
//! can assert against the actual compiled form.
//!
//! Consumers should only call these functions from `#[cfg(test)]` or
//! integration-test contexts — nothing in the production path depends on
//! this module.

use std::collections::{HashMap, HashSet};

use crate::attributes::NodeAttributes;
use crate::cuda::context::IconnxCudaContext;
use crate::cuda::inference::fusion::detect_fused_patterns;
use crate::cuda::tensor::GpuTensor;
use crate::ir::graph::GraphNode;
use crate::ir::plan::OpKind;

// Re-export the pattern types on the testing surface so integration
// tests can consume them without walking the private `fusion` submodule
// path directly. The underlying enum is already `pub`; this re-export
// is the only way external integration tests can reach them.
pub use crate::cuda::inference::fusion::{FusedPattern, FusedPatternInfo};

/// Testing mirror of the GraphNode shape — tests can construct these
/// directly. The conversion inside `detect_fused_patterns_for_tests`
/// widens them to `GraphNode`.
pub struct ExecutionNodeForTests {
    pub name: String,
    pub op_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attributes: NodeAttributes,
}

/// Run pattern detection on a test-fabricated node list.
///
/// The returned map is keyed by pattern-head node name (matching the
/// production map shape), and the set contains every node name that is
/// interior to a detected pattern and should be skipped during
/// execution.
pub fn detect_fused_patterns_for_tests(
    nodes: &[ExecutionNodeForTests],
    weights: &HashMap<String, GpuTensor>,
    ctx: &IconnxCudaContext,
) -> (
    HashMap<String, FusedPatternInfo>,
    HashSet<String>,
    HashMap<String, FusedPatternInfo>,
) {
    let converted: Vec<GraphNode> = nodes
        .iter()
        .map(|n| GraphNode {
            name: n.name.clone(),
            op_type: n.op_type.clone(),
            inputs: n.inputs.clone(),
            outputs: n.outputs.clone(),
            attributes: n.attributes.clone(),
        })
        .collect();
    detect_fused_patterns(&converted, weights, ctx)
}

/// Predicate: does this pattern info describe a `MulSinPowMulAdd` fusion?
///
/// Exposed as a helper so integration tests don't need to match on the
/// `FusedPattern` enum directly (and thus don't need to import every
/// variant they don't care about).
pub fn is_mul_sin_pow_mul_add(info: &FusedPatternInfo) -> bool {
    matches!(info.pattern, FusedPattern::MulSinPowMulAdd { .. })
}

/// Snapshot of a node's precomputed shape-op parameters, for test
/// inspection. Opaque to callers outside the testing facade — just a
/// read-only view of the three `Option<Vec<i64>>` fields on the node's
/// `NodePrecomputed` bundle in the compiled plan.
#[derive(Debug, Clone, Default)]
pub struct NodePrecomputedSnapshot {
    pub reshape_shape: Option<Vec<i64>>,
    pub unsqueeze_axes: Option<Vec<i64>>,
    pub squeeze_axes: Option<Vec<i64>>,
}

/// Look up a node in the executor's compiled plan by name and return a
/// snapshot of its precomputed shape-op fields. Returns `None` if no
/// node with that name is registered.
///
/// Forces compilation of the executor's graph so tests can inspect
/// precompute state before calling `run()`. The executor's cached plan
/// is populated as a side effect; subsequent `run()` calls reuse it.
pub fn get_node_precomputed(
    executor: &crate::cuda::inference::GpuGraphExecutor,
    node_name: &str,
) -> Option<NodePrecomputedSnapshot> {
    executor
        .compile()
        .expect("compile executor graph for test inspection");
    let plan_ref = executor.compiled_plan();
    let plan = plan_ref.as_ref()?;
    let op = plan.ops.iter().find(|op| op.node.name == node_name)?;
    Some(NodePrecomputedSnapshot {
        reshape_shape: op.precomputed.reshape_shape.clone(),
        unsqueeze_axes: op.precomputed.unsqueeze_axes.clone(),
        squeeze_axes: op.precomputed.squeeze_axes.clone(),
    })
}

/// Aggregate counts of how many Unsqueeze/Reshape/Squeeze nodes in the
/// executor have their precomputed fields populated. Used by the Kokoro
/// integration test to verify Cycle 2's precomputation mechanism is
/// firing on real model graphs.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PrecomputeStats {
    pub unsqueeze_hits: usize,
    pub unsqueeze_total: usize,
    pub reshape_hits: usize,
    pub reshape_total: usize,
    pub squeeze_hits: usize,
    pub squeeze_total: usize,
}

/// Count precompute hits and totals for Unsqueeze, Reshape, and Squeeze
/// nodes in the executor's compiled plan.
///
/// Forces compilation if the plan hasn't been built yet.
pub fn count_precomputed_shape_ops(
    executor: &crate::cuda::inference::GpuGraphExecutor,
) -> PrecomputeStats {
    executor
        .compile()
        .expect("compile executor graph for test inspection");
    let plan_ref = executor.compiled_plan();
    let plan = match plan_ref.as_ref() {
        Some(p) => p,
        None => return PrecomputeStats::default(),
    };
    let mut stats = PrecomputeStats::default();
    for op in &plan.ops {
        match op.node.op_type.as_str() {
            "Unsqueeze" => {
                stats.unsqueeze_total += 1;
                if op.precomputed.unsqueeze_axes.is_some() {
                    stats.unsqueeze_hits += 1;
                }
            }
            "Reshape" => {
                stats.reshape_total += 1;
                if op.precomputed.reshape_shape.is_some() {
                    stats.reshape_hits += 1;
                }
            }
            "Squeeze" => {
                stats.squeeze_total += 1;
                if op.precomputed.squeeze_axes.is_some() {
                    stats.squeeze_hits += 1;
                }
            }
            _ => {}
        }
    }
    stats
}

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
/// has registered on the given executor's compiled plan.
///
/// Counts both static patterns (`OpKind::FusedHead`) and dynamic
/// candidates (`plan.dynamic_candidates`). Forces compilation if the
/// plan hasn't been built yet.
pub fn count_fused_patterns(
    executor: &crate::cuda::inference::GpuGraphExecutor,
) -> PatternCount {
    executor
        .compile()
        .expect("compile executor graph for test inspection");
    let plan_ref = executor.compiled_plan();
    let plan = match plan_ref.as_ref() {
        Some(p) => p,
        None => return PatternCount::default(),
    };
    let mut count = PatternCount::default();
    for op in &plan.ops {
        if let OpKind::FusedHead(pattern) = &op.kind {
            increment(&mut count, pattern);
        }
    }
    for info in plan.dynamic_candidates.values() {
        increment(&mut count, &info.pattern);
    }
    count
}

fn increment(count: &mut PatternCount, pattern: &FusedPattern) {
    match pattern {
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
