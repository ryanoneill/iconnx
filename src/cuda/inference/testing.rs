//! Test-only bridge for exercising the fusion detection walker from
//! integration tests without exposing the private executor internals.
//!
//! The testing module lives inside `cuda::inference` so it can reach the
//! `pub(in crate::cuda::inference)` internals of `ExecutionNode` and the
//! fusion submodule's `pub(super)` re-exports. It publishes a minimal
//! public surface that only echoes already-public types (`FusedPattern`,
//! `FusedPatternInfo`) so integration tests can detect patterns against a
//! hand-built node list.
//!
//! Consumers should only call these functions from `#[cfg(test)]` or
//! integration-test contexts — nothing in the production path depends on
//! this module.

use std::collections::{HashMap, HashSet};

use crate::attributes::NodeAttributes;
use crate::cuda::context::IconnxCudaContext;
use crate::cuda::inference::fusion::detect_fused_patterns;
use crate::cuda::inference::ExecutionNode;
use crate::cuda::tensor::GpuTensor;

// Re-export the pattern types on the testing surface so integration
// tests can consume them without walking the private `fusion` submodule
// path directly. The underlying enum is already `pub`; this re-export
// is the only way external integration tests can reach them.
pub use crate::cuda::inference::fusion::{FusedPattern, FusedPatternInfo};

/// Testing mirror of the internal `ExecutionNode` shape.
///
/// Tests construct these directly and the conversion inside
/// `detect_fused_patterns_for_tests` fills in the private
/// `precomputed_slice` field (always `None` in tests, which is what the
/// real executor does for non-`Slice` nodes).
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
) -> (HashMap<String, FusedPatternInfo>, HashSet<String>) {
    let converted: Vec<ExecutionNode> = nodes
        .iter()
        .map(|n| ExecutionNode {
            name: n.name.clone(),
            op_type: n.op_type.clone(),
            inputs: n.inputs.clone(),
            outputs: n.outputs.clone(),
            attributes: n.attributes.clone(),
            precomputed_slice: None,
            precomputed_reshape_shape: None,
            precomputed_unsqueeze_axes: None,
            precomputed_squeeze_axes: None,
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
/// read-only view of the three `Option<Vec<i64>>` fields that Cycle 2
/// adds to `ExecutionNode`.
#[derive(Debug, Clone, Default)]
pub struct NodePrecomputedSnapshot {
    pub reshape_shape: Option<Vec<i64>>,
    pub unsqueeze_axes: Option<Vec<i64>>,
    pub squeeze_axes: Option<Vec<i64>>,
}

/// Look up a node in the executor's internal node list by name and return
/// a snapshot of its precomputed shape-op fields. Returns `None` if no
/// node with that name is registered.
///
/// This is a test-only helper — production code should never need to
/// inspect the executor's internal node state directly.
pub fn get_node_precomputed(
    executor: &crate::cuda::inference::GpuGraphExecutor,
    node_name: &str,
) -> Option<NodePrecomputedSnapshot> {
    let node = executor.nodes.iter().find(|n| n.name == node_name)?;
    Some(NodePrecomputedSnapshot {
        reshape_shape: node.precomputed_reshape_shape.clone(),
        unsqueeze_axes: node.precomputed_unsqueeze_axes.clone(),
        squeeze_axes: node.precomputed_squeeze_axes.clone(),
    })
}
