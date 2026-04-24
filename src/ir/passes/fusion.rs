//! Fusion-pattern detection. Thin adapter that bridges
//! `OptimizableGraph` into the existing `cuda::inference::fusion` detection
//! library. Operates entirely on CPU `Tensor` values — no GPU context
//! required, no D2H copies.

use std::collections::{HashMap, HashSet};

use crate::cuda::inference::fusion::{FusedPatternInfo, detect_fused_patterns_from_cpu};
use crate::ir::graph::OptimizableGraph;

/// Result of the fusion pass — attached to an `OptimizableGraph` so
/// lowering can thread the annotations into each `PlannedOp`.
#[derive(Default, Debug, Clone)]
pub struct FusionAnnotations {
    pub heads: HashMap<String, FusedPatternInfo>,
    pub skipped: HashSet<String>,
    pub dynamic_candidates: HashMap<String, FusedPatternInfo>,
}

/// Detect fusion patterns in an `OptimizableGraph`.
///
/// The detection library operates directly on `&[GraphNode]`. Initializer
/// values are read from `graph.initializers` without any D2H round-trip.
pub fn detect_fusion(graph: &OptimizableGraph) -> FusionAnnotations {
    let (heads, skipped, dynamic_candidates) =
        detect_fused_patterns_from_cpu(&graph.nodes, &graph.initializers);
    FusionAnnotations { heads, skipped, dynamic_candidates }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attributes::NodeAttributes;
    use crate::cuda::inference::fusion::FusedPattern;
    use crate::ir::graph::OptimizableGraphBuilder;
    use crate::tensor::Tensor;

    #[test]
    fn detect_returns_empty_on_empty_graph() {
        let g = OptimizableGraphBuilder::new().build();
        let a = detect_fusion(&g);
        assert!(a.heads.is_empty());
        assert!(a.skipped.is_empty());
        assert!(a.dynamic_candidates.is_empty());
    }

    // Pattern-specific detection tests live alongside the existing fusion
    // detection tests (`tests/fusion_detection_test.rs`) — those
    // still run against the now-shared `detect_with_scalar_lookup` core.
    //
    // In addition, the test below exercises the *CPU-path* scalar-lookup
    // closure used by `detect_fused_patterns_from_cpu`. It verifies the
    // adapter correctly reads a `Tensor::Float32` initializer as a
    // scalar (the branch the GPU-side path covers via `to_host_f32`).
    // Without this test the Float32-initializer branch of the closure is
    // not exercised in a pure-CPU unit test.
    #[test]
    fn detect_finds_gelu_with_float32_initializer_exponent() {
        // GELU pattern:
        //   pow    = Pow(x, 3)
        //   mul1   = Mul(pow, 0.044715)
        //   add1   = Add(x, mul1)          # inner sum
        //   mul2   = Mul(add1, sqrt(2/pi)) # tanh pre-scale
        //   tanh_  = Tanh(mul2)
        //   add2   = Add(tanh_, 1)
        //   half   = Mul(x, 0.5)           # parallel path
        //   y      = Mul(half, add2)       # final Mul — tail
        //
        // The walker keys on the `Pow` node, then follows outputs.
        // `exp3` is the only input that needs to be read as a scalar; all
        // the other weights need only be present (their values aren't
        // inspected here).
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("x".into(), vec![]);
        b.add_initializer(
            "exp3".into(),
            Tensor::from_vec_f32(vec![3.0], vec![1]),
        );
        b.add_initializer(
            "c_inner".into(),
            Tensor::from_vec_f32(vec![0.044715], vec![1]),
        );
        b.add_initializer(
            "c_sqrt2pi".into(),
            Tensor::from_vec_f32(vec![0.797_884_6], vec![1]),
        );
        b.add_initializer(
            "c_one".into(),
            Tensor::from_vec_f32(vec![1.0], vec![1]),
        );
        b.add_initializer(
            "c_half".into(),
            Tensor::from_vec_f32(vec![0.5], vec![1]),
        );

        b.add_node(
            "pow",
            "Pow",
            vec!["x".into(), "exp3".into()],
            vec!["t_pow".into()],
            NodeAttributes::new(),
        );
        b.add_node(
            "mul1",
            "Mul",
            vec!["t_pow".into(), "c_inner".into()],
            vec!["t_mul1".into()],
            NodeAttributes::new(),
        );
        b.add_node(
            "add1",
            "Add",
            vec!["x".into(), "t_mul1".into()],
            vec!["t_add1".into()],
            NodeAttributes::new(),
        );
        b.add_node(
            "mul2",
            "Mul",
            vec!["t_add1".into(), "c_sqrt2pi".into()],
            vec!["t_mul2".into()],
            NodeAttributes::new(),
        );
        b.add_node(
            "tanh_",
            "Tanh",
            vec!["t_mul2".into()],
            vec!["t_tanh".into()],
            NodeAttributes::new(),
        );
        b.add_node(
            "add2",
            "Add",
            vec!["t_tanh".into(), "c_one".into()],
            vec!["t_add2".into()],
            NodeAttributes::new(),
        );
        b.add_node(
            "half",
            "Mul",
            vec!["x".into(), "c_half".into()],
            vec!["t_half".into()],
            NodeAttributes::new(),
        );
        b.add_node(
            "tail",
            "Mul",
            vec!["t_half".into(), "t_add2".into()],
            vec!["y".into()],
            NodeAttributes::new(),
        );
        b.add_output("y".into());

        let annotations = detect_fusion(&b.build());

        // The GELU walker keys on the `Pow` node, so "pow" is the head.
        let head = annotations
            .heads
            .get("pow")
            .expect("GELU head registered on Pow node");
        match &head.pattern {
            FusedPattern::Gelu { x_input, output_name } => {
                assert_eq!(x_input, "x");
                assert_eq!(output_name, "y");
            }
            other => panic!("expected FusedPattern::Gelu, got {:?}", other),
        }

        // All interior nodes should be marked for skip.
        for node in ["mul1", "add1", "mul2", "tanh_", "add2", "half", "tail"] {
            assert!(
                annotations.skipped.contains(node),
                "expected '{}' to be in skipped set",
                node
            );
        }
    }
}
