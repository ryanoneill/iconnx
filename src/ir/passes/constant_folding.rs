//! Constant-folding pass — promotes initializer-rooted fold chains into
//! synthesized initializers. The fold loop itself is in
//! `folding_core`; this module is the initializer-only configuration.

use crate::ir::graph::OptimizableGraph;
use crate::ir::passes::folding_core;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attributes::NodeAttributes;
    use crate::ir::OptimizableGraphBuilder;
    use crate::tensor::Tensor;

    #[test]
    fn mul_of_two_initializers_folds() {
        let mut b = OptimizableGraphBuilder::new();
        b.add_initializer("a".into(), Tensor::from_vec_f32(vec![2.0, 3.0], vec![2]));
        b.add_initializer("b".into(), Tensor::from_vec_f32(vec![4.0, 5.0], vec![2]));
        b.add_node(
            "m",
            "Mul",
            vec!["a".into(), "b".into()],
            vec!["y".into()],
            NodeAttributes::new(),
        );
        let g = constant_folding(b.build());
        assert!(
            g.nodes.is_empty(),
            "Mul of two constants should be folded away"
        );
        let y = g.initializers.get("y").expect("folded y initializer present");
        assert_eq!(y.as_slice(), &[8.0, 15.0]);
    }

    #[test]
    fn chain_add_mul_folds_transitively() {
        let mut b = OptimizableGraphBuilder::new();
        b.add_initializer("c1".into(), Tensor::from_vec_f32(vec![1.0], vec![1]));
        b.add_initializer("c2".into(), Tensor::from_vec_f32(vec![2.0], vec![1]));
        b.add_initializer("c3".into(), Tensor::from_vec_f32(vec![3.0], vec![1]));
        b.add_node(
            "add1",
            "Add",
            vec!["c1".into(), "c2".into()],
            vec!["t0".into()],
            NodeAttributes::new(),
        );
        b.add_node(
            "mul1",
            "Mul",
            vec!["t0".into(), "c3".into()],
            vec!["y".into()],
            NodeAttributes::new(),
        );
        let g = constant_folding(b.build());
        assert!(g.nodes.is_empty(), "chain of constants should fold");
        assert_eq!(g.initializers.get("y").unwrap().as_slice(), &[9.0]); // (1+2)*3 = 9
    }

    #[test]
    fn semi_constant_does_not_fold() {
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("x".into(), vec![Some(1)]);
        b.add_initializer("c".into(), Tensor::from_vec_f32(vec![5.0], vec![1]));
        b.add_node(
            "add",
            "Add",
            vec!["x".into(), "c".into()],
            vec!["y".into()],
            NodeAttributes::new(),
        );
        let g = constant_folding(b.build());
        assert_eq!(g.nodes.len(), 1, "op with a dynamic input should not fold");
    }

    #[test]
    fn unknown_op_does_not_fold() {
        let mut b = OptimizableGraphBuilder::new();
        b.add_initializer("c".into(), Tensor::from_vec_f32(vec![0.5], vec![1]));
        b.add_node(
            "wacky",
            "WackyUnknownOp",
            vec!["c".into()],
            vec!["y".into()],
            NodeAttributes::new(),
        );
        let g = constant_folding(b.build());
        assert_eq!(g.nodes.len(), 1, "unknown op stays");
        assert!(g.initializers.get("y").is_none());
    }

    #[test]
    fn output_budget_respected() {
        // A Mul of two 1001x1001 f32 tensors produces 1,002,001 elements,
        // exceeding FOLD_OUTPUT_ELEMENT_BUDGET (1M). Should not fold.
        let big = vec![0.5_f32; 1001 * 1001];
        let mut b = OptimizableGraphBuilder::new();
        b.add_initializer(
            "a".into(),
            Tensor::from_vec_f32(big.clone(), vec![1001, 1001]),
        );
        b.add_initializer("b".into(), Tensor::from_vec_f32(big, vec![1001, 1001]));
        b.add_node(
            "huge_mul",
            "Mul",
            vec!["a".into(), "b".into()],
            vec!["y".into()],
            NodeAttributes::new(),
        );
        let g = constant_folding(b.build());
        assert_eq!(g.nodes.len(), 1, "output-size-capped op should not fold");
    }

    #[test]
    fn single_op_fold_matches_direct_forward_1e_7() {
        let a = Tensor::from_vec_f32(vec![1.23, 4.56, 7.89], vec![3]);
        let b = Tensor::from_vec_f32(vec![0.5, 0.25, 0.125], vec![3]);
        let attrs = NodeAttributes::new();

        // Direct forward.
        let direct = crate::operators::mul::Mul::forward(&[a.clone(), b.clone()], &attrs);

        // Via constant_folding.
        let mut builder = OptimizableGraphBuilder::new();
        builder.add_initializer("a".into(), a);
        builder.add_initializer("b".into(), b);
        builder.add_node(
            "m",
            "Mul",
            vec!["a".into(), "b".into()],
            vec!["y".into()],
            NodeAttributes::new(),
        );
        let g = constant_folding(builder.build());
        let folded = g.initializers.get("y").expect("folded");

        // Single op → 1e-7 per spec.
        let max_diff = direct
            .as_slice()
            .iter()
            .zip(folded.as_slice().iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff < 1e-7,
            "single-op fold must match direct forward to 1e-7, got {}",
            max_diff
        );
    }

    #[test]
    fn two_op_chain_fold_matches_manual_1e_6() {
        // (a + b) * c computed manually vs folded through constant_folding.
        let a = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::from_vec_f32(vec![0.1, 0.2, 0.3], vec![3]);
        let c = Tensor::from_vec_f32(vec![10.0, 20.0, 30.0], vec![3]);
        let attrs = NodeAttributes::new();

        let sum = crate::operators::add::Add::forward(&[a.clone(), b.clone()], &attrs);
        let manual = crate::operators::mul::Mul::forward(&[sum, c.clone()], &attrs);

        let mut builder = OptimizableGraphBuilder::new();
        builder.add_initializer("a".into(), a);
        builder.add_initializer("b".into(), b);
        builder.add_initializer("c".into(), c);
        builder.add_node(
            "add1",
            "Add",
            vec!["a".into(), "b".into()],
            vec!["t0".into()],
            attrs.clone(),
        );
        builder.add_node(
            "mul1",
            "Mul",
            vec!["t0".into(), "c".into()],
            vec!["y".into()],
            attrs,
        );
        let g = constant_folding(builder.build());
        let folded = g.initializers.get("y").expect("folded");

        let max_diff = manual
            .as_slice()
            .iter()
            .zip(folded.as_slice().iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff < 1e-6,
            "≥2-op chain fold must match manual to 1e-6, got {}",
            max_diff
        );
    }
}
