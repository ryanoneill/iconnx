//! Constant-promotion pass — promotes single-output `Constant` *nodes*
//! whose `value` attribute carries a static tensor into
//! `OptimizableGraph::initializers`.
//!
//! # Why this exists
//!
//! ONNX serializers have two equivalent ways to represent a static
//! tensor in a graph:
//!
//! 1. A **graph initializer** — appears in
//!    `OptimizableGraph::initializers` and is looked up by name when
//!    any downstream op consumes it.
//! 2. A **`Constant` node** — appears as a graph node whose output is
//!    the static tensor and whose value lives on the node's `value`
//!    attribute.
//!
//! ORT treats the two as interchangeable. Most exporters use form (1),
//! but several real-world exporters (paddle2onnx's `_rapidai` variants
//! among them) emit every static tensor as form (2). In particular,
//! `en_number_mobile_v2.0_rec_rapidai.onnx` exports its LSTM
//! `(W, R, B)` weights through a runtime chain of
//! `Constant → Unsqueeze → Concat → Slice → Concat` nodes — every
//! foldable leaf is a `Constant` node, and the graph has **zero**
//! initializers.
//!
//! `pack_lstm_weights_eagerly` (`src/ir/lowering.rs`) looks the LSTM
//! `W`/`R`/`B` input names up in the initializer map; without the
//! promotion below, the rec model fails at lowering time with
//! `Kernel("LSTM '...' W initializer 'Concat_6' not found")`.
//!
//! # What this pass does
//!
//! For every `Constant` node with **exactly one non-empty output** and
//! a resolvable static tensor value, this pass:
//!
//! - inserts `(output_name, tensor)` into `graph.initializers`, and
//! - drops the `Constant` node from `graph.nodes`.
//!
//! The downstream `constant_folding` pass then collapses any chain of
//! foldable ops rooted at those initializers (`Concat`, `Slice`,
//! `Unsqueeze`, etc.) into a single static initializer — at which
//! point `pack_lstm_weights_eagerly` finds the LSTM weights.
//!
//! # Why this is principled, not a hack
//!
//! Constant-vs-initializer is purely a serialization artifact in ONNX
//! — there is no execution-semantic difference between the two. Every
//! op that reads a Constant's output reads a static tensor with the
//! same dtype and shape it would read from an initializer. ORT does
//! exactly this internally during graph optimization. The promotion is
//! semantically equivalent to inlining.
//!
//! # Pass ordering
//!
//! Must run **before** `constant_folding`. The folding pass's
//! `try_fold` reads inputs from `(initializers, folded)` — so once a
//! Constant's output is an initializer, the entire foldable subgraph
//! rooted at it collapses in the existing pass.
//!
//! # Why we don't extend the folding dispatcher instead
//!
//! `dispatch_cpu_forward` in `folding_core.rs` intentionally excludes
//! `Constant` because folding's contract is "evaluate a CPU forward
//! with the input tensors"; `Constant` has zero inputs and its output
//! lives on the node attribute, not on any input. Promotion is a
//! cleaner seam: it makes the Constant→initializer equivalence
//! explicit at the IR level, then folding gets to work without
//! special-casing Constants.
//!
//! # Coverage
//!
//! The existing Int64 path in
//! `src/ir/passes/precompute.rs::collect_i64_values` and
//! `src/ir/lowering.rs::collect_static_i64` walks Constant nodes
//! directly and is left in place as a defense-in-depth fallback (it is
//! harmless once promotion has folded the Constants — they're already
//! initializers). Unifying those into this pass is left for a future
//! refactor so this change stays focused on unblocking M6.5.

use crate::ir::graph::OptimizableGraph;

/// Promote every `Constant` node with a single non-empty output and a
/// resolvable static tensor `value` into `graph.initializers`.
///
/// Constants whose `value` does not resolve to a tensor (e.g.
/// `value_string` / `sparse_value` — both currently unsupported by
/// `NodeAttributes::resolve_constant_value`), or whose output is empty,
/// or that emit more than one output, are preserved verbatim. The
/// downstream dispatch will then either pick them up via the
/// `Constant` if-let guard (executor) or surface a real error.
///
/// Idempotent: running the pass twice produces the same graph (any
/// Constants that were promotable were promoted on the first pass).
pub fn constant_promotion(mut graph: OptimizableGraph) -> OptimizableGraph {
    let mut kept = Vec::with_capacity(graph.nodes.len());

    for node in graph.nodes.into_iter() {
        if !is_promotable(&node) {
            kept.push(node);
            continue;
        }

        let Some(value) = node.attributes.resolve_constant_value() else {
            // Value isn't carried in a supported attribute (e.g.
            // `value_string` or `sparse_value`). Preserve the node so
            // any downstream consumer surfaces a real error rather
            // than a silent drop.
            kept.push(node);
            continue;
        };

        // Safe: `is_promotable` already verified `outputs.len() == 1`
        // and `outputs[0]` is non-empty.
        let output_name = node.outputs[0].clone();

        // `insert` overwrites: if a same-named initializer already
        // exists, the Constant node wins. This matches ORT's behavior
        // (Constant nodes shadow initializers of the same name) and
        // also catches the pathological case of an exporter emitting
        // both forms — exceedingly rare in practice but well-defined.
        graph.initializers.insert(output_name, value);
        // The node itself is dropped — its output is now an
        // initializer.
    }

    graph.nodes = kept;
    graph
}

/// A `Constant` node is promotable iff it has exactly one output and
/// that output name is non-empty. Multi-output Constants do not exist
/// in any released ONNX opset, but the guard future-proofs the pass
/// against an opset that adds one.
fn is_promotable(node: &crate::ir::graph::GraphNode) -> bool {
    if node.op_type != "Constant" {
        return false;
    }
    if node.outputs.len() != 1 {
        return false;
    }
    !node.outputs[0].is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attributes::NodeAttributes;
    use crate::ir::passes::{constant_folding_pass, topo_sort};
    use crate::ir::OptimizableGraphBuilder;
    use crate::tensor::Tensor;

    #[test]
    fn float32_constant_node_becomes_initializer() {
        // The single most important behavior: Float32 Constants
        // promote so that downstream folding can pick them up.
        let mut b = OptimizableGraphBuilder::new();
        let mut attrs = NodeAttributes::new();
        attrs.add_tensor(
            "value".into(),
            Tensor::from_vec_f32(vec![1.0, 2.0, 3.0], vec![3]),
        );
        b.add_node("c0", "Constant", vec![], vec!["c0_out".into()], attrs);

        let g = constant_promotion(b.build());
        assert!(g.nodes.is_empty(), "Constant node should be removed");
        let t = g.initializers.get("c0_out").expect("initializer");
        assert_eq!(t.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn int64_constant_node_becomes_initializer() {
        // Int64 Constants (axes/indices/shape sources) promote too —
        // the existing `collect_i64_values` / `collect_static_i64`
        // paths still work but no longer have to walk the node list.
        let mut b = OptimizableGraphBuilder::new();
        let mut attrs = NodeAttributes::new();
        attrs.add_tensor(
            "value".into(),
            Tensor::from_vec_i64(vec![0, 1, 2], vec![3]),
        );
        b.add_node("c0", "Constant", vec![], vec!["c0_out".into()], attrs);

        let g = constant_promotion(b.build());
        assert!(g.nodes.is_empty());
        let t = g.initializers.get("c0_out").expect("initializer");
        assert_eq!(t.as_slice_i64(), vec![0i64, 1, 2]);
    }

    #[test]
    fn int32_constant_node_becomes_initializer() {
        let mut b = OptimizableGraphBuilder::new();
        let mut attrs = NodeAttributes::new();
        attrs.add_tensor(
            "value".into(),
            Tensor::from_vec_i32(vec![7, 8], vec![2]),
        );
        b.add_node("c0", "Constant", vec![], vec!["c0_out".into()], attrs);

        let g = constant_promotion(b.build());
        assert!(g.nodes.is_empty());
        let t = g.initializers.get("c0_out").expect("initializer");
        assert_eq!(t.as_slice_i32(), vec![7i32, 8]);
    }

    #[test]
    fn bool_constant_node_becomes_initializer() {
        let mut b = OptimizableGraphBuilder::new();
        let mut attrs = NodeAttributes::new();
        attrs.add_tensor(
            "value".into(),
            Tensor::from_vec_bool(vec![true, false, true], vec![3]),
        );
        b.add_node("c0", "Constant", vec![], vec!["c0_out".into()], attrs);

        let g = constant_promotion(b.build());
        assert!(g.nodes.is_empty());
        let t = g.initializers.get("c0_out").expect("initializer");
        assert_eq!(t.as_slice_bool(), vec![true, false, true]);
    }

    #[test]
    fn float16_constant_node_becomes_initializer() {
        // Float16 / BFloat16 Constants are rare but turn up in FP16
        // exports. Verifying coverage prevents a future surprise.
        use half::f16;
        let mut b = OptimizableGraphBuilder::new();
        let mut attrs = NodeAttributes::new();
        attrs.add_tensor(
            "value".into(),
            Tensor::from_vec_f16(
                vec![f16::from_f32(1.5), f16::from_f32(-0.25)],
                vec![2],
            ),
        );
        b.add_node("c0", "Constant", vec![], vec!["c0_out".into()], attrs);

        let g = constant_promotion(b.build());
        assert!(g.nodes.is_empty());
        let t = g.initializers.get("c0_out").expect("initializer");
        let got = t.as_slice_f16();
        assert_eq!(got, vec![f16::from_f32(1.5), f16::from_f32(-0.25)]);
    }

    #[test]
    fn constant_with_empty_output_is_preserved() {
        // Empty output name is the "no output" sentinel — preserve
        // the node verbatim so any later pass surfaces the real
        // condition rather than a silent drop.
        let mut b = OptimizableGraphBuilder::new();
        let mut attrs = NodeAttributes::new();
        attrs.add_tensor(
            "value".into(),
            Tensor::from_vec_f32(vec![1.0], vec![1]),
        );
        b.add_node("c0", "Constant", vec![], vec!["".into()], attrs);

        let g = constant_promotion(b.build());
        assert_eq!(g.nodes.len(), 1, "Constant with empty output stays");
        assert!(g.initializers.is_empty());
    }

    #[test]
    fn constant_without_resolvable_value_is_preserved() {
        // No `value` / `value_int` / `value_float` / etc. — promotion
        // can't recover a tensor, so leave the node verbatim.
        let mut b = OptimizableGraphBuilder::new();
        b.add_node(
            "c0",
            "Constant",
            vec![],
            vec!["c0_out".into()],
            NodeAttributes::new(),
        );

        let g = constant_promotion(b.build());
        assert_eq!(g.nodes.len(), 1, "unresolvable Constant stays");
        assert!(!g.initializers.contains_key("c0_out"));
    }

    #[test]
    fn non_constant_op_is_left_alone() {
        let mut b = OptimizableGraphBuilder::new();
        b.add_initializer("a".into(), Tensor::from_vec_f32(vec![1.0], vec![1]));
        b.add_initializer("b".into(), Tensor::from_vec_f32(vec![2.0], vec![1]));
        b.add_node(
            "add",
            "Add",
            vec!["a".into(), "b".into()],
            vec!["c".into()],
            NodeAttributes::new(),
        );

        let g = constant_promotion(b.build());
        assert_eq!(g.nodes.len(), 1);
        assert_eq!(g.nodes[0].op_type, "Add");
    }

    #[test]
    fn idempotent_second_run_is_noop() {
        let mut b = OptimizableGraphBuilder::new();
        let mut attrs = NodeAttributes::new();
        attrs.add_tensor("value".into(), Tensor::from_vec_f32(vec![1.0], vec![1]));
        b.add_node("c0", "Constant", vec![], vec!["c0_out".into()], attrs);

        let g1 = constant_promotion(b.build());
        let n_after_first = g1.nodes.len();
        let inits_after_first = g1.initializers.len();
        let g2 = constant_promotion(g1);
        assert_eq!(g2.nodes.len(), n_after_first);
        assert_eq!(g2.initializers.len(), inits_after_first);
        assert!(g2.initializers.contains_key("c0_out"));
    }

    #[test]
    fn paddle_rec_style_constant_chain_folds_after_promotion() {
        // The acceptance test for the M6.5 rec bug, as a synthetic
        // graph: build a `Constant → Unsqueeze → Concat → Slice →
        // Concat` chain (the exact shape paddle2onnx's _rapidai rec
        // model uses for every LSTM W/R/B), promote the Constants,
        // then verify the entire chain folds away leaving a single
        // static initializer downstream consumers can find.
        //
        // The arithmetic exercises every link in the chain:
        // - Two scalar f32 Constants c0, c1 → Unsqueeze along axis 0
        //   each → two rank-1 [1] tensors
        // - Concat along axis 0 → rank-1 [2] tensor [c0_val, c1_val]
        // - Slice [0:2] (no-op slice) → rank-1 [2] tensor
        // - Concat with itself along axis 0 → rank-1 [4] tensor
        //
        // Result: post-promotion + folding, the graph has zero
        // nodes and the chain output is a single concrete
        // initializer.
        let mut b = OptimizableGraphBuilder::new();

        // The two leaf scalar Constants (f32 — the dtype that the
        // existing folding excluded). Shape [] (scalar).
        let mut c0_attrs = NodeAttributes::new();
        c0_attrs.add_tensor(
            "value".into(),
            Tensor::from_vec_f32(vec![1.5], vec![]),
        );
        b.add_node("c0", "Constant", vec![], vec!["c0_out".into()], c0_attrs);

        let mut c1_attrs = NodeAttributes::new();
        c1_attrs.add_tensor(
            "value".into(),
            Tensor::from_vec_f32(vec![-2.5], vec![]),
        );
        b.add_node("c1", "Constant", vec![], vec!["c1_out".into()], c1_attrs);

        // Unsqueeze axis=0 inputs are themselves Int64 Constants
        // (this is the real shape — axes is an input, not an
        // attribute, in opset ≥ 13).
        let mut axes_attrs = NodeAttributes::new();
        axes_attrs.add_tensor(
            "value".into(),
            Tensor::from_vec_i64(vec![0], vec![1]),
        );
        b.add_node(
            "axes_const",
            "Constant",
            vec![],
            vec!["axes_out".into()],
            axes_attrs,
        );

        b.add_node(
            "u0",
            "Unsqueeze",
            vec!["c0_out".into(), "axes_out".into()],
            vec!["u0_out".into()],
            NodeAttributes::new(),
        );
        b.add_node(
            "u1",
            "Unsqueeze",
            vec!["c1_out".into(), "axes_out".into()],
            vec!["u1_out".into()],
            NodeAttributes::new(),
        );

        // Concat axis=0 of the two [1]-shaped tensors → [2].
        let mut cat0_attrs = NodeAttributes::new();
        cat0_attrs.add_int("axis".into(), 0);
        b.add_node(
            "cat0",
            "Concat",
            vec!["u0_out".into(), "u1_out".into()],
            vec!["cat0_out".into()],
            cat0_attrs,
        );

        // Slice [0:2] along axis 0 — no-op slice that exercises the
        // op nevertheless. Slice inputs: data, starts, ends, axes,
        // steps. Build the three Int64 Constants for those inputs.
        let mut starts_attrs = NodeAttributes::new();
        starts_attrs.add_tensor(
            "value".into(),
            Tensor::from_vec_i64(vec![0], vec![1]),
        );
        b.add_node(
            "starts_const",
            "Constant",
            vec![],
            vec!["starts_out".into()],
            starts_attrs,
        );
        let mut ends_attrs = NodeAttributes::new();
        ends_attrs.add_tensor(
            "value".into(),
            Tensor::from_vec_i64(vec![2], vec![1]),
        );
        b.add_node(
            "ends_const",
            "Constant",
            vec![],
            vec!["ends_out".into()],
            ends_attrs,
        );
        let mut slice_axes_attrs = NodeAttributes::new();
        slice_axes_attrs.add_tensor(
            "value".into(),
            Tensor::from_vec_i64(vec![0], vec![1]),
        );
        b.add_node(
            "slice_axes_const",
            "Constant",
            vec![],
            vec!["slice_axes_out".into()],
            slice_axes_attrs,
        );

        b.add_node(
            "sl",
            "Slice",
            vec![
                "cat0_out".into(),
                "starts_out".into(),
                "ends_out".into(),
                "slice_axes_out".into(),
            ],
            vec!["sl_out".into()],
            NodeAttributes::new(),
        );

        // Final Concat axis=0 of sl_out with itself → [4].
        let mut cat1_attrs = NodeAttributes::new();
        cat1_attrs.add_int("axis".into(), 0);
        b.add_node(
            "cat1",
            "Concat",
            vec!["sl_out".into(), "sl_out".into()],
            vec!["W".into()],
            cat1_attrs,
        );

        b.add_output("W".into());

        // Promote, then topo sort (so folding sees nodes in order),
        // then fold.
        let g = constant_promotion(b.build());

        // Every Constant should be gone post-promotion.
        assert!(
            g.nodes.iter().all(|n| n.op_type != "Constant"),
            "all Constant nodes should be promoted to initializers"
        );
        assert!(g.initializers.contains_key("c0_out"));
        assert!(g.initializers.contains_key("c1_out"));
        assert!(g.initializers.contains_key("axes_out"));
        assert!(g.initializers.contains_key("starts_out"));
        assert!(g.initializers.contains_key("ends_out"));
        assert!(g.initializers.contains_key("slice_axes_out"));

        // Re-topo-sort and fold.
        let g = topo_sort(g).expect("topo sort");
        let g = constant_folding_pass(g);

        // The entire foldable chain should have collapsed: no nodes
        // remain (the Unsqueeze/Concat/Slice/Concat chain all folded
        // to initializers).
        assert!(
            g.nodes.is_empty(),
            "promotion + folding should collapse the entire chain, \
             but {} nodes remain: {:?}",
            g.nodes.len(),
            g.nodes.iter().map(|n| &n.op_type).collect::<Vec<_>>()
        );

        let w = g
            .initializers
            .get("W")
            .expect("chain output W should be a static initializer");
        // Expected: [1.5, -2.5, 1.5, -2.5] (cat0 → [1.5, -2.5],
        // slice no-op → [1.5, -2.5], cat1 with itself → [1.5, -2.5,
        // 1.5, -2.5]).
        assert_eq!(w.as_slice(), &[1.5_f32, -2.5, 1.5, -2.5]);
    }
}
