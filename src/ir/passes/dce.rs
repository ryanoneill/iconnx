//! Dead-code elimination sweep — removes ops whose outputs feed nothing.

use std::collections::HashSet;

use crate::ir::graph::{GraphNode, OptimizableGraph};

/// Remove ops whose outputs are neither consumed by any other op nor
/// declared as graph outputs. Mark-sweep from `graph.outputs`, backwards.
///
/// Algorithm:
///
/// 1. Seed the `live` tensor-name set with `graph.outputs`.
/// 2. Iterate ops in reverse; any op whose output lands in `live` contributes
///    all of its non-empty inputs to `live`. Repeat until fixed point.
/// 3. Keep only ops that have at least one live output (multi-output ops
///    with any live output survive; empty-string output slots are ignored).
pub fn dead_code_elimination(mut graph: OptimizableGraph) -> OptimizableGraph {
    // Live tensor names: start from graph outputs.
    let mut live: HashSet<String> = graph.outputs.iter().cloned().collect();

    // Fixed-point expansion: repeatedly add inputs of live ops until stable.
    //
    // Implementation detail: an op is live iff any of its outputs is live.
    // All of that op's inputs then become live too.
    //
    // We iterate over `graph.nodes` in reverse topological order for
    // efficiency — since topo_sort runs before DCE, later nodes in
    // `graph.nodes` are typically downstream.
    loop {
        let mut added = false;
        for node in graph.nodes.iter().rev() {
            if node.outputs.iter().any(|o| !o.is_empty() && live.contains(o)) {
                for inp in &node.inputs {
                    if !inp.is_empty() && live.insert(inp.clone()) {
                        added = true;
                    }
                }
            }
        }
        if !added {
            break;
        }
    }

    // Retain only ops with at least one live output.
    let kept: Vec<GraphNode> = graph
        .nodes
        .into_iter()
        .filter(|node| {
            node.outputs
                .iter()
                .any(|o| !o.is_empty() && live.contains(o))
        })
        .collect();

    graph.nodes = kept;
    graph
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::cuda::IconnxCudaContext;
    use crate::ir;
    use crate::onnx_parser::OnnxParser;

    #[test]
    fn empty_graph_is_noop() {
        let graph = crate::ir::OptimizableGraphBuilder::new().build();
        let out = super::dead_code_elimination(graph);
        assert!(out.nodes.is_empty());
    }

    #[test]
    fn unconsumed_constant_is_removed() {
        use crate::attributes::NodeAttributes;
        use crate::tensor::Tensor;

        let mut attrs = NodeAttributes::new();
        attrs.add_tensor("value".into(), Tensor::from_vec_f32(vec![1.0], vec![1]));

        let mut b = crate::ir::OptimizableGraphBuilder::new();
        b.add_node("dead_const", "Constant", vec![], vec!["x".into()], attrs);
        // No output named "x" is consumed anywhere.
        let out = super::dead_code_elimination(b.build());
        assert!(
            out.nodes.is_empty(),
            "dead Constant should have been removed"
        );
    }

    #[test]
    fn live_op_survives() {
        use crate::attributes::NodeAttributes;

        let mut b = crate::ir::OptimizableGraphBuilder::new();
        b.add_input("a".into(), vec![Some(4)]);
        b.add_node(
            "n1",
            "Add",
            vec!["a".into(), "a".into()],
            vec!["y".into()],
            NodeAttributes::new(),
        );
        b.add_output("y".into());
        let out = super::dead_code_elimination(b.build());
        assert_eq!(out.nodes.len(), 1);
        assert_eq!(out.nodes[0].name, "n1");
    }

    #[test]
    fn multi_output_op_stays_if_any_output_is_live() {
        use crate::attributes::NodeAttributes;

        let mut b = crate::ir::OptimizableGraphBuilder::new();
        b.add_input("x".into(), vec![]);
        // Fake multi-output op (we don't need a real LSTM for the graph-level test).
        b.add_node(
            "multi",
            "LSTM",
            vec!["x".into()],
            vec!["y".into(), "y_h".into(), "y_c".into()],
            NodeAttributes::new(),
        );
        b.add_output("y".into()); // y is live; y_h and y_c are not
        let out = super::dead_code_elimination(b.build());
        assert_eq!(out.nodes.len(), 1, "op with any live output must be kept");
    }

    #[test]
    fn chain_of_dead_ops_all_removed() {
        use crate::attributes::NodeAttributes;

        let mut a1 = NodeAttributes::new();
        a1.add_tensor(
            "value".into(),
            crate::tensor::Tensor::from_vec_f32(vec![0.0], vec![1]),
        );
        let mut b = crate::ir::OptimizableGraphBuilder::new();
        b.add_node("c0", "Constant", vec![], vec!["a".into()], a1);
        b.add_node(
            "mul_dead",
            "Mul",
            vec!["a".into(), "a".into()],
            vec!["b".into()],
            NodeAttributes::new(),
        );
        // Nothing consumes "b" and no graph outputs.
        let out = super::dead_code_elimination(b.build());
        assert!(out.nodes.is_empty());
    }

    /// Pre-flight sanity check per the Phase 3 spec. Determines whether
    /// `dead_code_elimination` lands under the no-op clause (if zero
    /// orphans in Kokoro's post-Phase-2 plan) or falls under the
    /// regular Δratio gate.
    #[test]
    #[ignore = "requires GPU + kokoro model; one-shot determination"]
    fn kokoro_has_no_preexisting_orphans() {
        let model_path = "/home/ryano/workspace/ryanoneill/rust-ai-explorations/kokoro-v1.0.onnx";
        let model = OnnxParser::parse_file(model_path).expect("parse kokoro");

        // Build OptimizableGraph from the parsed ONNX model. The parser's
        // `GraphNode` has no `name` field, so we synthesize names as
        // `node_{i}` to match the pattern used by the `GpuGraphExecutor`
        // adapter in `tests/kokoro_inference_test.rs`.
        let mut builder = ir::OptimizableGraphBuilder::new();
        for (name, tensor) in model.extract_weights() {
            builder.add_initializer(name, tensor);
        }
        let graph_inputs = model.inputs();
        let graph_outputs = model.outputs();
        for name in &graph_inputs {
            builder.add_input(name.clone(), Vec::new());
        }
        for name in &graph_outputs {
            builder.add_output(name.clone());
        }
        let computation_graph = model.computation_graph();
        for (idx, node) in computation_graph.nodes().iter().enumerate() {
            builder.add_node(
                format!("node_{}", idx),
                node.op_type.clone(),
                node.inputs.clone(),
                node.outputs.clone(),
                node.attributes.clone(),
            );
        }

        let graph = builder.build();
        let ctx = IconnxCudaContext::new().expect("ctx");
        let plan = ir::lower(graph, &ctx).expect("lower");

        // Orphan definition: a PlannedOp whose outputs are
        //   - not present in any op's input set, AND
        //   - not in `plan.graph_outputs`.
        let consumed: HashSet<String> = plan
            .ops
            .iter()
            .flat_map(|op| op.node.inputs.iter().cloned())
            .chain(plan.graph_outputs.iter().cloned())
            .collect();

        let orphan_count: usize = plan
            .ops
            .iter()
            .filter(|op| {
                op.node
                    .outputs
                    .iter()
                    .all(|o| !o.is_empty() && !consumed.contains(o.as_str()))
            })
            .count();

        println!(
            "Kokoro plan: {} ops, {} orphans",
            plan.ops.len(),
            orphan_count
        );
        assert_eq!(
            orphan_count, 0,
            "expected 0 orphans in today's lower(kokoro); got {}. DCE must land under the \
             Δratio gate instead of the no-op clause.",
            orphan_count
        );
    }
}
