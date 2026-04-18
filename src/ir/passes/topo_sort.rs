//! Topological-sort pass — reorders `OptimizableGraph::nodes` so that every
//! node's inputs are produced by an earlier node (or by an initializer /
//! graph input). Returns `Err` on cycles.

use std::collections::{HashMap, HashSet};

use crate::cuda::CudaError;
use crate::ir::graph::{GraphNode, OptimizableGraph};

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
                visit(
                    dep_idx,
                    nodes,
                    node_by_output,
                    available,
                    visited,
                    on_stack,
                    sorted,
                )?;
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
    use crate::ir::graph::{GraphInput, OptimizableGraphBuilder};

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
        b.add_node(
            "n1",
            "Add",
            vec!["a".into()],
            vec!["b".into()],
            NodeAttributes::new(),
        );
        b.add_node(
            "n2",
            "Mul",
            vec!["b".into()],
            vec!["c".into()],
            NodeAttributes::new(),
        );
        let sorted = topo_sort(b.build()).unwrap();
        let names: Vec<_> = sorted.nodes.iter().map(|n| n.name.clone()).collect();
        assert_eq!(names, vec!["n1", "n2"]);
    }

    #[test]
    fn reverse_order_is_reordered() {
        let mut g = OptimizableGraph::default();
        g.inputs.push(GraphInput {
            name: "a".into(),
            shape: vec![],
        });
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
