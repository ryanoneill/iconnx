//! Pure-CPU computation graph ready for optimization passes.
//!
//! `OptimizableGraph` is the input to every pass and to `lower()`.
//! It has no GPU state: initializers are CPU-side `crate::tensor::Tensor`
//! values so constant-folding passes (Phase 3) can read them directly.

use std::collections::HashMap;

use crate::attributes::NodeAttributes;
use crate::tensor::Tensor;

/// A named graph input with optional dimensions.
///
/// `shape[i] == None` means the dimension is dynamic. Phase 3's
/// shape-inference pass fills in concrete values where possible.
#[derive(Clone, Debug)]
pub struct GraphInput {
    pub name: String,
    pub shape: Vec<Option<usize>>,
}

/// A single ONNX-style node.
#[derive(Clone, Debug)]
pub struct GraphNode {
    pub name: String,
    pub op_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attributes: NodeAttributes,
}

/// A computation graph before lowering — pure CPU data.
#[derive(Clone, Debug, Default)]
pub struct OptimizableGraph {
    pub inputs: Vec<GraphInput>,
    pub outputs: Vec<String>,
    pub nodes: Vec<GraphNode>,
    pub initializers: HashMap<String, Tensor>,
}

/// Mutable builder that accumulates initializers and nodes, then produces
/// an `OptimizableGraph` via [`OptimizableGraphBuilder::build`]. The
/// builder preserves insertion order of nodes (the topo-sort pass reorders
/// later if the insertion order isn't already a valid topological order).
#[derive(Clone, Default)]
pub struct OptimizableGraphBuilder {
    inputs: Vec<GraphInput>,
    outputs: Vec<String>,
    nodes: Vec<GraphNode>,
    initializers: HashMap<String, Tensor>,
}

impl OptimizableGraphBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_input(&mut self, name: String, shape: Vec<Option<usize>>) -> &mut Self {
        self.inputs.push(GraphInput { name, shape });
        self
    }

    pub fn add_output(&mut self, name: String) -> &mut Self {
        self.outputs.push(name);
        self
    }

    pub fn add_initializer(&mut self, name: String, tensor: Tensor) -> &mut Self {
        self.initializers.insert(name, tensor);
        self
    }

    pub fn add_node(
        &mut self,
        name: impl Into<String>,
        op_type: impl Into<String>,
        inputs: Vec<String>,
        outputs: Vec<String>,
        attributes: NodeAttributes,
    ) -> &mut Self {
        self.nodes.push(GraphNode {
            name: name.into(),
            op_type: op_type.into(),
            inputs,
            outputs,
            attributes,
        });
        self
    }

    pub fn build(self) -> OptimizableGraph {
        OptimizableGraph {
            inputs: self.inputs,
            outputs: self.outputs,
            nodes: self.nodes,
            initializers: self.initializers,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optimizable_graph_default_is_empty() {
        let g = OptimizableGraph::default();
        assert!(g.inputs.is_empty());
        assert!(g.outputs.is_empty());
        assert!(g.nodes.is_empty());
        assert!(g.initializers.is_empty());
    }

    #[test]
    fn graph_node_stores_fields() {
        let node = GraphNode {
            name: "n1".into(),
            op_type: "Add".into(),
            inputs: vec!["a".into(), "b".into()],
            outputs: vec!["c".into()],
            attributes: NodeAttributes::new(),
        };
        assert_eq!(node.name, "n1");
        assert_eq!(node.inputs.len(), 2);
        assert_eq!(node.outputs.len(), 1);
    }

    #[test]
    fn builder_records_nodes_in_insertion_order() {
        let mut b = OptimizableGraphBuilder::new();
        b.add_node(
            "n1",
            "Add",
            vec!["a".into(), "b".into()],
            vec!["c".into()],
            NodeAttributes::new(),
        );
        b.add_node(
            "n2",
            "Mul",
            vec!["c".into(), "d".into()],
            vec!["e".into()],
            NodeAttributes::new(),
        );
        let g = b.build();
        assert_eq!(g.nodes.len(), 2);
        assert_eq!(g.nodes[0].name, "n1");
        assert_eq!(g.nodes[1].name, "n2");
    }

    #[test]
    fn builder_records_initializers_by_name() {
        let mut b = OptimizableGraphBuilder::new();
        let t = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0], vec![3]);
        b.add_initializer("w".into(), t);
        let g = b.build();
        assert!(g.initializers.contains_key("w"));
    }

    #[test]
    fn builder_records_inputs_and_outputs() {
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("in0".into(), vec![None, Some(4)]);
        b.add_output("out0".into());
        let g = b.build();
        assert_eq!(g.inputs[0].name, "in0");
        assert_eq!(g.outputs, vec!["out0".to_string()]);
    }
}
