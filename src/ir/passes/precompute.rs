//! Precompute slice/reshape/unsqueeze/squeeze parameters from initializer
//! or Constant-node values. Pure function: reads `OptimizableGraph::initializers`
//! and Constant nodes, returns a map of node-name -> `NodePrecomputed`.

use std::collections::HashMap;

use crate::ir::graph::{GraphNode, OptimizableGraph};
use crate::ir::plan::{NodePrecomputed, PrecomputedSlice};
use crate::tensor::Tensor;

pub fn precompute_params(graph: &OptimizableGraph) -> HashMap<String, NodePrecomputed> {
    let i64_by_name = collect_i64_values(graph);
    let mut out = HashMap::new();
    for node in &graph.nodes {
        let pre = match node.op_type.as_str() {
            "Slice" => NodePrecomputed {
                slice: precompute_slice(node, &i64_by_name),
                ..Default::default()
            },
            "Reshape" => NodePrecomputed {
                reshape_shape: precompute_reshape_shape(node, &i64_by_name),
                ..Default::default()
            },
            "Unsqueeze" => NodePrecomputed {
                unsqueeze_axes: precompute_axes(node, &i64_by_name),
                ..Default::default()
            },
            "Squeeze" => NodePrecomputed {
                squeeze_axes: precompute_axes(node, &i64_by_name),
                ..Default::default()
            },
            _ => continue,
        };
        out.insert(node.name.clone(), pre);
    }
    out
}

/// Build a lookup of all Int64 tensor values available at lowering:
/// initializers + output tensors of Constant nodes whose value attribute
/// is an Int64 tensor.
fn collect_i64_values(graph: &OptimizableGraph) -> HashMap<String, Vec<i64>> {
    let mut out: HashMap<String, Vec<i64>> = HashMap::new();
    for (name, tensor) in &graph.initializers {
        if let Tensor::Int64(_) = tensor {
            out.insert(name.clone(), tensor.as_slice_i64());
        }
    }
    for node in &graph.nodes {
        if node.op_type != "Constant" {
            continue;
        }
        if let Some(value) = node.attributes.get_tensor("value") {
            if let Tensor::Int64(_) = value {
                if let Some(output) = node.outputs.first() {
                    out.insert(output.clone(), value.as_slice_i64());
                }
            }
        }
    }
    out
}

fn precompute_slice(
    node: &GraphNode,
    i64_by_name: &HashMap<String, Vec<i64>>,
) -> Option<PrecomputedSlice> {
    // ONNX Slice: inputs = [data, starts, ends, axes?, steps?]
    if node.inputs.len() < 3 {
        return None;
    }
    let mut pre = PrecomputedSlice::default();
    pre.starts = i64_by_name.get(&node.inputs[1]).cloned();
    pre.ends = i64_by_name.get(&node.inputs[2]).cloned();
    if let Some(ax_name) = node.inputs.get(3) {
        pre.axes = i64_by_name.get(ax_name).cloned();
    }
    if let Some(st_name) = node.inputs.get(4) {
        pre.steps = i64_by_name.get(st_name).cloned();
    }
    if pre.starts.is_none() && pre.ends.is_none() && pre.axes.is_none() && pre.steps.is_none() {
        None
    } else {
        Some(pre)
    }
}

fn precompute_reshape_shape(
    node: &GraphNode,
    i64_by_name: &HashMap<String, Vec<i64>>,
) -> Option<Vec<i64>> {
    // ONNX Reshape: inputs = [data, shape]
    node.inputs.get(1).and_then(|n| i64_by_name.get(n)).cloned()
}

fn precompute_axes(
    node: &GraphNode,
    i64_by_name: &HashMap<String, Vec<i64>>,
) -> Option<Vec<i64>> {
    // ONNX Unsqueeze/Squeeze: inputs = [data, axes]
    node.inputs.get(1).and_then(|n| i64_by_name.get(n)).cloned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attributes::NodeAttributes;
    use crate::ir::graph::OptimizableGraphBuilder;

    #[test]
    fn slice_with_initializer_starts_ends_is_populated() {
        let mut b = OptimizableGraphBuilder::new();
        b.add_initializer("starts".into(), Tensor::from_vec_i64(vec![0, 0], vec![2]));
        b.add_initializer("ends".into(), Tensor::from_vec_i64(vec![4, 8], vec![2]));
        b.add_node(
            "sl",
            "Slice",
            vec!["data".into(), "starts".into(), "ends".into()],
            vec!["out".into()],
            NodeAttributes::new(),
        );
        let map = precompute_params(&b.build());
        let pre = map.get("sl").unwrap();
        assert_eq!(
            pre.slice.as_ref().unwrap().starts.as_deref(),
            Some(&[0i64, 0][..])
        );
        assert_eq!(
            pre.slice.as_ref().unwrap().ends.as_deref(),
            Some(&[4i64, 8][..])
        );
        assert!(pre.slice.as_ref().unwrap().axes.is_none());
    }

    #[test]
    fn reshape_shape_from_constant_node_is_picked_up() {
        let mut b = OptimizableGraphBuilder::new();
        let mut attrs = NodeAttributes::new();
        attrs.add_tensor("value".into(), Tensor::from_vec_i64(vec![-1, 8], vec![2]));
        b.add_node("shape_const", "Constant", vec![], vec!["shp".into()], attrs);
        b.add_node(
            "rs",
            "Reshape",
            vec!["data".into(), "shp".into()],
            vec!["out".into()],
            NodeAttributes::new(),
        );
        let map = precompute_params(&b.build());
        assert_eq!(map["rs"].reshape_shape.as_deref(), Some(&[-1i64, 8][..]));
    }

    #[test]
    fn non_applicable_op_has_no_entry() {
        let mut b = OptimizableGraphBuilder::new();
        b.add_node(
            "n1",
            "Add",
            vec!["a".into(), "b".into()],
            vec!["c".into()],
            NodeAttributes::new(),
        );
        let map = precompute_params(&b.build());
        assert!(!map.contains_key("n1"));
    }
}
