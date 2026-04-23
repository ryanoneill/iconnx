//! Precompute slice/reshape/unsqueeze/squeeze parameters from initializer
//! or Constant-node values. Pure function: reads `OptimizableGraph::initializers`
//! and Constant nodes, returns a map of node-name -> `NodePrecomputed`.
//!
//! As of Phase 3 commit #3 this pass also consumes the shape map produced
//! by `tensor_shapes_from_graph`. When the data tensor's inferred shape is
//! **fully known** (non-empty, every dim `Some(_)`), Reshape's shape input
//! is pre-resolved: any `0` placeholder is replaced with the corresponding
//! data-tensor dimension (ONNX "copy from input"), and any single `-1` is
//! replaced with the remaining element count. Any `None` dim or the empty
//! "fully unknown" sentinel causes the pass to fall through to the raw
//! shape vector — the runtime resolver in
//! `cuda/executor/layout.rs::Reshape` handles those cases correctly
//! against actual tensors.
//!
//! This strict "fully known" guard is deliberate: an upstream inferencer
//! that produces a wrong-but-plausible dim (e.g. `Some(512)` at a batch
//! axis where runtime batch=1) would otherwise get silently baked in and
//! override the guaranteed-correct runtime resolution.

use std::collections::HashMap;

use crate::ir::graph::{GraphNode, OptimizableGraph};
use crate::ir::plan::{NodePrecomputed, PrecomputedSlice};
use crate::tensor::Tensor;

pub fn precompute_params(
    graph: &OptimizableGraph,
    tensor_shapes: &HashMap<String, Vec<Option<usize>>>,
) -> HashMap<String, NodePrecomputed> {
    let i64_by_name = collect_i64_values(graph);
    let mut out = HashMap::new();
    for node in &graph.nodes {
        let pre = match node.op_type.as_str() {
            "Slice" => NodePrecomputed {
                slice: precompute_slice(node, &i64_by_name),
                ..Default::default()
            },
            "Reshape" => NodePrecomputed {
                reshape_shape: precompute_reshape_shape(node, &i64_by_name, tensor_shapes),
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
        if let Some(value) = node.attributes.resolve_constant_value() {
            if let Tensor::Int64(_) = &value {
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
    tensor_shapes: &HashMap<String, Vec<Option<usize>>>,
) -> Option<Vec<i64>> {
    // ONNX Reshape: inputs = [data, shape].
    let raw_shape = node.inputs.get(1).and_then(|n| i64_by_name.get(n)).cloned()?;

    // Fast path: no `0` placeholder AND no `-1` → nothing to pre-resolve.
    let has_zero = raw_shape.iter().any(|&d| d == 0);
    let has_neg_one = raw_shape.iter().any(|&d| d == -1);
    if !has_zero && !has_neg_one {
        return Some(raw_shape);
    }

    // The `Shape::new()` sentinel (an empty `Vec`) means "fully unknown"
    // throughout this codebase. It is NOT a known rank-0 tensor and must
    // never contribute a concrete rank or element count — treating it as
    // rank-0 was half of the earlier Kokoro cascade (the fold accumulator
    // started at `1`, turning `-1` into `1` and feeding a downstream Conv
    // a shape with too few elements). See
    // `reshape_neg_one_left_alone_when_data_shape_is_empty_sentinel` below.
    let data_shape = node
        .inputs
        .first()
        .and_then(|name| tensor_shapes.get(name))
        .filter(|s| !s.is_empty());
    let Some(data_shape) = data_shape else {
        return Some(raw_shape);
    };

    // Step 1: resolve `0` placeholders per-dim. ONNX Reshape defines `0`
    // at position `i` as "copy `data_shape[i]`", so we only need that
    // specific dim to be known — not the entire rank. Dims that are
    // `None` stay as `0` and are resolved by the runtime path
    // (`cuda/executor/layout.rs::Reshape`) against the actual tensor.
    //
    // This per-dim strategy is safe BECAUSE every inferencer in
    // `shape_inference` propagates unknown-on-unknown (see `infer_gather`
    // — previously the one outlier that fabricated a concrete dim from an
    // unknown indices input). Trusting a `Some(_)` dim produced by those
    // inferencers cannot introduce a wrong-but-plausible value.
    let mut resolved: Vec<i64> = raw_shape
        .iter()
        .enumerate()
        .map(|(i, &d)| {
            if d == 0 {
                match data_shape.get(i).copied().flatten() {
                    Some(n) => n as i64,
                    // Dim unknown — leave `0` so the runtime resolves it.
                    None => 0,
                }
            } else {
                d
            }
        })
        .collect();

    // Step 2: resolve a single `-1` using the known element count. Unlike
    // step 1, this requires EVERY dim of the data tensor to be known,
    // because the element count is the product of all dims. If any dim
    // is `None` we cannot compute the product and must leave `-1`
    // alone for the runtime to resolve.
    if has_neg_one {
        let all_known: Option<Vec<usize>> =
            data_shape.iter().copied().collect::<Option<Vec<_>>>();
        if let Some(data_dims) = all_known {
            let data_elements: i64 = data_dims.iter().map(|&n| n as i64).product();
            let known_product: Option<i64> = resolved
                .iter()
                .filter(|&&d| d != -1)
                .try_fold(1i64, |acc, &d| if d > 0 { Some(acc * d) } else { None });
            if let Some(known_product) = known_product {
                if known_product > 0 && data_elements % known_product == 0 {
                    let inferred = data_elements / known_product;
                    for d in resolved.iter_mut() {
                        if *d == -1 {
                            *d = inferred;
                        }
                    }
                }
            }
        }
    }

    Some(resolved)
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

    fn empty_shapes() -> HashMap<String, Vec<Option<usize>>> {
        HashMap::new()
    }

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
        let map = precompute_params(&b.build(), &empty_shapes());
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
        let map = precompute_params(&b.build(), &empty_shapes());
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
        let map = precompute_params(&b.build(), &empty_shapes());
        assert!(!map.contains_key("n1"));
    }

    #[test]
    fn reshape_resolves_zero_placeholder_from_inferred_shape() {
        // Shape input has a `0` — should be replaced with data's dim at that index.
        let mut b = OptimizableGraphBuilder::new();
        b.add_initializer(
            "shp".into(),
            Tensor::from_vec_i64(vec![0, 512], vec![2]),
        );
        b.add_node(
            "rs",
            "Reshape",
            vec!["data".into(), "shp".into()],
            vec!["out".into()],
            NodeAttributes::new(),
        );
        let mut shapes = HashMap::new();
        shapes.insert("data".into(), vec![Some(8usize), Some(512)]);
        let map = precompute_params(&b.build(), &shapes);
        assert_eq!(map["rs"].reshape_shape.as_deref(), Some(&[8i64, 512][..]));
    }

    #[test]
    fn reshape_resolves_neg_one_from_inferred_shape() {
        // [-1, 64] against data [4, 8, 16] (total 512) → [-1, 64] → [8, 64].
        let mut b = OptimizableGraphBuilder::new();
        b.add_initializer(
            "shp".into(),
            Tensor::from_vec_i64(vec![-1, 64], vec![2]),
        );
        b.add_node(
            "rs",
            "Reshape",
            vec!["data".into(), "shp".into()],
            vec!["out".into()],
            NodeAttributes::new(),
        );
        let mut shapes = HashMap::new();
        shapes.insert("data".into(), vec![Some(4usize), Some(8), Some(16)]);
        let map = precompute_params(&b.build(), &shapes);
        assert_eq!(map["rs"].reshape_shape.as_deref(), Some(&[8i64, 64][..]));
    }

    #[test]
    fn reshape_neg_one_left_alone_when_data_shape_has_dynamic_dim() {
        // Data shape has a None dim → `-1` can't be resolved, stays `-1`.
        let mut b = OptimizableGraphBuilder::new();
        b.add_initializer(
            "shp".into(),
            Tensor::from_vec_i64(vec![-1, 64], vec![2]),
        );
        b.add_node(
            "rs",
            "Reshape",
            vec!["data".into(), "shp".into()],
            vec!["out".into()],
            NodeAttributes::new(),
        );
        let mut shapes = HashMap::new();
        shapes.insert("data".into(), vec![None, Some(8), Some(16)]);
        let map = precompute_params(&b.build(), &shapes);
        assert_eq!(map["rs"].reshape_shape.as_deref(), Some(&[-1i64, 64][..]));
    }

    #[test]
    fn reshape_zero_left_alone_when_inferred_dim_unknown() {
        // Shape input `[0, 64]`, data shape dim 0 is None → stays 0.
        let mut b = OptimizableGraphBuilder::new();
        b.add_initializer(
            "shp".into(),
            Tensor::from_vec_i64(vec![0, 64], vec![2]),
        );
        b.add_node(
            "rs",
            "Reshape",
            vec!["data".into(), "shp".into()],
            vec!["out".into()],
            NodeAttributes::new(),
        );
        let mut shapes = HashMap::new();
        shapes.insert("data".into(), vec![None, Some(64)]);
        let map = precompute_params(&b.build(), &shapes);
        assert_eq!(map["rs"].reshape_shape.as_deref(), Some(&[0i64, 64][..]));
    }

    #[test]
    fn reshape_resolves_zero_per_dim_when_only_that_dim_is_known() {
        // Shape input `[0, -1, 12, 64]` against partial data shape
        // `[Some(2), None, Some(768)]`. Dim 0 of data is `Some(2)` so the
        // `0` placeholder at position 0 MUST be pre-resolved to `2`.
        // The `-1` stays (we don't know the total element count), and
        // the remaining fixed dims pass through unchanged.
        //
        // This is the Kokoro BERT / LSTM attention Reshape shape: the
        // `0` at the batch axis is statically known from the graph
        // input seed, but the sequence-length axis stays dynamic. The
        // previous "all dims known" guard discarded ALL pre-resolution
        // in this case, leaving a `0` in the precomputed shape that
        // forced a D2H on every call. Per-dim resolution is safe
        // because every shape_inference op now propagates
        // unknown-on-unknown.
        let mut b = OptimizableGraphBuilder::new();
        b.add_initializer(
            "shp".into(),
            Tensor::from_vec_i64(vec![0, -1, 12, 64], vec![4]),
        );
        b.add_node(
            "rs",
            "Reshape",
            vec!["data".into(), "shp".into()],
            vec!["out".into()],
            NodeAttributes::new(),
        );
        let mut shapes = HashMap::new();
        shapes.insert("data".into(), vec![Some(2usize), None, Some(768)]);
        let map = precompute_params(&b.build(), &shapes);
        assert_eq!(
            map["rs"].reshape_shape.as_deref(),
            Some(&[2i64, -1, 12, 64][..])
        );
    }

    #[test]
    fn reshape_neg_one_stays_when_data_shape_is_partial() {
        // `-1` resolution requires ALL dims to be known (we need the
        // product). Partial shape → `-1` must stay `-1`, even if the
        // `0` placeholder is resolvable per-dim.
        let mut b = OptimizableGraphBuilder::new();
        b.add_initializer(
            "shp".into(),
            Tensor::from_vec_i64(vec![-1, 64], vec![2]),
        );
        b.add_node(
            "rs",
            "Reshape",
            vec!["data".into(), "shp".into()],
            vec!["out".into()],
            NodeAttributes::new(),
        );
        let mut shapes = HashMap::new();
        shapes.insert("data".into(), vec![None, Some(8), Some(16)]);
        let map = precompute_params(&b.build(), &shapes);
        assert_eq!(map["rs"].reshape_shape.as_deref(), Some(&[-1i64, 64][..]));
    }

    #[test]
    fn reshape_neg_one_left_alone_when_data_shape_is_empty_sentinel() {
        // Repro for Phase 3 Commit 3 bug: several shape inferencers return
        // `Shape::new()` (an empty Vec) to signal "fully unknown" — e.g.
        // Reshape itself, Slice, an LSTM with a malformed direction attr.
        // Pre-resolution must treat an empty `Shape` the same as a missing
        // entry: it is NOT a valid rank-0 scalar just because the fold
        // accumulator starts at 1. Baking `-1` → `1` here produced the
        // Kokoro cudnn `bad parameter` cascade (a downstream Conv fed a
        // reshape of shape `[1, 1]` from an input with many more elements).
        let mut b = OptimizableGraphBuilder::new();
        b.add_initializer(
            "shp".into(),
            Tensor::from_vec_i64(vec![1, -1], vec![2]),
        );
        b.add_node(
            "rs",
            "Reshape",
            vec!["data".into(), "shp".into()],
            vec!["out".into()],
            NodeAttributes::new(),
        );
        let mut shapes = HashMap::new();
        // Empty Vec = inferencer's "fully unknown" sentinel.
        shapes.insert("data".into(), Vec::new());
        let map = precompute_params(&b.build(), &shapes);
        assert_eq!(map["rs"].reshape_shape.as_deref(), Some(&[1i64, -1][..]));
    }

    #[test]
    fn reshape_zero_left_alone_when_data_shape_is_empty_sentinel() {
        // Same sentinel rule for `0` resolution — an empty Shape from the
        // inferencer must not be treated as a known rank-0 tensor.
        let mut b = OptimizableGraphBuilder::new();
        b.add_initializer(
            "shp".into(),
            Tensor::from_vec_i64(vec![0, 64], vec![2]),
        );
        b.add_node(
            "rs",
            "Reshape",
            vec!["data".into(), "shp".into()],
            vec!["out".into()],
            NodeAttributes::new(),
        );
        let mut shapes = HashMap::new();
        shapes.insert("data".into(), Vec::new());
        let map = precompute_params(&b.build(), &shapes);
        assert_eq!(map["rs"].reshape_shape.as_deref(), Some(&[0i64, 64][..]));
    }
}
