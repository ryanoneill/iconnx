//! Shape-extraction folding.
//!
//! Folds Shape-rooted chains terminating at extraction ops whose
//! extracted indices hit only CONCRETE dims of a (possibly
//! partially-dynamic) Shape input. Designed for graphs where
//! `seq_len`-like symbolic dims propagate through tensor_shapes as
//! `None`, making the all-or-nothing `Shape`-foldability predicate
//! unable to fire (see spec §Amendment history, 2026-04-22).
//!
//! Three supported patterns (exactly these; additions are explicit
//! spec amendments):
//!   1. `Shape(x) -> Gather(indices=I)`
//!   2. `Shape(x) -> Cast(to=Int64) -> Gather(indices=I)`
//!   3. `Shape(x) -> Slice(start=s, end=e)`
//!
//! Pattern 1 is implemented in Task 2.2; patterns 2 and 3 land in
//! Tasks 2.3 and 2.4.
//!
//! Must run AFTER shape_inference has populated tensor_shapes.

use std::cell::Cell;
use std::collections::HashMap;

use crate::ir::graph::{GraphNode, OptimizableGraph};
use crate::ir::passes::folding_core;
use crate::tensor::Tensor;

thread_local! {
    pub(crate) static FOLD_COUNT_SHAPE_GATHER: Cell<usize> = const { Cell::new(0) };
    pub(crate) static FOLD_COUNT_SHAPE_CAST_GATHER: Cell<usize> = const { Cell::new(0) };
    pub(crate) static FOLD_COUNT_SHAPE_SLICE: Cell<usize> = const { Cell::new(0) };
}

pub(crate) fn reset_fold_counters() {
    FOLD_COUNT_SHAPE_GATHER.with(|c| c.set(0));
    FOLD_COUNT_SHAPE_CAST_GATHER.with(|c| c.set(0));
    FOLD_COUNT_SHAPE_SLICE.with(|c| c.set(0));
}

fn bump_counter(cell: &'static std::thread::LocalKey<Cell<usize>>) {
    cell.with(|c| c.set(c.get() + 1));
}

/// Public snapshot of per-pattern fold counts. Produced by
/// [`ShapeExtractionFoldingCounts::snapshot`] so tests and diagnostics can
/// inspect how many extraction chains each pattern resolved in the current
/// thread. Counts are cumulative within the thread until `reset_fold_counters`
/// runs.
#[derive(Clone, Debug, Default)]
pub struct ShapeExtractionFoldingCounts {
    pub shape_gather: usize,
    pub shape_cast_gather: usize,
    pub shape_slice: usize,
    pub total: usize,
}

impl ShapeExtractionFoldingCounts {
    pub fn snapshot() -> Self {
        let g = FOLD_COUNT_SHAPE_GATHER.with(|c| c.get());
        let cg = FOLD_COUNT_SHAPE_CAST_GATHER.with(|c| c.get());
        let s = FOLD_COUNT_SHAPE_SLICE.with(|c| c.get());
        Self {
            shape_gather: g,
            shape_cast_gather: cg,
            shape_slice: s,
            total: g + cg + s,
        }
    }
}

pub fn shape_extraction_folding(
    graph: OptimizableGraph,
    tensor_shapes: &HashMap<String, Vec<Option<usize>>>,
) -> OptimizableGraph {
    // Build a producer-name → GraphNode index so the predicate can walk
    // back from an extraction op to its Shape source without O(n) list
    // traversal per call.
    let producers = build_producer_map(&graph);
    // Clone initializers so the closure can resolve Gather.indices /
    // Slice.starts+ends by name. `Tensor` is internally shallow-cheap
    // (ndarray's ArrayD holds its data in a Vec — the clone isn't free
    // but this runs once per lower()).
    let initializers = graph.initializers.clone();

    let folded = folding_core::fold_with_predicate(graph, move |node, folded| {
        match node.op_type.as_str() {
            "Gather" => {
                try_fold_shape_gather(node, &producers, tensor_shapes, &initializers, folded)
            }
            "Slice" => {
                try_fold_shape_slice(node, &producers, tensor_shapes, &initializers, folded)
            }
            _ => None,
        }
    });

    // Local orphan sweep: remove Shape and Cast nodes whose outputs are
    // no longer referenced by any remaining node or graph output. These
    // become dead once this pass folds an extraction chain that was
    // their sole consumer. A full DCE pass runs later in the pipeline,
    // but doing it here keeps the pass self-contained and guarantees
    // "entire chain collapses" as a user-visible invariant (the spec's
    // Commit #2 behaviour, tested directly in the unit tests).
    prune_orphaned_shape_chain_nodes(folded)
}

/// Remove Shape / Cast nodes whose outputs are not referenced by any
/// remaining node (as an input) and are not listed as graph outputs.
/// Scoped to these two op types so we don't accidentally eliminate work
/// the rest of the pipeline depends on.
fn prune_orphaned_shape_chain_nodes(mut graph: OptimizableGraph) -> OptimizableGraph {
    loop {
        let referenced: std::collections::HashSet<String> = graph
            .nodes
            .iter()
            .flat_map(|n| n.inputs.iter().cloned())
            .chain(graph.outputs.iter().cloned())
            .collect();
        let before = graph.nodes.len();
        graph.nodes.retain(|n| {
            let is_chain_op = matches!(n.op_type.as_str(), "Shape" | "Cast");
            if !is_chain_op {
                return true;
            }
            // Keep if any output of this node is still referenced
            // somewhere reachable.
            n.outputs
                .iter()
                .any(|o| !o.is_empty() && referenced.contains(o))
        });
        if graph.nodes.len() == before {
            break;
        }
    }
    graph
}

/// Maps an output tensor name to the node that produces it.
/// Uses HashMap<String, GraphNode> because GraphNode is Clone-cheap
/// and the predicate borrow is local.
fn build_producer_map(graph: &OptimizableGraph) -> HashMap<String, GraphNode> {
    let mut map = HashMap::new();
    for node in graph.nodes.iter() {
        for out_name in node.outputs.iter() {
            if !out_name.is_empty() {
                map.insert(out_name.clone(), node.clone());
            }
        }
    }
    map
}

/// Fold `Shape(x) → Gather(indices=I)` when every index in I is a
/// concrete dim of x.
fn try_fold_shape_gather(
    gather_node: &GraphNode,
    producers: &HashMap<String, GraphNode>,
    tensor_shapes: &HashMap<String, Vec<Option<usize>>>,
    initializers: &HashMap<String, Tensor>,
    folded: &HashMap<String, Tensor>,
) -> Option<Tensor> {
    // Gather has two (sometimes three) inputs: data (input 0) and
    // indices (input 1). For this pattern to fire, input 0 must be
    // produced by a Shape op — optionally through a Cast(to=Int64)
    // which is a no-op transform on Shape's native Int64 output.
    let data_name = gather_node.inputs.first()?;
    let mut producer = producers.get(data_name)?;
    let mut walked_cast = false;

    // Pattern 2: Shape → Cast(Int64) → Gather. Walk past the Cast if
    // present and target dtype is Int64 (ONNX DataType 7).
    if producer.op_type == "Cast" {
        let cast_to = producer.attributes.get_int("to")?;
        if cast_to != 7 {
            return None;
        }
        let cast_input = producer.inputs.first()?;
        producer = producers.get(cast_input)?;
        walked_cast = true;
    }

    if producer.op_type != "Shape" {
        return None;
    }

    // The Shape op's input is the tensor whose rank + (partial) shape
    // we inspect.
    let shape_input_name = producer.inputs.first()?;
    let inferred = tensor_shapes.get(shape_input_name)?;

    // Reject the empty-Vec "fully unknown" sentinel — no dims known
    // (mirrors the invariant preserved from commit 8350f67: an empty
    // inferred shape would synthesize an Int64 tensor of shape [0] and
    // downstream folds would trip out-of-bounds checks).
    if inferred.is_empty() {
        return None;
    }

    // Gather.indices: must be an initializer (directly or via earlier
    // fold in this pass). Runtime-produced indices aren't fold-able.
    let indices_name = gather_node.inputs.get(1)?;
    let indices_tensor = initializers
        .get(indices_name)
        .or_else(|| folded.get(indices_name))?;
    if !indices_tensor.is_int64() {
        // Gather.indices is ONNX-typed Int64 in every real graph; reject
        // anything else as out-of-scope rather than risking a
        // conversion-related miscompile.
        return None;
    }
    let indices = indices_tensor.to_array_i64();
    let indices_slice = indices.as_slice()?;

    // Read Gather axis attribute. For this pattern we require axis 0
    // (Shape output is rank-1, so only axis 0 is valid). Reject
    // anything else as out-of-scope.
    let axis = gather_node.attributes.get_int("axis").unwrap_or(0);
    if axis != 0 {
        return None;
    }

    // Resolve each index against inferred shape. Every referenced dim
    // must be Some(_); out-of-bounds or None dim → bail.
    let rank = inferred.len() as i64;
    let mut output: Vec<i64> = Vec::with_capacity(indices_slice.len());
    for &idx in indices_slice {
        // ONNX allows negative indices (wraps around rank).
        let normalized = if idx < 0 { idx + rank } else { idx };
        if normalized < 0 || normalized as usize >= inferred.len() {
            return None;
        }
        let dim = inferred[normalized as usize]?; // None → bail
        output.push(dim as i64);
    }

    // Output shape: ONNX Gather output rank = data rank + indices rank - 1.
    // Shape output is rank 1, so gather output shape is just the indices
    // shape (rank of data − gather axis = 0 remaining, indices rank kept).
    let output_shape = indices_tensor.shape().to_vec();
    if walked_cast {
        bump_counter(&FOLD_COUNT_SHAPE_CAST_GATHER);
    } else {
        bump_counter(&FOLD_COUNT_SHAPE_GATHER);
    }
    Some(Tensor::from_vec_i64(output, output_shape))
}

/// Fold `Shape(x) → Slice(start=s, end=e, (axes)=[0], (steps)=[1])` when
/// every dim in the `[start, end)` range is a concrete dim of x.
fn try_fold_shape_slice(
    slice_node: &GraphNode,
    producers: &HashMap<String, GraphNode>,
    tensor_shapes: &HashMap<String, Vec<Option<usize>>>,
    initializers: &HashMap<String, Tensor>,
    folded: &HashMap<String, Tensor>,
) -> Option<Tensor> {
    // ONNX Slice inputs: [data, starts, ends, (axes), (steps)].
    let data_name = slice_node.inputs.first()?;
    let producer = producers.get(data_name)?;
    if producer.op_type != "Shape" {
        return None;
    }
    let shape_input_name = producer.inputs.first()?;
    let inferred = tensor_shapes.get(shape_input_name)?;
    if inferred.is_empty() {
        return None;
    }

    // Resolve starts, ends as Int64 initializers / earlier folds.
    let starts_name = slice_node.inputs.get(1)?;
    let ends_name = slice_node.inputs.get(2)?;
    let starts = initializers
        .get(starts_name)
        .or_else(|| folded.get(starts_name))?;
    let ends = initializers
        .get(ends_name)
        .or_else(|| folded.get(ends_name))?;
    if !starts.is_int64() || !ends.is_int64() {
        return None;
    }
    let starts_arr = starts.to_array_i64();
    let ends_arr = ends.to_array_i64();
    let start = *starts_arr.as_slice()?.first()?;
    let end = *ends_arr.as_slice()?.first()?;

    // Optional axes input — if provided and non-empty, must be [0]
    // (Shape output is rank-1, only axis 0 is meaningful).
    if let Some(axes_name) = slice_node.inputs.get(3) {
        if !axes_name.is_empty() {
            let axes = initializers
                .get(axes_name)
                .or_else(|| folded.get(axes_name))?;
            if !axes.is_int64() {
                return None;
            }
            let axes_arr = axes.to_array_i64();
            let axis = *axes_arr.as_slice()?.first()?;
            if axis != 0 {
                return None;
            }
        }
    }
    // Optional steps input — if provided, must be [1].
    if let Some(steps_name) = slice_node.inputs.get(4) {
        if !steps_name.is_empty() {
            let steps = initializers
                .get(steps_name)
                .or_else(|| folded.get(steps_name))?;
            if !steps.is_int64() {
                return None;
            }
            let steps_arr = steps.to_array_i64();
            let step = *steps_arr.as_slice()?.first()?;
            if step != 1 {
                return None;
            }
        }
    }

    // Clamp start/end to [0, rank] per ONNX Slice semantics.
    let rank = inferred.len() as i64;
    let start_n = if start < 0 {
        (start + rank).max(0)
    } else {
        start.min(rank)
    };
    let end_n = if end < 0 {
        (end + rank).max(0)
    } else {
        end.min(rank)
    };
    if start_n >= end_n {
        return None;
    }

    // Every dim in [start_n, end_n) must be Some(_).
    let mut output: Vec<i64> = Vec::with_capacity((end_n - start_n) as usize);
    for i in start_n..end_n {
        let dim = inferred[i as usize]?;
        output.push(dim as i64);
    }

    bump_counter(&FOLD_COUNT_SHAPE_SLICE);
    let len = output.len();
    Some(Tensor::from_vec_i64(output, vec![len]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attributes::NodeAttributes;
    use crate::ir::graph::OptimizableGraphBuilder;

    fn shapes_map(entries: &[(&str, Vec<Option<usize>>)]) -> HashMap<String, Vec<Option<usize>>> {
        entries
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    #[test]
    fn gather_of_shape_with_all_concrete_indices_folds() {
        // Shape(x) where x: [Some(1), None, Some(512)].
        // Gather([0, 2]) extracts dim 0 (=1) and dim 2 (=512).
        // Both concrete → fold to initializer [1, 512].
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("x".to_string(), vec![Some(1), None, Some(512)]);
        b.add_initializer(
            "indices".to_string(),
            Tensor::from_vec_i64(vec![0, 2], vec![2]),
        );
        b.add_node(
            "shape_of_x",
            "Shape",
            vec!["x".to_string()],
            vec!["shape_out".to_string()],
            NodeAttributes::new(),
        );
        let mut gather_attrs = NodeAttributes::new();
        gather_attrs.add_int("axis".to_string(), 0);
        b.add_node(
            "gather_dims",
            "Gather",
            vec!["shape_out".to_string(), "indices".to_string()],
            vec!["dims".to_string()],
            gather_attrs,
        );
        let graph = b.build();

        let shapes = shapes_map(&[("x", vec![Some(1), None, Some(512)])]);
        let folded = shape_extraction_folding(graph, &shapes);

        // Both Shape and Gather should be folded away.
        assert!(
            folded
                .nodes
                .iter()
                .all(|n| n.op_type != "Gather" && n.op_type != "Shape")
        );
        let dims = folded
            .initializers
            .get("dims")
            .expect("dims synthesized");
        assert_eq!(dims.to_array_i64().as_slice().unwrap(), &[1i64, 512]);
    }

    #[test]
    fn gather_of_shape_with_none_at_extracted_index_does_not_fold() {
        // Gather([1]) targets the None dim — must not fold.
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("x".to_string(), vec![Some(1), None, Some(512)]);
        b.add_initializer(
            "indices".to_string(),
            Tensor::from_vec_i64(vec![1], vec![1]),
        );
        b.add_node(
            "shape_of_x",
            "Shape",
            vec!["x".to_string()],
            vec!["shape_out".to_string()],
            NodeAttributes::new(),
        );
        let mut gather_attrs = NodeAttributes::new();
        gather_attrs.add_int("axis".to_string(), 0);
        b.add_node(
            "gather_dims",
            "Gather",
            vec!["shape_out".to_string(), "indices".to_string()],
            vec!["dims".to_string()],
            gather_attrs,
        );
        let graph = b.build();

        let shapes = shapes_map(&[("x", vec![Some(1), None, Some(512)])]);
        let folded = shape_extraction_folding(graph, &shapes);

        // Neither folded.
        assert!(folded.nodes.iter().any(|n| n.op_type == "Shape"));
        assert!(folded.nodes.iter().any(|n| n.op_type == "Gather"));
        assert!(!folded.initializers.contains_key("dims"));
    }

    #[test]
    fn gather_on_non_shape_input_does_not_fold() {
        // Gather's data input is an initializer, NOT a Shape output.
        // Our predicate must decline (over-matching guard). Note that
        // folding_core's standard all-inputs-constant path may still fold
        // it — this test asserts we don't crash on non-Shape-rooted
        // Gather, not that it necessarily stays unfolded.
        let mut b = OptimizableGraphBuilder::new();
        b.add_initializer(
            "data".to_string(),
            Tensor::from_vec_i64(vec![10, 20, 30], vec![3]),
        );
        b.add_initializer(
            "indices".to_string(),
            Tensor::from_vec_i64(vec![0], vec![1]),
        );
        let mut gather_attrs = NodeAttributes::new();
        gather_attrs.add_int("axis".to_string(), 0);
        b.add_node(
            "gather",
            "Gather",
            vec!["data".to_string(), "indices".to_string()],
            vec!["out".to_string()],
            gather_attrs,
        );
        let graph = b.build();

        let shapes = shapes_map(&[]);
        let folded = shape_extraction_folding(graph, &shapes);

        // Smoke: no crash; our predicate did not fire (standard path
        // may have — either outcome is correct behavior).
        let _ = folded;
    }

    #[test]
    fn out_of_bounds_gather_index_does_not_fold() {
        // Gather([5]) targets dim 5, but x has only 3 dims. Must
        // return None from the predicate, not panic.
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("x".to_string(), vec![Some(1), Some(2), Some(3)]);
        b.add_initializer(
            "indices".to_string(),
            Tensor::from_vec_i64(vec![5], vec![1]),
        );
        b.add_node(
            "shape_of_x",
            "Shape",
            vec!["x".to_string()],
            vec!["shape_out".to_string()],
            NodeAttributes::new(),
        );
        let mut gather_attrs = NodeAttributes::new();
        gather_attrs.add_int("axis".to_string(), 0);
        b.add_node(
            "gather_oob",
            "Gather",
            vec!["shape_out".to_string(), "indices".to_string()],
            vec!["dims".to_string()],
            gather_attrs,
        );
        let graph = b.build();

        let shapes = shapes_map(&[("x", vec![Some(1), Some(2), Some(3)])]);
        let folded = shape_extraction_folding(graph, &shapes);

        // Must not panic; must not produce a (wrong) initializer.
        assert!(!folded.initializers.contains_key("dims"));
    }

    #[test]
    fn negative_gather_index_resolves_via_rank_wrap() {
        // Gather([-1]) should extract the LAST dim.
        // x: [Some(1), None, Some(512)]. rank=3. idx=-1 → 2 → 512.
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("x".to_string(), vec![Some(1), None, Some(512)]);
        b.add_initializer(
            "indices".to_string(),
            Tensor::from_vec_i64(vec![-1], vec![1]),
        );
        b.add_node(
            "shape_of_x",
            "Shape",
            vec!["x".to_string()],
            vec!["shape_out".to_string()],
            NodeAttributes::new(),
        );
        let mut gather_attrs = NodeAttributes::new();
        gather_attrs.add_int("axis".to_string(), 0);
        b.add_node(
            "gather_last",
            "Gather",
            vec!["shape_out".to_string(), "indices".to_string()],
            vec!["last_dim".to_string()],
            gather_attrs,
        );
        let graph = b.build();

        let shapes = shapes_map(&[("x", vec![Some(1), None, Some(512)])]);
        let folded = shape_extraction_folding(graph, &shapes);

        let last_dim = folded
            .initializers
            .get("last_dim")
            .expect("last_dim synthesized");
        assert_eq!(last_dim.to_array_i64().as_slice().unwrap(), &[512i64]);
    }

    #[test]
    fn cast_in_between_shape_and_gather_is_transparent() {
        // Shape(x) → Cast(Int64) → Gather([0])
        // x: [Some(1), None, Some(512)]. Expect: out = [1].
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("x".to_string(), vec![Some(1), None, Some(512)]);
        b.add_initializer(
            "indices".to_string(),
            Tensor::from_vec_i64(vec![0], vec![1]),
        );
        b.add_node(
            "shape_of_x",
            "Shape",
            vec!["x".to_string()],
            vec!["shape_out".to_string()],
            NodeAttributes::new(),
        );
        let mut cast_attrs = NodeAttributes::new();
        cast_attrs.add_int("to".to_string(), 7); // ONNX DataType 7 = INT64
        b.add_node(
            "cast_shape",
            "Cast",
            vec!["shape_out".to_string()],
            vec!["casted".to_string()],
            cast_attrs,
        );
        let mut gather_attrs = NodeAttributes::new();
        gather_attrs.add_int("axis".to_string(), 0);
        b.add_node(
            "gather_first",
            "Gather",
            vec!["casted".to_string(), "indices".to_string()],
            vec!["first_dim".to_string()],
            gather_attrs,
        );
        let graph = b.build();

        let shapes = shapes_map(&[("x", vec![Some(1), None, Some(512)])]);
        let folded = shape_extraction_folding(graph, &shapes);

        // All three chain nodes (Shape, Cast, Gather) should be folded away.
        assert!(folded.nodes.is_empty(), "all three chain nodes should fold");
        let first_dim = folded
            .initializers
            .get("first_dim")
            .expect("first_dim synthesized");
        assert_eq!(first_dim.to_array_i64().as_slice().unwrap(), &[1i64]);
    }

    #[test]
    fn cast_to_non_int64_does_not_fold() {
        // Cast(to=Float32) is NOT transparent for Shape's Int64 output —
        // the value semantics change, so we must not fold.
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("x".to_string(), vec![Some(1), Some(2), Some(3)]);
        b.add_initializer(
            "indices".to_string(),
            Tensor::from_vec_i64(vec![0], vec![1]),
        );
        b.add_node(
            "shape_of_x",
            "Shape",
            vec!["x".to_string()],
            vec!["shape_out".to_string()],
            NodeAttributes::new(),
        );
        let mut cast_attrs = NodeAttributes::new();
        cast_attrs.add_int("to".to_string(), 1); // ONNX DataType 1 = FLOAT
        b.add_node(
            "cast_shape",
            "Cast",
            vec!["shape_out".to_string()],
            vec!["casted".to_string()],
            cast_attrs,
        );
        let mut gather_attrs = NodeAttributes::new();
        gather_attrs.add_int("axis".to_string(), 0);
        b.add_node(
            "gather",
            "Gather",
            vec!["casted".to_string(), "indices".to_string()],
            vec!["out".to_string()],
            gather_attrs,
        );
        let graph = b.build();

        let shapes = shapes_map(&[("x", vec![Some(1), Some(2), Some(3)])]);
        let folded = shape_extraction_folding(graph, &shapes);

        // Cast-to-Float blocks the fold. Cast (non-Int64) stays.
        assert!(folded.nodes.iter().any(|n| n.op_type == "Cast"));
    }

    #[test]
    fn slice_of_shape_with_all_concrete_range_folds() {
        // Shape(x) → Slice(start=0, end=2). x: [Some(4), Some(8), None].
        // Dims [0..2) = [4, 8] — both concrete, fold.
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("x".to_string(), vec![Some(4), Some(8), None]);
        b.add_initializer(
            "starts".to_string(),
            Tensor::from_vec_i64(vec![0], vec![1]),
        );
        b.add_initializer(
            "ends".to_string(),
            Tensor::from_vec_i64(vec![2], vec![1]),
        );
        b.add_node(
            "shape_of_x",
            "Shape",
            vec!["x".to_string()],
            vec!["shape_out".to_string()],
            NodeAttributes::new(),
        );
        b.add_node(
            "slice",
            "Slice",
            vec!["shape_out".to_string(), "starts".to_string(), "ends".to_string()],
            vec!["prefix".to_string()],
            NodeAttributes::new(),
        );
        let graph = b.build();

        let shapes = shapes_map(&[("x", vec![Some(4), Some(8), None])]);
        let folded = shape_extraction_folding(graph, &shapes);

        let prefix = folded
            .initializers
            .get("prefix")
            .expect("prefix synthesized");
        assert_eq!(prefix.to_array_i64().as_slice().unwrap(), &[4i64, 8]);
    }

    #[test]
    fn slice_of_shape_with_none_in_range_does_not_fold() {
        // Shape(x) → Slice(start=1, end=3). x: [Some(4), Some(8), None].
        // Dim 2 is None, in range → must NOT fold.
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("x".to_string(), vec![Some(4), Some(8), None]);
        b.add_initializer(
            "starts".to_string(),
            Tensor::from_vec_i64(vec![1], vec![1]),
        );
        b.add_initializer(
            "ends".to_string(),
            Tensor::from_vec_i64(vec![3], vec![1]),
        );
        b.add_node(
            "shape_of_x",
            "Shape",
            vec!["x".to_string()],
            vec!["shape_out".to_string()],
            NodeAttributes::new(),
        );
        b.add_node(
            "slice",
            "Slice",
            vec!["shape_out".to_string(), "starts".to_string(), "ends".to_string()],
            vec!["suffix".to_string()],
            NodeAttributes::new(),
        );
        let graph = b.build();

        let shapes = shapes_map(&[("x", vec![Some(4), Some(8), None])]);
        let folded = shape_extraction_folding(graph, &shapes);

        assert!(folded.nodes.iter().any(|n| n.op_type == "Slice"));
        assert!(!folded.initializers.contains_key("suffix"));
    }

    #[test]
    fn standalone_shape_without_extraction_consumer_does_not_fold() {
        // Shape(x) whose output feeds no Gather/Slice — must NOT fold.
        // The output is also registered as a graph output, so the orphan
        // sweep keeps the Shape node alive.
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("x".to_string(), vec![Some(1), Some(2)]);
        b.add_node(
            "shape_of_x",
            "Shape",
            vec!["x".to_string()],
            vec!["shape_out".to_string()],
            NodeAttributes::new(),
        );
        b.add_output("shape_out".to_string());
        let graph = b.build();

        let shapes = shapes_map(&[("x", vec![Some(1), Some(2)])]);
        let folded = shape_extraction_folding(graph, &shapes);

        // Shape stays; no synthesized initializer.
        assert!(folded.nodes.iter().any(|n| n.op_type == "Shape"));
        assert!(!folded.initializers.contains_key("shape_out"));
    }

    #[test]
    fn shape_extraction_folding_is_idempotent() {
        let build_graph = || {
            let mut b = OptimizableGraphBuilder::new();
            b.add_input("x".to_string(), vec![Some(1), None, Some(512)]);
            b.add_initializer(
                "indices".to_string(),
                Tensor::from_vec_i64(vec![0], vec![1]),
            );
            b.add_node(
                "shape_of_x",
                "Shape",
                vec!["x".to_string()],
                vec!["shape_out".to_string()],
                NodeAttributes::new(),
            );
            let mut gather_attrs = NodeAttributes::new();
            gather_attrs.add_int("axis".to_string(), 0);
            b.add_node(
                "gather_first",
                "Gather",
                vec!["shape_out".to_string(), "indices".to_string()],
                vec!["first".to_string()],
                gather_attrs,
            );
            b.build()
        };
        let shapes = shapes_map(&[("x", vec![Some(1), None, Some(512)])]);

        let once = shape_extraction_folding(build_graph(), &shapes);
        let twice = shape_extraction_folding(build_graph(), &shapes);
        let thrice = shape_extraction_folding(once, &shapes);

        assert_eq!(twice.nodes.len(), 0);
        assert_eq!(thrice.nodes.len(), 0);
        assert_eq!(
            twice
                .initializers
                .get("first")
                .map(|t| t.to_array_i64().as_slice().unwrap().to_vec()),
            thrice
                .initializers
                .get("first")
                .map(|t| t.to_array_i64().as_slice().unwrap().to_vec())
        );
    }
}
