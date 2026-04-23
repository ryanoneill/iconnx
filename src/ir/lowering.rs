//! Lower an `OptimizableGraph` into an `ExecutionPlan`.
//!
//! Orchestrates the passes (topo sort -> precompute -> fusion) on CPU,
//! uploads initializers to GPU, eagerly packs LSTM weights, and assembles
//! the final `Vec<PlannedOp>` with each op's `OpKind` + `NodePrecomputed`
//! attached.

use std::cell::RefCell;
use std::collections::HashMap;

use crate::cuda::cudnn::pack_lstm_weights_for_cudnn;
use crate::cuda::{CudaError, GpuTensor, IconnxCudaContext};
use crate::ir::graph::OptimizableGraph;
use crate::ir::passes::{
    constant_folding_pass, dead_code_elimination, detect_fusion,
    elementwise_fusion_pass, fusion_annotations_from_graph, precompute_params,
    shape_extraction_folding, shape_inference, tensor_shapes_from_graph, topo_sort,
    FusionAnnotations,
};
use crate::ir::plan::{
    ExecutionPlan, LstmPlanCache, LstmWeightCache, OpKind, PlannedOp,
};
use crate::tensor::Tensor;

/// Lower an `OptimizableGraph` to an `ExecutionPlan`.
///
/// Runs the topo-sort, precompute, and fusion-detection passes; uploads
/// initializers to the GPU; eagerly packs LSTM weights; and builds the
/// final `Vec<PlannedOp>`. The resulting plan is model-specific and
/// self-contained: any compatible `Executor` can run it directly.
pub fn lower(
    graph: OptimizableGraph,
    ctx: &IconnxCudaContext,
) -> Result<ExecutionPlan, CudaError> {
    // 1. Topological sort.
    let graph = topo_sort(graph)?;

    // 2. Constant folding — evaluate all-static subgraphs on CPU and promote
    // their results to initializers. Runs BEFORE DCE so the DCE sweep can
    // reclaim any ops that became orphaned by folding.
    let graph = constant_folding_pass(graph);

    // 3. DCE sweep — remove ops orphaned by constant folding.
    let graph = dead_code_elimination(graph);

    // 4. Shape inference — populate inferred shapes for downstream passes.
    //
    // The `shape_inference` call is an intentional identity pass: it exists
    // purely to keep the pipeline uniform (every step has the
    // `fn(OptimizableGraph) -> OptimizableGraph` signature). It is NOT a
    // placeholder or TODO. The actual shape map is computed by
    // `tensor_shapes_from_graph`, which returns a `HashMap<String, Shape>`
    // that gets attached to `ExecutionPlan::tensor_shapes` below — a
    // separate output from the graph itself. See the docs on
    // `shape_inference` for the full rationale.
    //
    // 4b. Shape-extraction folding (Phase 4 B) — collapse Shape-rooted
    // chains terminating at Gather/Slice (and optionally through a
    // Cast→Int64) when the requested dims are all concrete. Must run
    // AFTER `tensor_shapes_from_graph` so the predicate sees inferred
    // shapes. Collapsed chain outputs are promoted to initializers and
    // orphaned Shape/Cast nodes are removed in a local sweep inside the
    // pass; the DCE step at the end of this block picks up any downstream
    // consumers that became unreferenced.
    //
    // The shape-inference <-> shape-extraction-folding interaction is
    // iterated in a bounded fixpoint per spec §Amendment history
    // ("2026-04-22 (fourth refinement)"): a fold that synthesizes an
    // initializer for e.g. `Reshape.shape` can unlock further shape
    // refinements, which in turn unlock further folds. The loop is
    // capped at `SHAPE_FOLDING_FIXPOINT_MAX_ITERATIONS`; it exits early
    // the first iteration that adds zero initializers AND refines zero
    // `None` dims to `Some`. Only these two passes are iterated —
    // constant_folding / DCE / precompute_params / elementwise_fusion
    // stay strictly one-shot.
    const SHAPE_FOLDING_FIXPOINT_MAX_ITERATIONS: usize = 3;

    let mut graph = shape_inference(graph);
    let mut tensor_shapes = tensor_shapes_from_graph(&graph);

    for _iter in 0..SHAPE_FOLDING_FIXPOINT_MAX_ITERATIONS {
        let initializer_count_before = graph.initializers.len();
        let none_dim_count_before = count_none_dims(&tensor_shapes);

        graph = shape_extraction_folding(graph, &tensor_shapes);
        // Re-run shape_inference (identity today) + tensor_shapes_from_graph
        // to pick up synthesized initializers from the fold.
        graph = shape_inference(graph);
        tensor_shapes = tensor_shapes_from_graph(&graph);

        let initializer_count_after = graph.initializers.len();
        let none_dim_count_after = count_none_dims(&tensor_shapes);

        let added_inits = initializer_count_after.saturating_sub(initializer_count_before);
        let resolved_dims = none_dim_count_before.saturating_sub(none_dim_count_after);

        if added_inits == 0 && resolved_dims == 0 {
            // Strict progress-assertion early-break: no new initializers AND
            // no None->Some refinements means no further progress is possible.
            break;
        }
    }

    // Keep the DCE invariant — cheap, and paves the way for future passes.
    let graph = dead_code_elimination(graph);

    // 5. General elementwise fusion — detects arbitrary chains and
    // emits `FusedPattern::GeneralChain` for ones that don't match the
    // 8 specialized variants. Pure-CPU, no GPU state touched.
    let graph = elementwise_fusion_pass(graph);
    let fusion_annotations = fusion_annotations_from_graph(&graph);

    // 6. DCE again in case elementwise_fusion exposed dead ops (it's a
    // no-op today since the pass is annotation-only, but keeping the
    // sweep here future-proofs the pipeline: any later change that
    // actually rewrites nodes will benefit without a separate touch).
    let graph = dead_code_elimination(graph);

    // 7. Precompute params (pure, no GPU) — now uses inferred shapes to
    // resolve Reshape `0`/`-1` placeholders when possible.
    let precomputed_by_name = precompute_params(&graph, &tensor_shapes);

    // 8. Detect fusion (pure, no GPU) using the legacy detector, then
    // merge with the new elementwise_fusion annotations. Primary wins
    // on head-name conflict; the legacy pass fills in dynamic
    // candidates and any head not claimed by the new pass.
    let legacy_fusion = detect_fusion(&graph);
    let fusion = merge_fusion_annotations(fusion_annotations, legacy_fusion);

    // 7. Upload initializers to GPU.
    let weights = upload_initializers(&graph, ctx)?;

    // 8. Build static_i64 for fast lookup at runtime.
    let static_i64 = collect_static_i64(&graph);

    // 9. Pack LSTM weights eagerly.
    let lstm_weights = pack_lstm_weights_eagerly(&graph, &weights, ctx)?;

    // 10. Build PlannedOp sequence from sorted nodes + fusion + precomputed.
    let ops: Vec<PlannedOp> = graph
        .nodes
        .iter()
        .map(|node| {
            let kind = if fusion.skipped.contains(&node.name) {
                OpKind::Skip
            } else if let Some(head) = fusion.heads.get(&node.name) {
                OpKind::FusedHead(head.pattern.clone())
            } else {
                OpKind::Regular
            };
            let precomputed = precomputed_by_name
                .get(&node.name)
                .cloned()
                .unwrap_or_default();
            PlannedOp {
                node: node.clone(),
                kind,
                precomputed,
            }
        })
        .collect();

    // 11. Compute tensor_last_use for memory-pool return timing.
    let tensor_last_use = compute_tensor_last_use(&ops, &graph.outputs, &weights);

    Ok(ExecutionPlan {
        ops,
        weights,
        static_i64,
        lstm_weights,
        lstm_plan_cache: RefCell::new(LstmPlanCache::new()),
        dynamic_candidates: fusion.dynamic_candidates,
        tensor_last_use,
        graph_inputs: graph.inputs,
        graph_outputs: graph.outputs,
        tensor_shapes,
    })
}

/// Merge two `FusionAnnotations` instances. The primary (from the new
/// `elementwise_fusion` pass) takes precedence on head-name conflict;
/// the secondary (from the legacy `detect_fusion`) fills in any head
/// the primary didn't claim, plus contributes all dynamic candidates.
///
/// Skipped-node sets are unioned (safe: if a node is skipped by either
/// pass, its execution is redundant under the other's head). Dynamic
/// candidates are unioned with primary wins — primary currently emits
/// no dynamic candidates, so this is a simple merge today.
fn merge_fusion_annotations(
    primary: FusionAnnotations,
    secondary: FusionAnnotations,
) -> FusionAnnotations {
    let mut merged = primary;
    for (head, info) in secondary.heads {
        merged.heads.entry(head).or_insert(info);
    }
    merged.skipped.extend(secondary.skipped);
    for (name, info) in secondary.dynamic_candidates {
        merged.dynamic_candidates.entry(name).or_insert(info);
    }
    merged
}

/// Upload every initializer tensor to the GPU, mirroring today's
/// `GpuGraphExecutor::add_initializer` behaviour: Float32, Int64, and
/// Int32 are uploaded; all other dtypes (Float64, Bool) are skipped so
/// the downstream dispatch raises a runtime error that matches the
/// existing executor.
fn upload_initializers(
    graph: &OptimizableGraph,
    ctx: &IconnxCudaContext,
) -> Result<HashMap<String, GpuTensor>, CudaError> {
    let mut out = HashMap::new();
    for (name, tensor) in &graph.initializers {
        let gpu = match tensor {
            Tensor::Float32(_) => {
                GpuTensor::from_host_f32(ctx, &tensor.as_slice(), tensor.shape().to_vec())?
            }
            Tensor::Int64(_) => {
                GpuTensor::from_host_i64(ctx, &tensor.as_slice_i64(), tensor.shape().to_vec())?
            }
            Tensor::Int32(_) => {
                GpuTensor::from_host_i32(ctx, &tensor.as_slice_i32(), tensor.shape().to_vec())?
            }
            // Matches today's GpuGraphExecutor::add_initializer: any other
            // dtype is simply not added to the weights map.
            _ => continue,
        };
        out.insert(name.clone(), gpu);
    }
    Ok(out)
}

/// Collect every statically-known Int64 vector in the graph: Int64
/// initializers plus the output of every `Constant` node whose `value`
/// attribute is an Int64 tensor. Used at runtime by ops that read Int64
/// shape/axes/pad/indices inputs without round-tripping through a GPU
/// D2H copy.
fn collect_static_i64(graph: &OptimizableGraph) -> HashMap<String, Vec<i64>> {
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

/// For each LSTM node, pack its (W, R, B) weights into cuDNN's packed
/// weight buffer once at lowering time, keyed on `(W_ptr, R_ptr)` so two
/// LSTM nodes that share weights share a packed buffer. Mirrors today's
/// on-demand packing path in `src/cuda/inference/mod.rs` (around line
/// 1743) but runs eagerly so inference-time dispatch is allocation-free.
fn pack_lstm_weights_eagerly(
    graph: &OptimizableGraph,
    weights: &HashMap<String, GpuTensor>,
    ctx: &IconnxCudaContext,
) -> Result<LstmWeightCache, CudaError> {
    let mut out = LstmWeightCache::new();
    for node in &graph.nodes {
        if node.op_type != "LSTM" {
            continue;
        }

        // ONNX LSTM input order: X, W, R, B?, sequence_lens?, initial_h?, initial_c?, P?
        let w_name = node.inputs.get(1).ok_or_else(|| {
            CudaError::Kernel(format!("LSTM '{}' missing W input", node.name))
        })?;
        let r_name = node.inputs.get(2).ok_or_else(|| {
            CudaError::Kernel(format!("LSTM '{}' missing R input", node.name))
        })?;

        let w = weights.get(w_name).ok_or_else(|| {
            CudaError::Kernel(format!(
                "LSTM '{}' W initializer '{}' not found",
                node.name, w_name
            ))
        })?;
        let r = weights.get(r_name).ok_or_else(|| {
            CudaError::Kernel(format!(
                "LSTM '{}' R initializer '{}' not found",
                node.name, r_name
            ))
        })?;
        // B is optional; empty string in inputs means "absent" per ONNX.
        let b = node
            .inputs
            .get(3)
            .and_then(|n| if n.is_empty() { None } else { weights.get(n) });

        let key = (
            w.device_ptr(ctx) as usize,
            r.device_ptr(ctx) as usize,
        );
        if out.contains_key(&key) {
            // Shared weights across multiple LSTM nodes — pack once.
            continue;
        }

        let w_shape = w.shape();
        let num_directions = w_shape[0];
        let hidden_size = w_shape[1] / 4;
        let input_size = w_shape[2];

        let packed = pack_lstm_weights_for_cudnn(
            ctx,
            w,
            r,
            b,
            hidden_size,
            input_size,
            num_directions,
        )?;
        out.insert(key, packed);
    }
    Ok(out)
}

/// Compute the "last use" index for every non-weight tensor referenced
/// in the op sequence. Mirrors the logic today in
/// `src/cuda/inference/mod.rs::compute_tensor_last_use` (around line
/// 1063): for each op, record the op index as the last use of every
/// non-empty, non-weight input; graph outputs are pinned to `ops.len()`
/// so the memory pool never reclaims them early.
fn compute_tensor_last_use(
    ops: &[PlannedOp],
    graph_outputs: &[String],
    weights: &HashMap<String, GpuTensor>,
) -> HashMap<String, usize> {
    let mut last_use: HashMap<String, usize> = HashMap::new();
    for (idx, op) in ops.iter().enumerate() {
        for input in &op.node.inputs {
            if !input.is_empty() && !weights.contains_key(input) {
                last_use.insert(input.clone(), idx);
            }
        }
    }
    for out_name in graph_outputs {
        if !weights.contains_key(out_name) {
            last_use.insert(out_name.clone(), ops.len());
        }
    }
    last_use
}

/// Sum the number of `None` entries across every shape in the map.
/// Used by the Phase 4 B fixpoint loop in [`lower`] to detect
/// per-iteration progress: a `None` dim that becomes `Some` somewhere
/// in `tensor_shapes` is one of the two progress signals (the other is
/// a newly-synthesized initializer). The loop exits the first iteration
/// that produces zero of both.
fn count_none_dims(shapes: &HashMap<String, Vec<Option<usize>>>) -> usize {
    shapes
        .values()
        .map(|s| s.iter().filter(|d| d.is_none()).count())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attributes::NodeAttributes;
    use crate::ir::graph::OptimizableGraphBuilder;
    use crate::ir::passes::shape_extraction_folding::{
        reset_fold_counters, ShapeExtractionFoldingCounts,
    };
    use crate::tensor::Tensor;

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn lower_empty_graph_produces_empty_plan() {
        let ctx = IconnxCudaContext::new().expect("ctx");
        let g = OptimizableGraphBuilder::new().build();
        let plan = lower(g, &ctx).expect("lower empty graph");
        assert!(plan.ops.is_empty());
        assert!(plan.weights.is_empty());
        assert!(plan.lstm_weights.is_empty());
        assert!(plan.static_i64.is_empty());
        assert!(plan.tensor_last_use.is_empty());
    }

    #[test]
    fn count_none_dims_counts_none_entries_only() {
        let mut shapes: HashMap<String, Vec<Option<usize>>> = HashMap::new();
        shapes.insert("all_concrete".into(), vec![Some(1), Some(5), Some(512)]);
        shapes.insert("mixed".into(), vec![None, Some(5), None]);
        shapes.insert("all_none".into(), vec![None, None]);
        shapes.insert("empty".into(), vec![]);
        // 0 + 2 + 2 + 0 = 4 None dims.
        assert_eq!(count_none_dims(&shapes), 4);
    }

    /// Fixpoint termination test — a `Shape -> Gather` chain against a
    /// concrete input must converge within the iteration cap and the
    /// fold must fire exactly once. Exercises the fixpoint body in
    /// [`lower`] directly (skipping the CUDA-dependent portions) so it
    /// runs without a GPU.
    ///
    /// We don't chain two patterns back-to-back here because the Phase 3
    /// inferencer today is an identity pass — `shape_inference` itself
    /// makes no progress, so a synthetic chain-of-two would converge in
    /// one iteration identically to a single fold. What this test
    /// asserts is that (a) the loop drives at least one successful
    /// iteration, (b) it early-breaks on zero progress before the cap,
    /// and (c) the fold counter increments exactly once for a
    /// single-chain input.
    #[test]
    fn shape_folding_fixpoint_terminates_within_three_iterations() {
        use crate::ir::passes::{
            dead_code_elimination, shape_extraction_folding, shape_inference,
            tensor_shapes_from_graph, topo_sort,
        };

        reset_fold_counters();

        // Build: Shape(x[1,5,512]) -> Gather(indices=[0,2]) -> "dims".
        // A single extraction chain — one fold is available.
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("x".to_string(), vec![Some(1), Some(5), Some(512)]);
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
        b.add_output("dims".to_string());
        let graph = b.build();

        // Mirror lower()'s pre-fixpoint pipeline, minus CUDA paths.
        let graph = topo_sort(graph).expect("topo_sort");
        let mut graph = shape_inference(graph);
        let mut tensor_shapes = tensor_shapes_from_graph(&graph);

        // Run the same fixpoint body as lower(). Track iteration count so
        // we can assert early-break behavior.
        const CAP: usize = 3;
        let mut iterations_executed = 0usize;
        for _iter in 0..CAP {
            let inits_before = graph.initializers.len();
            let none_before = count_none_dims(&tensor_shapes);

            graph = shape_extraction_folding(graph, &tensor_shapes);
            graph = shape_inference(graph);
            tensor_shapes = tensor_shapes_from_graph(&graph);

            iterations_executed += 1;

            let inits_after = graph.initializers.len();
            let none_after = count_none_dims(&tensor_shapes);

            let added = inits_after.saturating_sub(inits_before);
            let resolved = none_before.saturating_sub(none_after);

            if added == 0 && resolved == 0 {
                break;
            }
        }
        let _graph = dead_code_elimination(graph);

        // Fold occurred.
        let counts = ShapeExtractionFoldingCounts::snapshot();
        assert_eq!(
            counts.total, 1,
            "expected exactly one Shape->Gather fold, got counts={:?}",
            counts
        );

        // Early break — we added 1 initializer on iteration 1 and 0 on
        // iteration 2 (no more chains left to fold), so the loop should
        // have run either 1 or 2 times but strictly less than CAP+1.
        // The crucial invariant is that we did NOT run the full 3
        // iterations; if we had, it would mean the zero-progress early
        // break never fired.
        assert!(
            iterations_executed < CAP + 1 && iterations_executed >= 1,
            "iterations_executed={} outside expected range [1, {}]",
            iterations_executed,
            CAP,
        );
        // Stronger: after the first iteration folded, the second should
        // have zero progress and break immediately.
        assert!(
            iterations_executed <= 2,
            "fixpoint failed to early-break after fold exhausted; ran {} iterations",
            iterations_executed,
        );
    }
}
