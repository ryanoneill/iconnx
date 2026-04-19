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
use crate::ir::passes::{dead_code_elimination, detect_fusion, precompute_params, topo_sort};
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

    // 2. DCE sweep. Structural no-op at landing per Kokoro sanity check;
    // becomes active once constant_folding and elementwise_fusion start
    // creating orphans.
    let graph = dead_code_elimination(graph);

    // 3. Precompute params (pure, no GPU).
    let precomputed_by_name = precompute_params(&graph);

    // 4. Detect fusion (pure, no GPU).
    let fusion = detect_fusion(&graph);

    // 5. Upload initializers to GPU.
    let weights = upload_initializers(&graph, ctx)?;

    // 6. Build static_i64 for fast lookup at runtime.
    let static_i64 = collect_static_i64(&graph);

    // 7. Pack LSTM weights eagerly.
    let lstm_weights = pack_lstm_weights_eagerly(&graph, &weights, ctx)?;

    // 8. Build PlannedOp sequence from sorted nodes + fusion + precomputed.
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

    // 9. Compute tensor_last_use for memory-pool return timing.
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
    })
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::graph::OptimizableGraphBuilder;

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
}
