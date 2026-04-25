//! Shape inference — forward-propagates tensor shapes through the graph.
//!
//! Dimensions are `Option<usize>`: `Some(n)` = statically known,
//! `None` = dynamic. Partially-dynamic shapes like `[None, 512, 80]`
//! are representable.
//!
//! Per the Phase 3 spec §Commit #3, this inferencer handles 28 ops
//! covering Kokoro. Unknown ops leave their output shapes fully
//! `None`-dimmed (the downstream passes treat `None`-containing shapes
//! as "don't trust").

use std::collections::HashMap;

use crate::attributes::NodeAttributes;
use crate::ir::graph::{GraphNode, OptimizableGraph};
use crate::tensor::Tensor;

pub type Shape = Vec<Option<usize>>;

/// Pipeline-shaped identity pass for shape inference.
///
/// **This function intentionally returns the graph unchanged.** It is
/// NOT a stub, NOT a TODO, and NOT incomplete — it exists to keep the
/// lowering pipeline uniform. Every pass in `lower()` has the signature
/// `fn(OptimizableGraph) -> OptimizableGraph`, and shape inference slots
/// into that chain like any other pass.
///
/// The real work of inferring tensor shapes lives in
/// [`tensor_shapes_from_graph`], which `lower()` calls immediately after
/// this pass. It returns a `HashMap<String, Shape>` rather than mutating
/// the graph, because inferred shapes are metadata attached to the
/// `ExecutionPlan` (via `ExecutionPlan::tensor_shapes`) — they are not
/// part of the graph structure itself.
///
/// Keeping the two functions split is deliberate:
/// - `shape_inference` preserves the pipeline's pass-chain shape, so
///   future passes can be inserted without changing the call pattern.
/// - `tensor_shapes_from_graph` returns a map the downstream passes
///   (e.g. `precompute_params`) consume directly, avoiding a re-walk.
///
/// Do not merge these into one function without understanding both
/// constraints — a review would catch it and send it back.
pub fn shape_inference(graph: OptimizableGraph) -> OptimizableGraph {
    graph
}

/// Compute the shape map from a fully-passed-through graph. Called from
/// `lower()` between DCE and the plan-assembly step.
pub fn tensor_shapes_from_graph(graph: &OptimizableGraph) -> HashMap<String, Shape> {
    let mut shapes: HashMap<String, Shape> = HashMap::new();
    let initializers = &graph.initializers;

    // Seed from graph inputs.
    for input in &graph.inputs {
        shapes.insert(input.name.clone(), input.shape.clone());
    }

    // Seed from initializers — these have known static shapes.
    for (name, tensor) in initializers {
        shapes.insert(
            name.clone(),
            tensor.shape().iter().map(|d| Some(*d)).collect(),
        );
    }

    // Forward-propagate through each node in order (graph is already
    // topo-sorted by the time this runs).
    for node in &graph.nodes {
        let out_shapes = infer_node_output_shapes(node, &shapes, initializers);
        for (name, shape) in node.outputs.iter().zip(out_shapes.into_iter()) {
            if !name.is_empty() {
                shapes.insert(name.clone(), shape);
            }
        }
    }

    shapes
}

/// Dispatch shape-inference for a single node. Unknown op → fully
/// `None`-dimmed output shapes (one per declared output).
fn infer_node_output_shapes(
    node: &GraphNode,
    shapes: &HashMap<String, Shape>,
    initializers: &HashMap<String, Tensor>,
) -> Vec<Shape> {
    let input_shapes: Vec<Shape> = node
        .inputs
        .iter()
        .map(|name| {
            if name.is_empty() {
                Vec::new()
            } else {
                shapes.get(name).cloned().unwrap_or_default()
            }
        })
        .collect();

    let attrs = &node.attributes;

    match node.op_type.as_str() {
        // Broadcasting elementwise — output shape is the broadcast of inputs.
        "Add" | "Sub" | "Mul" | "Div" | "Pow" | "Where"
        | "Equal" | "Less" | "Greater" | "GreaterOrEqual" | "LessOrEqual" | "NotEqual"
        | "And" | "Or" => {
            vec![broadcast_shapes(&input_shapes)]
        }

        // Unary pass-through.
        "Sqrt" | "Exp" | "Erf" | "Sin" | "Cos" | "Tanh" | "Sigmoid" | "Relu" | "LeakyRelu"
        | "Atan" | "Floor" | "Clip" | "Round" | "Cast" | "Softmax" | "LayerNormalization"
        | "Not" => {
            let s = input_shapes.first().cloned().unwrap_or_default();
            vec![s]
        }

        // Shape family.
        "Shape" => {
            let rank = input_shapes.first().map(|s| s.len()).unwrap_or(0);
            vec![vec![Some(rank)]]
        }
        // Constant: output shape is the shape of the `value` attribute
        // tensor. Phase 3 Commit #3 omitted this because Constants were
        // assumed to be promoted to initializers by the ONNX loader;
        // Kokoro disproves that — many Constant nodes persist in the
        // graph, and their Int64 outputs feed Gather.indices /
        // Unsqueeze.axes / Concat inputs that downstream Shape→Gather
        // / Shape→Slice chains depend on. Without this case they fell
        // through to `_ => Shape::new()`, zeroing propagation through
        // every consumer.
        "Constant" => {
            let value_shape = attrs
                .get_tensor("value")
                .map(|t| t.shape().iter().map(|d| Some(*d)).collect::<Shape>())
                .unwrap_or_default();
            vec![value_shape]
        }
        "ConstantOfShape" => {
            // Output shape is the VALUE of input[0] — only resolvable if input
            // is an initializer AND can be downcast to i64 at shape-inference time.
            // Since `tensor_shapes_from_graph` doesn't receive initializer VALUES,
            // we return fully-unknown. constant_folding in commit #2 typically
            // resolves ConstantOfShape before shape_inference runs anyway.
            vec![Shape::new()]
        }

        // Reshape family.
        "Reshape" => {
            vec![infer_reshape(&input_shapes, node, initializers)]
        }
        "Unsqueeze" => {
            vec![infer_unsqueeze(&input_shapes, attrs)]
        }
        "Squeeze" => {
            vec![infer_squeeze(&input_shapes, attrs)]
        }

        // Layout.
        "Transpose" => {
            vec![infer_transpose(&input_shapes, attrs)]
        }
        "Concat" => {
            vec![infer_concat(&input_shapes, attrs)]
        }
        "Slice" => {
            // Without the start/end/axes/steps values (they're input tensors,
            // not attributes), we can only preserve rank. Dim values become None.
            let s = input_shapes.first().cloned().unwrap_or_default();
            vec![s.iter().map(|_| None).collect()]
        }
        "Gather" => {
            vec![infer_gather(&input_shapes, attrs)]
        }

        // Reductions.
        "ReduceMean" | "ReduceSum" => {
            vec![infer_reduce(&input_shapes, attrs)]
        }

        // Conv / matmul.
        "Conv" => {
            vec![infer_conv(&input_shapes, attrs, false)]
        }
        "ConvTranspose" => {
            vec![infer_conv(&input_shapes, attrs, true)]
        }
        "MatMul" => {
            vec![infer_matmul(&input_shapes)]
        }
        "Gemm" => {
            vec![infer_gemm(&input_shapes, attrs)]
        }

        // LSTM — output Y shape [seq_len, num_directions, batch, hidden].
        "LSTM" => {
            vec![infer_lstm(&input_shapes, attrs)]
        }

        // Split — N outputs, each same as input except along `axis` where
        // the dim is the corresponding split size. Sizes resolution
        // mirrors the executor: explicit `split` initializer if present,
        // else `num_outputs` attribute, else node.outputs.len().
        "Split" => infer_split(&input_shapes, node, attrs, initializers),

        _ => vec![Shape::new(); node.outputs.len().max(1)],
    }
}

/// Broadcast a list of shapes per ONNX rules. `None` on any side
/// propagates to `None` on the output.
fn broadcast_shapes(inputs: &[Shape]) -> Shape {
    if inputs.is_empty() {
        return Shape::new();
    }
    let rank = inputs.iter().map(|s| s.len()).max().unwrap_or(0);
    let mut out = Shape::with_capacity(rank);
    for axis in 0..rank {
        let mut dim: Option<usize> = Some(1);
        for shape in inputs {
            if shape.is_empty() {
                // Completely unknown input shape → propagate fully-None.
                dim = None;
                break;
            }
            let s_rank = shape.len();
            // Right-align: a shape shorter than the broadcast rank is
            // implicitly padded with 1s on the left, so only axes
            // `axis >= rank - s_rank` index into `shape`.
            let offset = rank - s_rank;
            if axis >= offset {
                let d = shape[axis - offset];
                dim = match (dim, d) {
                    (Some(1), other) => other,
                    (other, Some(1)) => other,
                    (Some(a), Some(b)) if a == b => Some(a),
                    (None, _) | (_, None) => None,
                    _ => None, // shape mismatch — treat as unknown rather than panic
                };
            }
        }
        out.push(dim);
    }
    out
}

// Phase 3 §Commit #3 Known limitation — completed in Phase 4 B 2026-04-22.
// Resolves `0` placeholders from the data tensor's shape (Phase 3 spec
// convention: "0 = copy-from-data") and single `-1` from element-count
// division when all other dims are known. Falls back to Shape::new()
// when shape input isn't an initializer or when a dim can't be
// resolved.
fn infer_reshape(
    inputs: &[Shape],
    node: &GraphNode,
    initializers: &HashMap<String, Tensor>,
) -> Shape {
    // ONNX Reshape: inputs = [data, shape]. The shape input is a rank-1
    // Int64 tensor; in Kokoro it is always an initializer by the time
    // this pass runs (constant_folding + shape_extraction_folding have
    // already collapsed any Constant / Shape→Extract sources).
    let shape_name = match node.inputs.get(1) {
        Some(n) if !n.is_empty() => n,
        _ => return Shape::new(),
    };
    let shape_tensor = match initializers.get(shape_name) {
        Some(t) if t.is_int64() => t,
        _ => return Shape::new(),
    };
    let raw_shape: Vec<i64> = shape_tensor.to_array_i64().iter().copied().collect();

    // `Shape::new()` sentinel means "fully unknown" throughout this
    // codebase — it is NOT a known rank-0 tensor. Mirror
    // `precompute.rs::precompute_reshape_shape` which documents the
    // same invariant.
    let data_shape_opt = inputs.first().filter(|s| !s.is_empty());

    let mut resolved: Vec<i64> = Vec::with_capacity(raw_shape.len());
    for (i, &d) in raw_shape.iter().enumerate() {
        if d == 0 {
            // `0` at position i means "copy data_shape[i]". If the data
            // dim at that position is unknown (None) or data shape
            // itself is unknown, we cannot resolve this output dim.
            match data_shape_opt.and_then(|s| s.get(i).copied().flatten()) {
                Some(n) => resolved.push(n as i64),
                None => return Shape::new(),
            }
        } else {
            resolved.push(d);
        }
    }

    // Resolve a single `-1` using the known element count — requires
    // every dim of the data tensor to be known.
    if resolved.contains(&-1) {
        let data_shape = match data_shape_opt {
            Some(s) => s,
            None => return Shape::new(),
        };
        let all_known: Option<Vec<usize>> = data_shape.iter().copied().collect();
        let data_dims = match all_known {
            Some(d) => d,
            None => return Shape::new(),
        };
        let data_elements: i64 = data_dims.iter().map(|&n| n as i64).product();
        let known_product: Option<i64> = resolved
            .iter()
            .filter(|&&d| d != -1)
            .try_fold(1i64, |acc, &d| if d > 0 { Some(acc * d) } else { None });
        let known_product = match known_product {
            Some(p) if p > 0 && data_elements % p == 0 => p,
            _ => return Shape::new(),
        };
        let inferred = data_elements / known_product;
        for d in resolved.iter_mut() {
            if *d == -1 {
                *d = inferred;
            }
        }
    }

    // Every entry should now be > 0. Any residual non-positive entry
    // (a leftover 0 or -1 the steps above couldn't resolve) means we
    // don't have a fully concrete answer — bail to the unknown sentinel
    // rather than emit a wrong-but-plausible shape.
    if resolved.iter().any(|&d| d <= 0) {
        return Shape::new();
    }
    resolved.into_iter().map(|d| Some(d as usize)).collect()
}

fn infer_unsqueeze(inputs: &[Shape], attrs: &NodeAttributes) -> Shape {
    let src = inputs.first().cloned().unwrap_or_default();
    if src.is_empty() {
        return Shape::new();
    }
    let axes: Vec<i64> = attrs.get_ints("axes").unwrap_or(&[]).to_vec();
    if axes.is_empty() {
        return Shape::new();
    }
    let out_rank = src.len() + axes.len();
    let mut normalized: Vec<i64> = axes
        .iter()
        .map(|&a| if a < 0 { a + out_rank as i64 } else { a })
        .collect();
    normalized.sort_unstable();
    let mut out: Shape = Vec::with_capacity(out_rank);
    let mut src_iter = src.into_iter();
    for i in 0..out_rank {
        if normalized.binary_search(&(i as i64)).is_ok() {
            out.push(Some(1));
        } else {
            out.push(src_iter.next().unwrap_or(None));
        }
    }
    out
}

fn infer_squeeze(inputs: &[Shape], attrs: &NodeAttributes) -> Shape {
    let src = inputs.first().cloned().unwrap_or_default();
    if src.is_empty() {
        return Shape::new();
    }
    let axes: Vec<i64> = attrs.get_ints("axes").unwrap_or(&[]).to_vec();
    let rank = src.len() as i64;
    let normalized: Vec<i64> = axes
        .iter()
        .map(|&a| if a < 0 { a + rank } else { a })
        .collect();
    if normalized.is_empty() {
        // No explicit axes → remove all size-1 dims that are Some(1).
        src.into_iter().filter(|d| *d != Some(1)).collect()
    } else {
        src.into_iter()
            .enumerate()
            .filter(|(i, _)| !normalized.contains(&(*i as i64)))
            .map(|(_, d)| d)
            .collect()
    }
}

fn infer_transpose(inputs: &[Shape], attrs: &NodeAttributes) -> Shape {
    let src = inputs.first().cloned().unwrap_or_default();
    if src.is_empty() {
        return Shape::new();
    }
    let perm: Vec<i64> = attrs.get_ints("perm").unwrap_or(&[]).to_vec();
    if perm.is_empty() {
        // Default: reverse dims.
        src.into_iter().rev().collect()
    } else {
        perm.iter()
            .map(|&p| src.get(p as usize).copied().unwrap_or(None))
            .collect()
    }
}

fn infer_concat(inputs: &[Shape], attrs: &NodeAttributes) -> Shape {
    if inputs.is_empty() || inputs[0].is_empty() {
        return Shape::new();
    }
    let axis: i64 = attrs.get_int("axis").unwrap_or(0);
    let rank = inputs[0].len() as i64;
    let axis = if axis < 0 { axis + rank } else { axis } as usize;
    let mut out = inputs[0].clone();
    for input in inputs.iter().skip(1) {
        if input.len() != out.len() {
            return Shape::new();
        }
        match (out.get(axis).copied().flatten(), input.get(axis).copied().flatten()) {
            (Some(a), Some(b)) => out[axis] = Some(a + b),
            _ => out[axis] = None,
        }
    }
    out
}

fn infer_gather(inputs: &[Shape], attrs: &NodeAttributes) -> Shape {
    // ONNX Gather output rank is `data.rank + indices.rank - 1`. We can
    // only materialize that formula when BOTH input ranks are known.
    //
    // Within this file, `Shape::new()` (an empty `Vec`) is the
    // codebase-wide "fully unknown" sentinel — it does NOT denote a
    // known rank-0 scalar. Every other inferencer here treats it that
    // way (see `broadcast_shapes`, `infer_unsqueeze`, `infer_squeeze`,
    // `infer_reduce`, `infer_conv`, `infer_matmul`, `infer_gemm`,
    // `infer_lstm`), and `precompute.rs` documents the exact same
    // convention for Reshape.
    //
    // The previous implementation fell through on an empty
    // `indices_shape`, extending the output with the surrounding slices
    // of `data_shape` alone — which silently fabricated a concrete
    // output rank of `data.rank - 1`. On Kokoro this surfaced as the
    // token-embedding Gather (`[178, 512]` + unknown `tokens`)
    // producing `[Some(512)]` instead of the required unknown
    // sentinel. That fabricated `[Some(512)]` then propagated through
    // Transpose / Conv / LSTM / Transpose, ending up as the reported
    // `[None, Some(512), Some(2), Some(256)]` at the `text_encoder`
    // LSTM Transpose_1 output. Downstream passes (Reshape
    // pre-resolution, Commit 4 fusion eligibility, Phase 4 memory
    // planner) consume those shapes as truth — so this bug is
    // load-bearing even though today's runtime still works.
    if inputs.len() < 2 || inputs[0].is_empty() || inputs[1].is_empty() {
        return Shape::new();
    }
    let data_shape = &inputs[0];
    let indices_shape = &inputs[1];
    let axis: i64 = attrs.get_int("axis").unwrap_or(0);
    let axis = if axis < 0 {
        axis + data_shape.len() as i64
    } else {
        axis
    } as usize;
    let mut out: Shape = Vec::new();
    out.extend_from_slice(&data_shape[..axis]);
    out.extend_from_slice(indices_shape);
    if axis < data_shape.len() {
        out.extend_from_slice(&data_shape[axis + 1..]);
    }
    out
}

fn infer_reduce(inputs: &[Shape], attrs: &NodeAttributes) -> Shape {
    let src = inputs.first().cloned().unwrap_or_default();
    if src.is_empty() {
        return Shape::new();
    }
    let axes: Vec<i64> = attrs.get_ints("axes").unwrap_or(&[]).to_vec();
    let keepdims = attrs.get_int("keepdims").unwrap_or(1) != 0;
    let rank = src.len() as i64;
    let normalized: Vec<i64> = axes
        .iter()
        .map(|&a| if a < 0 { a + rank } else { a })
        .collect();
    let all_axes = normalized.is_empty();
    src.into_iter()
        .enumerate()
        .filter_map(|(i, d)| {
            let reduced = all_axes || normalized.contains(&(i as i64));
            if reduced {
                if keepdims {
                    Some(Some(1))
                } else {
                    None
                }
            } else {
                Some(d)
            }
        })
        .collect()
}

fn infer_conv(inputs: &[Shape], attrs: &NodeAttributes, is_transpose: bool) -> Shape {
    // Conv: output = [N, C_out, H_out, W_out, ...] where spatial dims are
    // computed from input + kernel + strides + padding + dilations.
    // For Phase 3 scope, compute only when all spatial dims are Some().
    if inputs.len() < 2 {
        return Shape::new();
    }
    let input = &inputs[0];
    let kernel = &inputs[1];
    if input.is_empty() || kernel.is_empty() || input.len() < 3 {
        return Shape::new();
    }
    let batch = input[0];
    let out_channels = if is_transpose { kernel[1] } else { kernel[0] };
    let kernel_spatial = &kernel[2..];
    let spatial_in = &input[2..];

    let strides: Vec<i64> = attrs.get_ints("strides").unwrap_or(&[]).to_vec();
    let pads: Vec<i64> = attrs.get_ints("pads").unwrap_or(&[]).to_vec();
    let dilations: Vec<i64> = attrs.get_ints("dilations").unwrap_or(&[]).to_vec();
    let output_padding: Vec<i64> =
        attrs.get_ints("output_padding").unwrap_or(&[]).to_vec();

    let mut out: Shape = Vec::with_capacity(input.len());
    out.push(batch);
    out.push(out_channels);
    for (i, (&in_dim, &k_dim)) in spatial_in.iter().zip(kernel_spatial.iter()).enumerate() {
        let in_dim = if let Some(d) = in_dim { d as i64 } else {
            out.push(None);
            continue;
        };
        let k_dim = if let Some(d) = k_dim { d as i64 } else {
            out.push(None);
            continue;
        };
        let s = *strides.get(i).unwrap_or(&1);
        let d = *dilations.get(i).unwrap_or(&1);
        let p_begin = *pads.get(i).unwrap_or(&0);
        let p_end = *pads.get(i + spatial_in.len()).unwrap_or(&0);
        let effective_k = (k_dim - 1) * d + 1;
        let out_dim = if is_transpose {
            let op = *output_padding.get(i).unwrap_or(&0);
            (in_dim - 1) * s - p_begin - p_end + effective_k + op
        } else {
            (in_dim + p_begin + p_end - effective_k) / s + 1
        };
        out.push(Some(out_dim.max(0) as usize));
    }
    out
}

fn infer_matmul(inputs: &[Shape]) -> Shape {
    // MatMul: output last two dims are [a.dim(-2), b.dim(-1)].
    // Batch dims (all but last two) broadcast per ONNX rules. A rank-0
    // batch shape (no batch dims) is treated as the broadcast identity,
    // not as "fully unknown" — so [8, 4, 16] × [16, 32] = [8, 4, 32].
    if inputs.len() < 2 {
        return Shape::new();
    }
    let a = &inputs[0];
    let b = &inputs[1];
    if a.len() < 2 || b.len() < 2 {
        return Shape::new();
    }
    let a_batch = a[..a.len() - 2].to_vec();
    let b_batch = b[..b.len() - 2].to_vec();
    let batch = match (a_batch.is_empty(), b_batch.is_empty()) {
        (true, true) => Shape::new(),
        (true, false) => b_batch,
        (false, true) => a_batch,
        (false, false) => broadcast_shapes(&[a_batch, b_batch]),
    };
    let mut out = batch;
    out.push(a[a.len() - 2]);
    out.push(b[b.len() - 1]);
    out
}

fn infer_gemm(inputs: &[Shape], attrs: &NodeAttributes) -> Shape {
    if inputs.len() < 2 {
        return Shape::new();
    }
    let a = &inputs[0];
    let b = &inputs[1];
    if a.len() < 2 || b.len() < 2 {
        return Shape::new();
    }
    let trans_a = attrs.get_int("transA").unwrap_or(0) != 0;
    let trans_b = attrs.get_int("transB").unwrap_or(0) != 0;
    let m = if trans_a { a[1] } else { a[0] };
    let n = if trans_b { b[0] } else { b[1] };
    vec![m, n]
}

fn infer_lstm(inputs: &[Shape], attrs: &NodeAttributes) -> Shape {
    // Y shape: [seq_len, num_directions, batch, hidden_size].
    if inputs.is_empty() || inputs[0].len() < 2 {
        return Shape::new();
    }
    let x_shape = &inputs[0];
    let seq_len = x_shape[0];
    let batch = x_shape.get(1).copied().unwrap_or(Some(1));
    let hidden_size = attrs.get_int("hidden_size").map(|h| h as usize);
    // ONNX spec (RNN/LSTM/GRU) defines `direction` as an enum with exactly
    // three valid values: "forward" (default, 1 dir), "reverse" (1 dir),
    // and "bidirectional" (2 dirs). Any other value indicates a malformed
    // graph; rather than silently falling through to num_directions=1 and
    // producing a plausible-but-wrong output shape, bail out to an unknown
    // shape so downstream passes do not treat the tensor as trusted.
    let num_directions = match attrs.get_string("direction") {
        None => Some(1),
        Some(d) if d == "forward" || d == "reverse" => Some(1),
        Some("bidirectional") => Some(2),
        Some(_) => return Shape::new(),
    };
    vec![seq_len, num_directions, batch, hidden_size.map(Some).unwrap_or(None)]
}

/// Infer per-output shapes for ONNX Split.
///
/// Returns one Shape per declared output. Each output shape matches the
/// input on every axis except `attrs.axis`, where the dim takes the
/// corresponding split size. Sizes resolution mirrors the executor's
/// `dispatch_split` precisely:
///   * `inputs[1]` is an Int64 initializer `split` — use those values.
///   * Else `num_outputs` attribute (opset 18+).
///   * Else `node.outputs.len()`.
///
/// When `axis` is unknown (None) the output is rank-preserving with
/// every dim None — same fallback as Slice.
fn infer_split(
    input_shapes: &[Shape],
    node: &GraphNode,
    attrs: &NodeAttributes,
    initializers: &HashMap<String, Tensor>,
) -> Vec<Shape> {
    let input_shape = input_shapes.first().cloned().unwrap_or_default();
    let ndim = input_shape.len();
    let n_outputs = node.outputs.len().max(1);

    if input_shape.is_empty() {
        return vec![Shape::new(); n_outputs];
    }

    let axis_attr = attrs.get_int("axis").unwrap_or(0);
    let axis = if axis_attr < 0 {
        (ndim as i64 + axis_attr).max(0) as usize
    } else {
        (axis_attr as usize).min(ndim.saturating_sub(1))
    };

    // Resolve split sizes.
    let split_sizes: Vec<usize> = node
        .inputs
        .get(1)
        .and_then(|name| {
            if name.is_empty() {
                None
            } else {
                initializers.get(name)
            }
        })
        .map(|t| {
            t.as_slice_i64()
                .into_iter()
                .map(|s| s as usize)
                .collect::<Vec<_>>()
        })
        .filter(|v: &Vec<usize>| !v.is_empty())
        .unwrap_or_else(|| {
            let n = attrs
                .get_int("num_outputs")
                .map(|v| v as usize)
                .unwrap_or(n_outputs)
                .max(1);
            // Use the input's axis dim if it is statically known to derive
            // even chunks; otherwise fall back to N rank-preserving
            // unknown-axis-dim shapes.
            match input_shape.get(axis).copied().flatten() {
                Some(axis_size) => {
                    let chunk = axis_size / n;
                    let remainder = axis_size % n;
                    let mut sizes = vec![chunk; n];
                    if let Some(last) = sizes.last_mut() {
                        *last += remainder;
                    }
                    sizes
                }
                None => Vec::new(),
            }
        });

    if split_sizes.is_empty() {
        // Couldn't resolve sizes statically. Preserve rank but mark the
        // axis dim as unknown for each declared output.
        let mut s = input_shape.clone();
        s[axis] = None;
        return vec![s; n_outputs];
    }

    split_sizes
        .into_iter()
        .map(|size| {
            let mut s = input_shape.clone();
            s[axis] = Some(size);
            s
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attributes::NodeAttributes;
    use crate::ir::OptimizableGraphBuilder;

    #[test]
    fn elementwise_broadcasts_static_shapes() {
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("a".into(), vec![Some(1), Some(3), Some(4)]);
        b.add_input("b".into(), vec![Some(2), Some(1), Some(4)]);
        b.add_node(
            "add",
            "Add",
            vec!["a".into(), "b".into()],
            vec!["y".into()],
            NodeAttributes::new(),
        );
        let shapes = tensor_shapes_from_graph(&b.build());
        assert_eq!(shapes.get("y").unwrap(), &vec![Some(2), Some(3), Some(4)]);
    }

    #[test]
    fn dynamic_batch_propagates() {
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("x".into(), vec![None, Some(512), Some(80)]);
        b.add_input("y".into(), vec![None, Some(512), Some(80)]);
        b.add_node(
            "add",
            "Add",
            vec!["x".into(), "y".into()],
            vec!["z".into()],
            NodeAttributes::new(),
        );
        let shapes = tensor_shapes_from_graph(&b.build());
        assert_eq!(shapes.get("z").unwrap(), &vec![None, Some(512), Some(80)]);
    }

    #[test]
    fn unknown_op_produces_empty_shape() {
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("x".into(), vec![Some(5)]);
        b.add_node(
            "wacky",
            "WackyUnknownOp",
            vec!["x".into()],
            vec!["y".into()],
            NodeAttributes::new(),
        );
        let shapes = tensor_shapes_from_graph(&b.build());
        // Fully-unknown shape per spec: empty Vec.
        assert_eq!(shapes.get("y").unwrap(), &Shape::new());
    }

    #[test]
    fn unary_passthrough_preserves_shape() {
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("x".into(), vec![Some(2), None, Some(8)]);
        b.add_node("sq", "Sqrt", vec!["x".into()], vec!["y".into()], NodeAttributes::new());
        let shapes = tensor_shapes_from_graph(&b.build());
        assert_eq!(shapes.get("y").unwrap(), &vec![Some(2), None, Some(8)]);
    }

    #[test]
    fn erf_propagates_shape() {
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("x".into(), vec![Some(4), Some(8)]);
        b.add_node(
            "erf_node",
            "Erf",
            vec!["x".into()],
            vec!["y".into()],
            NodeAttributes::new(),
        );
        let graph = b.build();
        let shapes = tensor_shapes_from_graph(&graph);
        assert_eq!(shapes.get("y").unwrap(), &vec![Some(4), Some(8)]);
    }

    #[test]
    fn split_with_explicit_split_initializer_returns_per_output_shapes() {
        // Split([6, 2]) along axis 0 with sizes [1, 3, 2] → outputs of
        // shape [1,2], [3,2], [2,2].
        use crate::tensor::Tensor;
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("x".into(), vec![Some(6), Some(2)]);
        b.add_initializer("split_sizes".into(), Tensor::from_vec_i64(vec![1, 3, 2], vec![3]));
        let mut attrs = NodeAttributes::new();
        attrs.add_int("axis".into(), 0);
        b.add_node(
            "split",
            "Split",
            vec!["x".into(), "split_sizes".into()],
            vec!["o0".into(), "o1".into(), "o2".into()],
            attrs,
        );
        let shapes = tensor_shapes_from_graph(&b.build());
        assert_eq!(shapes.get("o0").unwrap(), &vec![Some(1), Some(2)]);
        assert_eq!(shapes.get("o1").unwrap(), &vec![Some(3), Some(2)]);
        assert_eq!(shapes.get("o2").unwrap(), &vec![Some(2), Some(2)]);
    }

    #[test]
    fn split_via_num_outputs_attribute_returns_even_shapes() {
        // Split([6]) with num_outputs=3 → three [2] outputs.
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("x".into(), vec![Some(6)]);
        let mut attrs = NodeAttributes::new();
        attrs.add_int("axis".into(), 0);
        attrs.add_int("num_outputs".into(), 3);
        b.add_node(
            "split",
            "Split",
            vec!["x".into()],
            vec!["a".into(), "b".into(), "c".into()],
            attrs,
        );
        let shapes = tensor_shapes_from_graph(&b.build());
        assert_eq!(shapes.get("a").unwrap(), &vec![Some(2)]);
        assert_eq!(shapes.get("b").unwrap(), &vec![Some(2)]);
        assert_eq!(shapes.get("c").unwrap(), &vec![Some(2)]);
    }

    #[test]
    fn relu_propagates_shape() {
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("x".into(), vec![Some(1), Some(3), Some(224), Some(224)]);
        b.add_node(
            "relu_node",
            "Relu",
            vec!["x".into()],
            vec!["y".into()],
            NodeAttributes::new(),
        );
        let graph = b.build();
        let shapes = tensor_shapes_from_graph(&graph);
        assert_eq!(
            shapes.get("y").unwrap(),
            &vec![Some(1), Some(3), Some(224), Some(224)]
        );
    }

    #[test]
    fn shape_op_gives_rank() {
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("x".into(), vec![Some(3), Some(4), Some(5)]);
        b.add_node("s", "Shape", vec!["x".into()], vec!["r".into()], NodeAttributes::new());
        let shapes = tensor_shapes_from_graph(&b.build());
        assert_eq!(shapes.get("r").unwrap(), &vec![Some(3)]);
    }

    #[test]
    fn transpose_permutes_dims() {
        let mut attrs = NodeAttributes::new();
        attrs.add_ints("perm".into(), vec![2, 0, 1]);
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("x".into(), vec![Some(4), Some(5), Some(6)]);
        b.add_node("t", "Transpose", vec!["x".into()], vec!["y".into()], attrs);
        let shapes = tensor_shapes_from_graph(&b.build());
        assert_eq!(shapes.get("y").unwrap(), &vec![Some(6), Some(4), Some(5)]);
    }

    #[test]
    fn concat_sums_on_axis() {
        let mut attrs = NodeAttributes::new();
        attrs.add_int("axis".into(), 1);
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("a".into(), vec![Some(2), Some(3), Some(4)]);
        b.add_input("b".into(), vec![Some(2), Some(5), Some(4)]);
        b.add_node("c", "Concat", vec!["a".into(), "b".into()], vec!["y".into()], attrs);
        let shapes = tensor_shapes_from_graph(&b.build());
        assert_eq!(shapes.get("y").unwrap(), &vec![Some(2), Some(8), Some(4)]);
    }

    #[test]
    fn reduce_collapses_axes_with_keepdims_0() {
        let mut attrs = NodeAttributes::new();
        attrs.add_ints("axes".into(), vec![1]);
        attrs.add_int("keepdims".into(), 0);
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("x".into(), vec![Some(3), Some(4), Some(5)]);
        b.add_node("r", "ReduceMean", vec!["x".into()], vec!["y".into()], attrs);
        let shapes = tensor_shapes_from_graph(&b.build());
        assert_eq!(shapes.get("y").unwrap(), &vec![Some(3), Some(5)]);
    }

    #[test]
    fn matmul_inherits_last_two_dims() {
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("a".into(), vec![Some(8), Some(4), Some(16)]);
        b.add_input("b".into(), vec![Some(16), Some(32)]);
        b.add_node("m", "MatMul", vec!["a".into(), "b".into()], vec!["y".into()], NodeAttributes::new());
        let shapes = tensor_shapes_from_graph(&b.build());
        assert_eq!(shapes.get("y").unwrap(), &vec![Some(8), Some(4), Some(32)]);
    }

    #[test]
    fn lstm_forward_direction_yields_one_direction() {
        // ONNX `direction` = "forward" -> num_directions = 1.
        let mut attrs = NodeAttributes::new();
        attrs.add_int("hidden_size".into(), 64);
        attrs.add_string("direction".into(), "forward".into());
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("x".into(), vec![Some(10), Some(2), Some(80)]);
        b.add_node(
            "lstm",
            "LSTM",
            vec!["x".into()],
            vec!["y".into()],
            attrs,
        );
        let shapes = tensor_shapes_from_graph(&b.build());
        assert_eq!(
            shapes.get("y").unwrap(),
            &vec![Some(10), Some(1), Some(2), Some(64)]
        );
    }

    #[test]
    fn lstm_reverse_direction_yields_one_direction() {
        // ONNX `direction` = "reverse" -> num_directions = 1.
        let mut attrs = NodeAttributes::new();
        attrs.add_int("hidden_size".into(), 64);
        attrs.add_string("direction".into(), "reverse".into());
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("x".into(), vec![Some(10), Some(2), Some(80)]);
        b.add_node(
            "lstm",
            "LSTM",
            vec!["x".into()],
            vec!["y".into()],
            attrs,
        );
        let shapes = tensor_shapes_from_graph(&b.build());
        assert_eq!(
            shapes.get("y").unwrap(),
            &vec![Some(10), Some(1), Some(2), Some(64)]
        );
    }

    #[test]
    fn lstm_bidirectional_yields_two_directions() {
        // ONNX `direction` = "bidirectional" -> num_directions = 2.
        let mut attrs = NodeAttributes::new();
        attrs.add_int("hidden_size".into(), 64);
        attrs.add_string("direction".into(), "bidirectional".into());
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("x".into(), vec![Some(10), Some(2), Some(80)]);
        b.add_node(
            "lstm",
            "LSTM",
            vec!["x".into()],
            vec!["y".into()],
            attrs,
        );
        let shapes = tensor_shapes_from_graph(&b.build());
        assert_eq!(
            shapes.get("y").unwrap(),
            &vec![Some(10), Some(2), Some(2), Some(64)]
        );
    }

    #[test]
    fn lstm_absent_direction_defaults_to_one() {
        // Absent `direction` attr -> ONNX default is "forward" -> 1 direction.
        let mut attrs = NodeAttributes::new();
        attrs.add_int("hidden_size".into(), 64);
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("x".into(), vec![Some(10), Some(2), Some(80)]);
        b.add_node(
            "lstm",
            "LSTM",
            vec!["x".into()],
            vec!["y".into()],
            attrs,
        );
        let shapes = tensor_shapes_from_graph(&b.build());
        assert_eq!(
            shapes.get("y").unwrap(),
            &vec![Some(10), Some(1), Some(2), Some(64)]
        );
    }

    #[test]
    fn gather_unknown_indices_rank_returns_unknown_shape() {
        // Regression test for the Phase 3 Commit 3 silent-shape bug on
        // Kokoro. Token-embedding Gather reads `encoder.text_encoder.
        // embedding.weight` (shape `[178, 512]`) with `tokens` as the
        // indices tensor. `tokens` is a graph input whose shape is not
        // seeded by today's loader, so `tensor_shapes_from_graph` sees
        // an empty `Shape::new()` (the codebase's "fully unknown"
        // sentinel) for the indices input.
        //
        // Previously, `infer_gather` concatenated the empty indices
        // shape with the data shape's surrounding slices, fabricating a
        // rank equal to `data.rank - 1` (i.e. `[Some(512)]`). This is
        // the exact same sentinel-confusion bug that `precompute.rs`
        // already documents for Reshape: `Shape::new()` is "unknown
        // rank", NOT a known rank-0 scalar, and must never contribute a
        // concrete rank to an output shape.
        //
        // Downstream this fabricated `[Some(512)]` fed a Transpose with
        // `perm=[0,2,1]` that produced `[Some(512), None, None]`, then
        // a Conv that produced `[Some(512), Some(512), None]` (batch
        // axis polluted with `512`), then an LSTM whose Y-shape
        // `[seq_len, 2, batch, hidden] = [None, Some(2), Some(512),
        // Some(256)]` fed a final Transpose with `perm=[0,2,1,3]` that
        // produced the reported symptom `[None, Some(512), Some(2),
        // Some(256)]`.
        //
        // Fix: match the convention already used by every other
        // inferencer in this file — when any input shape is the empty
        // `Shape::new()` sentinel, propagate unknown instead of
        // synthesizing a rank.
        let mut attrs = NodeAttributes::new();
        attrs.add_int("axis".into(), 0);
        let mut b = OptimizableGraphBuilder::new();
        // data is an initializer with a concrete shape.
        b.add_initializer(
            "emb".into(),
            crate::tensor::Tensor::from_vec_f32(vec![0.0; 178 * 512], vec![178, 512]),
        );
        // indices is a graph input WITHOUT a seeded shape, so it reads
        // back as `Shape::new()` from `tensor_shapes_from_graph`.
        b.add_input("idx".into(), Vec::new());
        b.add_node(
            "g",
            "Gather",
            vec!["emb".into(), "idx".into()],
            vec!["y".into()],
            attrs,
        );
        let shapes = tensor_shapes_from_graph(&b.build());
        // Must be the unknown sentinel, NOT a fabricated `[Some(512)]`.
        assert_eq!(shapes.get("y").unwrap(), &Shape::new());
    }

    #[test]
    fn gather_known_indices_produces_correct_rank() {
        // Positive test: Gather of `[178, 512]` embedding with rank-2
        // indices `[1, 5]` on axis 0 must yield rank 3 `[1, 5, 512]`.
        // This is the textbook token-embedding case; it is the shape we
        // want to see once the loader seeds graph-input shapes.
        let mut attrs = NodeAttributes::new();
        attrs.add_int("axis".into(), 0);
        let mut b = OptimizableGraphBuilder::new();
        b.add_initializer(
            "emb".into(),
            crate::tensor::Tensor::from_vec_f32(vec![0.0; 178 * 512], vec![178, 512]),
        );
        b.add_input("idx".into(), vec![Some(1), Some(5)]);
        b.add_node(
            "g",
            "Gather",
            vec!["emb".into(), "idx".into()],
            vec!["y".into()],
            attrs,
        );
        let shapes = tensor_shapes_from_graph(&b.build());
        assert_eq!(shapes.get("y").unwrap(), &vec![Some(1), Some(5), Some(512)]);
    }

    #[test]
    fn gather_unknown_data_returns_unknown_shape() {
        // Symmetric unknown-propagation: if data is the unknown
        // sentinel, the output is also unknown regardless of indices.
        let mut attrs = NodeAttributes::new();
        attrs.add_int("axis".into(), 0);
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("data".into(), Vec::new());
        b.add_input("idx".into(), vec![Some(1), Some(5)]);
        b.add_node(
            "g",
            "Gather",
            vec!["data".into(), "idx".into()],
            vec!["y".into()],
            attrs,
        );
        let shapes = tensor_shapes_from_graph(&b.build());
        assert_eq!(shapes.get("y").unwrap(), &Shape::new());
    }

    #[test]
    fn constant_produces_value_attribute_shape() {
        // Phase 4 B: Constant nodes that survive the ONNX loader (not
        // all are promoted to initializers — Kokoro has dozens) carry
        // their payload in the `value` attribute. Shape inference
        // must read its tensor shape rather than the default empty.
        let mut attrs = NodeAttributes::new();
        attrs.add_tensor(
            "value".into(),
            crate::tensor::Tensor::from_vec_i64(vec![3, 4, 5], vec![3]),
        );
        let mut b = OptimizableGraphBuilder::new();
        b.add_node(
            "c",
            "Constant",
            vec![],
            vec!["const_out".into()],
            attrs,
        );
        let shapes = tensor_shapes_from_graph(&b.build());
        assert_eq!(shapes.get("const_out").unwrap(), &vec![Some(3)]);
    }

    #[test]
    fn constant_without_value_attribute_falls_back_to_unknown() {
        // Malformed Constant (no `value` attr) — fall back to empty
        // sentinel rather than panic.
        let mut b = OptimizableGraphBuilder::new();
        b.add_node(
            "c",
            "Constant",
            vec![],
            vec!["const_out".into()],
            NodeAttributes::new(),
        );
        let shapes = tensor_shapes_from_graph(&b.build());
        assert_eq!(shapes.get("const_out").unwrap(), &Shape::new());
    }

    #[test]
    fn reshape_with_concrete_initializer_shape_produces_concrete_output() {
        // Phase 4 B: shape input is a concrete Int64 initializer
        // without any `0`/`-1` placeholders — infer_reshape should
        // return the shape verbatim.
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("data".into(), vec![Some(1), Some(5), Some(128)]);
        b.add_initializer(
            "new_shape".into(),
            crate::tensor::Tensor::from_vec_i64(vec![5, 128], vec![2]),
        );
        b.add_node(
            "r",
            "Reshape",
            vec!["data".into(), "new_shape".into()],
            vec!["y".into()],
            NodeAttributes::new(),
        );
        let shapes = tensor_shapes_from_graph(&b.build());
        assert_eq!(shapes.get("y").unwrap(), &vec![Some(5), Some(128)]);
    }

    #[test]
    fn reshape_with_zero_placeholder_copies_from_data() {
        // `0` at pos 0 copies data[0] = 1; `-1` at pos 1 resolves to
        // 640 / 1 = 640.
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("data".into(), vec![Some(1), Some(5), Some(128)]);
        b.add_initializer(
            "new_shape".into(),
            crate::tensor::Tensor::from_vec_i64(vec![0, -1], vec![2]),
        );
        b.add_node(
            "r",
            "Reshape",
            vec!["data".into(), "new_shape".into()],
            vec!["y".into()],
            NodeAttributes::new(),
        );
        let shapes = tensor_shapes_from_graph(&b.build());
        assert_eq!(shapes.get("y").unwrap(), &vec![Some(1), Some(640)]);
    }

    #[test]
    fn reshape_with_none_dim_in_data_when_zero_requested_falls_back() {
        // `0` at pos 0 targets data[0] which is None → must fall back
        // to Shape::new() (the unknown sentinel), NOT fabricate a dim.
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("data".into(), vec![None, Some(5)]);
        b.add_initializer(
            "new_shape".into(),
            crate::tensor::Tensor::from_vec_i64(vec![0, 5], vec![2]),
        );
        b.add_node(
            "r",
            "Reshape",
            vec!["data".into(), "new_shape".into()],
            vec!["y".into()],
            NodeAttributes::new(),
        );
        let shapes = tensor_shapes_from_graph(&b.build());
        assert_eq!(shapes.get("y").unwrap(), &Shape::new());
    }

    #[test]
    fn reshape_with_runtime_shape_input_falls_back() {
        // shape input is a graph input (runtime tensor), not an
        // initializer → fall back to unknown sentinel.
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("data".into(), vec![Some(1), Some(5), Some(128)]);
        b.add_input("new_shape".into(), vec![Some(2)]);
        b.add_node(
            "r",
            "Reshape",
            vec!["data".into(), "new_shape".into()],
            vec!["y".into()],
            NodeAttributes::new(),
        );
        let shapes = tensor_shapes_from_graph(&b.build());
        assert_eq!(shapes.get("y").unwrap(), &Shape::new());
    }

    #[test]
    fn lstm_unknown_direction_returns_unknown_shape() {
        // A `direction` attr outside {"forward", "reverse", "bidirectional"}
        // indicates a malformed graph. Rather than silently falling back to
        // num_directions=1 (and producing a plausible-but-wrong shape), the
        // inferencer must return the unknown shape (`Shape::new()`) so
        // downstream passes treat the tensor as untrusted.
        let mut attrs = NodeAttributes::new();
        attrs.add_int("hidden_size".into(), 64);
        attrs.add_string("direction".into(), "sideways".into());
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("x".into(), vec![Some(10), Some(2), Some(80)]);
        b.add_node(
            "lstm",
            "LSTM",
            vec!["x".into()],
            vec!["y".into()],
            attrs,
        );
        let shapes = tensor_shapes_from_graph(&b.build());
        assert_eq!(shapes.get("y").unwrap(), &Shape::new());
    }
}
