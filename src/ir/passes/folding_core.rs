//! Shared fold-loop primitive for `constant_folding` (initializer-only)
//! and `shape_aware_folding` (Shape-predicate extension).
//!
//! The fold loop walks nodes in topo order, folds any node whose inputs
//! are all constant (initializer or earlier folded-in-pass) OR whose
//! `extra_foldable` predicate returns `Some(tensor)`, promotes folded
//! outputs to initializers, and drops the folded node. The loop runs to
//! fixpoint internally — chains of foldable ops collapse in one pass.

use std::collections::HashMap;

use crate::ir::graph::{GraphNode, OptimizableGraph};
use crate::tensor::Tensor;

/// Maximum element count for a folded output. Folds that would produce
/// tensors larger than this are skipped (folding a 1M-element Conv on
/// CPU is not a win — and could explode the initializer table).
pub(crate) const FOLD_OUTPUT_ELEMENT_BUDGET: usize = 1_000_000;

/// Fold every node whose inputs are constant (initializer, earlier-folded,
/// or `extra_foldable` returns `Some`). Returns the graph with folded
/// nodes removed and their outputs promoted to initializers.
///
/// `extra_foldable` is called once per non-initializer-only node; it receives
/// the node and the in-pass folded map (not the initializers map, since
/// initializers are already checked). Returning `Some(tensor)` folds the
/// node with the provided value; `None` defers to the standard
/// initializer-only fold path.
///
/// The predicate receives:
/// - `node`: the graph node being considered for folding.
/// - `folded`: the in-pass `folded` map (name → tensor) of outputs
///   already folded earlier in the same pass. Not currently consumed
///   by `shape_aware_folding`'s Shape predicate (it consults
///   `tensor_shapes` directly), but exposed so a future extension that
///   needs intra-pass chain state can access it without a signature
///   break.
pub fn fold_with_predicate(
    mut graph: OptimizableGraph,
    extra_foldable: impl Fn(&GraphNode, &HashMap<String, Tensor>) -> Option<Tensor>,
) -> OptimizableGraph {
    let mut folded: HashMap<String, Tensor> = HashMap::new();
    let mut kept_nodes: Vec<GraphNode> = Vec::with_capacity(graph.nodes.len());

    for node in graph.nodes.into_iter() {
        if let Some(outputs) = try_fold(&node, &graph.initializers, &folded, &extra_foldable) {
            for (name, tensor) in node.outputs.iter().zip(outputs.into_iter()) {
                if !name.is_empty() {
                    folded.insert(name.clone(), tensor);
                }
            }
        } else {
            kept_nodes.push(node);
        }
    }

    for (name, tensor) in folded {
        graph.initializers.insert(name, tensor);
    }
    graph.nodes = kept_nodes;
    graph
}

/// Attempt to fold one node. Returns `Some(outputs)` iff standard
/// initializer-only folding succeeds OR `extra_foldable` returns a value.
///
/// Multi-output ops (e.g., LSTM producing Y / Y_h / Y_c) are not folded
/// by this pass — their multi-output forward signature is out of scope.
fn try_fold(
    node: &GraphNode,
    initializers: &HashMap<String, Tensor>,
    folded: &HashMap<String, Tensor>,
    extra_foldable: &impl Fn(&GraphNode, &HashMap<String, Tensor>) -> Option<Tensor>,
) -> Option<Vec<Tensor>> {
    // Extension path first — lets passes override standard semantics
    // (e.g., shape_aware_folding's Shape-uses-tensor_shapes rule).
    if let Some(tensor) = extra_foldable(node, folded) {
        if tensor.len() > FOLD_OUTPUT_ELEMENT_BUDGET {
            return None;
        }
        return Some(vec![tensor]);
    }

    // Standard initializer-only path.
    let mut input_tensors: Vec<Tensor> = Vec::with_capacity(node.inputs.len());
    for name in &node.inputs {
        if name.is_empty() {
            continue;
        }
        let t = initializers
            .get(name)
            .or_else(|| folded.get(name))?
            .clone();
        input_tensors.push(t);
    }

    let output = dispatch_cpu_forward(&node.op_type, &input_tensors, &node.attributes)?;

    if output.len() > FOLD_OUTPUT_ELEMENT_BUDGET {
        return None;
    }
    Some(vec![output])
}

/// Route a node to its CPU forward implementation. Returns `None` for
/// any op we don't support yet.
///
/// Mirrors the authoritative CPU dispatch in
/// `src/graph_executor.rs::execute_operator`, but single-op (not
/// whole-graph) so the constant-folding pass can call it in isolation.
fn dispatch_cpu_forward(
    op_type: &str,
    inputs: &[Tensor],
    attributes: &crate::attributes::NodeAttributes,
) -> Option<Tensor> {
    use crate::operators::*;
    Some(match op_type {
        // Binary elementwise arithmetic.
        "Add" => add::Add::forward(inputs, attributes),
        "Sub" => sub::Sub::forward(inputs, attributes),
        "Mul" => mul::Mul::forward(inputs, attributes),
        "Div" => div::Div::forward(inputs, attributes),
        "Pow" => pow::Pow::forward(inputs, attributes),

        // Unary elementwise.
        "Sqrt" => sqrt::Sqrt::forward(inputs, attributes),
        "Exp" => exp::Exp::forward(inputs, attributes),
        "Sin" => sin::Sin::forward(inputs, attributes),
        "Cos" => cos::Cos::forward(inputs, attributes),
        "Tanh" => tanh::Tanh::forward(inputs, attributes),
        "Erf" => erf::Erf::forward(inputs, attributes),
        "Relu" => relu::Relu::forward(inputs, attributes),
        "Sigmoid" => sigmoid::Sigmoid::forward(inputs, attributes),
        "Atan" => atan::Atan::forward(inputs, attributes),
        "Clip" => clip::Clip::forward(inputs, attributes),
        "Floor" => floor::Floor::forward(inputs, attributes),
        "Round" => round::Round::forward(inputs, attributes),
        "Cast" => cast::Cast::forward(inputs, attributes),

        // Shape / layout.
        "Reshape" => reshape::Reshape::forward(inputs, attributes),
        "Flatten" => flatten::Flatten::forward(inputs, attributes),
        "GlobalAveragePool" => global_average_pool::GlobalAveragePool::forward(inputs, attributes),
        "Unsqueeze" => unsqueeze::Unsqueeze::forward(inputs, attributes),
        "Squeeze" => squeeze::Squeeze::forward(inputs, attributes),
        "Transpose" => transpose::Transpose::forward(inputs, attributes),
        "Shape" => shape::Shape::forward(inputs, attributes),
        "ConstantOfShape" => constant_of_shape::ConstantOfShape::forward(inputs, attributes),
        "Gather" => gather::Gather::forward(inputs, attributes),
        "Concat" => concat::Concat::forward(inputs, attributes),
        "Slice" => slice::Slice::forward(inputs, attributes),
        "Range" => range::Range::forward(inputs, attributes),

        // Comparisons — today's CPU operators internally dispatch on input
        // dtype, so calling the single entry-point handles Float32 / Int64 /
        // Int32 correctly without a sub-match here.
        "Equal" => equal::Equal::forward(inputs, attributes),
        "Less" => less::Less::forward(inputs, attributes),
        "Greater" => greater::Greater::forward(inputs, attributes),
        "GreaterOrEqual" => greater_or_equal::GreaterOrEqual::forward(inputs, attributes),

        // Everything else (Conv, ConvTranspose, MatMul, Gemm, LSTM, STFT,
        // LayerNormalization, ReduceMean, ReduceSum, ScatterND, NonZero,
        // Softmax, LeakyRelu, Pad, Resize, Where, CumSum, And, Or, Not,
        // Expand, Constant) is intentionally excluded. Either they rarely
        // have all-constant inputs in real ONNX graphs, or their CPU
        // forward is prohibitively expensive at fold time, or their
        // signature is multi-output.
        _ => return None,
    })
}
