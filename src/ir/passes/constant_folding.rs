//! Constant folding — evaluate ops whose inputs are all static on CPU,
//! replacing them with new initializers. Unlocks downstream DCE.
//!
//! Only folds ops that have a `crate::operators::<op>::forward`
//! implementation (today's CPU operator set). Unknown op → unchanged.
//! Only folds when every non-empty input is in `graph.initializers`
//! — partial constant propagation is out of scope.

use std::collections::HashMap;

use crate::ir::graph::{GraphNode, OptimizableGraph};
use crate::tensor::Tensor;

/// Output-size cap per folded op. Folding a 1M-element Conv on CPU is not a win.
const FOLD_OUTPUT_ELEMENT_BUDGET: usize = 1_000_000;

/// Evaluate every op whose non-empty inputs are all in `graph.initializers`
/// (or in an earlier-folded output within this pass) on the CPU, promote the
/// result to a new initializer, and drop the op. Ops with any dynamic input,
/// unknown op types, multi-output ops, and ops whose output exceeds the
/// element budget are preserved verbatim.
pub fn constant_folding(mut graph: OptimizableGraph) -> OptimizableGraph {
    // Accumulated folded outputs — reused within this pass so chains of
    // const-ops fold transitively without requiring a second pass.
    let mut folded: HashMap<String, Tensor> = HashMap::new();

    let mut kept_nodes: Vec<GraphNode> = Vec::with_capacity(graph.nodes.len());

    for node in graph.nodes.into_iter() {
        if let Some(outputs) = try_fold(&node, &graph.initializers, &folded) {
            // Every output tensor is now folded — record into `folded` and
            // drop the op.
            for (name, tensor) in node.outputs.iter().zip(outputs.into_iter()) {
                if !name.is_empty() {
                    folded.insert(name.clone(), tensor);
                }
            }
            // Do NOT push `node` — it's been folded away.
        } else {
            kept_nodes.push(node);
        }
    }

    // Promote folded tensors into initializers so later passes (shape
    // inference, precompute, fusion) treat them as first-class constants.
    for (name, tensor) in folded {
        graph.initializers.insert(name, tensor);
    }

    graph.nodes = kept_nodes;
    graph
}

/// Attempt to fold one node. Returns `Some(outputs)` iff every non-empty
/// input is already constant (initializer or earlier folded) AND the op
/// has a CPU `forward` implementation AND the computed output size is
/// within the element budget.
///
/// Multi-output ops (e.g., LSTM producing Y / Y_h / Y_c) are not folded
/// by this pass — their multi-output forward signature is out of scope.
fn try_fold(
    node: &GraphNode,
    initializers: &HashMap<String, Tensor>,
    folded: &HashMap<String, Tensor>,
) -> Option<Vec<Tensor>> {
    // Gather input tensors in order. Empty-named inputs are valid "absent
    // optional" markers per ONNX — skip them.
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

    // Dispatch the CPU operator. Unknown op → bail out.
    let output = dispatch_cpu_forward(&node.op_type, &input_tensors, &node.attributes)?;

    // Enforce output-size cap. `Tensor::len()` is dtype-agnostic (works for
    // Float32 / Int64 / Int32 / Bool outputs); `as_slice()` would panic for
    // non-Float32, so we deliberately use `len()` here.
    if output.len() > FOLD_OUTPUT_ELEMENT_BUDGET {
        return None;
    }

    // Single-output ops only for now — every op in the dispatch table below
    // has exactly one output. Future extension: return Vec<Tensor> from
    // `dispatch_cpu_forward` to cover multi-output ops like LSTM.
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
        "Sigmoid" => sigmoid::Sigmoid::forward(inputs, attributes),
        "Atan" => atan::Atan::forward(inputs, attributes),
        "Clip" => clip::Clip::forward(inputs, attributes),
        "Floor" => floor::Floor::forward(inputs, attributes),
        "Round" => round::Round::forward(inputs, attributes),
        "Cast" => cast::Cast::forward(inputs, attributes),

        // Shape / layout.
        "Reshape" => reshape::Reshape::forward(inputs, attributes),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attributes::NodeAttributes;
    use crate::ir::OptimizableGraphBuilder;

    #[test]
    fn mul_of_two_initializers_folds() {
        let mut b = OptimizableGraphBuilder::new();
        b.add_initializer("a".into(), Tensor::from_vec_f32(vec![2.0, 3.0], vec![2]));
        b.add_initializer("b".into(), Tensor::from_vec_f32(vec![4.0, 5.0], vec![2]));
        b.add_node(
            "m",
            "Mul",
            vec!["a".into(), "b".into()],
            vec!["y".into()],
            NodeAttributes::new(),
        );
        let g = constant_folding(b.build());
        assert!(
            g.nodes.is_empty(),
            "Mul of two constants should be folded away"
        );
        let y = g.initializers.get("y").expect("folded y initializer present");
        assert_eq!(y.as_slice(), &[8.0, 15.0]);
    }

    #[test]
    fn chain_add_mul_folds_transitively() {
        let mut b = OptimizableGraphBuilder::new();
        b.add_initializer("c1".into(), Tensor::from_vec_f32(vec![1.0], vec![1]));
        b.add_initializer("c2".into(), Tensor::from_vec_f32(vec![2.0], vec![1]));
        b.add_initializer("c3".into(), Tensor::from_vec_f32(vec![3.0], vec![1]));
        b.add_node(
            "add1",
            "Add",
            vec!["c1".into(), "c2".into()],
            vec!["t0".into()],
            NodeAttributes::new(),
        );
        b.add_node(
            "mul1",
            "Mul",
            vec!["t0".into(), "c3".into()],
            vec!["y".into()],
            NodeAttributes::new(),
        );
        let g = constant_folding(b.build());
        assert!(g.nodes.is_empty(), "chain of constants should fold");
        assert_eq!(g.initializers.get("y").unwrap().as_slice(), &[9.0]); // (1+2)*3 = 9
    }

    #[test]
    fn semi_constant_does_not_fold() {
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("x".into(), vec![Some(1)]);
        b.add_initializer("c".into(), Tensor::from_vec_f32(vec![5.0], vec![1]));
        b.add_node(
            "add",
            "Add",
            vec!["x".into(), "c".into()],
            vec!["y".into()],
            NodeAttributes::new(),
        );
        let g = constant_folding(b.build());
        assert_eq!(g.nodes.len(), 1, "op with a dynamic input should not fold");
    }

    #[test]
    fn unknown_op_does_not_fold() {
        let mut b = OptimizableGraphBuilder::new();
        b.add_initializer("c".into(), Tensor::from_vec_f32(vec![0.5], vec![1]));
        b.add_node(
            "wacky",
            "WackyUnknownOp",
            vec!["c".into()],
            vec!["y".into()],
            NodeAttributes::new(),
        );
        let g = constant_folding(b.build());
        assert_eq!(g.nodes.len(), 1, "unknown op stays");
        assert!(g.initializers.get("y").is_none());
    }

    #[test]
    fn output_budget_respected() {
        // A Mul of two 1001x1001 f32 tensors produces 1,002,001 elements,
        // exceeding FOLD_OUTPUT_ELEMENT_BUDGET (1M). Should not fold.
        let big = vec![0.5_f32; 1001 * 1001];
        let mut b = OptimizableGraphBuilder::new();
        b.add_initializer(
            "a".into(),
            Tensor::from_vec_f32(big.clone(), vec![1001, 1001]),
        );
        b.add_initializer("b".into(), Tensor::from_vec_f32(big, vec![1001, 1001]));
        b.add_node(
            "huge_mul",
            "Mul",
            vec!["a".into(), "b".into()],
            vec!["y".into()],
            NodeAttributes::new(),
        );
        let g = constant_folding(b.build());
        assert_eq!(g.nodes.len(), 1, "output-size-capped op should not fold");
    }

    #[test]
    fn single_op_fold_matches_direct_forward_1e_7() {
        let a = Tensor::from_vec_f32(vec![1.23, 4.56, 7.89], vec![3]);
        let b = Tensor::from_vec_f32(vec![0.5, 0.25, 0.125], vec![3]);
        let attrs = NodeAttributes::new();

        // Direct forward.
        let direct = crate::operators::mul::Mul::forward(&[a.clone(), b.clone()], &attrs);

        // Via constant_folding.
        let mut builder = OptimizableGraphBuilder::new();
        builder.add_initializer("a".into(), a);
        builder.add_initializer("b".into(), b);
        builder.add_node(
            "m",
            "Mul",
            vec!["a".into(), "b".into()],
            vec!["y".into()],
            NodeAttributes::new(),
        );
        let g = constant_folding(builder.build());
        let folded = g.initializers.get("y").expect("folded");

        // Single op → 1e-7 per spec.
        let max_diff = direct
            .as_slice()
            .iter()
            .zip(folded.as_slice().iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff < 1e-7,
            "single-op fold must match direct forward to 1e-7, got {}",
            max_diff
        );
    }

    #[test]
    fn two_op_chain_fold_matches_manual_1e_6() {
        // (a + b) * c computed manually vs folded through constant_folding.
        let a = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::from_vec_f32(vec![0.1, 0.2, 0.3], vec![3]);
        let c = Tensor::from_vec_f32(vec![10.0, 20.0, 30.0], vec![3]);
        let attrs = NodeAttributes::new();

        let sum = crate::operators::add::Add::forward(&[a.clone(), b.clone()], &attrs);
        let manual = crate::operators::mul::Mul::forward(&[sum, c.clone()], &attrs);

        let mut builder = OptimizableGraphBuilder::new();
        builder.add_initializer("a".into(), a);
        builder.add_initializer("b".into(), b);
        builder.add_initializer("c".into(), c);
        builder.add_node(
            "add1",
            "Add",
            vec!["a".into(), "b".into()],
            vec!["t0".into()],
            attrs.clone(),
        );
        builder.add_node(
            "mul1",
            "Mul",
            vec!["t0".into(), "c".into()],
            vec!["y".into()],
            attrs,
        );
        let g = constant_folding(builder.build());
        let folded = g.initializers.get("y").expect("folded");

        let max_diff = manual
            .as_slice()
            .iter()
            .zip(folded.as_slice().iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff < 1e-6,
            "≥2-op chain fold must match manual to 1e-6, got {}",
            max_diff
        );
    }
}
