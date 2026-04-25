//! General elementwise fusion — detects chains of elementwise ops and
//! emits `FusedPattern` annotations. Additive wrt the 8 hand-coded
//! patterns (option (a) lock per Phase 3 spec): the pass first tries to
//! match each chain against the known specialized variants; any chain
//! that doesn't match falls back to `FusedPattern::GeneralChain` and is
//! dispatched via the NVRTC-compiled general kernel.
//!
//! The pass itself is a pure function over `OptimizableGraph`; it does
//! not mutate the graph's node list. Annotations are emitted by the
//! companion helper [`fusion_annotations_from_graph`], which is called at
//! lowering time to produce the `FusionAnnotations` structure consumed
//! by `PlannedOp` assembly.
//!
//! The 8 specialized matchers live in the sibling `matchers` submodule;
//! this file owns the pass API, the chain grower, and the GeneralChain
//! builder.

use std::collections::{HashMap, HashSet};

use crate::cuda::inference::fusion::{FusedPattern, FusedPatternInfo};
use crate::ir::graph::{GraphNode, OptimizableGraph};
use crate::ir::passes::fusion::FusionAnnotations;

mod matchers;

#[cfg(test)]
mod equivalence_tests;

/// Maximum chain length. GELU is 7 ops; 8 leaves 1 slot of margin.
const CHAIN_CAP: usize = 8;

/// Allowlist of op types that fuse as pure-per-element. Includes binary
/// ops because the 8 specialized `FusedPattern` variants (DivRsqrt,
/// AddMulAdd, etc.) chain through Add/Sub/Mul/Div/Pow. The `GeneralChain`
/// fallback only fires when a chain is pure-unary (see
/// [`UNARY_ELEMENTWISE_OPS`]); binary-containing chains that fail to
/// match a specialized variant are dropped unfused.
const ELEMENTWISE_OPS: &[&str] = &[
    "Add", "Sub", "Mul", "Div", "Pow", "Sqrt", "Exp", "Erf", "Sin", "Cos", "Tanh", "Sigmoid",
    "Relu", "LeakyRelu", "Atan", "Floor", "Clip", "Round",
];

/// Subset of [`ELEMENTWISE_OPS`] that the single-input `GeneralChain`
/// kernel codegen can emit. Any chain where every op is in this list is
/// eligible for `GeneralChain` fallback; chains with a binary op that
/// didn't match a specialized variant are dropped unfused.
const UNARY_ELEMENTWISE_OPS: &[&str] = &[
    "Sqrt", "Exp", "Erf", "Sin", "Cos", "Tanh", "Sigmoid", "Relu", "LeakyRelu", "Atan", "Floor",
    "Clip", "Round",
];

/// Pure-function pass. Does not mutate `graph.nodes` — annotations are
/// computed separately by [`fusion_annotations_from_graph`] so lowering
/// can call them after every other pass has run (including DCE).
pub fn elementwise_fusion(graph: OptimizableGraph) -> OptimizableGraph {
    // This pass does not mutate graph.nodes; annotations are emitted by
    // `fusion_annotations_from_graph(graph)` called at lowering time.
    // Returning the graph unchanged keeps the pipeline's pure-fn invariant.
    graph
}

/// Compute fusion annotations after all other passes have run.
///
/// Walks the graph looking for chains of elementwise ops connected
/// single-use end-to-end. Each detected chain is either matched against
/// one of the 8 specialized `FusedPattern` variants (preserving today's
/// hand-coded kernels) or, when no match is found, emitted as a
/// `FusedPattern::GeneralChain` — the NVRTC-compiled general kernel
/// handles these.
pub fn fusion_annotations_from_graph(graph: &OptimizableGraph) -> FusionAnnotations {
    // Build a consumer map: tensor name → Vec<node_idx>.
    let mut consumers: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, node) in graph.nodes.iter().enumerate() {
        for input in &node.inputs {
            if !input.is_empty() {
                consumers.entry(input.as_str()).or_default().push(i);
            }
        }
    }

    // Producer map: tensor name → producer node index. Used by matchers
    // that need to walk to a side-input's producer (e.g. GELU's `Mul(x,
    // 0.5)`, which feeds the chain's tail Mul from outside the chain).
    let mut producers: HashMap<&str, usize> = HashMap::new();
    for (i, node) in graph.nodes.iter().enumerate() {
        for output in &node.outputs {
            if !output.is_empty() {
                producers.insert(output.as_str(), i);
            }
        }
    }

    let mut annotations = FusionAnnotations::default();
    let mut consumed: HashSet<usize> = HashSet::new();

    for (start_idx, start_node) in graph.nodes.iter().enumerate() {
        if consumed.contains(&start_idx) || !is_elementwise(&start_node.op_type) {
            continue;
        }
        // Try to extend a chain forward.
        let chain = grow_chain(start_idx, &graph.nodes, &consumers, &consumed);
        if chain.len() < 2 {
            continue;
        }

        // Emit a FusedPattern. Match against existing 8 variants first;
        // fall back to GeneralChain ONLY if the chain is pure-unary (the
        // general kernel codegen emits a single-input kernel; binary ops
        // that failed to match a specialized variant can't safely be
        // fused by the general path).
        let match_result = matchers::match_known_pattern(
            &chain,
            &graph.nodes,
            &producers,
            &graph.initializers,
        );
        let (pattern, extra_skip) = match match_result {
            Some(m) => (m.pattern, m.extra_skip_nodes),
            None => {
                if !chain_is_pure_unary(&chain, &graph.nodes) {
                    continue;
                }
                (build_general_chain(&chain, &graph.nodes), Vec::new())
            }
        };

        let head_name = graph.nodes[chain[0]].name.clone();
        let mut nodes_to_skip: Vec<String> = chain[1..]
            .iter()
            .map(|&i| graph.nodes[i].name.clone())
            .collect();
        for name in &extra_skip {
            if !nodes_to_skip.contains(name) {
                nodes_to_skip.push(name.clone());
            }
        }
        annotations.heads.insert(
            head_name.clone(),
            FusedPatternInfo {
                pattern,
                head_node: head_name,
                nodes_to_skip: nodes_to_skip.clone(),
            },
        );
        for &i in chain.iter() {
            consumed.insert(i);
            if i != chain[0] {
                annotations.skipped.insert(graph.nodes[i].name.clone());
            }
        }
        for name in &extra_skip {
            annotations.skipped.insert(name.clone());
            if let Some(&idx) = producers.get(name.as_str()) {
                consumed.insert(idx);
            }
        }
    }

    annotations
}

fn is_elementwise(op_type: &str) -> bool {
    ELEMENTWISE_OPS.contains(&op_type)
}

fn is_unary_elementwise(op_type: &str) -> bool {
    UNARY_ELEMENTWISE_OPS.contains(&op_type)
}

/// Returns true iff every op in the chain is a unary elementwise op
/// codegen can emit.
fn chain_is_pure_unary(chain: &[usize], nodes: &[GraphNode]) -> bool {
    chain
        .iter()
        .all(|&i| is_unary_elementwise(&nodes[i].op_type))
}

/// Extend a chain from `start_idx` forward. An op joins the chain iff:
/// - it's in `ELEMENTWISE_OPS`,
/// - it is the UNIQUE consumer of the previous op's sole output,
/// - it has not already been consumed by an earlier chain.
///
/// Chain is capped at [`CHAIN_CAP`].
fn grow_chain(
    start_idx: usize,
    nodes: &[GraphNode],
    consumers: &HashMap<&str, Vec<usize>>,
    consumed: &HashSet<usize>,
) -> Vec<usize> {
    let mut chain = vec![start_idx];
    let mut current_idx = start_idx;
    while chain.len() < CHAIN_CAP {
        // Find unique consumer of current's output.
        let current_output = nodes[current_idx].outputs.first().map(|s| s.as_str());
        let Some(out_name) = current_output else { break; };
        let Some(cs) = consumers.get(out_name) else { break; };
        if cs.len() != 1 {
            break;
        }
        let next_idx = cs[0];
        if consumed.contains(&next_idx) || !is_elementwise(&nodes[next_idx].op_type) {
            break;
        }
        chain.push(next_idx);
        current_idx = next_idx;
    }
    chain
}

/// Build a `GeneralChain` variant from a chain of elementwise ops.
///
/// `chain_signature` is a canonical string form of the op sequence that
/// keys the compiled-kernel cache. Identical sequences with identical
/// attributes produce identical signatures and reuse the same NVRTC
/// compile.
fn build_general_chain(chain: &[usize], nodes: &[GraphNode]) -> FusedPattern {
    let ops: Vec<(String, crate::attributes::NodeAttributes)> = chain
        .iter()
        .map(|&i| (nodes[i].op_type.clone(), nodes[i].attributes.clone()))
        .collect();
    let chain_signature = ops
        .iter()
        .map(|(op, attrs)| signature_for_op(op, attrs))
        .collect::<Vec<String>>()
        .join("|");
    let input_name = nodes[chain[0]].inputs.first().cloned().unwrap_or_default();
    let output_name = nodes[*chain.last().unwrap()]
        .outputs
        .first()
        .cloned()
        .unwrap_or_default();
    FusedPattern::GeneralChain {
        chain_signature,
        ops,
        input_name,
        output_name,
    }
}

/// Encode an op's type + attrs as a canonical signature fragment. The
/// codegen module parses the same fragment to regenerate the emitted CUDA
/// source, so the format must round-trip losslessly for every op in
/// [`ELEMENTWISE_OPS`].
fn signature_for_op(op_type: &str, attrs: &crate::attributes::NodeAttributes) -> String {
    // Use Debug format ({:?}) to match emit_op's kernel-body formatting so
    // two floats that differ only under Debug (e.g. subnormal edge cases)
    // cannot silently hit the same cache key while producing different
    // kernel bodies.
    match op_type {
        "LeakyRelu" => {
            let alpha = attrs.get_float("alpha").unwrap_or(0.01);
            format!("LeakyRelu(alpha={alpha:?})")
        }
        "Clip" => {
            let min = attrs.get_float("min").unwrap_or(f32::NEG_INFINITY);
            let max = attrs.get_float("max").unwrap_or(f32::INFINITY);
            format!("Clip(min={min:?},max={max:?})")
        }
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attributes::NodeAttributes;
    use crate::ir::graph::OptimizableGraphBuilder;
    use crate::tensor::Tensor;

    fn scalar(v: f32) -> Tensor {
        Tensor::from_vec_f32(vec![v], vec![1])
    }

    #[test]
    fn single_op_not_fused() {
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("x".into(), vec![Some(4)]);
        b.add_node(
            "a",
            "Sqrt",
            vec!["x".into()],
            vec!["y".into()],
            NodeAttributes::new(),
        );
        let a = fusion_annotations_from_graph(&b.build());
        assert!(a.heads.is_empty());
    }

    #[test]
    fn chain_of_two_elementwise_fused() {
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("x".into(), vec![Some(4)]);
        b.add_node(
            "a",
            "Sqrt",
            vec!["x".into()],
            vec!["t".into()],
            NodeAttributes::new(),
        );
        b.add_node(
            "b",
            "Exp",
            vec!["t".into()],
            vec!["y".into()],
            NodeAttributes::new(),
        );
        let a = fusion_annotations_from_graph(&b.build());
        assert_eq!(a.heads.len(), 1);
        assert!(a.heads.contains_key("a"));
        assert!(a.skipped.contains("b"));
    }

    #[test]
    fn chain_cap_respected() {
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("x".into(), vec![Some(4)]);
        let mut prev = "x".to_string();
        for i in 0..10 {
            let name = format!("n{}", i);
            let out = format!("t{}", i);
            b.add_node(
                name,
                "Sqrt",
                vec![prev.clone()],
                vec![out.clone()],
                NodeAttributes::new(),
            );
            prev = out;
        }
        let a = fusion_annotations_from_graph(&b.build());
        // First chain covers n0..n7 (8 ops, cap). Second chain covers n8..n9 (2 ops).
        assert_eq!(a.heads.len(), 2, "10-op run should split at cap=8");
    }

    #[test]
    fn non_elementwise_breaks_chain() {
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("x".into(), vec![Some(4)]);
        b.add_node(
            "a",
            "Sqrt",
            vec!["x".into()],
            vec!["t".into()],
            NodeAttributes::new(),
        );
        b.add_node(
            "r",
            "Reshape",
            vec!["t".into(), "_".into()],
            vec!["s".into()],
            NodeAttributes::new(),
        );
        b.add_node(
            "b",
            "Exp",
            vec!["s".into()],
            vec!["y".into()],
            NodeAttributes::new(),
        );
        let a = fusion_annotations_from_graph(&b.build());
        // Sqrt and Exp are separated by Reshape; neither chain reaches length 2.
        assert!(a.heads.is_empty());
    }

    #[test]
    fn matches_div_rsqrt_on_known_pattern() {
        // y = numerator / sqrt(variance + eps)
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("numerator".into(), vec![Some(4)]);
        b.add_input("variance".into(), vec![Some(4)]);
        b.add_initializer("eps".into(), scalar(1e-5));
        b.add_node(
            "add",
            "Add",
            vec!["variance".into(), "eps".into()],
            vec!["t_add".into()],
            NodeAttributes::new(),
        );
        b.add_node(
            "sq",
            "Sqrt",
            vec!["t_add".into()],
            vec!["t_sq".into()],
            NodeAttributes::new(),
        );
        b.add_node(
            "div",
            "Div",
            vec!["numerator".into(), "t_sq".into()],
            vec!["y".into()],
            NodeAttributes::new(),
        );
        let a = fusion_annotations_from_graph(&b.build());
        let head = a.heads.get("add").expect("DivRsqrt head on Add");
        assert!(
            matches!(head.pattern, FusedPattern::DivRsqrt { .. }),
            "expected DivRsqrt, got {:?}",
            head.pattern
        );
        assert!(a.skipped.contains("sq"));
        assert!(a.skipped.contains("div"));
    }

    #[test]
    fn matches_add_mul_add_on_known_pattern() {
        // y = (x + a) * b + c, with b and c as initializers.
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("x".into(), vec![Some(4)]);
        b.add_input("a".into(), vec![Some(4)]);
        b.add_initializer("bw".into(), Tensor::from_vec_f32(vec![2.0; 4], vec![4]));
        b.add_initializer("cw".into(), Tensor::from_vec_f32(vec![3.0; 4], vec![4]));
        b.add_node(
            "add1",
            "Add",
            vec!["x".into(), "a".into()],
            vec!["t1".into()],
            NodeAttributes::new(),
        );
        b.add_node(
            "mul",
            "Mul",
            vec!["t1".into(), "bw".into()],
            vec!["t2".into()],
            NodeAttributes::new(),
        );
        b.add_node(
            "add2",
            "Add",
            vec!["t2".into(), "cw".into()],
            vec!["y".into()],
            NodeAttributes::new(),
        );
        let a = fusion_annotations_from_graph(&b.build());
        let head = a.heads.get("add1").expect("AddMulAdd head on Add1");
        assert!(
            matches!(head.pattern, FusedPattern::AddMulAdd { .. }),
            "expected AddMulAdd, got {:?}",
            head.pattern
        );
    }

    #[test]
    fn matches_gelu_on_known_pattern() {
        // The canonical GELU op sequence.
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("x".into(), vec![Some(4)]);
        b.add_initializer("exp3".into(), scalar(3.0));
        b.add_initializer("c_inner".into(), scalar(0.044715));
        b.add_initializer("c_sqrt2pi".into(), scalar(0.797_884_6));
        b.add_initializer("c_one".into(), scalar(1.0));
        b.add_initializer("c_half".into(), scalar(0.5));
        b.add_node(
            "pow",
            "Pow",
            vec!["x".into(), "exp3".into()],
            vec!["t_pow".into()],
            NodeAttributes::new(),
        );
        b.add_node(
            "mul1",
            "Mul",
            vec!["t_pow".into(), "c_inner".into()],
            vec!["t_mul1".into()],
            NodeAttributes::new(),
        );
        b.add_node(
            "add1",
            "Add",
            vec!["x".into(), "t_mul1".into()],
            vec!["t_add1".into()],
            NodeAttributes::new(),
        );
        b.add_node(
            "mul2",
            "Mul",
            vec!["t_add1".into(), "c_sqrt2pi".into()],
            vec!["t_mul2".into()],
            NodeAttributes::new(),
        );
        b.add_node(
            "tanh_",
            "Tanh",
            vec!["t_mul2".into()],
            vec!["t_tanh".into()],
            NodeAttributes::new(),
        );
        b.add_node(
            "add2",
            "Add",
            vec!["t_tanh".into(), "c_one".into()],
            vec!["t_add2".into()],
            NodeAttributes::new(),
        );
        // The parallel Mul(x, 0.5) branch (must come BEFORE mul3 in
        // topological order so mul3's inputs are both available).
        b.add_node(
            "half",
            "Mul",
            vec!["x".into(), "c_half".into()],
            vec!["t_half".into()],
            NodeAttributes::new(),
        );
        b.add_node(
            "mul3",
            "Mul",
            vec!["t_half".into(), "t_add2".into()],
            vec!["y".into()],
            NodeAttributes::new(),
        );
        let a = fusion_annotations_from_graph(&b.build());
        let head = a.heads.get("pow").expect("GELU head on Pow");
        assert!(
            matches!(head.pattern, FusedPattern::Gelu { .. }),
            "expected Gelu, got {:?}",
            head.pattern
        );
        // `half` is the parallel Mul(x, 0.5) — must be in extra_skip.
        assert!(
            a.skipped.contains("half"),
            "GELU's parallel Mul(x, 0.5) must be skipped: got skipped={:?}",
            a.skipped
        );
    }

    #[test]
    fn matches_mul_add_on_known_pattern() {
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("a".into(), vec![Some(4)]);
        b.add_initializer("bw".into(), Tensor::from_vec_f32(vec![2.0; 4], vec![4]));
        b.add_initializer("cw".into(), Tensor::from_vec_f32(vec![3.0; 4], vec![4]));
        b.add_node(
            "mul",
            "Mul",
            vec!["a".into(), "bw".into()],
            vec!["t".into()],
            NodeAttributes::new(),
        );
        b.add_node(
            "add",
            "Add",
            vec!["t".into(), "cw".into()],
            vec!["y".into()],
            NodeAttributes::new(),
        );
        let a = fusion_annotations_from_graph(&b.build());
        let head = a.heads.get("mul").expect("MulAdd head on Mul");
        assert!(
            matches!(head.pattern, FusedPattern::MulAdd { .. }),
            "expected MulAdd, got {:?}",
            head.pattern
        );
    }

    /// WS-4 M4.7 regression: when `Mul`'s static side has a different shape
    /// than the bias initializer, the fused `gpu_fused_mul_add` wrapper has
    /// no path that fits — both supported layouts require `b.shape == c.shape`.
    /// Exercising fusion in that case panics at runtime (DistilBERT-INT8
    /// surfaces this when a per-tensor scalar scale meets a per-channel
    /// bias). The matcher must refuse fusion so the unfused Mul + Add fall
    /// back to separate elementwise ops.
    #[test]
    fn refuses_mul_add_when_static_b_shape_differs_from_c() {
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("a".into(), vec![Some(1), Some(5), Some(768)]);
        // Scalar scale (DistilBERT-INT8 combined dequant scale) vs.
        // per-channel bias [768]. Wrapper has no path for (a=[1,5,768],
        // b=[], c=[768]).
        b.add_initializer("scale".into(), Tensor::from_vec_f32(vec![0.001], vec![]));
        b.add_initializer(
            "bias".into(),
            Tensor::from_vec_f32(vec![0.0; 768], vec![768]),
        );
        b.add_node(
            "mul",
            "Mul",
            vec!["a".into(), "scale".into()],
            vec!["t".into()],
            NodeAttributes::new(),
        );
        b.add_node(
            "add",
            "Add",
            vec!["t".into(), "bias".into()],
            vec!["y".into()],
            NodeAttributes::new(),
        );
        let a = fusion_annotations_from_graph(&b.build());
        assert!(
            !a.heads.contains_key("mul"),
            "Mul→Add must not fuse when static-side shape ({:?}) doesn't \
             match the bias shape ({:?}); unfused fallback is required",
            vec![] as Vec<usize>,
            vec![768usize]
        );
    }

    /// MulAdd matcher must also refuse fusion when neither `Mul` input is
    /// a static initializer — Path 1 (a==b==c) requires runtime shape
    /// agreement that the matcher cannot verify, and Path 2 requires `b`
    /// to be the static side. Letting it through gambles a runtime panic.
    #[test]
    fn refuses_mul_add_when_both_mul_inputs_dynamic() {
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("a".into(), vec![Some(4)]);
        b.add_input("b_dyn".into(), vec![Some(4)]);
        b.add_initializer("cw".into(), Tensor::from_vec_f32(vec![3.0; 4], vec![4]));
        b.add_node(
            "mul",
            "Mul",
            vec!["a".into(), "b_dyn".into()],
            vec!["t".into()],
            NodeAttributes::new(),
        );
        b.add_node(
            "add",
            "Add",
            vec!["t".into(), "cw".into()],
            vec!["y".into()],
            NodeAttributes::new(),
        );
        let a = fusion_annotations_from_graph(&b.build());
        assert!(
            !a.heads.contains_key("mul"),
            "Mul→Add must not fuse when both Mul inputs are dynamic — \
             matcher cannot verify the wrapper's b/c shape contract"
        );
    }

    #[test]
    fn matches_add_mul_on_known_pattern() {
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("a".into(), vec![Some(4)]);
        b.add_input("bi".into(), vec![Some(4)]);
        b.add_initializer("cw".into(), Tensor::from_vec_f32(vec![3.0; 4], vec![4]));
        b.add_node(
            "add",
            "Add",
            vec!["a".into(), "bi".into()],
            vec!["t".into()],
            NodeAttributes::new(),
        );
        b.add_node(
            "mul",
            "Mul",
            vec!["t".into(), "cw".into()],
            vec!["y".into()],
            NodeAttributes::new(),
        );
        let a = fusion_annotations_from_graph(&b.build());
        let head = a.heads.get("add").expect("AddMul head on Add");
        assert!(
            matches!(head.pattern, FusedPattern::AddMul { .. }),
            "expected AddMul, got {:?}",
            head.pattern
        );
    }

    #[test]
    fn matches_sub_mul_on_known_pattern() {
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("a".into(), vec![Some(4)]);
        b.add_input("bi".into(), vec![Some(4)]);
        b.add_initializer("cw".into(), Tensor::from_vec_f32(vec![3.0; 4], vec![4]));
        b.add_node(
            "sub",
            "Sub",
            vec!["a".into(), "bi".into()],
            vec!["t".into()],
            NodeAttributes::new(),
        );
        b.add_node(
            "mul",
            "Mul",
            vec!["t".into(), "cw".into()],
            vec!["y".into()],
            NodeAttributes::new(),
        );
        let a = fusion_annotations_from_graph(&b.build());
        let head = a.heads.get("sub").expect("SubMul head on Sub");
        assert!(
            matches!(head.pattern, FusedPattern::SubMul { .. }),
            "expected SubMul, got {:?}",
            head.pattern
        );
    }

    #[test]
    fn matches_div_mul_on_known_pattern() {
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("a".into(), vec![Some(4)]);
        b.add_input("bi".into(), vec![Some(4)]);
        b.add_initializer("cw".into(), Tensor::from_vec_f32(vec![3.0; 4], vec![4]));
        b.add_node(
            "div",
            "Div",
            vec!["a".into(), "bi".into()],
            vec!["t".into()],
            NodeAttributes::new(),
        );
        b.add_node(
            "mul",
            "Mul",
            vec!["t".into(), "cw".into()],
            vec!["y".into()],
            NodeAttributes::new(),
        );
        let a = fusion_annotations_from_graph(&b.build());
        let head = a.heads.get("div").expect("DivMul head on Div");
        assert!(
            matches!(head.pattern, FusedPattern::DivMul { .. }),
            "expected DivMul, got {:?}",
            head.pattern
        );
    }

    #[test]
    fn matches_mul_sin_pow_mul_add_on_known_pattern() {
        // y = sin(x * w0)^p * w1 + b
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("x".into(), vec![Some(4)]);
        b.add_initializer("w0".into(), Tensor::from_vec_f32(vec![2.0; 4], vec![4]));
        b.add_initializer("w1".into(), Tensor::from_vec_f32(vec![3.0; 4], vec![4]));
        b.add_initializer("p_exp".into(), scalar(2.0));
        b.add_initializer("bias".into(), Tensor::from_vec_f32(vec![0.5; 4], vec![4]));
        b.add_node(
            "hmul",
            "Mul",
            vec!["x".into(), "w0".into()],
            vec!["t0".into()],
            NodeAttributes::new(),
        );
        b.add_node(
            "sin",
            "Sin",
            vec!["t0".into()],
            vec!["t1".into()],
            NodeAttributes::new(),
        );
        b.add_node(
            "pow",
            "Pow",
            vec!["t1".into(), "p_exp".into()],
            vec!["t2".into()],
            NodeAttributes::new(),
        );
        b.add_node(
            "tmul",
            "Mul",
            vec!["t2".into(), "w1".into()],
            vec!["t3".into()],
            NodeAttributes::new(),
        );
        b.add_node(
            "badd",
            "Add",
            vec!["t3".into(), "bias".into()],
            vec!["y".into()],
            NodeAttributes::new(),
        );
        let a = fusion_annotations_from_graph(&b.build());
        let head = a.heads.get("hmul").expect("MulSinPowMulAdd head on hmul");
        assert!(
            matches!(head.pattern, FusedPattern::MulSinPowMulAdd { .. }),
            "expected MulSinPowMulAdd, got {:?}",
            head.pattern
        );
    }
}
