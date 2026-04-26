//! Fusion-pattern detection. Thin adapter that bridges
//! `OptimizableGraph` into the existing `cuda::inference::fusion` detection
//! library. Operates entirely on CPU `Tensor` values — no GPU context
//! required, no D2H copies.

use std::collections::{HashMap, HashSet};

use crate::cuda::inference::fusion::{FusedPattern, FusedPatternInfo, detect_fused_patterns_from_cpu};
use crate::ir::graph::OptimizableGraph;

/// Result of the fusion pass — attached to an `OptimizableGraph` so
/// lowering can thread the annotations into each `PlannedOp`.
#[derive(Default, Debug, Clone)]
pub struct FusionAnnotations {
    pub heads: HashMap<String, FusedPatternInfo>,
    pub skipped: HashSet<String>,
    pub dynamic_candidates: HashMap<String, FusedPatternInfo>,
}

/// Detect fusion patterns in an `OptimizableGraph`.
///
/// The detection library operates directly on `&[GraphNode]`. Initializer
/// values are read from `graph.initializers` without any D2H round-trip.
pub fn detect_fusion(graph: &OptimizableGraph) -> FusionAnnotations {
    let (heads, skipped, dynamic_candidates) =
        detect_fused_patterns_from_cpu(&graph.nodes, &graph.initializers);
    FusionAnnotations { heads, skipped, dynamic_candidates }
}

/// Drop fused patterns whose participating static values (initializers
/// or `Constant` node outputs) carry a non-Float32 dtype (notably
/// Float16 for Whisper-FP16). `gpu_fused_div_rsqrt` and the other fused
/// kernels are sealed to f32 today, so the op-by-op fallback is the
/// only correct dispatch path for FP16 graphs. Gating here keeps fused
/// dispatch correct rather than panicking inside
/// `eps_tensor.to_host_f32(...)`. A full FP16 fused kernel suite is a
/// separate post-Phase-3 effort.
///
/// Constant node outputs are inspected because Whisper-FP16 exports its
/// LayerNorm `eps` as a `Constant` node (not an initializer). Only
/// checking `graph.initializers` would miss this case. Both surfaces
/// are valid producer locations for a "static" tensor.
///
/// Applies to both static heads and dynamic candidates. Static heads
/// also have their interior nodes cleaned out of `skipped` so the
/// fallback path can execute them.
pub fn gate_fusion_to_float32(
    mut annotations: FusionAnnotations,
    graph: &OptimizableGraph,
) -> FusionAnnotations {
    let static_dtypes = collect_static_dtypes(graph);
    drop_non_f32_patterns(
        &mut annotations.heads,
        &mut annotations.skipped,
        &static_dtypes,
        true,
    );
    drop_non_f32_patterns(
        &mut annotations.dynamic_candidates,
        &mut annotations.skipped,
        &static_dtypes,
        false,
    );
    annotations
}

/// Build a `name → dtype-name` map covering every static tensor in the
/// graph: initializers and `Constant` node outputs. Dtype is recorded as
/// the string from `Tensor::dtype()` so the helper can be dtype-agnostic.
fn collect_static_dtypes(graph: &OptimizableGraph) -> HashMap<String, &'static str> {
    let mut dtypes: HashMap<String, &'static str> = HashMap::new();
    for (name, tensor) in &graph.initializers {
        dtypes.insert(name.clone(), tensor.dtype());
    }
    for node in &graph.nodes {
        if node.op_type != "Constant" {
            continue;
        }
        if let Some(value) = node.attributes.resolve_constant_value() {
            if let Some(out) = node.outputs.first() {
                dtypes.insert(out.clone(), value.dtype());
            }
        }
    }
    dtypes
}

/// Remove patterns whose participating tensor names point at a static
/// value (initializer or Constant node output) with non-Float32 dtype.
/// Static patterns (`heads`) also clean their interior nodes out of
/// `skipped` so the fallback path can execute them. Dynamic candidates
/// don't contribute to `skipped` (per detector docs), so the
/// `clean_skipped` flag flips that behavior.
fn drop_non_f32_patterns(
    patterns: &mut HashMap<String, FusedPatternInfo>,
    skipped: &mut HashSet<String>,
    static_dtypes: &HashMap<String, &'static str>,
    clean_skipped: bool,
) {
    let drop_keys: Vec<String> = patterns
        .iter()
        .filter_map(|(head, info)| {
            if pattern_touches_non_f32_static(&info.pattern, static_dtypes) {
                Some(head.clone())
            } else {
                None
            }
        })
        .collect();

    for head in drop_keys {
        let info = patterns.remove(&head).expect("just collected");
        if clean_skipped {
            for interior in &info.nodes_to_skip {
                skipped.remove(interior);
            }
        }
    }
}

/// True iff any of the pattern's input tensor names corresponds to a
/// static value (initializer or Constant) whose dtype is not `"float32"`.
/// Inputs not present in the static-dtype map (i.e. runtime values) are
/// ignored — their dtype isn't known at IR-pass time, so they don't
/// influence the gate.
fn pattern_touches_non_f32_static(
    pattern: &FusedPattern,
    static_dtypes: &HashMap<String, &'static str>,
) -> bool {
    let inputs = pattern_input_names(pattern);
    inputs.iter().any(|name| {
        static_dtypes
            .get(*name)
            .map(|dt| *dt != "float32")
            .unwrap_or(false)
    })
}

/// All tensor input names referenced by a fused pattern. Output names
/// are excluded — only inputs participate in the dtype gate.
fn pattern_input_names(pattern: &FusedPattern) -> Vec<&str> {
    match pattern {
        FusedPattern::DivRsqrt {
            numerator_input,
            variance_input,
            eps_input,
            ..
        } => vec![numerator_input, variance_input, eps_input],
        FusedPattern::AddMulAdd {
            x_input,
            a_input,
            b_input,
            c_input,
            ..
        } => vec![x_input, a_input, b_input, c_input],
        FusedPattern::Gelu { x_input, .. } => vec![x_input],
        FusedPattern::MulAdd {
            a_input,
            b_input,
            c_input,
            ..
        } => vec![a_input, b_input, c_input],
        FusedPattern::AddMul {
            a_input,
            b_input,
            c_input,
            ..
        } => vec![a_input, b_input, c_input],
        FusedPattern::SubMul {
            a_input,
            b_input,
            c_input,
            ..
        } => vec![a_input, b_input, c_input],
        FusedPattern::DivMul {
            a_input,
            b_input,
            c_input,
            ..
        } => vec![a_input, b_input, c_input],
        FusedPattern::MulSinPowMulAdd {
            x_input,
            w0_input,
            w1_input,
            b_input,
            p_input,
            ..
        } => vec![x_input, w0_input, w1_input, b_input, p_input],
        FusedPattern::GeneralChain { input_name, .. } => vec![input_name],
    }
    .into_iter()
    .map(String::as_str)
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attributes::NodeAttributes;
    use crate::cuda::inference::fusion::FusedPattern;
    use crate::ir::graph::OptimizableGraphBuilder;
    use crate::tensor::Tensor;

    #[test]
    fn detect_returns_empty_on_empty_graph() {
        let g = OptimizableGraphBuilder::new().build();
        let a = detect_fusion(&g);
        assert!(a.heads.is_empty());
        assert!(a.skipped.is_empty());
        assert!(a.dynamic_candidates.is_empty());
    }

    // Pattern-specific detection tests live alongside the existing fusion
    // detection tests (`tests/fusion_detection_test.rs`) — those
    // still run against the now-shared `detect_with_scalar_lookup` core.
    //
    // In addition, the test below exercises the *CPU-path* scalar-lookup
    // closure used by `detect_fused_patterns_from_cpu`. It verifies the
    // adapter correctly reads a `Tensor::Float32` initializer as a
    // scalar (the branch the GPU-side path covers via `to_host_f32`).
    // Without this test the Float32-initializer branch of the closure is
    // not exercised in a pure-CPU unit test.
    #[test]
    fn detect_finds_gelu_with_float32_initializer_exponent() {
        // GELU pattern:
        //   pow    = Pow(x, 3)
        //   mul1   = Mul(pow, 0.044715)
        //   add1   = Add(x, mul1)          # inner sum
        //   mul2   = Mul(add1, sqrt(2/pi)) # tanh pre-scale
        //   tanh_  = Tanh(mul2)
        //   add2   = Add(tanh_, 1)
        //   half   = Mul(x, 0.5)           # parallel path
        //   y      = Mul(half, add2)       # final Mul — tail
        //
        // The walker keys on the `Pow` node, then follows outputs.
        // `exp3` is the only input that needs to be read as a scalar; all
        // the other weights need only be present (their values aren't
        // inspected here).
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("x".into(), vec![]);
        b.add_initializer(
            "exp3".into(),
            Tensor::from_vec_f32(vec![3.0], vec![1]),
        );
        b.add_initializer(
            "c_inner".into(),
            Tensor::from_vec_f32(vec![0.044715], vec![1]),
        );
        b.add_initializer(
            "c_sqrt2pi".into(),
            Tensor::from_vec_f32(vec![0.797_884_6], vec![1]),
        );
        b.add_initializer(
            "c_one".into(),
            Tensor::from_vec_f32(vec![1.0], vec![1]),
        );
        b.add_initializer(
            "c_half".into(),
            Tensor::from_vec_f32(vec![0.5], vec![1]),
        );

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
        b.add_node(
            "half",
            "Mul",
            vec!["x".into(), "c_half".into()],
            vec!["t_half".into()],
            NodeAttributes::new(),
        );
        b.add_node(
            "tail",
            "Mul",
            vec!["t_half".into(), "t_add2".into()],
            vec!["y".into()],
            NodeAttributes::new(),
        );
        b.add_output("y".into());

        let annotations = detect_fusion(&b.build());

        // The GELU walker keys on the `Pow` node, so "pow" is the head.
        let head = annotations
            .heads
            .get("pow")
            .expect("GELU head registered on Pow node");
        match &head.pattern {
            FusedPattern::Gelu { x_input, output_name } => {
                assert_eq!(x_input, "x");
                assert_eq!(output_name, "y");
            }
            other => panic!("expected FusedPattern::Gelu, got {:?}", other),
        }

        // All interior nodes should be marked for skip.
        for node in ["mul1", "add1", "mul2", "tanh_", "add2", "half", "tail"] {
            assert!(
                annotations.skipped.contains(node),
                "expected '{}' to be in skipped set",
                node
            );
        }
    }
}
