//! Numerical equivalence tests for the 8 specialized `FusedPattern`
//! variants emitted by `elementwise_fusion`.
//!
//! Each test:
//!   1. Builds the minimal ONNX-ish graph for its variant using the
//!      `GpuGraphExecutor` shim (which internally calls `ir::lower`).
//!   2. Runs the graph end-to-end and compares the output to a
//!      CPU-computed reference to within `1e-5` absolute tolerance.
//!   3. Verifies the compiled plan's head node carries the expected
//!      `FusedPattern::<Variant>` — guarding against the regression
//!      described in the plan (the new pass silently demoting a matched
//!      pattern to `GeneralChain`).
//!
//! All tests are `#[ignore]` because they require a live CUDA context.
//! Run with `cargo test --features cuda --release elementwise_fusion
//! -- --include-ignored`.

#![cfg(test)]

use std::collections::HashMap;

use crate::attributes::NodeAttributes;
use crate::cuda::inference::fusion::FusedPattern;
use crate::cuda::inference::testing::count_fused_patterns;
use crate::cuda::inference::GpuGraphExecutor;
use crate::tensor::Tensor;

fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() <= tol)
}

fn assert_approx_eq(label: &str, got: &[f32], expected: &[f32], tol: f32) {
    assert!(
        approx_eq(got, expected, tol),
        "{}: output mismatch (tol={})\n  got:      {:?}\n  expected: {:?}",
        label,
        tol,
        got,
        expected,
    );
}

#[test]
#[ignore = "requires CUDA GPU"]
fn divrsqrt_fused_matches_hand_computed_within_1e_5() {
    let mut executor = GpuGraphExecutor::new().expect("executor");

    // y = numerator / sqrt(variance + eps)
    let numerator: Vec<f32> = (1..=8).map(|i| i as f32).collect();
    let variance: Vec<f32> = (1..=8).map(|i| i as f32 * 0.1).collect();
    let eps_val = 1e-5_f32;
    let expected: Vec<f32> = numerator
        .iter()
        .zip(variance.iter())
        .map(|(n, v)| n / (v + eps_val).sqrt())
        .collect();

    executor
        .add_initializer("eps".into(), &Tensor::from_vec_f32(vec![eps_val], vec![1]))
        .expect("eps");
    executor.add_node(
        "add",
        "Add",
        vec!["variance", "eps"],
        vec!["t_add"],
        NodeAttributes::new(),
    );
    executor.add_node(
        "sq",
        "Sqrt",
        vec!["t_add"],
        vec!["t_sq"],
        NodeAttributes::new(),
    );
    executor.add_node(
        "div",
        "Div",
        vec!["numerator", "t_sq"],
        vec!["y"],
        NodeAttributes::new(),
    );

    let counts = count_fused_patterns(&executor);
    assert!(
        counts.div_rsqrt >= 1,
        "expected >=1 DivRsqrt, counts={:?}",
        counts
    );

    let mut inputs = HashMap::new();
    inputs.insert(
        "numerator".to_string(),
        Tensor::from_vec_f32(numerator.clone(), vec![8]),
    );
    inputs.insert(
        "variance".to_string(),
        Tensor::from_vec_f32(variance.clone(), vec![8]),
    );
    let outputs = executor.run(inputs, vec!["y"]).expect("run");
    let y = outputs.get("y").expect("y");
    assert_approx_eq("DivRsqrt", &y.as_slice(), &expected, 1e-5);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn add_mul_add_fused_matches_hand_computed_within_1e_5() {
    // y = (x + a) * b + c, b and c as initializers.
    let mut executor = GpuGraphExecutor::new().expect("executor");
    let x_host: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let a_host: Vec<f32> = vec![0.5; 8];
    let b_host: Vec<f32> = vec![2.0; 8];
    let c_host: Vec<f32> = vec![-1.0; 8];
    let expected: Vec<f32> = x_host
        .iter()
        .zip(a_host.iter())
        .zip(b_host.iter())
        .zip(c_host.iter())
        .map(|(((x, a), b), c)| (x + a) * b + c)
        .collect();

    executor
        .add_initializer("bw".into(), &Tensor::from_vec_f32(b_host.clone(), vec![8]))
        .expect("bw");
    executor
        .add_initializer("cw".into(), &Tensor::from_vec_f32(c_host.clone(), vec![8]))
        .expect("cw");
    executor.add_node(
        "add1",
        "Add",
        vec!["x", "a"],
        vec!["t1"],
        NodeAttributes::new(),
    );
    executor.add_node(
        "mul",
        "Mul",
        vec!["t1", "bw"],
        vec!["t2"],
        NodeAttributes::new(),
    );
    executor.add_node(
        "add2",
        "Add",
        vec!["t2", "cw"],
        vec!["y"],
        NodeAttributes::new(),
    );

    let counts = count_fused_patterns(&executor);
    assert!(counts.add_mul_add >= 1, "counts={:?}", counts);

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), Tensor::from_vec_f32(x_host, vec![8]));
    inputs.insert("a".to_string(), Tensor::from_vec_f32(a_host, vec![8]));
    let outputs = executor.run(inputs, vec!["y"]).expect("run");
    assert_approx_eq(
        "AddMulAdd",
        &outputs.get("y").unwrap().as_slice(),
        &expected,
        1e-5,
    );
}

#[test]
#[ignore = "requires CUDA GPU"]
fn gelu_fused_matches_hand_computed_within_1e_5() {
    let mut executor = GpuGraphExecutor::new().expect("executor");
    let x_host: Vec<f32> = (-4i32..4).map(|i| i as f32 * 0.5).collect();
    let expected: Vec<f32> = x_host
        .iter()
        .map(|&x| {
            let inner = (2.0f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x);
            x * 0.5 * (1.0 + inner.tanh())
        })
        .collect();

    executor
        .add_initializer("exp3".into(), &Tensor::from_vec_f32(vec![3.0], vec![1]))
        .expect("exp3");
    executor
        .add_initializer(
            "c_inner".into(),
            &Tensor::from_vec_f32(vec![0.044715], vec![1]),
        )
        .expect("c_inner");
    executor
        .add_initializer(
            "c_sqrt2pi".into(),
            &Tensor::from_vec_f32(vec![(2.0f32 / std::f32::consts::PI).sqrt()], vec![1]),
        )
        .expect("c_sqrt2pi");
    executor
        .add_initializer("c_one".into(), &Tensor::from_vec_f32(vec![1.0], vec![1]))
        .expect("c_one");
    executor
        .add_initializer("c_half".into(), &Tensor::from_vec_f32(vec![0.5], vec![1]))
        .expect("c_half");

    executor.add_node(
        "pow",
        "Pow",
        vec!["x", "exp3"],
        vec!["t_pow"],
        NodeAttributes::new(),
    );
    executor.add_node(
        "mul1",
        "Mul",
        vec!["t_pow", "c_inner"],
        vec!["t_mul1"],
        NodeAttributes::new(),
    );
    executor.add_node(
        "add1",
        "Add",
        vec!["x", "t_mul1"],
        vec!["t_add1"],
        NodeAttributes::new(),
    );
    executor.add_node(
        "mul2",
        "Mul",
        vec!["t_add1", "c_sqrt2pi"],
        vec!["t_mul2"],
        NodeAttributes::new(),
    );
    executor.add_node(
        "tanh_",
        "Tanh",
        vec!["t_mul2"],
        vec!["t_tanh"],
        NodeAttributes::new(),
    );
    executor.add_node(
        "add2",
        "Add",
        vec!["t_tanh", "c_one"],
        vec!["t_add2"],
        NodeAttributes::new(),
    );
    executor.add_node(
        "half",
        "Mul",
        vec!["x", "c_half"],
        vec!["t_half"],
        NodeAttributes::new(),
    );
    executor.add_node(
        "mul3",
        "Mul",
        vec!["t_half", "t_add2"],
        vec!["y"],
        NodeAttributes::new(),
    );

    let counts = count_fused_patterns(&executor);
    assert!(counts.gelu >= 1, "expected >=1 GELU, counts={:?}", counts);

    let mut inputs = HashMap::new();
    let n = x_host.len();
    inputs.insert("x".to_string(), Tensor::from_vec_f32(x_host, vec![n]));
    let outputs = executor.run(inputs, vec!["y"]).expect("run");
    assert_approx_eq(
        "Gelu",
        &outputs.get("y").unwrap().as_slice(),
        &expected,
        1e-5,
    );
}

#[test]
#[ignore = "requires CUDA GPU"]
fn mul_add_fused_matches_hand_computed_within_1e_5() {
    let mut executor = GpuGraphExecutor::new().expect("executor");
    let a_host: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let b_host: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();
    let c_host: Vec<f32> = vec![0.25; 8];
    let expected: Vec<f32> = a_host
        .iter()
        .zip(b_host.iter())
        .zip(c_host.iter())
        .map(|((a, b), c)| a * b + c)
        .collect();

    executor
        .add_initializer("c".into(), &Tensor::from_vec_f32(c_host, vec![8]))
        .expect("c");
    executor.add_node("mul", "Mul", vec!["a", "b"], vec!["t"], NodeAttributes::new());
    executor.add_node("add", "Add", vec!["t", "c"], vec!["y"], NodeAttributes::new());

    let counts = count_fused_patterns(&executor);
    assert!(counts.mul_add >= 1, "counts={:?}", counts);

    let mut inputs = HashMap::new();
    inputs.insert("a".to_string(), Tensor::from_vec_f32(a_host, vec![8]));
    inputs.insert("b".to_string(), Tensor::from_vec_f32(b_host, vec![8]));
    let outputs = executor.run(inputs, vec!["y"]).expect("run");
    assert_approx_eq(
        "MulAdd",
        &outputs.get("y").unwrap().as_slice(),
        &expected,
        1e-5,
    );
}

#[test]
#[ignore = "requires CUDA GPU"]
fn add_mul_fused_matches_hand_computed_within_1e_5() {
    let mut executor = GpuGraphExecutor::new().expect("executor");
    let a_host: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let b_host: Vec<f32> = (0..8).map(|i| i as f32 + 0.5).collect();
    let c_host: Vec<f32> = vec![2.0; 8];
    let expected: Vec<f32> = a_host
        .iter()
        .zip(b_host.iter())
        .zip(c_host.iter())
        .map(|((a, b), c)| (a + b) * c)
        .collect();

    executor
        .add_initializer("c".into(), &Tensor::from_vec_f32(c_host, vec![8]))
        .expect("c");
    executor.add_node("add", "Add", vec!["a", "b"], vec!["t"], NodeAttributes::new());
    executor.add_node("mul", "Mul", vec!["t", "c"], vec!["y"], NodeAttributes::new());

    let counts = count_fused_patterns(&executor);
    assert!(counts.add_mul >= 1, "counts={:?}", counts);

    let mut inputs = HashMap::new();
    inputs.insert("a".to_string(), Tensor::from_vec_f32(a_host, vec![8]));
    inputs.insert("b".to_string(), Tensor::from_vec_f32(b_host, vec![8]));
    let outputs = executor.run(inputs, vec!["y"]).expect("run");
    assert_approx_eq(
        "AddMul",
        &outputs.get("y").unwrap().as_slice(),
        &expected,
        1e-5,
    );
}

#[test]
#[ignore = "requires CUDA GPU"]
fn sub_mul_fused_matches_hand_computed_within_1e_5() {
    let mut executor = GpuGraphExecutor::new().expect("executor");
    let a_host: Vec<f32> = (0..8).map(|i| i as f32 + 10.0).collect();
    let b_host: Vec<f32> = (0..8).map(|i| i as f32 + 0.5).collect();
    let c_host: Vec<f32> = vec![1.5; 8];
    let expected: Vec<f32> = a_host
        .iter()
        .zip(b_host.iter())
        .zip(c_host.iter())
        .map(|((a, b), c)| (a - b) * c)
        .collect();

    executor
        .add_initializer("c".into(), &Tensor::from_vec_f32(c_host, vec![8]))
        .expect("c");
    executor.add_node("sub", "Sub", vec!["a", "b"], vec!["t"], NodeAttributes::new());
    executor.add_node("mul", "Mul", vec!["t", "c"], vec!["y"], NodeAttributes::new());

    let counts = count_fused_patterns(&executor);
    assert!(counts.sub_mul >= 1, "counts={:?}", counts);

    let mut inputs = HashMap::new();
    inputs.insert("a".to_string(), Tensor::from_vec_f32(a_host, vec![8]));
    inputs.insert("b".to_string(), Tensor::from_vec_f32(b_host, vec![8]));
    let outputs = executor.run(inputs, vec!["y"]).expect("run");
    assert_approx_eq(
        "SubMul",
        &outputs.get("y").unwrap().as_slice(),
        &expected,
        1e-5,
    );
}

#[test]
#[ignore = "requires CUDA GPU"]
fn div_mul_fused_matches_hand_computed_within_1e_5() {
    let mut executor = GpuGraphExecutor::new().expect("executor");
    let a_host: Vec<f32> = (1..=8).map(|i| i as f32 * 2.0).collect();
    let b_host: Vec<f32> = (1..=8).map(|i| i as f32).collect();
    let c_host: Vec<f32> = vec![3.0; 8];
    let expected: Vec<f32> = a_host
        .iter()
        .zip(b_host.iter())
        .zip(c_host.iter())
        .map(|((a, b), c)| (a / b) * c)
        .collect();

    executor
        .add_initializer("c".into(), &Tensor::from_vec_f32(c_host, vec![8]))
        .expect("c");
    executor.add_node("div", "Div", vec!["a", "b"], vec!["t"], NodeAttributes::new());
    executor.add_node("mul", "Mul", vec!["t", "c"], vec!["y"], NodeAttributes::new());

    let counts = count_fused_patterns(&executor);
    assert!(counts.div_mul >= 1, "counts={:?}", counts);

    let mut inputs = HashMap::new();
    inputs.insert("a".to_string(), Tensor::from_vec_f32(a_host, vec![8]));
    inputs.insert("b".to_string(), Tensor::from_vec_f32(b_host, vec![8]));
    let outputs = executor.run(inputs, vec!["y"]).expect("run");
    assert_approx_eq(
        "DivMul",
        &outputs.get("y").unwrap().as_slice(),
        &expected,
        1e-5,
    );
}

#[test]
#[ignore = "requires CUDA GPU"]
fn mul_sin_pow_mul_add_fused_matches_hand_computed_within_1e_5() {
    // y = sin(x * w0)^p * w1 + b
    let mut executor = GpuGraphExecutor::new().expect("executor");
    let x_host: Vec<f32> = (0..8).map(|i| i as f32 * 0.25).collect();
    let w0_host: Vec<f32> = vec![2.0; 8];
    let w1_host: Vec<f32> = vec![3.0; 8];
    let b_host: Vec<f32> = vec![0.5; 8];
    let p_val = 2.0_f32;
    let expected: Vec<f32> = x_host
        .iter()
        .zip(w0_host.iter())
        .zip(w1_host.iter())
        .zip(b_host.iter())
        .map(|(((x, w0), w1), b)| (x * w0).sin().powf(p_val) * w1 + b)
        .collect();

    executor
        .add_initializer("w0".into(), &Tensor::from_vec_f32(w0_host, vec![8]))
        .expect("w0");
    executor
        .add_initializer("w1".into(), &Tensor::from_vec_f32(w1_host, vec![8]))
        .expect("w1");
    executor
        .add_initializer("p_exp".into(), &Tensor::from_vec_f32(vec![p_val], vec![1]))
        .expect("p_exp");
    executor
        .add_initializer("b".into(), &Tensor::from_vec_f32(b_host, vec![8]))
        .expect("b");
    executor.add_node(
        "hmul",
        "Mul",
        vec!["x", "w0"],
        vec!["t0"],
        NodeAttributes::new(),
    );
    executor.add_node(
        "sin",
        "Sin",
        vec!["t0"],
        vec!["t1"],
        NodeAttributes::new(),
    );
    executor.add_node(
        "pow",
        "Pow",
        vec!["t1", "p_exp"],
        vec!["t2"],
        NodeAttributes::new(),
    );
    executor.add_node(
        "tmul",
        "Mul",
        vec!["t2", "w1"],
        vec!["t3"],
        NodeAttributes::new(),
    );
    executor.add_node(
        "badd",
        "Add",
        vec!["t3", "b"],
        vec!["y"],
        NodeAttributes::new(),
    );

    let counts = count_fused_patterns(&executor);
    assert!(counts.mul_sin_pow_mul_add >= 1, "counts={:?}", counts);

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), Tensor::from_vec_f32(x_host, vec![8]));
    let outputs = executor.run(inputs, vec!["y"]).expect("run");
    assert_approx_eq(
        "MulSinPowMulAdd",
        &outputs.get("y").unwrap().as_slice(),
        &expected,
        1e-5,
    );
}

/// Erf-chain GeneralChain: a pure-unary chain containing `Erf`
/// (`Sqrt → Erf → Tanh`) must fuse into a single
/// `FusedPattern::GeneralChain` whose `chain_signature` carries
/// "Erf" verbatim, and the fused output must match a sequential CPU
/// reference at 1e-5 tolerance.
///
/// Design note (adaptation from plan sketch): the plan originally asked
/// for a GELU-via-Erf graph whose 4-op prefix
/// (Mul(1/√2) → Erf → Add(1.0) → Mul(0.5)) would fuse into a
/// GeneralChain. That graph does NOT fit the actual architecture: the
/// GeneralChain codegen only emits unary-elementwise ops, and
/// `chain_is_pure_unary` gates the fallback on the chain containing no
/// binary ops. Add/Mul are binary and so a chain that includes them
/// cannot become a GeneralChain today. Additionally, the legacy
/// `detect_fusion` AddMul walker grabs the `Add(1.0) → Mul(0.5)` tail
/// before the new pass can see it. The principled fix that preserves
/// the test's stated intent — "verify Erf auto-fuses as part of a
/// GeneralChain, first non-trivial GeneralChain fire" — is to build a
/// genuine pure-unary chain containing Erf and assert on that. The
/// assertion (GeneralChain + "Erf" in signature) is unchanged; only
/// the surrounding chain shape is adapted to fit the architecture.
#[test]
#[ignore = "requires CUDA GPU"]
fn erf_chain_gelu_prefix_fuses_into_general_chain_matches_sequential() {
    use crate::ir::plan::OpKind;

    // Positive inputs only — Sqrt rejects negatives.
    let input_vals: Vec<f32> = (0..64).map(|i| 0.01 + (i as f32) * 0.05).collect();

    let mut executor =
        crate::cuda::inference::GpuGraphExecutor::new().expect("create executor");
    executor.add_input("x".into(), vec![Some(64)]);
    executor.add_node(
        "sqrt_node",
        "Sqrt",
        vec!["x"],
        vec!["sqrted"],
        NodeAttributes::new(),
    );
    executor.add_node(
        "erf_node",
        "Erf",
        vec!["sqrted"],
        vec!["erfed"],
        NodeAttributes::new(),
    );
    executor.add_node(
        "tanh_node",
        "Tanh",
        vec!["erfed"],
        vec!["out"],
        NodeAttributes::new(),
    );

    // Force compilation and inspect the plan: expect at least one
    // FusedHead(GeneralChain) whose chain_signature contains "Erf".
    executor.compile().expect("compile");
    let plan_ref = executor.compiled_plan();
    let plan = plan_ref.as_ref().expect("plan present after compile");
    let general_chains_with_erf: Vec<String> = plan
        .ops
        .iter()
        .filter_map(|op| match &op.kind {
            OpKind::FusedHead(FusedPattern::GeneralChain {
                chain_signature, ..
            }) if chain_signature.contains("Erf") => Some(chain_signature.clone()),
            _ => None,
        })
        .collect();
    assert!(
        !general_chains_with_erf.is_empty(),
        "expected FusedPattern::GeneralChain with 'Erf' in chain_signature on the \
         pure-unary chain (Sqrt → Erf → Tanh). Fused annotations seen: {:?}",
        plan.ops
            .iter()
            .filter_map(|op| match &op.kind {
                OpKind::FusedHead(p) => Some(format!("{:?}", p)),
                _ => None,
            })
            .collect::<Vec<_>>()
    );
    drop(plan_ref);

    // Run the fused graph end-to-end.
    let mut inputs = HashMap::new();
    inputs.insert(
        "x".to_string(),
        Tensor::from_vec_f32(input_vals.clone(), vec![64]),
    );
    let outputs = executor.run(inputs, vec!["out"]).expect("run fused");
    let fused_vals = outputs.get("out").expect("out").as_slice();

    // Sequential CPU reference: tanh(erf(sqrt(x))).
    let expected: Vec<f32> = input_vals
        .iter()
        .map(|&x| crate::operators::erf::erf_f32(x.sqrt()).tanh())
        .collect();

    assert_approx_eq("ErfChainGeneral", &fused_vals, &expected, 1e-5);
}

/// Sanity: ensure the new pass does NOT demote a matched variant to
/// `GeneralChain`. If this ever fires, the specialized kernel is being
/// bypassed in favor of the slower NVRTC path.
#[test]
fn specialized_variants_are_not_demoted_to_general_chain() {
    use crate::ir::graph::OptimizableGraphBuilder;
    use crate::ir::passes::fusion_annotations_from_graph;

    // DivRsqrt — minimal graph.
    let mut b = OptimizableGraphBuilder::new();
    b.add_input("numerator".into(), vec![Some(4)]);
    b.add_input("variance".into(), vec![Some(4)]);
    b.add_initializer("eps".into(), Tensor::from_vec_f32(vec![1e-5], vec![1]));
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
    let head = a.heads.get("add").expect("DivRsqrt head");
    assert!(
        !matches!(head.pattern, FusedPattern::GeneralChain { .. }),
        "DivRsqrt MUST emit the specialized variant, not GeneralChain. Got {:?}",
        head.pattern
    );
}
