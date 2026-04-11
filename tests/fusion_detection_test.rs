//! Tests for the fused-pattern detection walker in
//! `src/cuda/inference/fusion/detection.rs`.
//!
//! These tests construct synthetic `ExecutionNode` lists — no real ONNX
//! model, no real CUDA graph. The detection function is pure (aside
//! from a CUDA context reference used to read constant scalar exponents
//! via `to_host_f32`), so we can unit-test it cheaply.

#![cfg(feature = "cuda")]

use std::collections::HashMap;

use iconnx::attributes::NodeAttributes;
use iconnx::cuda::inference::testing::{
    detect_fused_patterns_for_tests, is_mul_sin_pow_mul_add, ExecutionNodeForTests,
};
use iconnx::cuda::{GpuTensor, IconnxCudaContext};

/// Helper: build a node with the given name, op type, inputs, and outputs
/// (no attributes, no precomputed slice).
fn node(name: &str, op: &str, inputs: &[&str], outputs: &[&str]) -> ExecutionNodeForTests {
    ExecutionNodeForTests {
        name: name.to_string(),
        op_type: op.to_string(),
        inputs: inputs.iter().map(|s| s.to_string()).collect(),
        outputs: outputs.iter().map(|s| s.to_string()).collect(),
        attributes: NodeAttributes::default(),
    }
}

/// Build a dummy weights map containing a constant scalar `p` tensor under
/// the given name.
fn weights_with_scalar(
    ctx: &IconnxCudaContext,
    name: &str,
    value: f32,
) -> HashMap<String, GpuTensor> {
    let mut w = HashMap::new();
    let t = GpuTensor::from_host_f32(ctx, &[value], vec![1]).expect("scalar upload");
    w.insert(name.to_string(), t);
    w
}

#[test]
fn detects_mul_sin_pow_mul_add_chain() {
    let ctx = IconnxCudaContext::new().expect("CUDA context");

    // Build the 5-op chain:
    //   t0 = Mul(x, w0)
    //   t1 = Sin(t0)
    //   t2 = Pow(t1, p)
    //   t3 = Mul(t2, w1)
    //   y  = Add(t3, b)
    let nodes = vec![
        node("head_mul", "Mul", &["x", "w0"], &["t0"]),
        node("sin_node", "Sin", &["t0"], &["t1"]),
        node("pow_node", "Pow", &["t1", "p"], &["t2"]),
        node("tail_mul", "Mul", &["t2", "w1"], &["t3"]),
        node("bias_add", "Add", &["t3", "b"], &["y"]),
    ];

    // p must be a constant scalar weight for the pattern to be eligible.
    let weights = weights_with_scalar(&ctx, "p", 2.0);

    let (patterns, skip_set) = detect_fused_patterns_for_tests(&nodes, &weights, &ctx);

    assert!(
        patterns.contains_key("head_mul"),
        "head Mul was not registered as a pattern head. Registered: {:?}",
        patterns.keys().collect::<Vec<_>>()
    );
    let info = patterns.get("head_mul").unwrap();
    assert!(
        is_mul_sin_pow_mul_add(info),
        "head_mul was registered but not as MulSinPowMulAdd pattern"
    );
    assert!(skip_set.contains("sin_node"));
    assert!(skip_set.contains("pow_node"));
    assert!(skip_set.contains("tail_mul"));
    assert!(skip_set.contains("bias_add"));
}

#[test]
fn rejects_when_intermediate_has_multiple_consumers() {
    let ctx = IconnxCudaContext::new().expect("CUDA context");

    // Same chain but t2 is consumed by a second downstream node as well,
    // so the fusion is not safe.
    let nodes = vec![
        node("head_mul", "Mul", &["x", "w0"], &["t0"]),
        node("sin_node", "Sin", &["t0"], &["t1"]),
        node("pow_node", "Pow", &["t1", "p"], &["t2"]),
        node("tail_mul", "Mul", &["t2", "w1"], &["t3"]),
        node("bias_add", "Add", &["t3", "b"], &["y"]),
        // Second consumer of t2 — breaks single-use invariant
        node("other_mul", "Mul", &["t2", "q"], &["y2"]),
    ];

    let weights = weights_with_scalar(&ctx, "p", 2.0);

    let (patterns, _skip_set) = detect_fused_patterns_for_tests(&nodes, &weights, &ctx);

    // The MulSinPowMulAdd pattern must NOT be registered against head_mul.
    assert!(
        !patterns
            .get("head_mul")
            .map(is_mul_sin_pow_mul_add)
            .unwrap_or(false),
        "MulSinPowMulAdd was registered despite multi-use intermediate"
    );
}
