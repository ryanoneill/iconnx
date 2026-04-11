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
    detect_fused_patterns_for_tests, is_mul_sin_pow_mul_add, ExecutionNodeForTests, FusedPattern,
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

/// Build a weights map with `p` as a scalar integer exponent AND
/// `w0`, `w1`, `b` as full-shape weight tensors. This mirrors the
/// realistic shape of the Kokoro vocoder's periodic activation.
fn weights_for_realistic_chain(
    ctx: &IconnxCudaContext,
    channels: usize,
) -> HashMap<String, GpuTensor> {
    let mut w = HashMap::new();
    w.insert(
        "p".to_string(),
        GpuTensor::from_host_f32(ctx, &[2.0], vec![1]).expect("p upload"),
    );
    w.insert(
        "w0".to_string(),
        GpuTensor::from_host_f32(ctx, &vec![0.5; channels], vec![1, channels])
            .expect("w0 upload"),
    );
    w.insert(
        "w1".to_string(),
        GpuTensor::from_host_f32(ctx, &vec![1.0; channels], vec![1, channels])
            .expect("w1 upload"),
    );
    w.insert(
        "b".to_string(),
        GpuTensor::from_host_f32(ctx, &vec![0.0; channels], vec![1, channels])
            .expect("b upload"),
    );
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

    // Verify the registered pattern's fields match the graph structure.
    match &info.pattern {
        FusedPattern::MulSinPowMulAdd {
            x_input,
            w0_input,
            w1_input,
            b_input,
            p_input,
            output_name,
        } => {
            assert_eq!(x_input, "x");
            assert_eq!(w0_input, "w0");
            assert_eq!(w1_input, "w1");
            assert_eq!(b_input, "b");
            assert_eq!(p_input, "p");
            assert_eq!(output_name, "y");
        }
        other => panic!("expected MulSinPowMulAdd, got {:?}", other),
    }
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

#[test]
fn detects_chain_when_all_operands_are_weights() {
    let ctx = IconnxCudaContext::new().expect("CUDA context");

    // Same 5-op chain as the basic test, but now w0, w1, b are ALL weights
    // (matching the realistic Kokoro vocoder shape). The MulAdd walker
    // must NOT preempt the MulSinPowMulAdd walker on tail_mul + bias_add.
    let nodes = vec![
        node("head_mul", "Mul", &["x", "w0"], &["t0"]),
        node("sin_node", "Sin", &["t0"], &["t1"]),
        node("pow_node", "Pow", &["t1", "p"], &["t2"]),
        node("tail_mul", "Mul", &["t2", "w1"], &["t3"]),
        node("bias_add", "Add", &["t3", "b"], &["y"]),
    ];

    let weights = weights_for_realistic_chain(&ctx, 8);

    let (patterns, skip_set) = detect_fused_patterns_for_tests(&nodes, &weights, &ctx);

    // Must be registered as MulSinPowMulAdd specifically (not MulAdd!).
    let info = patterns.get("head_mul").unwrap_or_else(|| {
        panic!(
            "head_mul not registered. Registered patterns: {:?}",
            patterns.keys().collect::<Vec<_>>()
        )
    });
    assert!(
        is_mul_sin_pow_mul_add(info),
        "head_mul was registered but not as MulSinPowMulAdd. \
         This means a shorter pattern (likely MulAdd) claimed tail_mul + bias_add \
         before MulSinPowMulAdd had a chance to run. Fix walker ordering in \
         detect_fused_patterns.",
    );
    assert!(skip_set.contains("sin_node"));
    assert!(skip_set.contains("pow_node"));
    assert!(skip_set.contains("tail_mul"));
    assert!(skip_set.contains("bias_add"));

    // No rogue MulAdd pattern should have claimed tail_mul.
    assert!(
        !patterns.contains_key("tail_mul"),
        "tail_mul should not be registered as its own pattern head; \
         it's owned by MulSinPowMulAdd"
    );
}

#[test]
fn rejects_non_integer_exponent() {
    let ctx = IconnxCudaContext::new().expect("CUDA context");

    // Same chain as the basic test but p = 2.5 (non-integer).
    // The fused kernel's integer squaring loop is the only supported
    // exponent path, so the walker must refuse to register the pattern.
    let nodes = vec![
        node("head_mul", "Mul", &["x", "w0"], &["t0"]),
        node("sin_node", "Sin", &["t0"], &["t1"]),
        node("pow_node", "Pow", &["t1", "p"], &["t2"]),
        node("tail_mul", "Mul", &["t2", "w1"], &["t3"]),
        node("bias_add", "Add", &["t3", "b"], &["y"]),
    ];

    let mut weights = HashMap::new();
    weights.insert(
        "p".to_string(),
        GpuTensor::from_host_f32(&ctx, &[2.5], vec![1]).expect("p upload"),
    );

    let (patterns, _skip_set) = detect_fused_patterns_for_tests(&nodes, &weights, &ctx);

    assert!(
        !patterns
            .get("head_mul")
            .map(is_mul_sin_pow_mul_add)
            .unwrap_or(false),
        "Non-integer exponent p=2.5 must not be detected as MulSinPowMulAdd"
    );
}

#[test]
fn executor_runs_mul_sin_pow_mul_add_and_records_profile_entry() {
    use iconnx::cuda::GpuGraphExecutor;
    use iconnx::tensor::Tensor;
    use iconnx::NodeAttributes;

    // Minimal graph: the same 5-op chain, with x as a runtime input and
    // w0, w1, b, p registered as model initializers.
    let mut executor = GpuGraphExecutor::new().expect("executor");

    let shape = vec![2usize, 4usize];
    let n = 2 * 4;
    let w0_data: Vec<f32> = (0..n).map(|i| 0.5 + 0.05 * i as f32).collect();
    let w1_data: Vec<f32> = (0..n).map(|i| 1.0 - 0.03 * i as f32).collect();
    let b_data: Vec<f32> = (0..n).map(|i| 0.1 * i as f32).collect();

    executor
        .add_initializer(
            "w0".to_string(),
            &Tensor::from_vec_f32(w0_data.clone(), shape.clone()),
        )
        .expect("add w0");
    executor
        .add_initializer(
            "w1".to_string(),
            &Tensor::from_vec_f32(w1_data.clone(), shape.clone()),
        )
        .expect("add w1");
    executor
        .add_initializer(
            "b".to_string(),
            &Tensor::from_vec_f32(b_data.clone(), shape.clone()),
        )
        .expect("add b");
    executor
        .add_initializer(
            "p".to_string(),
            &Tensor::from_vec_f32(vec![2.0], vec![1]),
        )
        .expect("add p");

    executor.add_node(
        "head_mul",
        "Mul",
        vec!["x", "w0"],
        vec!["t0"],
        NodeAttributes::default(),
    );
    executor.add_node(
        "sin_node",
        "Sin",
        vec!["t0"],
        vec!["t1"],
        NodeAttributes::default(),
    );
    executor.add_node(
        "pow_node",
        "Pow",
        vec!["t1", "p"],
        vec!["t2"],
        NodeAttributes::default(),
    );
    executor.add_node(
        "tail_mul",
        "Mul",
        vec!["t2", "w1"],
        vec!["t3"],
        NodeAttributes::default(),
    );
    executor.add_node(
        "bias_add",
        "Add",
        vec!["t3", "b"],
        vec!["y"],
        NodeAttributes::default(),
    );

    let x_data: Vec<f32> = (0..n).map(|i| 0.01 * i as f32 + 0.2).collect();
    let mut inputs = std::collections::HashMap::new();
    inputs.insert(
        "x".to_string(),
        Tensor::from_vec_f32(x_data.clone(), shape.clone()),
    );

    let (outputs, profile) = executor
        .run_with_profiling(inputs, vec!["y"])
        .expect("run");

    // The profile must contain the new fused pattern entry.
    let entry = profile
        .op_times
        .get("Fused_MulSinPowMulAdd")
        .unwrap_or_else(|| {
            panic!(
                "Fused_MulSinPowMulAdd missing from profile. Got: {:?}",
                profile.op_times.keys().collect::<Vec<_>>()
            )
        });
    assert_eq!(
        entry.1, 1,
        "expected exactly one fused-pattern execution, got {}",
        entry.1
    );

    // Intermediate ops that the pattern consumed should be absent from the
    // profile (they were skipped during execution).
    for skipped in ["Mul", "Sin", "Pow", "Add"] {
        assert!(
            !profile.op_times.contains_key(skipped),
            "{skipped} should have been skipped by fusion, but appears in profile: {:?}",
            profile.op_times.keys().collect::<Vec<_>>()
        );
    }

    // Numerical check: compute the same thing with plain Rust and compare.
    let y_tensor = outputs
        .get("y")
        .expect("output y missing")
        .as_slice()
        .to_vec();
    let mut expected = vec![0.0f32; n];
    for i in 0..n {
        let s = (x_data[i] * w0_data[i]).sin();
        expected[i] = s * s * w1_data[i] + b_data[i];
    }
    for i in 0..n {
        let diff = (y_tensor[i] - expected[i]).abs();
        assert!(
            diff <= 1e-5,
            "y[{i}] = {}, expected {}, diff {}",
            y_tensor[i],
            expected[i],
            diff
        );
    }
}
