//! GPU-side Kokoro fusion validation test.
//!
//! Loads the full Kokoro v1.0 ONNX model into a `GpuGraphExecutor`, runs
//! inference with profiling, and asserts that the `MulSinPowMulAdd` fused
//! pattern actually fires on the real graph. This is the Cycle 1 Phase E
//! success criterion for the fusion work: detecting and dispatching the
//! pattern is only useful if it actually matches the model's graph.
//!
//! The test is `#[ignore]` by default because it requires:
//!   - The real Kokoro v1.0 ONNX file at `./kokoro-v1.0.onnx` (symlink OK).
//!   - A working CUDA device.
//!   - Several minutes of wall-clock time on a modest GPU.
//!
//! Run manually with:
//!   cargo test --features cuda --test kokoro_gpu_fusion_test \
//!       -- --ignored --nocapture

#![cfg(feature = "cuda")]

use std::collections::HashMap;
use std::path::Path;

use iconnx::cuda::GpuGraphExecutor;
use iconnx::onnx_parser::OnnxParser;
use iconnx::tensor::Tensor;

/// Build the same GPU executor used by the other validation tests, loaded
/// with the full Kokoro graph if the model file is reachable.
fn setup_kokoro_gpu_executor() -> Option<GpuGraphExecutor> {
    let model_path = Path::new("kokoro-v1.0.onnx");
    if !model_path.exists() {
        return None;
    }

    let model = OnnxParser::parse_file(model_path).ok()?;
    let weights = model.extract_weights();
    let graph = model.computation_graph();
    let nodes = graph.nodes();

    let mut executor = GpuGraphExecutor::new().ok()?;

    for (name, tensor) in &weights {
        executor.add_initializer(name.clone(), tensor).ok()?;
    }

    for (idx, node) in nodes.iter().enumerate() {
        let fallback_name = format!("node_{}", idx);
        let node_name = node
            .outputs
            .first()
            .map(|s| s.as_str())
            .unwrap_or(&fallback_name);
        let inputs: Vec<&str> = node.inputs.iter().map(|s| s.as_str()).collect();
        let outputs: Vec<&str> = node.outputs.iter().map(|s| s.as_str()).collect();
        executor.add_node(
            node_name,
            &node.op_type,
            inputs,
            outputs,
            node.attributes.clone(),
        );
    }

    Some(executor)
}

/// Dummy inputs that match the Kokoro input schema: short token sequence,
/// fixed-size style embedding, scalar speed.
fn kokoro_inputs(seq_len: usize) -> HashMap<String, Tensor> {
    let tokens: Vec<i64> = (0..seq_len).map(|i| (i as i64) % 50).collect();
    let style: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01).sin()).collect();
    let speed = vec![1.0f32];

    let mut inputs = HashMap::new();
    inputs.insert(
        "tokens".to_string(),
        Tensor::from_vec_i64(tokens, vec![1, seq_len]),
    );
    inputs.insert(
        "style".to_string(),
        Tensor::from_vec_f32(style, vec![1, 256]),
    );
    inputs.insert(
        "speed".to_string(),
        Tensor::from_vec_f32(speed, vec![1]),
    );

    inputs
}

/// Headline Phase E success criterion: the `MulSinPowMulAdd` pattern must
/// actually fire on the real Kokoro graph, not just on our synthetic unit
/// tests. Expected occurrences (per the Leadline bench findings doc):
/// ~48 per forward pass, corresponding to the vocoder's periodic
/// activation in each residual block.
#[test]
#[ignore] // Requires full Kokoro model + CUDA device; several minutes runtime
fn fusion_fires_on_full_kokoro_at_seq_len_5() {
    let executor = match setup_kokoro_gpu_executor() {
        Some(e) => e,
        None => {
            println!("⚠️  Kokoro model not found at ./kokoro-v1.0.onnx — skipping");
            return;
        }
    };

    let inputs = kokoro_inputs(5);
    let (_outputs, profile) = executor
        .run_with_profiling(inputs, vec!["audio"])
        .expect("run_with_profiling failed");

    // Print the full op_times breakdown so the operator can see what fired.
    eprintln!("\n=== Kokoro seq_len=5 profile (iconnx) ===");
    for (op, (total_us, count)) in &profile.op_times {
        eprintln!("  {:35} {:>8} us ({:>5} calls)", op, total_us, count);
    }

    // Headline check: Fused_MulSinPowMulAdd must appear in the profile.
    let fused_entry = profile.op_times.get("Fused_MulSinPowMulAdd");
    assert!(
        fused_entry.is_some(),
        "Fused_MulSinPowMulAdd did NOT fire on the real Kokoro graph. \
         The detection walker recognized something in synthetic tests \
         but not in the real model. Either the pattern shape in Kokoro \
         differs from the walker's assumption, or an upstream walker \
         claimed the constituent nodes first. Observed profile keys: {:?}",
        profile.op_times.keys().collect::<Vec<_>>()
    );

    let (_fused_us, fused_calls) = fused_entry.unwrap();

    // Expected ~48 from the Leadline analysis. Allow ±10 because some
    // instances may legitimately not match (e.g. an intermediate has a
    // non-single-use consumer in certain branches of the graph).
    assert!(
        (38..=58).contains(fused_calls),
        "Fused_MulSinPowMulAdd fired {} times, expected ~48 (within ±10). \
         If the count is much lower, detection is missing real instances; \
         if much higher, the walker is over-matching and the fused kernel \
         may be running on graphs it shouldn't.",
        fused_calls
    );

    eprintln!(
        "✅ Fused_MulSinPowMulAdd fired {} times (expected ~48)",
        fused_calls
    );
}
