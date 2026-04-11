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
    inputs.insert("speed".to_string(), Tensor::from_vec_f32(speed, vec![1]));

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

    // === GPU-side timing assertions (Cycle 1.5a) ===

    // The new gpu_op_times map must be populated whenever
    // run_with_profiling is called.
    assert!(
        !profile.gpu_op_times.is_empty(),
        "gpu_op_times must be populated when run_with_profiling is called. \
         Got empty map; check that GpuEventTimer is wired into the execution loop."
    );

    // Key parity invariant: the two maps must have the exact same key set.
    // Divergence breaks Leadline's CPU/GPU join silently.
    let cpu_keys: std::collections::HashSet<&String> = profile.op_times.keys().collect();
    let gpu_keys: std::collections::HashSet<&String> = profile.gpu_op_times.keys().collect();
    assert_eq!(
        cpu_keys,
        gpu_keys,
        "op_times and gpu_op_times keys must match exactly. \
         cpu_only: {:?}, gpu_only: {:?}",
        cpu_keys.difference(&gpu_keys).collect::<Vec<_>>(),
        gpu_keys.difference(&cpu_keys).collect::<Vec<_>>()
    );

    // Count parity: every key's call count must match between the two maps.
    // Note: there is no physical invariant between cpu_us and gpu_us because
    // op_times measures CPU-side async enqueue time (launch overhead only,
    // not wall-clock), while gpu_op_times measures actual GPU kernel
    // execution time via CUDA events. These are orthogonal measurements --
    // gpu_us > cpu_us is the normal case for non-trivial ops (kernels run
    // longer than their dispatch cost). So we only assert call-count parity
    // here, not any magnitude relationship.
    for (key, (_cpu_us, cpu_count)) in &profile.op_times {
        let (_gpu_us, gpu_count) = profile
            .gpu_op_times
            .get(key)
            .copied()
            .expect("key in op_times must also be in gpu_op_times (checked above)");
        assert_eq!(
            *cpu_count, gpu_count,
            "call count for '{}' diverges: op_times={}, gpu_op_times={}",
            key, cpu_count, gpu_count
        );
    }

    // Diagnostic printout: top 10 ops by GPU kernel time, showing CPU
    // enqueue overhead alongside GPU kernel wall time. These are orthogonal
    // measurements:
    //   cpu_us -- CPU-side async dispatch/enqueue time for this op
    //              (pure launch overhead, measured via Instant::now()
    //              around the dispatch call; does NOT include kernel
    //              execution because dispatch is fire-and-forget async)
    //   gpu_us -- GPU-side kernel wall time from CUDA events (the time
    //              the kernel spent actually executing on the GPU)
    //
    // This is the core Leadline diagnostic: high cpu_us relative to gpu_us
    // means the op is enqueue-bound (fusion or CUDA graph capture would
    // help reduce launches); low cpu_us relative to gpu_us means the op
    // is compute-bound (kernel micro-optimization would help more).
    let mut rows: Vec<(String, u64, u64, usize)> = profile
        .gpu_op_times
        .iter()
        .map(|(k, (gpu, count))| {
            let cpu = profile.op_times.get(k).map(|(t, _)| *t).unwrap_or(0);
            (k.clone(), cpu, *gpu, *count)
        })
        .collect();
    // Sort by gpu_us descending so the biggest kernel-time consumers
    // appear first.
    rows.sort_by_key(|(_, _, gpu, _)| std::cmp::Reverse(*gpu));

    eprintln!("\n=== Top 10 ops by GPU kernel time (us) ===");
    eprintln!(
        "  {:35} {:>12} {:>12} {:>6}",
        "op", "cpu_enqueue", "gpu_kernel", "calls"
    );
    for (k, cpu, gpu, count) in rows.iter().take(10) {
        eprintln!("  {:35} {:>12} {:>12} {:>6}", k, cpu, gpu, count);
    }

    eprintln!(
        "\n✅ gpu_op_times populated with {} keys, key parity verified",
        profile.gpu_op_times.len()
    );

    // === Cycle 2: zero-cost shape-only ops ===
    //
    // Primary gate: hit-rate assertions via the testing facade. The
    // precomputation mechanism's correctness is measured by how many
    // Unsqueeze/Reshape/Squeeze nodes in the graph have their precomputed
    // field populated (Some), not by raw GPU-side timing.
    //
    // Why: Cycle 1.5a's GPU event timer has a ~10-20 us per-call floor
    // from event creation + recording overhead. For metadata-only ops
    // like Unsqueeze (which do no real kernel work), this floor
    // dominates the measurement. The original 2 ms threshold from the
    // spec was based on a theoretical analysis that assumed D2H
    // elimination would drop GPU time to near-zero — that assumption was
    // wrong. Diagnostic instrumentation confirmed 192/192 Unsqueeze
    // precomp hits yet only a 10% GPU time reduction.
    //
    // The REAL savings from Cycle 2 are on the CPU driver-call side:
    // ~207 D2H calls eliminated per inference on Kokoro seq_len=5.
    // These don't show up in gpu_op_times but they DO reduce driver
    // pressure and latency jitter.
    use iconnx::cuda::inference::testing::count_precomputed_shape_ops;
    let precompute_stats = count_precomputed_shape_ops(&executor);

    eprintln!(
        "\n=== Cycle 2 precompute hit rates ===\n  \
         Unsqueeze: {}/{} hits ({:.0}%)\n  \
         Reshape:   {}/{} hits ({:.0}%)\n  \
         Squeeze:   {}/{} hits ({:.0}%)",
        precompute_stats.unsqueeze_hits,
        precompute_stats.unsqueeze_total,
        precompute_stats.unsqueeze_hits as f64 * 100.0
            / precompute_stats.unsqueeze_total.max(1) as f64,
        precompute_stats.reshape_hits,
        precompute_stats.reshape_total,
        precompute_stats.reshape_hits as f64 * 100.0
            / precompute_stats.reshape_total.max(1) as f64,
        precompute_stats.squeeze_hits,
        precompute_stats.squeeze_total,
        precompute_stats.squeeze_hits as f64 * 100.0
            / precompute_stats.squeeze_total.max(1) as f64,
    );

    // Hit-rate assertions: Cycle 2's mechanism must fire for every node
    // whose axes/shape input is statically resolvable.
    //
    // Unsqueeze: 100% hit expected on Kokoro. All 192 Unsqueeze nodes
    // have axes from Constant Int64 nodes.
    assert_eq!(
        precompute_stats.unsqueeze_hits, precompute_stats.unsqueeze_total,
        "Cycle 2 Unsqueeze precompute hit rate should be 100% on Kokoro. \
         Got {}/{}. If this regressed, check that add_node's Constant-Int64 \
         caching block is still populating static_i64_cache and that \
         try_precompute_unsqueeze_axes is being called.",
        precompute_stats.unsqueeze_hits,
        precompute_stats.unsqueeze_total,
    );

    // Squeeze: 100% hit expected on Kokoro. Low volume (only 5 nodes).
    assert_eq!(
        precompute_stats.squeeze_hits, precompute_stats.squeeze_total,
        "Cycle 2 Squeeze precompute hit rate should be 100% on Kokoro. \
         Got {}/{}.",
        precompute_stats.squeeze_hits,
        precompute_stats.squeeze_total,
    );

    // Reshape: lower hit rate is expected because 51 of Kokoro's 61
    // Reshape nodes have shapes computed at runtime from Shape -> Cast ->
    // Gather chains. Cycle 2's targeted fix can't reach those; a broader
    // shape-inference pass (Approach C) is needed. The assertion here is
    // a floor: at least the 10 Constant-sourced Reshapes must hit.
    //
    // Current baseline from Cycle 2 diagnostic: 10/61 hits (16%). Allow
    // ±2 for graph evolution but catch major regressions.
    assert!(
        precompute_stats.reshape_hits >= 8,
        "Cycle 2 Reshape precompute hit rate regressed: got {}/{} hits \
         (expected at least 8, baseline 10). This usually means \
         try_precompute_reshape_shape stopped finding shape inputs in \
         static_i64_cache.",
        precompute_stats.reshape_hits,
        precompute_stats.reshape_total,
    );

    // Secondary gate: loose GPU-time regression guard. The 2 ms target
    // from the original spec was wrong; a more realistic guard catches
    // gross regressions (e.g., if precomputation stopped firing and we
    // re-introduced D2H on every call) without being flaky on the
    // measurement floor.
    //
    // Post-Cycle-2 baselines from diagnostic run on RTX 4070 Laptop, seq_len=5:
    //   Unsqueeze: ~3962 us (192 calls, floor dominated by event overhead)
    //   Reshape:   ~7279 us diagnostic / ~11236 us second run
    //               (51 misses still do D2H + 10 hits near-zero; wide noise
    //                observed run-to-run, guard widened accordingly)
    //   Squeeze:    ~251 us (5 calls, negligible)
    //
    // Thresholds pad generously above the widest observed values to
    // tolerate noise. The guard catches a gross regression (precompute
    // failing) which would push these values substantially higher.
    const MAX_GPU_US_REGRESSION_GUARD_UNSQUEEZE: u64 = 6_000;
    const MAX_GPU_US_REGRESSION_GUARD_RESHAPE: u64 = 14_000;
    const MAX_GPU_US_REGRESSION_GUARD_SQUEEZE: u64 = 500;

    for (op, limit) in [
        ("Unsqueeze", MAX_GPU_US_REGRESSION_GUARD_UNSQUEEZE),
        ("Reshape", MAX_GPU_US_REGRESSION_GUARD_RESHAPE),
        ("Squeeze", MAX_GPU_US_REGRESSION_GUARD_SQUEEZE),
    ] {
        if let Some(&(gpu_us, count)) = profile.gpu_op_times.get(op) {
            assert!(
                gpu_us < limit,
                "{op} GPU time ({gpu_us} us across {count} calls) exceeds \
                 regression guard {limit} us. This is a gross regression, \
                 not a measurement-floor issue — check whether precomputation \
                 is still firing via the hit-rate assertions above.",
            );
            eprintln!(
                "  {:20} Cycle 2: {:>6} us / {:>4} calls — under {} us guard",
                op, gpu_us, count, limit,
            );
        } else {
            eprintln!(
                "  {:20} Cycle 2: not present in gpu_op_times (op did not run)",
                op,
            );
        }
    }
}
