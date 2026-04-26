//! M3.7 numerical-bisect probe: dump stats for selected intermediate
//! outputs of the Whisper-Tiny FP16 encoder so we can localize the
//! source of the 13× output divergence reported by leadline-bench.
//!
//! Per leadline 2026-04-26 triage: walk
//!   graph_input_cast_0 → /conv1/Conv_output_0 → first LayerNorm output
//! and compare against expectations. With zeros input every intermediate
//! has a known closed-form value, so we can spot a structural bug
//! without needing pre-saved ORT references.

#![cfg(feature = "cuda")]

use std::collections::HashMap;
use std::path::PathBuf;

use iconnx::cuda::inference::GpuGraphExecutor;
use iconnx::tensor::Tensor;

fn whisper_fp16_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("ICONNX_WHISPER_FP16_PATH") {
        let path = PathBuf::from(p);
        if path.exists() {
            return Some(path);
        }
    }
    let default = PathBuf::from(
        "/home/ryano/workspace/ryanoneill/rust-ai-explorations/\
         models/whisper-tiny.en-export/onnx/encoder_model_fp16.onnx",
    );
    if default.exists() {
        return Some(default);
    }
    None
}

fn stats(values: &[f32]) -> (f32, f32, f32, usize, usize) {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0_f64;
    let mut nan = 0;
    let mut inf = 0;
    for &v in values {
        if v.is_nan() {
            nan += 1;
        } else if v.is_infinite() {
            inf += 1;
        } else {
            min = min.min(v);
            max = max.max(v);
            sum += v as f64;
        }
    }
    let n = values.len().saturating_sub(nan + inf);
    let mean = if n > 0 { (sum / n as f64) as f32 } else { 0.0 };
    (min, max, mean, nan, inf)
}

fn report(name: &str, t: &Tensor) {
    let v: Vec<f32> = match t {
        Tensor::Float32(a) => a.iter().copied().collect(),
        Tensor::Float16(a) => a.iter().map(|h| h.to_f32()).collect(),
        Tensor::Int64(a) => a.iter().map(|&x| x as f32).collect(),
        Tensor::Int32(a) => a.iter().map(|&x| x as f32).collect(),
        Tensor::Int8(a) => a.iter().map(|&x| x as f32).collect(),
        Tensor::UInt8(a) => a.iter().map(|&x| x as f32).collect(),
        Tensor::Bool(a) => a.iter().map(|&x| if x { 1.0 } else { 0.0 }).collect(),
        Tensor::Float64(a) => a.iter().map(|&x| x as f32).collect(),
    };
    let (min, max, mean, nan, inf) = stats(&v);
    println!(
        "{:<40} dtype={:<8} shape={:?} min={:.6e} max={:.6e} mean={:.6e} nan={} inf={} n={}",
        name,
        t.dtype(),
        t.shape(),
        min,
        max,
        mean,
        nan,
        inf,
        v.len()
    );
    // Print first 8 and last 8 values for quick visual inspection.
    if v.len() <= 16 {
        println!("    full: {:?}", v);
    } else {
        let head: Vec<f32> = v[..8].to_vec();
        let tail: Vec<f32> = v[v.len() - 8..].to_vec();
        println!("    head: {:?}", head);
        println!("    tail: {:?}", tail);
    }
}

#[test]
#[ignore = "M3.7 numerical-bisect probe; requires Whisper-Tiny FP16 model"]
fn whisper_fp16_intermediate_probe() {
    let Some(path) = whisper_fp16_path() else {
        eprintln!("skipping: Whisper FP16 model not found");
        return;
    };

    let model = iconnx::OnnxParser::parse_file(&path).expect("parse");
    let weights = model.extract_weights().expect("extract_weights");
    let graph = model.computation_graph().expect("computation_graph");

    let mut executor = GpuGraphExecutor::new().expect("executor");
    for (name, tensor) in &weights {
        executor
            .add_initializer(name.clone(), tensor)
            .expect("add_initializer");
    }
    for (idx, node) in graph.nodes().iter().enumerate() {
        let fallback = format!("node_{idx}");
        let node_name = node
            .outputs
            .first()
            .map(|s| s.as_str())
            .unwrap_or(&fallback);
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

    // Random input matching mel-log-spectrogram range. Same seed as the
    // Python ORT comparison probe so iconnx and ORT see identical inputs.
    let mut rng_state: u64 = 0x9E3779B97F4A7C15;
    let mut next = || {
        // Xorshift64* — deterministic, no external dep.
        rng_state ^= rng_state >> 12;
        rng_state ^= rng_state << 25;
        rng_state ^= rng_state >> 27;
        rng_state.wrapping_mul(0x2545F4914F6CDD1D)
    };
    // Box-Muller standard-normal floats. Matches numpy.random.randn's
    // population (different seed/sequence, but same distribution).
    let mut samples = Vec::with_capacity(80 * 3000);
    while samples.len() < 80 * 3000 {
        let u1 = (next() as f64 / u64::MAX as f64).max(1e-10);
        let u2 = next() as f64 / u64::MAX as f64;
        let mag = (-2.0 * u1.ln()).sqrt();
        let z1 = mag * (2.0 * std::f64::consts::PI * u2).cos();
        let z2 = mag * (2.0 * std::f64::consts::PI * u2).sin();
        samples.push(z1 as f32);
        if samples.len() < 80 * 3000 {
            samples.push(z2 as f32);
        }
    }
    // Save inputs so the Python ORT probe uses identical values.
    let bytes: Vec<u8> = samples.iter().flat_map(|f| f.to_le_bytes()).collect();
    std::fs::write("/tmp/iconnx_whisper_input.bin", bytes).expect("write input");
    let inputs: HashMap<String, Tensor> = HashMap::from([(
        "input_features".to_string(),
        Tensor::from_vec_f32(samples, vec![1, 80, 3000]),
    )]);

    // Probe outputs in execution order. Order matters — first divergence
    // wins. Names sourced from the onnx graph by the python probe.
    let probes = [
        "graph_input_cast_0",
        "/conv1/Conv_output_0",
        "/conv2/Conv_output_0",
        // gelu(conv2) transposed to [1, 1500, 384].
        "/Transpose_output_0",
        // After + embed_positions.weight (sinusoidal pos embed).
        "/Add_2_output_0",
        // First transformer block prologue (variance+eps).
        "/layers.0/self_attn_layer_norm/Add_output_0",
        // First LayerNorm output (after scale+bias, [1,1500,384]).
        "/layers.0/self_attn_layer_norm/Add_1_output_0",
        // After self-attention residual.
        "/layers.0/Add_output_0",
        // After FFN residual (end of layer 0).
        "/layers.0/Add_1_output_0",
        // Layer 1, 2, 3 final outputs (FFN residual).
        "/layers.1/Add_1_output_0",
        "/layers.2/Add_1_output_0",
        "/layers.3/Add_1_output_0",
        "last_hidden_state",
    ];

    let outputs = executor
        .run(inputs, probes.to_vec())
        .expect("run with intermediate outputs");

    println!("\n=== Whisper FP16 intermediate probe (zeros input) ===");
    for name in probes {
        match outputs.get(name) {
            Some(t) => report(name, t),
            None => println!("{} : MISSING from outputs map", name),
        }
    }

    // Save every probe to a binary file so Python can do element-wise
    // diffs. FP16 saves as u16 (raw bits); FP32 as f32 LE.
    for name in probes {
        let Some(t) = outputs.get(name) else { continue };
        let safe_name = name.replace('/', "_");
        let path = format!("/tmp/iconnx_probe_{}.bin", safe_name.trim_start_matches('_'));
        let bytes: Vec<u8> = match t {
            Tensor::Float32(a) => a.iter().flat_map(|f| f.to_le_bytes()).collect(),
            Tensor::Float16(a) => a.iter().flat_map(|h| h.to_bits().to_le_bytes()).collect(),
            _ => continue,
        };
        std::fs::write(&path, bytes).expect("write probe");
    }
    println!("\nSaved all probes to /tmp/iconnx_probe_*.bin");
}
