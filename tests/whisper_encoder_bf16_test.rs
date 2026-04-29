//! WS-3.5 Y(5) Whisper-Tiny BF16 end-to-end integration test.
//!
//! Two tests, mirroring the Y(4) BERT-base BF16 sign-off pattern:
//!
//! 1. `whisper_bf16_parses_and_lowers` — CPU-side structural-load: parse +
//!    extract_weights + computation_graph. Asserts at least one BFloat16
//!    initializer is present so a misnamed FP32 / FP16 export at the
//!    BF16 path is caught loudly. Skips silently when the model file
//!    isn't on disk (CI runners; only Ryan's local box has it).
//!
//! 2. `whisper_bf16_first_inference_no_panic` — GPU-side: builds a
//!    `GpuGraphExecutor`, uploads initializers, drives one inference
//!    with Box-Muller seeded random `input_features` (Gaussian samples
//!    matching the mel-log-spectrogram value range; pattern reused from
//!    `whisper_fp16_intermediate_probe.rs`). Per
//!    `feedback_dtype_test_fixtures` memory: random fixtures, never
//!    zeros — zeros is pathological for transformer attention. Hardware
//!    gate is `compute_capability >= (8, 0)` per WS-3.5 spec §1
//!    (sm_80+ for native BF16 compute) and `#[ignore]`-gated since
//!    leadline-bench owns the numerical compare gate; this test only
//!    confirms the dispatch surface lights up end-to-end. Output shape
//!    `[1, 1500, 384]` (last_hidden_state) is asserted as a structural
//!    check on the boundary cast (BF16 internal → FP32 at exit per
//!    spec §4 Stage 5).
//!
//! Numerical comparison vs ORT FP32 reference happens out-of-tree in
//! leadline-bench's `WhisperEncoderBf16` ModelProfile (separate-repo
//! work, not iconnx's responsibility). A first-inference failure here
//! indicates a missed dispatch arm at the Whisper op surface
//! (Conv1D + attention + LayerNorm + Cast-at-boundary) — escalate
//! per WS-3.5 spec §6 Y(5) Step 2.
//!
//! Path resolution mirrors the WS-3 Whisper-FP16 / WS-3.5 Y(4)
//! BERT-base BF16 convention: `ICONNX_WHISPER_BF16_PATH` env first,
//! default at
//! `rust-ai-explorations/models/whisper-tiny.en-export/onnx/encoder_model_bf16.onnx`
//! second, silent skip otherwise.

#![cfg(feature = "cuda")]

use std::collections::HashMap;
use std::path::PathBuf;

use iconnx::cuda::inference::GpuGraphExecutor;
use iconnx::tensor::Tensor;
use iconnx::OnnxParser;

/// Resolve the Whisper-Tiny BF16 encoder model path. Honors
/// `ICONNX_WHISPER_BF16_PATH` first, falls back to the
/// convention-default under `rust-ai-explorations/`. Returns `None` if
/// neither resolves — callers skip silently in that branch (CI-style;
/// the model file lives outside the iconnx repo).
fn whisper_bf16_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("ICONNX_WHISPER_BF16_PATH") {
        let path = PathBuf::from(p);
        if path.exists() {
            return Some(path);
        }
    }
    let default = PathBuf::from(
        "/home/ryano/workspace/ryanoneill/rust-ai-explorations/\
         models/whisper-tiny.en-export/onnx/encoder_model_bf16.onnx",
    );
    if default.exists() {
        return Some(default);
    }
    None
}

/// Box-Muller-transformed unit-Gaussian samples produced from an
/// xorshift64* PRNG. Deterministic, no external dep — matches the
/// pattern from `whisper_fp16_intermediate_probe.rs` (which itself
/// borrowed it from leadline-bench's `de2829e`) so numerical-bisect
/// probes line up with the rest of the WS-3.x family.
///
/// Whisper mel-log-spectrogram features are FP32 floats roughly
/// distributed as standard-normal after preprocessing, so unit-Gaussian
/// samples are an appropriate fixture per the
/// `feedback_dtype_test_fixtures` memory (random, never zeros — zeros
/// is pathological for attention).
fn box_muller_seeded_mel(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut next_u64 = || -> u64 {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        state.wrapping_mul(0x2545F4914F6CDD1D)
    };
    // Map u64 to (0, 1) using the top 53 bits (matches the IEEE-754
    // double mantissa width; same trick numpy uses internally).
    let mut next_uniform = || -> f64 {
        ((next_u64() >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let u1 = next_uniform().max(1e-10); // avoid log(0)
        let u2 = next_uniform();
        let r = (-2.0_f64 * u1.ln()).sqrt();
        let theta = 2.0_f64 * std::f64::consts::PI * u2;
        out.push((r * theta.cos()) as f32);
        if out.len() < n {
            out.push((r * theta.sin()) as f32);
        }
    }
    out
}

#[test]
#[ignore = "requires Whisper-Tiny BF16 encoder model file; override path via ICONNX_WHISPER_BF16_PATH"]
fn whisper_bf16_parses_and_lowers() {
    let Some(path) = whisper_bf16_path() else {
        eprintln!("skipping: Whisper-Tiny BF16 model not found");
        return;
    };

    let model = OnnxParser::parse_file(&path).expect("parse Whisper-Tiny BF16");
    let weights = model.extract_weights().expect("extract_weights");
    let graph = model.computation_graph().expect("computation_graph");

    assert!(graph.node_count() > 0, "graph must have nodes");
    assert!(!weights.is_empty(), "weights must be non-empty");

    // Loud-fail if the file at the BF16 path is actually FP32 / FP16 —
    // catches a misnamed export before the numerical gate has to.
    let any_bf16 = weights.values().any(|t| t.is_bfloat16());
    assert!(
        any_bf16,
        "Whisper-Tiny BF16 must contain at least one BFloat16 initializer"
    );
}

#[test]
#[ignore = "requires CUDA GPU + Whisper-Tiny BF16 encoder model file; override path via ICONNX_WHISPER_BF16_PATH"]
fn whisper_bf16_first_inference_no_panic() {
    let Some(path) = whisper_bf16_path() else {
        eprintln!("skipping: Whisper-Tiny BF16 model not found");
        return;
    };

    let mut executor = GpuGraphExecutor::new().expect("executor");

    // Hardware gate: BF16 native compute is sm_80+ per WS-3.5 spec §5
    // "Hardware Support Errors". Skip on older GPUs rather than fail —
    // the executor's `zeros_bf16` ctor enforces this with a typed
    // `UnsupportedHardware` error, so running on sm_75 would surface as
    // a typed parse-time error, not a panic. The skip here is a
    // courtesy for mixed-hardware CI matrices.
    let (cc_major, cc_minor) = executor.context().compute_capability();
    if cc_major < 8 {
        eprintln!(
            "skipping: BF16 native compute requires sm_80+, device is sm_{}{}",
            cc_major, cc_minor
        );
        return;
    }

    let model = OnnxParser::parse_file(&path).expect("parse Whisper-Tiny BF16");
    let weights = model.extract_weights().expect("extract_weights");
    let graph = model.computation_graph().expect("computation_graph");

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

    // Whisper-Tiny encoder input (per spec §4 Stage 5): FP32 mel
    // features at shape [1, 80, 3000] — 80 mel bins × 3000 frames
    // (30 s of audio at the standard Whisper preprocessor rate). Cast
    // to BF16 internally at graph entry; cast back to FP32 at exit.
    //
    // Random fixtures per `feedback_dtype_test_fixtures` memory:
    // Box-Muller-seeded unit-Gaussian samples (mel features are
    // approximately standard-normal post-preprocessing; matches the
    // distribution leadline-bench uses for its compare run so the
    // first-inference signal lines up).
    let mel = box_muller_seeded_mel(80 * 3000, 0xa5a5_a5a5_a5a5_a5a5);

    let declared_inputs = model.inputs();
    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    for name in &declared_inputs {
        let tensor = match name.as_str() {
            "input_features" => Tensor::from_vec_f32(mel.clone(), vec![1, 80, 3000]),
            other => panic!(
                "unexpected Whisper encoder input `{}` — extend \
                 whisper_bf16_first_inference_no_panic to handle it",
                other
            ),
        };
        inputs.insert(name.clone(), tensor);
    }

    // Output is "last_hidden_state" at shape [1, 1500, 384] — BF16
    // internally, cast to FP32 at executor exit per spec §4 Stage 5.
    // 1500 = 3000/2 (Conv1D stride-2 downsampling); 384 = Whisper-Tiny
    // encoder hidden width. Caller receives FP32. Materialize the
    // owned-String vec first so the `&str` borrows passed to `run()`
    // outlive the call.
    let output_strings: Vec<String> = model.outputs();
    let output_refs: Vec<&str> = output_strings.iter().map(String::as_str).collect();

    let result = executor
        .run(inputs, output_refs.clone())
        .expect("first inference must not panic");

    assert!(
        !result.is_empty(),
        "executor must produce at least one output"
    );
    for name in &output_refs {
        let tensor = result
            .get(*name)
            .unwrap_or_else(|| panic!("missing output `{}` from inference", name));
        assert!(
            !tensor.is_empty(),
            "output `{}` must be non-empty",
            name
        );
        // Boundary-cast structural check: encoder output should land at
        // [1, 1500, 384] regardless of internal dtype, since Stage 5
        // exit cast goes back to FP32 dimensions-preserved. If the
        // shape is anything else, either the boundary cast is broken
        // or the model is misnamed.
        if name == &"last_hidden_state" {
            assert_eq!(
                tensor.shape(),
                &[1, 1500, 384],
                "last_hidden_state must be [1, 1500, 384] (BF16 → FP32 boundary cast)"
            );
        }
    }
}
