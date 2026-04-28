//! WS-3.5 Y(4) BERT-base BF16 end-to-end integration test.
//!
//! Two tests, mirroring the Whisper-FP16 sign-off pattern from WS-3:
//!
//! 1. `bert_bf16_parses_and_lowers` — CPU-side structural-load: parse +
//!    extract_weights + computation_graph. Asserts at least one BFloat16
//!    initializer is present so a misnamed FP32 / FP16 export at the
//!    BF16 path is caught loudly. Skips silently when the model file
//!    isn't on disk (CI runners; only Ryan's local box has it).
//!
//! 2. `bert_bf16_first_inference_no_panic` — GPU-side: builds a
//!    `GpuGraphExecutor`, uploads initializers, drives one inference
//!    with xorshift64*-seeded random `input_ids` (over a conservative
//!    BERT-base vocab range) and a non-all-ones `attention_mask`. Per
//!    `feedback_dtype_test_fixtures` memory: random fixtures, never
//!    zeros; zeros is pathological for transformer attention. Hardware
//!    gate is `compute_capability >= (8, 0)` per WS-3.5 spec §1
//!    (sm_80+ for native BF16 compute) and `#[ignore]`-gated since
//!    leadline-bench owns the numerical compare gate; this test only
//!    confirms the dispatch surface lights up end-to-end.
//!
//! Numerical comparison vs ORT FP32 reference happens out-of-tree in
//! leadline-bench's `BertBaseBf16` ModelProfile (separate-repo work,
//! not iconnx's responsibility). A first-inference failure here
//! indicates a missed dispatch arm at the BERT op surface — escalate
//! per WS-3.5 spec §6 Y(4) Step 2.
//!
//! Path resolution mirrors the WS-3 Whisper-FP16 / WS-4 DistilBERT-INT8
//! convention: `ICONNX_BERT_BF16_PATH` env first, default at
//! `rust-ai-explorations/models/bert-base-uncased-onnx/onnx/model_bf16.onnx`
//! second, silent skip otherwise.

#![cfg(feature = "cuda")]

use std::collections::HashMap;
use std::path::PathBuf;

use iconnx::cuda::inference::GpuGraphExecutor;
use iconnx::tensor::Tensor;
use iconnx::OnnxParser;

/// Resolve the BERT-base BF16 model path. Honors
/// `ICONNX_BERT_BF16_PATH` first, falls back to the convention-default
/// under `rust-ai-explorations/`. Returns `None` if neither resolves —
/// callers skip silently in that branch (CI-style; the model file lives
/// outside the iconnx repo).
fn bert_bf16_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("ICONNX_BERT_BF16_PATH") {
        let path = PathBuf::from(p);
        if path.exists() {
            return Some(path);
        }
    }
    let default = PathBuf::from(
        "/home/ryano/workspace/ryanoneill/rust-ai-explorations/\
         models/bert-base-uncased-onnx/onnx/model_bf16.onnx",
    );
    if default.exists() {
        return Some(default);
    }
    None
}

/// Xorshift64* PRNG with uniform-u32 + Bernoulli sampling helpers.
/// Deterministic, no external dep — matches the xorshift pattern from
/// `whisper_fp16_intermediate_probe.rs` so numerical-bisect probes (if
/// needed at Y(4) Step 4) line up with the rest of the WS-3.x family.
/// BERT inputs are int64 IDs and binary masks, so no Gaussian sampling
/// (Box-Muller) is needed here.
struct XorshiftRng {
    state: u64,
}

impl XorshiftRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Xorshift64* — deterministic uniform u64 in [0, u64::MAX].
    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        self.state.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// Uniform u32 in [0, n).
    fn next_below(&mut self, n: u32) -> u32 {
        (self.next_u64() % n as u64) as u32
    }

    /// One Bernoulli(p) sample as i64 (1 if hit, 0 otherwise).
    fn bernoulli_i64(&mut self, p: f64) -> i64 {
        let u = (self.next_u64() as f64) / (u64::MAX as f64);
        if u < p {
            1
        } else {
            0
        }
    }
}

#[test]
#[ignore = "requires BERT-base BF16 model file; override path via ICONNX_BERT_BF16_PATH"]
fn bert_bf16_parses_and_lowers() {
    let Some(path) = bert_bf16_path() else {
        eprintln!("skipping: BERT-base BF16 model not found");
        return;
    };

    let model = OnnxParser::parse_file(&path).expect("parse BERT-base BF16");
    let weights = model.extract_weights().expect("extract_weights");
    let graph = model.computation_graph().expect("computation_graph");

    assert!(graph.node_count() > 0, "graph must have nodes");
    assert!(!weights.is_empty(), "weights must be non-empty");

    // Loud-fail if the file at the BF16 path is actually FP32 / FP16 —
    // catches a misnamed export before the numerical gate has to.
    let any_bf16 = weights.values().any(|t| t.is_bfloat16());
    assert!(
        any_bf16,
        "BERT-base BF16 must contain at least one BFloat16 initializer"
    );
}

#[test]
#[ignore = "requires CUDA GPU + BERT-base BF16 model file; override path via ICONNX_BERT_BF16_PATH"]
fn bert_bf16_first_inference_no_panic() {
    let Some(path) = bert_bf16_path() else {
        eprintln!("skipping: BERT-base BF16 model not found");
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

    let model = OnnxParser::parse_file(&path).expect("parse BERT-base BF16");
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

    // BERT-base inputs (per spec §4 Stage 5): int64 token IDs + int64
    // attention mask. Some BERT exports also expose `token_type_ids`
    // (segment IDs). Drive whichever the graph declares — the parser's
    // `inputs()` returns names in graph-declaration order.
    //
    // Random fixtures per `feedback_dtype_test_fixtures` memory:
    // xorshift64*-seeded uniform `input_ids` in [0, 30522) (BERT-base
    // uncased vocab) and a Bernoulli(0.85)-sparse `attention_mask` so
    // it isn't degenerately all-ones. `token_type_ids` is all-zero by
    // convention (single-segment input).
    let seq_len: usize = 8;
    let mut rng = XorshiftRng::new(0x9E3779B97F4A7C15);
    let bert_base_uncased_vocab: u32 = 30522;

    let input_ids: Vec<i64> = (0..seq_len)
        .map(|_| rng.next_below(bert_base_uncased_vocab) as i64)
        .collect();
    let attention_mask: Vec<i64> = (0..seq_len).map(|_| rng.bernoulli_i64(0.85)).collect();
    let token_type_ids: Vec<i64> = vec![0; seq_len];

    let declared_inputs = model.inputs();
    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    for name in &declared_inputs {
        let tensor = match name.as_str() {
            "input_ids" => Tensor::from_vec_i64(input_ids.clone(), vec![1, seq_len]),
            "attention_mask" => Tensor::from_vec_i64(attention_mask.clone(), vec![1, seq_len]),
            "token_type_ids" => Tensor::from_vec_i64(token_type_ids.clone(), vec![1, seq_len]),
            other => panic!(
                "unexpected BERT input `{}` — extend bert_bf16_first_inference_no_panic\
                 to handle it",
                other
            ),
        };
        inputs.insert(name.clone(), tensor);
    }

    // Output is "logits" — BF16 internally, cast to FP32 at executor
    // exit per spec §4 Stage 5. Caller receives FP32. Materialize the
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
    }
}
