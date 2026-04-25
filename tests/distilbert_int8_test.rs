//! WS-4 M4.7 DistilBERT-INT8 end-to-end integration test.
//!
//! Structural verification that the parser and IR pass-chain accept the
//! full Xenova DistilBERT-INT8 export and that the three quantization ops
//! we landed in M4.4–M4.6 (`DequantizeLinear`, `DynamicQuantizeLinear`,
//! `MatMulInteger`) all appear in the resulting graph. Runtime
//! correctness against ORT is gated by the leadline-bench compare harness:
//!
//! ```bash
//! cargo run -p leadline-bench --bin compare -- \
//!     --model distilbert-int8 \
//!     --tolerance 1e-2
//! ```
//!
//! Marked `#[ignore]` because the model file lives outside this repo.
//! Path is overridable via `ICONNX_DISTILBERT_INT8_PATH`.

#![cfg(feature = "cuda")]

use std::path::PathBuf;

fn distilbert_int8_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("ICONNX_DISTILBERT_INT8_PATH") {
        let path = PathBuf::from(p);
        if path.exists() {
            return Some(path);
        }
    }
    let default = PathBuf::from(
        "/home/ryano/workspace/ryanoneill/rust-ai-explorations/\
         models/distilbert-base-uncased-onnx/onnx/model_int8.onnx",
    );
    if default.exists() {
        return Some(default);
    }
    None
}

#[test]
#[ignore = "requires DistilBERT-INT8 model file; override path via ICONNX_DISTILBERT_INT8_PATH"]
fn distilbert_int8_loads_and_contains_quantization_ops() {
    let Some(path) = distilbert_int8_path() else {
        eprintln!("skipping: DistilBERT-INT8 model not found");
        return;
    };

    let model = iconnx::OnnxParser::parse_file(&path).expect("parse DistilBERT-INT8");
    let weights = model.extract_weights().expect("extract_weights");
    let graph = model.computation_graph().expect("computation_graph");

    assert!(graph.node_count() > 0, "graph must have nodes");
    assert!(!weights.is_empty(), "weights must be non-empty");

    // Confirm the three M4.4–M4.6 ops all appear in the graph. Xenova's
    // dynamic-quantization export uses every one of them: weights are
    // dequantized via DequantizeLinear, activations are quantized inline
    // via DynamicQuantizeLinear, and the resulting integer GEMM is
    // MatMulInteger.
    let ops: std::collections::HashSet<String> = graph
        .nodes()
        .iter()
        .map(|n| n.op_type.clone())
        .collect();
    assert!(
        ops.contains("DequantizeLinear"),
        "DistilBERT-INT8 must contain DequantizeLinear (M4.4)"
    );
    assert!(
        ops.contains("DynamicQuantizeLinear"),
        "DistilBERT-INT8 must contain DynamicQuantizeLinear (M4.5)"
    );
    assert!(
        ops.contains("MatMulInteger"),
        "DistilBERT-INT8 must contain MatMulInteger (M4.6)"
    );
}
