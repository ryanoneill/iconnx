#![cfg(feature = "cuda")]
use std::path::PathBuf;

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

#[test]
#[ignore = "requires Whisper-Tiny encoder FP16 model file; override path via ICONNX_WHISPER_FP16_PATH"]
fn whisper_encoder_fp16_loads_and_lowers() {
    let Some(path) = whisper_fp16_path() else {
        eprintln!("skipping: Whisper-Tiny FP16 model not found");
        return;
    };
    let model = iconnx::OnnxParser::parse_file(&path).expect("parse");
    let weights = model.extract_weights().expect("weights");
    let graph = model.computation_graph().expect("graph");
    assert!(graph.node_count() > 0, "graph must have at least one node");
    assert!(!weights.is_empty(), "weights map must be non-empty");

    let any_fp16 = weights
        .values()
        .any(|t| matches!(t, iconnx::Tensor::Float16(_)));
    assert!(
        any_fp16,
        "Whisper FP16 must contain at least one Float16 initializer"
    );
}
