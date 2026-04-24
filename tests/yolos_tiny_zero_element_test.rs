//! WS-1 M1.2 zero-element refinement regression test.
//!
//! YOLOS-tiny carries Constant attribute tensors with `dims = [0]`
//! (e.g. `/vit/embeddings/interpolation/Constant_{2,3}`). These
//! failed pre-fix with `InvalidTensorData { reason: "empty raw_data
//! for FLOAT tensor" }` because parse_tensor_proto's strict-mode
//! check rejected empty data unconditionally. This test loads the
//! full YOLOS-tiny model and drives `computation_graph()` — which
//! is what actually exercises parse_attributes → parse_tensor_proto
//! for Constant nodes — to confirm the refinement holds end-to-end.
//!
//! Marked `#[ignore]` because YOLOS-tiny lives outside this repo.
//! Path is overridable via `ICONNX_YOLOS_TINY_PATH`.

use std::path::PathBuf;

fn yolos_tiny_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("ICONNX_YOLOS_TINY_PATH") {
        let path = PathBuf::from(p);
        if path.exists() {
            return Some(path);
        }
    }
    let default =
        PathBuf::from("/home/ryano/workspace/ryanoneill/rust-ai-explorations/models/yolos-tiny-onnx/onnx/model.onnx");
    if default.exists() {
        return Some(default);
    }
    None
}

#[test]
#[ignore = "requires YOLOS-tiny model; override path via ICONNX_YOLOS_TINY_PATH"]
fn yolos_tiny_zero_element_constants_parse() {
    let Some(path) = yolos_tiny_path() else {
        eprintln!("YOLOS-tiny model not found; skipping.");
        return;
    };

    let model = iconnx::OnnxParser::parse_file(&path).expect("parse YOLOS-tiny");
    let graph = model
        .computation_graph()
        .expect("computation_graph must succeed (fails pre-fix on Constant_{2,3})");
    let weights = model.extract_weights().expect("extract_weights");

    assert!(graph.node_count() > 0, "graph must have nodes");
    assert!(!weights.is_empty(), "weights must be non-empty");

    // The specific regression: Constant nodes with zero-element value
    // tensors must parse. We expect ≥2 (the two Constant nodes leadline
    // flagged at /vit/embeddings/interpolation/Constant_{2,3}).
    let zero_element_constants: usize = graph
        .nodes()
        .iter()
        .filter(|n| n.op_type == "Constant")
        .filter_map(|n| n.attributes.get_tensor("value"))
        .filter(|t| t.is_empty())
        .count();
    assert!(
        zero_element_constants >= 2,
        "expected ≥2 zero-element Constant value tensors, got {}",
        zero_element_constants
    );
}
