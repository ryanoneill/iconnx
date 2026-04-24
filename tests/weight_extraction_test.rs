/// Test weight extraction from ONNX models
#[path = "common/mod.rs"]
#[allow(clippy::duplicate_mod)] // shared helper loaded via `#[path]` in multiple integration-test binaries
mod common;

use iconnx::onnx_parser::OnnxParser;

/// Test: Extract weights from Kokoro model
#[test]
fn test_extract_kokoro_weights() {
    let model_path = common::kokoro_model::kokoro_model_path();
    let model = OnnxParser::parse_file(&model_path).expect("parse kokoro-v1.0.onnx");
    let weights = model.extract_weights().expect("extract weights");

    println!("\n=== Weight Extraction ===");
    println!("Total weights extracted: {}", weights.len());

    // Should have hundreds of weights
    assert!(weights.len() > 500, "Kokoro should have >500 weights");

    // Check a few weights exist and have reasonable properties
    let mut total_params = 0;
    for (name, tensor) in weights.iter().take(10) {
        let params = tensor.len();
        total_params += params;
        println!("  {}: shape={:?}, params={}", name, tensor.shape(), params);
    }

    println!("\nTotal parameters (first 10 weights): {}", total_params);
    println!("Weights are loaded as Tensor objects ✅");

    // Verify tensors have correct structure
    for (name, tensor) in weights.iter().take(5) {
        assert!(!tensor.is_empty(), "Weight {} should not be empty", name);
        assert!(
            !tensor.shape().is_empty() || tensor.len() == 1,
            "Weight {} should have valid shape",
            name
        );
    }
}
