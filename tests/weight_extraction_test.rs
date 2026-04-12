/// Test weight extraction from ONNX models
use iconnx::onnx_parser::OnnxParser;
use std::path::Path;

/// Test: Extract weights from Kokoro model
#[test]
fn test_extract_kokoro_weights() {
    let model_path = Path::new("kokoro-v1.0.onnx");

    if !model_path.exists() {
        println!("Skipping - kokoro-v1.0.onnx not found");
        return;
    }

    let model = OnnxParser::parse_file(model_path).unwrap();
    let weights = model.extract_weights();

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
