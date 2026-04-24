/// Test ONNX parsing for Kokoro model
///
/// TDD: These tests are written FIRST, before implementation.
/// They define the behavior we expect from the ONNX parser.
#[path = "common/mod.rs"]
mod common;

/// Test 1: Parse kokoro-v1.0.onnx file
///
/// This is our first test - it will FAIL until we implement OnnxParser.
#[test]
fn test_parse_kokoro_onnx_file() {
    let model_path = common::kokoro_model::kokoro_model_path();
    // TDD: This will fail initially - OnnxParser doesn't parse yet!
    let model = iconnx::onnx_parser::OnnxParser::parse_file(&model_path);

    // Once implemented, should succeed
    assert!(model.is_ok(), "Should parse Kokoro ONNX file");
}

/// Test 2: Extract model metadata
#[test]
fn test_extract_model_metadata() {
    let model_path = common::kokoro_model::kokoro_model_path();
    let model = iconnx::onnx_parser::OnnxParser::parse_file(&model_path).unwrap();

    // Should extract basic info
    let inputs = model.inputs();
    let outputs = model.outputs();

    // Kokoro has 3 inputs: tokens, style, speed
    assert_eq!(inputs.len(), 3, "Kokoro should have 3 inputs");

    // Kokoro has 1 output: audio
    assert_eq!(outputs.len(), 1, "Kokoro should have 1 output");

    // Print for visibility
    println!("Inputs: {:?}", inputs);
    println!("Outputs: {:?}", outputs);
}

/// Test 3: Extract initializer weights
#[test]
fn test_extract_weights_from_model() {
    let model_path = common::kokoro_model::kokoro_model_path();
    let model = iconnx::onnx_parser::OnnxParser::parse_file(&model_path).unwrap();
    let weight_names = model.list_weight_names();

    // Kokoro should have many weight tensors (hundreds)
    assert!(
        weight_names.len() > 100,
        "Kokoro should have >100 weight tensors, found {}",
        weight_names.len()
    );

    println!("Found {} weight tensors", weight_names.len());
    println!(
        "First 5 weights: {:?}",
        &weight_names[..5.min(weight_names.len())]
    );
}

/// Test 4: List all operators used by Kokoro
#[test]
fn test_list_unique_operators() {
    let model_path = common::kokoro_model::kokoro_model_path();
    let model = iconnx::onnx_parser::OnnxParser::parse_file(&model_path).unwrap();
    let operators = model.list_unique_operators();

    // Print for visibility
    println!("Operators used by Kokoro ({} unique):", operators.len());
    for op in &operators {
        println!("  - {}", op);
    }

    // Should include known operators from transformer architecture
    assert!(
        operators.contains(&"MatMul".to_string()),
        "Should have MatMul operator"
    );
    assert!(
        operators.contains(&"Add".to_string()),
        "Should have Add operator"
    );

    // Should have multiple operator types
    assert!(operators.len() > 10, "Should have >10 unique operators");
}

/// Test 5: Parse computation graph nodes
#[test]
fn test_parse_computation_graph() {
    let model_path = common::kokoro_model::kokoro_model_path();
    let model = iconnx::onnx_parser::OnnxParser::parse_file(&model_path).unwrap();
    let graph = model.computation_graph().expect("parse computation graph");

    // Should have many nodes for a complex model like Kokoro
    let node_count = graph.node_count();
    assert!(
        node_count > 100,
        "Kokoro should have >100 computation nodes, found {}",
        node_count
    );

    // Get first few nodes for visibility
    let nodes = graph.nodes();
    println!("Graph has {} nodes", node_count);
    println!("First 3 nodes:");
    for (i, node) in nodes.iter().take(3).enumerate() {
        println!(
            "  [{}] {}: {} inputs -> {} outputs",
            i,
            node.op_type,
            node.inputs.len(),
            node.outputs.len()
        );
    }
}
