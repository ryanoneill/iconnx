/// End-to-end validation test for ONYCHES
///
/// This test validates the complete inference stack:
/// 1. Parse ONNX model (our parser)
/// 2. Execute graph (our executor)
/// 3. Compare with ort (ground truth)
///
/// We'll use Kokoro itself as the test model, but only execute
/// a small subgraph that uses our implemented operators.
#[path = "common/mod.rs"]
mod common;

use iconnx::graph_executor::GraphExecutor;
use iconnx::onnx_parser::OnnxParser;
use iconnx::tensor::Tensor;
use std::collections::HashMap;

/// Test: Load Kokoro and extract a simple subgraph we can execute
///
/// We'll manually construct a graph using operators from Kokoro
/// that we've implemented (Add, Mul, MatMul)
#[test]
fn test_simple_linear_layer_simulation() {
    // Create a simple linear layer: y = x @ W + b
    let mut executor = GraphExecutor::new();

    // Weight matrix [3, 2]
    let weight = Tensor::from_vec(
        vec![
            1.0, 4.0, // Column 0, 1
            2.0, 5.0, // Column 0, 1
            3.0, 6.0, // Column 0, 1
        ],
        vec![3, 2],
    );

    // Bias vector [2]
    let bias = Tensor::from_vec(vec![10.0, 20.0], vec![2]);

    executor.add_initializer("weight".to_string(), weight);
    executor.add_initializer("bias".to_string(), bias);

    // Build graph
    // Node 1: matmul_out = MatMul(input, weight)
    executor.add_node(
        "matmul",
        "MatMul",
        vec!["input", "weight"],
        vec!["matmul_out"],
    );

    // Node 2: output = Add(matmul_out, bias)
    executor.add_node(
        "add_bias",
        "Add",
        vec!["matmul_out", "bias"],
        vec!["output"],
    );

    // Input [1, 3]
    let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
    let mut inputs = HashMap::new();
    inputs.insert("input".to_string(), input);

    // Execute
    let outputs = executor.run(inputs, vec!["output"]).unwrap();

    // Verify
    let result = outputs.get("output").unwrap();

    // Manual calculation:
    // MatMul: [1, 2, 3] @ [[1, 4], [2, 5], [3, 6]] = [1*1+2*2+3*3, 1*4+2*5+3*6] = [14, 32]
    // Add: [14, 32] + [10, 20] = [24, 52]
    assert_eq!(result.shape(), &[1, 2]);
    assert_eq!(result.as_slice(), &[24.0, 52.0]);

    println!("✅ Linear layer validation passed!");
    println!("   Input: [1.0, 2.0, 3.0]");
    println!("   Output: [24.0, 52.0]");
    println!("   Expected: [24.0, 52.0]");
}

/// Test: Validate ONYCHES can load and inspect Kokoro model
#[test]
fn test_load_kokoro_model_structure() {
    let model_path = common::kokoro_model::kokoro_model_path();
    // Load model
    let model = OnnxParser::parse_file(&model_path).expect("parse kokoro-v1.0.onnx");

    // Get model info
    let inputs = model.inputs();
    let outputs = model.outputs();
    let operators = model.list_unique_operators();
    let weights = model.list_weight_names();
    let graph = model.computation_graph();

    println!("\n=== Kokoro Model Structure ===");
    println!("Inputs: {:?}", inputs);
    println!("Outputs: {:?}", outputs);
    println!("Unique operators: {} types", operators.len());
    println!("Weight tensors: {}", weights.len());
    println!("Computation nodes: {}", graph.node_count());

    // Check which operators we can handle
    let implemented = ["Add", "Mul", "Sub", "Div", "MatMul"];
    let can_handle: Vec<_> = operators
        .iter()
        .filter(|op| implemented.contains(&op.as_str()))
        .collect();

    println!("\nOperators we CAN handle: {:?}", can_handle);
    println!(
        "Coverage: {}/{} operators ({:.1}%)",
        can_handle.len(),
        operators.len(),
        (can_handle.len() as f32 / operators.len() as f32) * 100.0
    );

    // Verify basic structure
    assert_eq!(inputs.len(), 3);
    assert_eq!(outputs.len(), 1);
    assert!(operators.len() >= 40); // Kokoro uses many operators
    assert!(weights.len() > 500); // Kokoro has hundreds of weights
}

/// Test: Compare simple Add operation between ONYCHES and ort
#[test]
#[ignore] // Requires creating a minimal ONNX model
fn test_compare_with_ort_add() {
    // This test would:
    // 1. Load a simple ONNX model (just Add)
    // 2. Run with ort
    // 3. Run with ONYCHES
    // 4. Compare outputs (should match!)
    //
    // TODO: Create minimal ONNX model for this test
    println!("TODO: Create minimal ONNX model with Add operator");
}
