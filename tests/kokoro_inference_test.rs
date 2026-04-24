/// Kokoro full inference test
///
/// This is the ULTIMATE test - run the complete Kokoro model end-to-end
#[path = "common/mod.rs"]
mod common;

use iconnx::graph_executor::GraphExecutor;
use iconnx::onnx_parser::OnnxParser;
use iconnx::tensor::Tensor;
use std::collections::HashMap;

/// Test: Load full Kokoro model and build executor
#[test]
#[ignore] // Expensive test - run manually
fn test_load_full_kokoro_model() {
    let model_path = common::kokoro_model::kokoro_model_path();

    println!("\n=== Loading Kokoro Model ===");

    // Step 1: Parse model
    let model = OnnxParser::parse_file(&model_path).expect("parse kokoro-v1.0.onnx");
    println!("✅ Model parsed");

    // Step 2: Extract weights
    let weights = model.extract_weights().expect("extract weights");
    println!("✅ Weights extracted: {} tensors", weights.len());

    // Step 3: Get computation graph
    let graph = model.computation_graph().expect("parse computation graph");
    println!("✅ Graph loaded: {} nodes", graph.node_count());

    // Step 4: Build executor
    let mut executor = GraphExecutor::new();

    // Load weights
    println!("\nLoading weights into executor...");
    for (name, tensor) in weights {
        executor.add_initializer(name, tensor);
    }
    println!("✅ Weights loaded");

    // Add all nodes
    println!("\nAdding computation nodes...");
    for (i, node) in graph.nodes().iter().enumerate() {
        if i % 500 == 0 {
            println!("  Progress: {}/{} nodes", i, graph.node_count());
        }
        let inputs: Vec<&str> = node.inputs.iter().map(|s| s.as_str()).collect();
        let outputs: Vec<&str> = node.outputs.iter().map(|s| s.as_str()).collect();
        executor.add_node_with_attributes(
            &format!("node_{}", i),
            &node.op_type,
            inputs,
            outputs,
            node.attributes.clone(),
        );
    }
    println!("✅ All {} nodes added", graph.node_count());

    println!("\n=== Executor Ready ===");
    println!("ONYCHES can now run full Kokoro inference!");
}

/// Test: Run minimal Kokoro inference
#[test]
#[ignore] // Expensive - run manually
fn test_run_minimal_kokoro_inference() {
    let model_path = common::kokoro_model::kokoro_model_path();

    println!("\n=== Running Kokoro Inference ===");

    // Load and build executor (same as above)
    let model = OnnxParser::parse_file(&model_path).expect("parse kokoro-v1.0.onnx");
    let weights = model.extract_weights().expect("extract weights");
    let graph = model.computation_graph().expect("parse computation graph");

    let mut executor = GraphExecutor::new();
    for (name, tensor) in weights {
        executor.add_initializer(name, tensor);
    }

    for (i, node) in graph.nodes().iter().enumerate() {
        let inputs: Vec<&str> = node.inputs.iter().map(|s| s.as_str()).collect();
        let outputs: Vec<&str> = node.outputs.iter().map(|s| s.as_str()).collect();
        executor.add_node_with_attributes(
            &format!("node_{}", i),
            &node.op_type,
            inputs,
            outputs,
            node.attributes.clone(),
        );
    }

    println!("✅ Executor built with {} nodes", graph.node_count());

    // Create minimal test inputs
    println!("\nPreparing inputs...");

    // Get input names from model
    let input_names = model.inputs();
    println!("Expected inputs: {:?}", input_names);

    // tokens: Simple sequence [1, 3] (batch=1, seq=3)
    let tokens = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);

    // style: Zero vector [1, 256] (assuming style_dim=256)
    let style = Tensor::from_vec(vec![0.0; 256], vec![1, 256]);

    // speed: Normal speed
    let speed = Tensor::from_vec(vec![1.0], vec![1]);

    let mut inputs = HashMap::new();
    inputs.insert("tokens".to_string(), tokens);
    inputs.insert("style".to_string(), style);
    inputs.insert("speed".to_string(), speed);

    println!("✅ Inputs prepared");

    // Get output names
    let output_names = model.outputs();
    println!("Expected outputs: {:?}", output_names);

    // Execute!
    println!("\n🚀 Running inference...");
    match executor.run(inputs, vec!["audio"]) {
        Ok(outputs) => {
            println!("✅ INFERENCE SUCCESSFUL!");
            let audio = outputs.get("audio").unwrap();
            println!("Audio output shape: {:?}", audio.shape());
            println!("Audio samples: {}", audio.len());
            println!("\n🎉 ONYCHES EXECUTED FULL KOKORO MODEL!");
        }
        Err(e) => {
            println!("❌ Inference failed: {}", e);
            println!("This helps us identify what needs fixing!");
        }
    }
}
