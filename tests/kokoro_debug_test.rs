/// Debug Kokoro execution issues
#[path = "common/mod.rs"]
mod common;

use iconnx::graph_executor::GraphExecutor;
use iconnx::onnx_parser::OnnxParser;
use iconnx::tensor::Tensor;
use std::collections::HashMap;

/// Test: Debug first 10 nodes of Kokoro
#[test]
fn test_debug_first_10_nodes() {
    let model_path = common::kokoro_model::kokoro_model_path();
    let model = OnnxParser::parse_file(&model_path).expect("parse kokoro-v1.0.onnx");
    let graph = model.computation_graph().expect("parse computation graph");
    let nodes = graph.nodes();

    println!("\n=== First 10 Kokoro Nodes ===");
    for (i, node) in nodes.iter().enumerate().take(10) {
        println!("\nNode {}: {}", i, node.op_type);
        println!("  Inputs: {:?}", node.inputs);
        println!("  Outputs: {:?}", node.outputs);

        // Check for Constant nodes
        if node.op_type == "Constant" {
            if let Some(value) = node.attributes.get_tensor("value") {
                println!("  Has 'value' attribute: shape={:?}", value.shape());
            } else {
                println!("  ⚠️  Missing 'value' attribute!");
            }
        }

        // Check for other attributes
        if let Some(axis) = node.attributes.get_int("axis") {
            println!("  axis={}", axis);
        }
    }
}

/// Test: Try to run just the first few Constant nodes
#[test]
fn test_run_first_constant_nodes() {
    let model_path = common::kokoro_model::kokoro_model_path();
    let model = OnnxParser::parse_file(&model_path).expect("parse kokoro-v1.0.onnx");
    let weights = model.extract_weights().expect("extract weights");
    let graph = model.computation_graph().expect("parse computation graph");
    let nodes = graph.nodes();

    let mut executor = GraphExecutor::new();

    // Load weights
    for (name, tensor) in weights {
        executor.add_initializer(name, tensor);
    }

    // Add only first 5 nodes
    println!("\n=== Adding First 5 Nodes ===");
    for (i, node) in nodes.iter().enumerate().take(5) {
        println!("Node {}: {} → {:?}", i, node.op_type, node.outputs);

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

    // Create dummy inputs (won't actually use them yet)
    let mut inputs = HashMap::new();
    inputs.insert("tokens".to_string(), Tensor::from_vec(vec![1.0], vec![1]));
    inputs.insert(
        "style".to_string(),
        Tensor::from_vec(vec![0.0; 256], vec![1, 256]),
    );
    inputs.insert("speed".to_string(), Tensor::from_vec(vec![1.0], vec![1]));

    // Try to run
    println!("\n🚀 Attempting execution of first 5 nodes...");

    // Get output from last node
    let last_output = &nodes[4].outputs[0];
    match executor.run(inputs, vec![last_output]) {
        Ok(outputs) => {
            println!("✅ SUCCESS! First 5 nodes executed");
            for (name, tensor) in outputs {
                println!("  Output '{}': shape={:?}", name, tensor.shape());
            }
        }
        Err(e) => {
            println!("❌ Failed: {}", e);
            println!("This tells us what's missing!");
        }
    }
}
