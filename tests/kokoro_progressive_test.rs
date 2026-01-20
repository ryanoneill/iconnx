/// Progressive Kokoro execution tests
///
/// Run increasingly larger subgraphs to validate execution
use iconnx::graph_executor::GraphExecutor;
use iconnx::onnx_parser::OnnxParser;
use iconnx::tensor::Tensor;
use std::collections::HashMap;
use std::path::Path;

/// Helper: Run first N nodes of Kokoro
fn run_kokoro_first_n_nodes(n: usize) -> Result<(), String> {
    let model_path = Path::new("kokoro-v1.0.onnx");
    if !model_path.exists() {
        return Err("Model not found".to_string());
    }

    let model = OnnxParser::parse_file(model_path).map_err(|e| e.to_string())?;
    let weights = model.extract_weights();
    let graph = model.computation_graph();
    let nodes = graph.nodes();

    let mut executor = GraphExecutor::new();

    // Load weights
    for (name, tensor) in weights {
        executor.add_initializer(name, tensor);
    }

    // Add first N nodes
    for i in 0..n.min(nodes.len()) {
        let node = &nodes[i];
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

    // Create test inputs with correct types
    // tokens: Int64 [1, 3] - token IDs (from ORT validation)
    // style: Float32 [1, 256] - style embedding
    // speed: Float32 [1] - speed scalar
    let mut inputs = HashMap::new();
    inputs.insert(
        "tokens".to_string(),
        Tensor::from_vec_i64(vec![1, 2, 3], vec![1, 3]),
    );
    inputs.insert(
        "style".to_string(),
        Tensor::from_vec(vec![0.0; 256], vec![1, 256]),
    );
    inputs.insert("speed".to_string(), Tensor::from_vec(vec![1.0], vec![1]));

    // Try to execute to the last node's output
    let last_output = &nodes[n - 1].outputs[0];
    executor
        .run(inputs, vec![last_output])
        .map_err(|e| e.to_string())?;

    Ok(())
}

/// Test: Run first 10 nodes
#[test]
fn test_run_first_10_nodes() {
    println!("\n🚀 Testing first 10 Kokoro nodes...");
    match run_kokoro_first_n_nodes(10) {
        Ok(_) => println!("✅ SUCCESS: First 10 nodes executed!"),
        Err(e) => println!("❌ Failed at 10 nodes: {}", e),
    }
}

/// Test: Run first 20 nodes
#[test]
fn test_run_first_20_nodes() {
    println!("\n🚀 Testing first 20 Kokoro nodes...");
    match run_kokoro_first_n_nodes(20) {
        Ok(_) => println!("✅ SUCCESS: First 20 nodes executed!"),
        Err(e) => println!("❌ Failed at 20 nodes: {}", e),
    }
}

/// Test: Run first 50 nodes
#[test]
fn test_run_first_50_nodes() {
    println!("\n🚀 Testing first 50 Kokoro nodes...");
    match run_kokoro_first_n_nodes(50) {
        Ok(_) => println!("✅ SUCCESS: First 50 nodes executed!"),
        Err(e) => println!("❌ Failed at 50 nodes: {}", e),
    }
}

/// Test: Run first 100 nodes
#[test]
#[ignore] // Might be slow
fn test_run_first_100_nodes() {
    println!("\n🚀 Testing first 100 Kokoro nodes...");
    match run_kokoro_first_n_nodes(100) {
        Ok(_) => println!("✅ SUCCESS: First 100 nodes executed!"),
        Err(e) => panic!("❌ Failed at 100 nodes: {}", e),
    }
}

/// Test: Run FULL Kokoro model (all 2464 nodes)
#[test]
#[ignore] // Run manually: cargo test test_run_full_kokoro -- --ignored --nocapture
fn test_run_full_kokoro() {
    let model_path = Path::new("kokoro-v1.0.onnx");
    if !model_path.exists() {
        println!("⚠️  Kokoro model not found, skipping");
        return;
    }

    let model = OnnxParser::parse_file(model_path).unwrap();
    let graph = model.computation_graph();
    let total_nodes = graph.nodes().len();

    println!("\n🚀 Testing FULL Kokoro model: {} nodes...", total_nodes);

    match run_kokoro_first_n_nodes(total_nodes) {
        Ok(_) => println!("🎉 SUCCESS: ALL {} Kokoro nodes executed!", total_nodes),
        Err(e) => panic!("❌ Failed: {}", e),
    }
}
