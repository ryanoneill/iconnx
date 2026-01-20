/// Kokoro model validation tests
///
/// Tests ONYCHES against real Kokoro model subgraphs
use iconnx::onnx_parser::OnnxParser;
use std::collections::HashSet;
use std::path::Path;

/// Find nodes in Kokoro that we can execute with our operators
#[test]
fn test_find_executable_kokoro_subgraph() {
    let model_path = Path::new("kokoro-v1.0.onnx");

    if !model_path.exists() {
        println!("Skipping - kokoro-v1.0.onnx not found");
        return;
    }

    // Our implemented operators (49 total - 100% Kokoro coverage!)
    let supported: HashSet<&str> = vec![
        "Add",
        "And",
        "Atan",
        "Cast",
        "Clip",
        "Concat",
        "Constant",
        "ConstantOfShape",
        "Conv",
        "ConvTranspose",
        "Cos",
        "CumSum",
        "Div",
        "Equal",
        "Exp",
        "Expand",
        "Floor",
        "Gather",
        "Gemm",
        "Greater",
        "GreaterOrEqual",
        "LayerNormalization",
        "LeakyRelu",
        "Less",
        "LSTM",
        "MatMul",
        "Mul",
        "NonZero",
        "Pad",
        "Pow",
        "Range",
        "ReduceMean",
        "ReduceSum",
        "Reshape",
        "Resize",
        "Round",
        "ScatterND",
        "Shape",
        "Sigmoid",
        "Sin",
        "Slice",
        "Softmax",
        "Sqrt",
        "Squeeze",
        "STFT",
        "Sub",
        "Tanh",
        "Transpose",
        "Unsqueeze",
        "Where",
    ]
    .into_iter()
    .collect();

    // Load Kokoro
    let model = OnnxParser::parse_file(model_path).unwrap();
    let graph = model.computation_graph();
    let nodes = graph.nodes();

    println!("\n=== Kokoro Graph Analysis ===");
    println!("Total nodes: {}", nodes.len());

    // Find longest executable sequence from start
    let mut executable_count = 0;
    let mut first_unsupported = None;

    for (i, node) in nodes.iter().enumerate() {
        if supported.contains(node.op_type.as_str()) {
            executable_count += 1;
        } else {
            first_unsupported = Some((i, node.op_type.clone()));
            break;
        }
    }

    println!("\nExecutable from start: {} nodes", executable_count);
    if let Some((idx, op)) = first_unsupported {
        println!("Stopped at node {}: {} (not implemented)", idx, op);
    }

    // Find all executable nodes (anywhere in graph)
    let all_executable: Vec<_> = nodes
        .iter()
        .filter(|n| supported.contains(n.op_type.as_str()))
        .collect();

    println!(
        "\nTotal executable nodes: {}/{}",
        all_executable.len(),
        nodes.len()
    );
    println!(
        "Executable percentage: {:.1}%",
        (all_executable.len() as f32 / nodes.len() as f32) * 100.0
    );

    // Group by operator type
    use std::collections::HashMap;
    let mut op_counts: HashMap<&str, usize> = HashMap::new();
    for node in &all_executable {
        *op_counts.entry(node.op_type.as_str()).or_insert(0) += 1;
    }

    println!("\nExecutable nodes by operator:");
    let mut sorted_ops: Vec<_> = op_counts.iter().collect();
    sorted_ops.sort_by_key(|(_, count)| std::cmp::Reverse(**count));
    for (op, count) in sorted_ops.iter().take(10) {
        println!("  {}: {} nodes", op, count);
    }

    // Find continuous executable sequences
    let mut sequences = Vec::new();
    let mut current_seq = Vec::new();

    for node in nodes {
        if supported.contains(node.op_type.as_str()) {
            current_seq.push(node.clone());
        } else {
            if !current_seq.is_empty() {
                sequences.push(current_seq.clone());
                current_seq.clear();
            }
        }
    }
    if !current_seq.is_empty() {
        sequences.push(current_seq);
    }

    println!("\nContinuous executable sequences: {}", sequences.len());
    if let Some(longest) = sequences.iter().max_by_key(|s| s.len()) {
        println!("Longest sequence: {} nodes", longest.len());
        println!(
            "First 5 ops: {:?}",
            longest
                .iter()
                .take(5)
                .map(|n| &n.op_type)
                .collect::<Vec<_>>()
        );
    }

    // Verify we have significant coverage
    assert!(
        executable_count > 0,
        "Should find at least some executable nodes"
    );
    assert!(
        all_executable.len() as f32 / nodes.len() as f32 > 0.3,
        "Should be able to execute >30% of nodes"
    );
}

/// Test: Check if we can find a complete subgraph path
#[test]
#[ignore] // Run manually when ready to extract subgraph
fn test_extract_and_execute_subgraph() {
    // This will:
    // 1. Find executable subgraph
    // 2. Extract needed weights from Kokoro
    // 3. Build executor with those weights
    // 4. Run inference
    // 5. Compare with ort

    println!("TODO: Implement subgraph extraction and execution");
    println!("This requires:");
    println!("1. Weight extraction from ONNX initializers");
    println!("2. Input tensor creation");
    println!("3. ort session for comparison");
}
