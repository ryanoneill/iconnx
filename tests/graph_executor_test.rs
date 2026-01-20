/// Test Graph Executor for Iconnx
///
/// TDD: Tests written FIRST
/// The executor runs computation graphs with operators we've implemented
use iconnx::tensor::Tensor;
use std::collections::HashMap;

/// Test 1: Execute simple 2-node graph
///
/// Graph: input -> Add(input, const_2) -> Mul(result, const_3) -> output
///
/// Computation:
/// 1. c = input + 2  (Add node)
/// 2. output = c * 3  (Mul node)
///
/// For input = [1, 2, 3]:
/// - After Add: [3, 4, 5]
/// - After Mul: [9, 12, 15]
#[test]
fn test_execute_simple_graph() {
    use iconnx::graph_executor::GraphExecutor;

    // Create executor
    let mut executor = GraphExecutor::new();

    // Add initializers (constants/weights)
    let const_2 = Tensor::from_vec(vec![2.0], vec![1]);
    let const_3 = Tensor::from_vec(vec![3.0], vec![1]);

    executor.add_initializer("const_2".to_string(), const_2);
    executor.add_initializer("const_3".to_string(), const_3);

    // Build graph nodes
    // Node 1: c = Add(input, const_2)
    executor.add_node("add_node", "Add", vec!["input", "const_2"], vec!["c"]);

    // Node 2: output = Mul(c, const_3)
    executor.add_node("mul_node", "Mul", vec!["c", "const_3"], vec!["output"]);

    // Set inputs
    let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
    let mut inputs = HashMap::new();
    inputs.insert("input".to_string(), input);

    // Execute graph
    let outputs = executor.run(inputs, vec!["output"]).unwrap();

    // Verify output
    assert_eq!(outputs.len(), 1);
    let result = outputs.get("output").unwrap();
    assert_eq!(result.shape(), &[3]);
    assert_eq!(result.as_slice(), &[9.0, 12.0, 15.0]);
}

/// Test 2: Execute graph with multiple outputs
#[test]
fn test_execute_graph_multiple_outputs() {
    use iconnx::graph_executor::GraphExecutor;

    let mut executor = GraphExecutor::new();

    // Initializers
    let const_10 = Tensor::from_vec(vec![10.0], vec![1]);
    executor.add_initializer("const_10".to_string(), const_10);

    // Node 1: a = Add(input, const_10)
    executor.add_node("add_node", "Add", vec!["input", "const_10"], vec!["a"]);

    // Node 2: b = Mul(input, const_10)
    executor.add_node("mul_node", "Mul", vec!["input", "const_10"], vec!["b"]);

    // Inputs
    let input = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
    let mut inputs = HashMap::new();
    inputs.insert("input".to_string(), input);

    // Execute - request both outputs
    let outputs = executor.run(inputs, vec!["a", "b"]).unwrap();

    assert_eq!(outputs.len(), 2);

    // a = input + 10 = [11, 12]
    let a = outputs.get("a").unwrap();
    assert_eq!(a.as_slice(), &[11.0, 12.0]);

    // b = input * 10 = [10, 20]
    let b = outputs.get("b").unwrap();
    assert_eq!(b.as_slice(), &[10.0, 20.0]);
}

/// Test 3: Execute graph with MatMul
#[test]
fn test_execute_graph_with_matmul() {
    use iconnx::graph_executor::GraphExecutor;

    let mut executor = GraphExecutor::new();

    // Weight matrix (3x2)
    let weight = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
    executor.add_initializer("weight".to_string(), weight);

    // Bias vector
    let bias = Tensor::from_vec(vec![10.0, 20.0], vec![2]);
    executor.add_initializer("bias".to_string(), bias);

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

    // Input (1x3)
    let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
    let mut inputs = HashMap::new();
    inputs.insert("input".to_string(), input);

    // Execute
    let outputs = executor.run(inputs, vec!["output"]).unwrap();

    let result = outputs.get("output").unwrap();
    assert_eq!(result.shape(), &[1, 2]);

    // MatMul: [1,2,3] @ [[1,2],[3,4],[5,6]] = [22, 28]
    // Add bias: [22,28] + [10,20] = [32, 48]
    assert_eq!(result.as_slice(), &[32.0, 48.0]);
}

/// Test 4: Topological execution order
///
/// Graph with dependencies:
///   a = Add(input, 1)
///   b = Add(input, 2)
///   c = Mul(a, b)
///
/// Must execute a and b before c
#[test]
fn test_execution_order() {
    use iconnx::graph_executor::GraphExecutor;

    let mut executor = GraphExecutor::new();

    let one = Tensor::from_vec(vec![1.0], vec![1]);
    let two = Tensor::from_vec(vec![2.0], vec![1]);
    executor.add_initializer("one".to_string(), one);
    executor.add_initializer("two".to_string(), two);

    // Add nodes in WRONG order to test topological sorting
    executor.add_node("mul_node", "Mul", vec!["a", "b"], vec!["c"]);
    executor.add_node("add_a", "Add", vec!["input", "one"], vec!["a"]);
    executor.add_node("add_b", "Add", vec!["input", "two"], vec!["b"]);

    let input = Tensor::from_vec(vec![3.0], vec![1]);
    let mut inputs = HashMap::new();
    inputs.insert("input".to_string(), input);

    let outputs = executor.run(inputs, vec!["c"]).unwrap();

    let result = outputs.get("c").unwrap();
    // a = 3+1=4, b = 3+2=5, c = 4*5=20
    assert_eq!(result.as_slice(), &[20.0]);
}
