/// Test LeakyRelu operator for Iconnx
///
/// TDD: Tests written FIRST
/// LeakyRelu: x if x > 0 else alpha * x
use iconnx::attributes::NodeAttributes;
use iconnx::tensor::Tensor;

/// Test 1: Basic LeakyRelu with default alpha (0.01)
#[test]
fn test_leaky_relu_default() {
    let data = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);

    // Use default alpha=0.01
    let result = iconnx::operators::leaky_relu::LeakyRelu::forward(
        &[data],
        &NodeAttributes::new(),
    );

    assert_eq!(result.shape(), &[5]);
    assert_eq!(result.as_slice(), vec![-0.02, -0.01, 0.0, 1.0, 2.0]);
}

/// Test 2: LeakyRelu with custom alpha
#[test]
fn test_leaky_relu_custom_alpha() {
    let data = Tensor::from_vec(vec![-10.0, -5.0, 0.0, 5.0, 10.0], vec![5]);

    // Set custom alpha=0.1 via attributes
    let mut attrs = NodeAttributes::new();
    attrs.add_float("alpha".to_string(), 0.1);

    let result = iconnx::operators::leaky_relu::LeakyRelu::forward(&[data], &attrs);

    assert_eq!(result.as_slice(), vec![-1.0, -0.5, 0.0, 5.0, 10.0]);
}

/// Test 3: LeakyRelu with alpha=0 (regular ReLU)
#[test]
fn test_leaky_relu_alpha_zero() {
    let data = Tensor::from_vec(vec![-3.0, -1.0, 0.0, 1.0, 3.0], vec![5]);

    // alpha=0 makes it regular ReLU
    let mut attrs = NodeAttributes::new();
    attrs.add_float("alpha".to_string(), 0.0);

    let result = iconnx::operators::leaky_relu::LeakyRelu::forward(&[data], &attrs);

    assert_eq!(result.as_slice(), vec![0.0, 0.0, 0.0, 1.0, 3.0]);
}
