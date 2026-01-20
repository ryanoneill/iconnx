use iconnx::attributes::NodeAttributes;
/// Test activation operators for Iconnx
///
/// TDD: Tests written FIRST
/// Activation functions: Sigmoid, Tanh
use iconnx::tensor::Tensor;

/// Test Sigmoid: 1 / (1 + exp(-x))
#[test]
fn test_sigmoid_basic() {
    let data = Tensor::from_vec(vec![0.0, 1.0, -1.0], vec![3]);

    let result =
        iconnx::operators::sigmoid::Sigmoid::forward(&[data], &NodeAttributes::new());

    assert_eq!(result.shape(), &[3]);

    // Sigmoid(0) = 0.5
    // Sigmoid(1) ≈ 0.731
    // Sigmoid(-1) ≈ 0.269
    let output = result.as_slice();
    assert!((output[0] - 0.5).abs() < 0.001);
    assert!((output[1] - 0.731).abs() < 0.01);
    assert!((output[2] - 0.269).abs() < 0.01);
}

/// Test Tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
#[test]
fn test_tanh_basic() {
    let data = Tensor::from_vec(vec![0.0, 1.0, -1.0], vec![3]);

    let result = iconnx::operators::tanh::Tanh::forward(&[data], &NodeAttributes::new());

    assert_eq!(result.shape(), &[3]);

    // Tanh(0) = 0
    // Tanh(1) ≈ 0.762
    // Tanh(-1) ≈ -0.762
    let output = result.as_slice();
    assert!(output[0].abs() < 0.001);
    assert!((output[1] - 0.762).abs() < 0.01);
    assert!((output[2] + 0.762).abs() < 0.01);
}

/// Test Sigmoid 2D
#[test]
fn test_sigmoid_2d() {
    let data = Tensor::from_vec(vec![0.0, 1.0, -1.0, 2.0], vec![2, 2]);

    let result =
        iconnx::operators::sigmoid::Sigmoid::forward(&[data], &NodeAttributes::new());

    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.len(), 4);
}

/// Test Tanh 2D
#[test]
fn test_tanh_2d() {
    let data = Tensor::from_vec(vec![0.0, 1.0, -1.0, 2.0], vec![2, 2]);

    let result = iconnx::operators::tanh::Tanh::forward(&[data], &NodeAttributes::new());

    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.len(), 4);
}
