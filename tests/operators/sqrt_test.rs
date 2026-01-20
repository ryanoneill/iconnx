use iconnx::attributes::NodeAttributes;
/// Test Sqrt operator for Iconnx
///
/// TDD: Tests written FIRST
/// Sqrt computes element-wise square root
use iconnx::tensor::Tensor;

/// Test 1: Basic square root
#[test]
fn test_sqrt_basic() {
    let data = Tensor::from_vec(vec![1.0, 4.0, 9.0, 16.0], vec![4]);

    let result = iconnx::operators::sqrt::Sqrt::forward(&[data], &NodeAttributes::new());

    assert_eq!(result.shape(), &[4]);
    assert_eq!(result.as_slice(), vec![1.0, 2.0, 3.0, 4.0]);
}

/// Test 2: Sqrt of zero
#[test]
fn test_sqrt_zero() {
    let data = Tensor::from_vec(vec![0.0], vec![1]);

    let result = iconnx::operators::sqrt::Sqrt::forward(&[data], &NodeAttributes::new());

    assert_eq!(result.as_slice(), vec![0.0]);
}

/// Test 3: Sqrt 2D
#[test]
fn test_sqrt_2d() {
    let data = Tensor::from_vec(vec![1.0, 4.0, 9.0, 16.0], vec![2, 2]);

    let result = iconnx::operators::sqrt::Sqrt::forward(&[data], &NodeAttributes::new());

    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.as_slice(), vec![1.0, 2.0, 3.0, 4.0]);
}
