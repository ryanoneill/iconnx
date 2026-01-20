use iconnx::attributes::NodeAttributes;
/// Test Div operator for Iconnx
///
/// TDD: Tests written FIRST
/// The Div operator implements element-wise division with broadcasting
use iconnx::tensor::Tensor;

/// Test 1: Basic division
#[test]
fn test_div_same_shape() {
    let a = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]);
    let b = Tensor::from_vec(vec![2.0, 4.0, 5.0, 8.0], vec![2, 2]);

    let result = iconnx::operators::div::Div::forward(&[a, b], &NodeAttributes::new());

    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.as_slice(), &[5.0, 5.0, 6.0, 5.0]);
}

/// Test 2: Divide by scalar
#[test]
fn test_div_scalar() {
    let a = Tensor::from_vec(vec![10.0, 20.0, 30.0], vec![3]);
    let b = Tensor::from_vec(vec![2.0], vec![1]);

    let result = iconnx::operators::div::Div::forward(&[a, b], &NodeAttributes::new());

    assert_eq!(result.as_slice(), &[5.0, 10.0, 15.0]);
}

/// Test 3: Division with negatives
#[test]
fn test_div_negative() {
    let a = Tensor::from_vec(vec![-10.0, 20.0, -30.0], vec![3]);
    let b = Tensor::from_vec(vec![2.0, -4.0, 5.0], vec![3]);

    let result = iconnx::operators::div::Div::forward(&[a, b], &NodeAttributes::new());

    assert_eq!(result.as_slice(), &[-5.0, -5.0, -6.0]);
}
