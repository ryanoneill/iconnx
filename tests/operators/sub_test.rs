use iconnx::attributes::NodeAttributes;
/// Test Sub operator for Iconnx
///
/// TDD: Tests written FIRST
/// The Sub operator implements element-wise subtraction with broadcasting
use iconnx::tensor::Tensor;

/// Test 1: Basic subtraction
#[test]
fn test_sub_same_shape() {
    let a = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]);
    let b = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    let result = iconnx::operators::sub::Sub::forward(&[a, b], &NodeAttributes::new());

    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.as_slice(), &[9.0, 18.0, 27.0, 36.0]);
}

/// Test 2: Subtract scalar
#[test]
fn test_sub_scalar() {
    let a = Tensor::from_vec(vec![10.0, 20.0, 30.0], vec![3]);
    let b = Tensor::from_vec(vec![5.0], vec![1]);

    let result = iconnx::operators::sub::Sub::forward(&[a, b], &NodeAttributes::new());

    assert_eq!(result.as_slice(), &[5.0, 15.0, 25.0]);
}

/// Test 3: Negative result
#[test]
fn test_sub_negative_result() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
    let b = Tensor::from_vec(vec![5.0, 5.0, 5.0], vec![3]);

    let result = iconnx::operators::sub::Sub::forward(&[a, b], &NodeAttributes::new());

    assert_eq!(result.as_slice(), &[-4.0, -3.0, -2.0]);
}
