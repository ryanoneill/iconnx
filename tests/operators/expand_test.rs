use iconnx::attributes::NodeAttributes;
/// Test Expand operator for Iconnx
///
/// TDD: Tests written FIRST
/// Expand: Broadcast tensor to target shape
use iconnx::tensor::Tensor;

/// Test 1: Expand 1D to 2D
#[test]
fn test_expand_1d_to_2d() {
    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
    let shape = Tensor::from_vec(vec![2.0, 3.0], vec![2]);

    let result = iconnx::operators::expand::Expand::forward(
        &[data, shape],
        &NodeAttributes::new(),
    );

    assert_eq!(result.shape(), &[2, 3]);
    assert_eq!(result.as_slice(), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
}

/// Test 2: Expand scalar to 1D
#[test]
fn test_expand_scalar_to_1d() {
    let data = Tensor::from_vec(vec![5.0], vec![1]);
    let shape = Tensor::from_vec(vec![4.0], vec![1]);

    let result = iconnx::operators::expand::Expand::forward(
        &[data, shape],
        &NodeAttributes::new(),
    );

    assert_eq!(result.shape(), &[4]);
    assert_eq!(result.as_slice(), vec![5.0, 5.0, 5.0, 5.0]);
}

/// Test 3: Expand with broadcast dimension
#[test]
fn test_expand_broadcast() {
    let data = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]);
    let shape = Tensor::from_vec(vec![3.0, 2.0], vec![2]);

    let result = iconnx::operators::expand::Expand::forward(
        &[data, shape],
        &NodeAttributes::new(),
    );

    assert_eq!(result.shape(), &[3, 2]);
    assert_eq!(result.as_slice(), vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
}
