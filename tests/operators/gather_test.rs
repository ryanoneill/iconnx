use iconnx::attributes::NodeAttributes;
/// Test Gather operator for Iconnx
///
/// TDD: Tests written FIRST
/// Gather: Select elements along an axis using indices
use iconnx::tensor::Tensor;

/// Test 1: Basic gather along axis 0
#[test]
fn test_gather_axis_0() {
    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
    let indices = Tensor::from_vec(vec![0.0, 2.0], vec![2]);
    let axis = Tensor::from_vec(vec![0.0], vec![1]);

    let result = iconnx::operators::gather::Gather::forward(
        &[data, indices, axis],
        &NodeAttributes::new(),
    );

    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.as_slice(), vec![1.0, 2.0, 5.0, 6.0]);
}

/// Test 2: Gather along axis 1
#[test]
fn test_gather_axis_1() {
    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
    let indices = Tensor::from_vec(vec![1.0], vec![1]);
    let axis = Tensor::from_vec(vec![1.0], vec![1]);

    let result = iconnx::operators::gather::Gather::forward(
        &[data, indices, axis],
        &NodeAttributes::new(),
    );

    assert_eq!(result.shape(), &[3, 1]);
    assert_eq!(result.as_slice(), vec![2.0, 4.0, 6.0]);
}

/// Test 3: Gather 1D
#[test]
fn test_gather_1d() {
    let data = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0], vec![5]);
    let indices = Tensor::from_vec(vec![0.0, 3.0, 1.0], vec![3]);
    let axis = Tensor::from_vec(vec![0.0], vec![1]);

    let result = iconnx::operators::gather::Gather::forward(
        &[data, indices, axis],
        &NodeAttributes::new(),
    );

    assert_eq!(result.shape(), &[3]);
    assert_eq!(result.as_slice(), vec![10.0, 40.0, 20.0]);
}
