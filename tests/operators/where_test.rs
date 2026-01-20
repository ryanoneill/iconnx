use iconnx::attributes::NodeAttributes;
/// Test Where operator for Iconnx
///
/// TDD: Tests written FIRST
/// Where: condition ? x : y (element-wise conditional selection)
use iconnx::tensor::Tensor;

/// Test 1: Basic conditional selection
#[test]
fn test_where_basic() {
    let condition = Tensor::from_vec(vec![1.0, 0.0, 1.0, 0.0], vec![4]);
    let x = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], vec![4]);
    let y = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]);

    let result = iconnx::operators::where_op::Where::forward(
        &[condition, x, y],
        &NodeAttributes::new(),
    );

    assert_eq!(result.shape(), &[4]);
    assert_eq!(result.as_slice(), vec![10.0, 2.0, 30.0, 4.0]);
}

/// Test 2: Where with 2D tensors
#[test]
fn test_where_2d() {
    let condition = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
    let x = Tensor::from_vec(vec![100.0, 200.0, 300.0, 400.0], vec![2, 2]);
    let y = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    let result = iconnx::operators::where_op::Where::forward(
        &[condition, x, y],
        &NodeAttributes::new(),
    );

    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.as_slice(), vec![100.0, 2.0, 3.0, 400.0]);
}

/// Test 3: All true
#[test]
fn test_where_all_true() {
    let condition = Tensor::from_vec(vec![1.0, 1.0, 1.0], vec![3]);
    let x = Tensor::from_vec(vec![10.0, 20.0, 30.0], vec![3]);
    let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

    let result = iconnx::operators::where_op::Where::forward(
        &[condition, x, y],
        &NodeAttributes::new(),
    );

    assert_eq!(result.as_slice(), vec![10.0, 20.0, 30.0]);
}

/// Test 4: All false
#[test]
fn test_where_all_false() {
    let condition = Tensor::from_vec(vec![0.0, 0.0, 0.0], vec![3]);
    let x = Tensor::from_vec(vec![10.0, 20.0, 30.0], vec![3]);
    let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

    let result = iconnx::operators::where_op::Where::forward(
        &[condition, x, y],
        &NodeAttributes::new(),
    );

    assert_eq!(result.as_slice(), vec![1.0, 2.0, 3.0]);
}
