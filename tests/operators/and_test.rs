use iconnx::attributes::NodeAttributes;
/// Test And operator for Iconnx
///
/// TDD: Tests written FIRST
/// And: Logical AND (element-wise), 1.0 = true, 0.0 = false
use iconnx::tensor::Tensor;

/// Test 1: Basic AND operation
#[test]
fn test_and_basic() {
    let a = Tensor::from_vec(vec![1.0, 1.0, 0.0, 0.0], vec![4]);
    let b = Tensor::from_vec(vec![1.0, 0.0, 1.0, 0.0], vec![4]);

    let result = iconnx::operators::and::And::forward(&[a, b], &NodeAttributes::new());

    assert_eq!(result.shape(), &[4]);
    assert_eq!(result.as_slice(), vec![1.0, 0.0, 0.0, 0.0]);
}

/// Test 2: AND with 2D tensors
#[test]
fn test_and_2d() {
    let a = Tensor::from_vec(vec![1.0, 0.0, 1.0, 1.0], vec![2, 2]);
    let b = Tensor::from_vec(vec![1.0, 1.0, 0.0, 1.0], vec![2, 2]);

    let result = iconnx::operators::and::And::forward(&[a, b], &NodeAttributes::new());

    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.as_slice(), vec![1.0, 0.0, 0.0, 1.0]);
}

/// Test 3: All true
#[test]
fn test_and_all_true() {
    let a = Tensor::from_vec(vec![1.0, 1.0, 1.0], vec![3]);
    let b = Tensor::from_vec(vec![1.0, 1.0, 1.0], vec![3]);

    let result = iconnx::operators::and::And::forward(&[a, b], &NodeAttributes::new());

    assert_eq!(result.as_slice(), vec![1.0, 1.0, 1.0]);
}

/// Test 4: All false
#[test]
fn test_and_all_false() {
    let a = Tensor::from_vec(vec![0.0, 0.0, 1.0], vec![3]);
    let b = Tensor::from_vec(vec![0.0, 1.0, 0.0], vec![3]);

    let result = iconnx::operators::and::And::forward(&[a, b], &NodeAttributes::new());

    assert_eq!(result.as_slice(), vec![0.0, 0.0, 0.0]);
}
