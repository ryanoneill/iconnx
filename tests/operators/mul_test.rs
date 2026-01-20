use iconnx::attributes::NodeAttributes;
/// Test Mul operator for Iconnx
///
/// TDD: Tests written FIRST to define expected behavior
/// The Mul operator implements element-wise multiplication with broadcasting
use iconnx::tensor::Tensor;

/// Test 1: Simple element-wise multiplication (same shape)
#[test]
fn test_mul_same_shape() {
    let a = Tensor::from_vec(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2]);
    let b = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    let result = iconnx::operators::mul::Mul::forward(&[a, b], &NodeAttributes::new());

    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.as_slice(), &[2.0, 6.0, 12.0, 20.0]);
}

/// Test 2: Multiply by scalar (broadcasting)
#[test]
fn test_mul_scalar_broadcast() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::from_vec(vec![2.0], vec![1]);

    let result = iconnx::operators::mul::Mul::forward(&[a, b], &NodeAttributes::new());

    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.as_slice(), &[2.0, 4.0, 6.0, 8.0]);
}

/// Test 3: Multiply with broadcasting
#[test]
fn test_mul_broadcast() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::from_vec(vec![2.0, 3.0], vec![2]);

    let result = iconnx::operators::mul::Mul::forward(&[a, b], &NodeAttributes::new());

    assert_eq!(result.shape(), &[2, 2]);
    // Row 0: [1*2, 2*3] = [2, 6]
    // Row 1: [3*2, 4*3] = [6, 12]
    assert_eq!(result.as_slice(), &[2.0, 6.0, 6.0, 12.0]);
}

/// Test 4: Multiply three inputs
#[test]
fn test_mul_three_inputs() {
    let a = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
    let b = Tensor::from_vec(vec![4.0, 5.0], vec![2]);
    let c = Tensor::from_vec(vec![6.0, 7.0], vec![2]);

    let result = iconnx::operators::mul::Mul::forward(&[a, b, c], &NodeAttributes::new());

    assert_eq!(result.shape(), &[2]);
    assert_eq!(result.as_slice(), &[48.0, 105.0]); // 2*4*6=48, 3*5*7=105
}

/// Test 5: Multiply with negative numbers
#[test]
fn test_mul_negative() {
    let a = Tensor::from_vec(vec![2.0, -3.0, 4.0], vec![3]);
    let b = Tensor::from_vec(vec![-1.0, 2.0, -3.0], vec![3]);

    let result = iconnx::operators::mul::Mul::forward(&[a, b], &NodeAttributes::new());

    assert_eq!(result.shape(), &[3]);
    assert_eq!(result.as_slice(), &[-2.0, -6.0, -12.0]);
}

/// Test 6: Multiply by zero
#[test]
fn test_mul_zero() {
    let a = Tensor::from_vec(vec![5.0, 10.0, 15.0], vec![3]);
    let b = Tensor::from_vec(vec![0.0], vec![1]);

    let result = iconnx::operators::mul::Mul::forward(&[a, b], &NodeAttributes::new());

    assert_eq!(result.shape(), &[3]);
    assert_eq!(result.as_slice(), &[0.0, 0.0, 0.0]);
}
