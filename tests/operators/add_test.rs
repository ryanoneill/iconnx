/// Test Add operator for Iconnx
///
/// TDD: Tests written FIRST to define expected behavior
/// The Add operator implements element-wise addition with broadcasting
use iconnx::attributes::NodeAttributes;
use iconnx::tensor::Tensor;

/// Test 1: Simple element-wise addition (same shape)
#[test]
fn test_add_same_shape() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

    let result = iconnx::operators::add::Add::forward(&[a, b], &NodeAttributes::new());

    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.as_slice(), &[6.0, 8.0, 10.0, 12.0]);
}

/// Test 2: Add scalar to tensor (broadcasting)
#[test]
fn test_add_scalar_broadcast() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::from_vec(vec![10.0], vec![1]); // Scalar-like [1] shape

    let result = iconnx::operators::add::Add::forward(&[a, b], &NodeAttributes::new());

    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.as_slice(), &[11.0, 12.0, 13.0, 14.0]);
}

/// Test 3: Add 1D vector to 2D matrix (row broadcasting)
#[test]
fn test_add_broadcast_1d_to_2d() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::from_vec(vec![10.0, 20.0], vec![2]); // Will broadcast to each row

    let result = iconnx::operators::add::Add::forward(&[a, b], &NodeAttributes::new());

    assert_eq!(result.shape(), &[2, 2]);
    // Row 0: [1+10, 2+20] = [11, 22]
    // Row 1: [3+10, 4+20] = [13, 24]
    assert_eq!(result.as_slice(), &[11.0, 22.0, 13.0, 24.0]);
}

/// Test 4: Add with 3 inputs (chained addition)
#[test]
fn test_add_three_inputs() {
    let a = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
    let b = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
    let c = Tensor::from_vec(vec![5.0, 6.0], vec![2]);

    let result = iconnx::operators::add::Add::forward(&[a, b, c], &NodeAttributes::new());

    assert_eq!(result.shape(), &[2]);
    assert_eq!(result.as_slice(), &[9.0, 12.0]); // 1+3+5=9, 2+4+6=12
}

/// Test 5: Single input (identity)
#[test]
fn test_add_single_input() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

    let result =
        iconnx::operators::add::Add::forward(&[a.clone()], &NodeAttributes::new());

    assert_eq!(result.shape(), &[3]);
    assert_eq!(result.as_slice(), &[1.0, 2.0, 3.0]);
}

/// Test 6: Add with negative numbers
#[test]
fn test_add_negative_numbers() {
    let a = Tensor::from_vec(vec![1.0, -2.0, 3.0], vec![3]);
    let b = Tensor::from_vec(vec![-1.0, 2.0, -3.0], vec![3]);

    let result = iconnx::operators::add::Add::forward(&[a, b], &NodeAttributes::new());

    assert_eq!(result.shape(), &[3]);
    assert_eq!(result.as_slice(), &[0.0, 0.0, 0.0]);
}
