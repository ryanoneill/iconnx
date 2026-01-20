use iconnx::attributes::NodeAttributes;
/// Test NonZero operator for Iconnx
///
/// TDD: Tests written FIRST
/// NonZero: Find indices of non-zero elements
use iconnx::tensor::Tensor;

/// Test 1: NonZero in 1D
#[test]
fn test_nonzero_1d() {
    let data = Tensor::from_vec(vec![0.0, 1.0, 0.0, 3.0, 0.0], vec![5]);

    let result =
        iconnx::operators::nonzero::NonZero::forward(&[data], &NodeAttributes::new());

    // Result should be 1x2 (1 dimension, 2 non-zero elements)
    assert_eq!(result.shape(), &[1, 2]);
    assert_eq!(result.as_slice(), vec![1.0, 3.0]);
}

/// Test 2: NonZero in 2D
#[test]
fn test_nonzero_2d() {
    let data = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);

    let result =
        iconnx::operators::nonzero::NonZero::forward(&[data], &NodeAttributes::new());

    // Result should be 2x2 (2 dimensions, 2 non-zero elements)
    // Non-zero at (0,0) and (1,1)
    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.as_slice(), vec![0.0, 1.0, 0.0, 1.0]);
}

/// Test 3: All zeros
#[test]
fn test_nonzero_all_zeros() {
    let data = Tensor::from_vec(vec![0.0, 0.0, 0.0], vec![3]);

    let result =
        iconnx::operators::nonzero::NonZero::forward(&[data], &NodeAttributes::new());

    // Result should be 1x0 (no non-zero elements)
    assert_eq!(result.shape(), &[1, 0]);
    assert_eq!(result.as_slice().len(), 0);
}

/// Test 4: All non-zero
#[test]
fn test_nonzero_all_nonzero() {
    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

    let result =
        iconnx::operators::nonzero::NonZero::forward(&[data], &NodeAttributes::new());

    // Result should be 1x3
    assert_eq!(result.shape(), &[1, 3]);
    assert_eq!(result.as_slice(), vec![0.0, 1.0, 2.0]);
}
