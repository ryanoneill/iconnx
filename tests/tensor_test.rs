/// Test Tensor implementation for Iconnx
///
/// TDD: These tests are written FIRST, before implementation.
/// They define the behavior we expect from the Tensor type.
use ndarray::Array2;

/// Test 1: Create tensor from f32 data
///
/// Most basic tensor operation - wrapping ndarray
#[test]
fn test_create_tensor_from_f32_vec() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = iconnx::tensor::Tensor::from_vec(data.clone(), vec![2, 2]);

    // Should have correct shape
    assert_eq!(tensor.shape(), &[2, 2]);

    // Should contain correct data
    let values = tensor.as_slice();
    assert_eq!(values, &[1.0, 2.0, 3.0, 4.0]);
}

/// Test 2: Create 1D tensor
#[test]
fn test_create_1d_tensor() {
    let data = vec![1.0, 2.0, 3.0];
    let tensor = iconnx::tensor::Tensor::from_vec(data, vec![3]);

    assert_eq!(tensor.shape(), &[3]);
    assert_eq!(tensor.ndim(), 1);
}

/// Test 3: Create 3D tensor
#[test]
fn test_create_3d_tensor() {
    let data = vec![1.0; 24]; // 2x3x4 = 24 elements
    let tensor = iconnx::tensor::Tensor::from_vec(data, vec![2, 3, 4]);

    assert_eq!(tensor.shape(), &[2, 3, 4]);
    assert_eq!(tensor.ndim(), 3);
}

/// Test 4: Get tensor element count
#[test]
fn test_tensor_len() {
    let tensor = iconnx::tensor::Tensor::from_vec(vec![1.0; 12], vec![3, 4]);

    assert_eq!(tensor.len(), 12);
}

/// Test 5: Convert from ndarray
#[test]
fn test_from_ndarray() {
    let arr = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let tensor = iconnx::tensor::Tensor::from_array(arr.into_dyn());

    assert_eq!(tensor.shape(), &[2, 3]);
    assert_eq!(tensor.len(), 6);
}

/// Test 6: Convert to ndarray
#[test]
fn test_to_ndarray() {
    let tensor = iconnx::tensor::Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
    let arr = tensor.to_array();

    assert_eq!(arr.shape(), &[3]);
    assert_eq!(arr.as_slice().unwrap(), &[1.0, 2.0, 3.0]);
}

/// Test 7: Validate shape mismatch panics
#[test]
#[should_panic(expected = "Shape mismatch")]
fn test_shape_mismatch_panics() {
    // 4 elements but shape says 6 (2x3)
    let _tensor = iconnx::tensor::Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 3]);
}

/// Test 8: Create scalar (0D tensor)
#[test]
fn test_create_scalar_tensor() {
    let tensor = iconnx::tensor::Tensor::from_vec(vec![42.0], vec![]);

    assert_eq!(tensor.shape(), &[] as &[usize]);
    assert_eq!(tensor.ndim(), 0);
    assert_eq!(tensor.len(), 1);
}
