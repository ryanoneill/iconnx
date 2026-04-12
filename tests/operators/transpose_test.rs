use iconnx::attributes::NodeAttributes;
/// Test Transpose operator for Iconnx
///
/// TDD: Tests written FIRST
/// Transpose permutes tensor dimensions
/// ONNX spec: https://onnx.ai/onnx/operators/onnx__Transpose.html
use iconnx::tensor::Tensor;

/// Test 1: Transpose 2D matrix
#[test]
fn test_transpose_2d() {
    // [[1, 2, 3],
    //  [4, 5, 6]]  (2x3)
    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

    let result =
        iconnx::operators::transpose::Transpose::forward(&[data], &NodeAttributes::new());

    // Should be transposed to 3x2:
    // [[1, 4],
    //  [2, 5],
    //  [3, 6]]
    assert_eq!(result.shape(), &[3, 2]);
    assert_eq!(result.as_slice(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

/// Test 2: Transpose 1D (no-op)
#[test]
fn test_transpose_1d() {
    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

    let result = iconnx::operators::transpose::Transpose::forward(
        std::slice::from_ref(&data),
        &NodeAttributes::new(),
    );

    // 1D transpose is identity
    assert_eq!(result.shape(), data.shape());
    assert_eq!(result.as_slice(), data.as_slice());
}

/// Test 3: Transpose with custom permutation (via attribute)
#[test]
fn test_transpose_3d_permutation() {
    // Shape [2, 3, 4]
    let data = Tensor::from_vec(vec![1.0; 24], vec![2, 3, 4]);

    // Permutation [2, 0, 1] means: dim2 -> dim0, dim0 -> dim1, dim1 -> dim2
    let mut attrs = NodeAttributes::new();
    attrs.add_ints("perm".to_string(), vec![2, 0, 1]);

    let result = iconnx::operators::transpose::Transpose::forward(&[data], &attrs);

    // Result shape: [4, 2, 3]
    assert_eq!(result.shape(), &[4, 2, 3]);
}
