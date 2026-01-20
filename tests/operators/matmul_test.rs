use iconnx::attributes::NodeAttributes;
/// Test MatMul operator for Iconnx
///
/// TDD: Tests written FIRST
/// MatMul performs matrix multiplication (not element-wise!)
/// ONNX spec: https://onnx.ai/onnx/operators/onnx__MatMul.html
use iconnx::tensor::Tensor;

/// Test 1: Simple 2D matrix multiplication
/// [2x3] @ [3x2] = [2x2]
#[test]
fn test_matmul_2d() {
    // A = [[1, 2, 3],
    //      [4, 5, 6]]  (2x3)
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

    // B = [[7, 8],
    //      [9, 10],
    //      [11, 12]]  (3x2)
    let b = Tensor::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);

    let result =
        iconnx::operators::matmul::MatMul::forward(&[a, b], &NodeAttributes::new());

    assert_eq!(result.shape(), &[2, 2]);

    // Result = [[1*7+2*9+3*11,  1*8+2*10+3*12],
    //           [4*7+5*9+6*11,  4*8+5*10+6*12]]
    //        = [[58, 64],
    //           [139, 154]]
    assert_eq!(result.as_slice(), &[58.0, 64.0, 139.0, 154.0]);
}

/// Test 2: Vector-matrix multiplication
/// [1x3] @ [3x2] = [1x2]
#[test]
fn test_matmul_vector_matrix() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
    let b = Tensor::from_vec(vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0], vec![3, 2]);

    let result =
        iconnx::operators::matmul::MatMul::forward(&[a, b], &NodeAttributes::new());

    assert_eq!(result.shape(), &[1, 2]);
    // [1*4+2*6+3*8, 1*5+2*7+3*9] = [40, 46]
    assert_eq!(result.as_slice(), &[40.0, 46.0]);
}

/// Test 3: Matrix-vector multiplication
/// [2x3] @ [3x1] = [2x1]
#[test]
fn test_matmul_matrix_vector() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = Tensor::from_vec(vec![7.0, 8.0, 9.0], vec![3, 1]);

    let result =
        iconnx::operators::matmul::MatMul::forward(&[a, b], &NodeAttributes::new());

    assert_eq!(result.shape(), &[2, 1]);
    // [1*7+2*8+3*9, 4*7+5*8+6*9] = [50, 122]
    assert_eq!(result.as_slice(), &[50.0, 122.0]);
}

/// Test 4: 1D vector dot product
/// [3] @ [3] = scalar (shape [])
#[test]
fn test_matmul_1d_dot_product() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
    let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], vec![3]);

    let result =
        iconnx::operators::matmul::MatMul::forward(&[a, b], &NodeAttributes::new());

    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    assert_eq!(result.shape(), &[] as &[usize]); // Scalar
    assert_eq!(result.as_slice(), &[32.0]);
}

/// Test 5: Batched matrix multiplication
/// [2, 2, 3] @ [2, 3, 2] = [2, 2, 2]
#[test]
fn test_matmul_batched() {
    // Batch of 2 matrices, each 2x3
    let a = Tensor::from_vec(
        vec![
            1.0, 2.0, 3.0, // Batch 0, row 0
            4.0, 5.0, 6.0, // Batch 0, row 1
            7.0, 8.0, 9.0, // Batch 1, row 0
            10.0, 11.0, 12.0, // Batch 1, row 1
        ],
        vec![2, 2, 3],
    );

    // Batch of 2 matrices, each 3x2
    let b = Tensor::from_vec(
        vec![
            1.0, 2.0, // Batch 0, row 0
            3.0, 4.0, // Batch 0, row 1
            5.0, 6.0, // Batch 0, row 2
            7.0, 8.0, // Batch 1, row 0
            9.0, 10.0, // Batch 1, row 1
            11.0, 12.0, // Batch 1, row 2
        ],
        vec![2, 3, 2],
    );

    let result =
        iconnx::operators::matmul::MatMul::forward(&[a, b], &NodeAttributes::new());

    assert_eq!(result.shape(), &[2, 2, 2]);
    assert_eq!(result.len(), 8);

    // Batch 0: [[1,2,3],[4,5,6]] @ [[1,2],[3,4],[5,6]]
    //        = [[22,28],[49,64]]
    // Batch 1: [[7,8,9],[10,11,12]] @ [[7,8],[9,10],[11,12]]
    //        = [[220,244],[301,334]]
    let expected = vec![
        22.0, 28.0, // Batch 0, row 0
        49.0, 64.0, // Batch 0, row 1
        220.0, 244.0, // Batch 1, row 0
        301.0, 334.0, // Batch 1, row 1
    ];
    assert_eq!(result.as_slice(), expected.as_slice());
}

/// Test 6: Identity matrix multiplication
#[test]
fn test_matmul_identity() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    // Identity matrix
    let identity = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);

    let result = iconnx::operators::matmul::MatMul::forward(
        &[a.clone(), identity],
        &NodeAttributes::new(),
    );

    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.as_slice(), a.as_slice()); // Should be unchanged
}
