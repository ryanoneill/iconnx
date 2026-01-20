use iconnx::attributes::NodeAttributes;
/// Test Reshape operator for Iconnx
///
/// TDD: Tests written FIRST
/// Reshape changes tensor dimensions without changing data
/// ONNX spec: https://onnx.ai/onnx/operators/onnx__Reshape.html
use iconnx::tensor::Tensor;

/// Test 1: Reshape 1D to 2D
#[test]
fn test_reshape_1d_to_2d() {
    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6]);
    let shape = Tensor::from_vec(vec![2.0, 3.0], vec![2]); // New shape as tensor

    let result = iconnx::operators::reshape::Reshape::forward(
        &[data, shape],
        &NodeAttributes::new(),
    );

    assert_eq!(result.shape(), &[2, 3]);
    assert_eq!(result.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

/// Test 2: Reshape 2D to 1D (flatten)
#[test]
fn test_reshape_2d_to_1d() {
    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let shape = Tensor::from_vec(vec![4.0], vec![1]);

    let result = iconnx::operators::reshape::Reshape::forward(
        &[data, shape],
        &NodeAttributes::new(),
    );

    assert_eq!(result.shape(), &[4]);
    assert_eq!(result.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
}

/// Test 3: Reshape with -1 (infer dimension)
#[test]
fn test_reshape_infer_dimension() {
    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6]);
    // Shape [2, -1] should infer -1 as 3 (since 6 / 2 = 3)
    let shape = Tensor::from_vec(vec![2.0, -1.0], vec![2]);

    let result = iconnx::operators::reshape::Reshape::forward(
        &[data, shape],
        &NodeAttributes::new(),
    );

    assert_eq!(result.shape(), &[2, 3]);
}

/// Test 4: Reshape 2D to 3D
#[test]
fn test_reshape_2d_to_3d() {
    let data = Tensor::from_vec(vec![1.0; 24], vec![4, 6]);
    let shape = Tensor::from_vec(vec![2.0, 3.0, 4.0], vec![3]);

    let result = iconnx::operators::reshape::Reshape::forward(
        &[data, shape],
        &NodeAttributes::new(),
    );

    assert_eq!(result.shape(), &[2, 3, 4]);
    assert_eq!(result.len(), 24);
}

/// Test 5: Reshape to scalar (0D)
#[test]
fn test_reshape_to_scalar() {
    let data = Tensor::from_vec(vec![42.0], vec![1]);
    let shape = Tensor::from_vec(vec![], vec![0]); // Empty shape = scalar

    let result = iconnx::operators::reshape::Reshape::forward(
        &[data, shape],
        &NodeAttributes::new(),
    );

    assert_eq!(result.shape(), &[] as &[usize]);
    assert_eq!(result.len(), 1);
}
