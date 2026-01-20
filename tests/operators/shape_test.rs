use iconnx::attributes::NodeAttributes;
/// Test Shape operator for Iconnx
///
/// TDD: Tests written FIRST
/// Shape: Get shape as 1D tensor
use iconnx::tensor::Tensor;

/// Test 1: Shape of 1D tensor
#[test]
fn test_shape_1d() {
    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]);

    let result =
        iconnx::operators::shape::Shape::forward(&[data], &NodeAttributes::new());

    assert_eq!(result.shape(), &[1]);
    assert!(result.is_int64(), "Shape should return Int64");
    assert_eq!(result.as_slice_i64(), vec![5]);
}

/// Test 2: Shape of 2D tensor
#[test]
fn test_shape_2d() {
    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

    let result =
        iconnx::operators::shape::Shape::forward(&[data], &NodeAttributes::new());

    assert_eq!(result.shape(), &[2]);
    assert!(result.is_int64(), "Shape should return Int64");
    assert_eq!(result.as_slice_i64(), vec![2, 3]);
}

/// Test 3: Shape of 3D tensor
#[test]
fn test_shape_3d() {
    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2]);

    let result =
        iconnx::operators::shape::Shape::forward(&[data], &NodeAttributes::new());

    assert_eq!(result.shape(), &[3]);
    assert!(result.is_int64(), "Shape should return Int64");
    assert_eq!(result.as_slice_i64(), vec![2, 2, 2]);
}

/// Test 4: Shape of scalar (0D)
#[test]
fn test_shape_scalar() {
    let data = Tensor::from_vec(vec![42.0], vec![1]);

    let result =
        iconnx::operators::shape::Shape::forward(&[data], &NodeAttributes::new());

    assert_eq!(result.shape(), &[1]);
    assert!(result.is_int64(), "Shape should return Int64");
    assert_eq!(result.as_slice_i64(), vec![1]);
}

/// Test 5: Shape with start attribute
#[test]
fn test_shape_with_start() {
    let data = Tensor::from_vec(vec![0.0; 24], vec![2, 3, 4]);

    let mut attrs = NodeAttributes::new();
    attrs.add_int("start".to_string(), 1);

    let result = iconnx::operators::shape::Shape::forward(&[data], &attrs);

    // Shape [2, 3, 4] with start=1 -> [3, 4]
    assert_eq!(result.shape(), &[2]);
    assert_eq!(result.as_slice_i64(), vec![3, 4]);
}

/// Test 6: Shape with end attribute
#[test]
fn test_shape_with_end() {
    let data = Tensor::from_vec(vec![0.0; 24], vec![2, 3, 4]);

    let mut attrs = NodeAttributes::new();
    attrs.add_int("end".to_string(), 2);

    let result = iconnx::operators::shape::Shape::forward(&[data], &attrs);

    // Shape [2, 3, 4] with end=2 -> [2, 3]
    assert_eq!(result.shape(), &[2]);
    assert_eq!(result.as_slice_i64(), vec![2, 3]);
}

/// Test 7: Shape with start and end
#[test]
fn test_shape_with_start_and_end() {
    let data = Tensor::from_vec(vec![0.0; 120], vec![2, 3, 4, 5]);

    let mut attrs = NodeAttributes::new();
    attrs.add_int("start".to_string(), 1);
    attrs.add_int("end".to_string(), 3);

    let result = iconnx::operators::shape::Shape::forward(&[data], &attrs);

    // Shape [2, 3, 4, 5] with start=1, end=3 -> [3, 4]
    assert_eq!(result.shape(), &[2]);
    assert_eq!(result.as_slice_i64(), vec![3, 4]);
}

/// Test 8: Shape with negative start
#[test]
fn test_shape_negative_start() {
    let data = Tensor::from_vec(vec![0.0; 24], vec![2, 3, 4]);

    let mut attrs = NodeAttributes::new();
    attrs.add_int("start".to_string(), -2);

    let result = iconnx::operators::shape::Shape::forward(&[data], &attrs);

    // Shape [2, 3, 4] with start=-2 -> [3, 4]
    assert_eq!(result.shape(), &[2]);
    assert_eq!(result.as_slice_i64(), vec![3, 4]);
}

/// Test 9: Shape with negative end
#[test]
fn test_shape_negative_end() {
    let data = Tensor::from_vec(vec![0.0; 24], vec![2, 3, 4]);

    let mut attrs = NodeAttributes::new();
    attrs.add_int("end".to_string(), -1);

    let result = iconnx::operators::shape::Shape::forward(&[data], &attrs);

    // Shape [2, 3, 4] with end=-1 -> [2, 3]
    assert_eq!(result.shape(), &[2]);
    assert_eq!(result.as_slice_i64(), vec![2, 3]);
}
