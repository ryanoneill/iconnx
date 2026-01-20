use iconnx::attributes::NodeAttributes;
/// Test ReduceSum operator for Iconnx
///
/// TDD: Tests written FIRST
/// ReduceSum: sum along specified axes
use iconnx::tensor::Tensor;

/// Test ReduceSum: sum all elements
#[test]
fn test_reduce_sum_all() {
    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]);

    // Sum of all elements = 1+2+3+4 = 10
    let result = iconnx::operators::reduce_sum::ReduceSum::forward(
        &[data],
        &NodeAttributes::new(),
    );

    assert_eq!(result.shape(), &[] as &[usize]); // Scalar
    assert_eq!(result.as_slice(), vec![10.0]);
}

/// Test ReduceSum along specific axis with keepdims=0 (explicit)
#[test]
fn test_reduce_sum_axis() {
    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let axes = Tensor::from_vec(vec![1.0], vec![1]); // Reduce along axis 1

    // Explicitly set keepdims=0 to remove reduced dimension
    let mut attrs = NodeAttributes::new();
    attrs.add_int("keepdims".to_string(), 0);

    let result = iconnx::operators::reduce_sum::ReduceSum::forward(&[data, axes], &attrs);

    // Sum along columns: [1+2+3, 4+5+6] = [6.0, 15.0]
    assert_eq!(result.shape(), &[2]);
    assert_eq!(result.as_slice(), vec![6.0, 15.0]);
}

/// Test ReduceSum 2D
#[test]
fn test_reduce_sum_2d() {
    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    let result = iconnx::operators::reduce_sum::ReduceSum::forward(
        &[data],
        &NodeAttributes::new(),
    );

    // Sum of all: 1+2+3+4 = 10
    assert_eq!(result.shape(), &[] as &[usize]);
    assert_eq!(result.as_slice(), vec![10.0]);
}

/// Test ReduceSum with keepdims=1 (ONNX default)
#[test]
fn test_reduce_sum_keepdims_true() {
    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let axes = Tensor::from_vec_i64(vec![1], vec![1]); // Reduce along axis 1

    // Default keepdims=1 preserves reduced dimension as size 1
    let result = iconnx::operators::reduce_sum::ReduceSum::forward(
        &[data, axes],
        &NodeAttributes::new(),
    );

    // Shape should be [2, 1] with keepdims=1
    assert_eq!(result.shape(), &[2, 1]);
    assert_eq!(result.as_slice(), vec![6.0, 15.0]);
}

/// Test ReduceSum with keepdims=0 (explicit)
#[test]
fn test_reduce_sum_keepdims_false() {
    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let axes = Tensor::from_vec_i64(vec![1], vec![1]); // Reduce along axis 1

    let mut attrs = NodeAttributes::new();
    attrs.add_int("keepdims".to_string(), 0);

    let result = iconnx::operators::reduce_sum::ReduceSum::forward(&[data, axes], &attrs);

    // Shape should be [2] with keepdims=0
    assert_eq!(result.shape(), &[2]);
    assert_eq!(result.as_slice(), vec![6.0, 15.0]);
}

/// Test ReduceSum axis 0 with keepdims
#[test]
fn test_reduce_sum_axis_0_keepdims() {
    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let axes = Tensor::from_vec_i64(vec![0], vec![1]);

    // keepdims=1 by default
    let result = iconnx::operators::reduce_sum::ReduceSum::forward(
        &[data, axes],
        &NodeAttributes::new(),
    );

    // Sum along axis 0: [1+3, 2+4] = [4, 6], shape [1, 2]
    assert_eq!(result.shape(), &[1, 2]);
    assert_eq!(result.as_slice(), vec![4.0, 6.0]);
}
