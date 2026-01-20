use iconnx::attributes::NodeAttributes;
/// Test reduction operators for Iconnx
///
/// TDD: Tests written FIRST
/// Reduction ops: ReduceMean, ReduceSum
use iconnx::tensor::Tensor;

/// Test ReduceMean: average along axis
#[test]
fn test_reduce_mean_all() {
    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]);

    // Mean of all elements = (1+2+3+4)/4 = 2.5
    let result = iconnx::operators::reduce_mean::ReduceMean::forward(
        &[data],
        &NodeAttributes::new(),
    );

    assert_eq!(result.shape(), &[] as &[usize]); // Scalar
    assert_eq!(result.as_slice(), vec![2.5]);
}

/// Test ReduceMean along specific axis with keepdims=0
#[test]
fn test_reduce_mean_axis() {
    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let axes = Tensor::from_vec(vec![1.0], vec![1]); // Reduce along axis 1

    // Explicitly set keepdims=0 to remove reduced dimension
    // (ONNX default is keepdims=1)
    let mut attrs = NodeAttributes::new();
    attrs.add_int("keepdims".to_string(), 0);

    let result =
        iconnx::operators::reduce_mean::ReduceMean::forward(&[data, axes], &attrs);

    // Mean along columns: [(1+2+3)/3, (4+5+6)/3] = [2.0, 5.0]
    assert_eq!(result.shape(), &[2]);
    assert_eq!(result.as_slice(), vec![2.0, 5.0]);
}
