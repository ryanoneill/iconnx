use iconnx::attributes::NodeAttributes;
use iconnx::operators::equal::Equal;
/// Test Equal operator for Iconnx
use iconnx::tensor::Tensor;

#[test]
fn test_equal_same_values() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
    let b = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

    let result = Equal::forward(&[a, b], &NodeAttributes::new());

    assert_eq!(result.as_slice(), vec![1.0, 1.0, 1.0]);
}

#[test]
fn test_equal_different_values() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
    let b = Tensor::from_vec(vec![1.0, 5.0, 3.0], vec![3]);

    let result = Equal::forward(&[a, b], &NodeAttributes::new());

    assert_eq!(result.as_slice(), vec![1.0, 0.0, 1.0]);
}

#[test]
fn test_equal_int64() {
    let a = Tensor::from_vec_i64(vec![1, 2, 3], vec![3]);
    let b = Tensor::from_vec_i64(vec![1, 2, 4], vec![3]);

    let result = Equal::forward(&[a, b], &NodeAttributes::new());

    assert_eq!(result.as_slice(), vec![1.0, 1.0, 0.0]);
}
