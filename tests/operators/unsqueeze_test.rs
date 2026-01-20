use iconnx::attributes::NodeAttributes;
use iconnx::operators::unsqueeze::Unsqueeze;
/// Test Unsqueeze operator for Iconnx
use iconnx::tensor::Tensor;

#[test]
fn test_unsqueeze_add_axis_0() {
    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
    let axes = Tensor::from_vec_i64(vec![0], vec![1]);

    let result = Unsqueeze::forward(&[data, axes], &NodeAttributes::new());

    assert_eq!(result.shape(), &[1, 3]);
    assert_eq!(result.as_slice(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_unsqueeze_add_axis_1() {
    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
    let axes = Tensor::from_vec_i64(vec![1], vec![1]);

    let result = Unsqueeze::forward(&[data, axes], &NodeAttributes::new());

    assert_eq!(result.shape(), &[3, 1]);
    assert_eq!(result.as_slice(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_unsqueeze_multiple_axes() {
    let data = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
    let axes = Tensor::from_vec_i64(vec![0, 2], vec![2]);

    let result = Unsqueeze::forward(&[data, axes], &NodeAttributes::new());

    assert_eq!(result.shape(), &[1, 2, 1]);
    assert_eq!(result.as_slice(), vec![1.0, 2.0]);
}
