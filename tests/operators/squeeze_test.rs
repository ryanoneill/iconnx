use iconnx::attributes::NodeAttributes;
use iconnx::operators::squeeze::Squeeze;
/// Test Squeeze operator for Iconnx
use iconnx::tensor::Tensor;

#[test]
fn test_squeeze_single_dim() {
    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
    let axes = Tensor::from_vec_i64(vec![0], vec![1]);

    let result = Squeeze::forward(&[data, axes], &NodeAttributes::new());

    assert_eq!(result.shape(), &[3]);
    assert_eq!(result.as_slice(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_squeeze_multiple_dims() {
    let data = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2, 1]);
    let axes = Tensor::from_vec_i64(vec![0, 2], vec![2]);

    let result = Squeeze::forward(&[data, axes], &NodeAttributes::new());

    assert_eq!(result.shape(), &[2]);
    assert_eq!(result.as_slice(), vec![1.0, 2.0]);
}
