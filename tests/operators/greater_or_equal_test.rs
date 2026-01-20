use iconnx::attributes::NodeAttributes;
use iconnx::operators::greater_or_equal::GreaterOrEqual;
/// Test GreaterOrEqual operator for Iconnx
use iconnx::tensor::Tensor;

#[test]
fn test_greater_or_equal_basic() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
    let b = Tensor::from_vec(vec![2.0, 2.0, 2.0], vec![3]);

    let result = GreaterOrEqual::forward(&[a, b], &NodeAttributes::new());

    assert_eq!(result.as_slice(), vec![0.0, 1.0, 1.0]);
}
