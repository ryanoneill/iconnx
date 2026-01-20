use iconnx::attributes::NodeAttributes;
use iconnx::operators::floor::Floor;
/// Test Floor operator for Iconnx
use iconnx::tensor::Tensor;

#[test]
fn test_floor_basic() {
    let data = Tensor::from_vec(vec![1.7, 2.3, -1.5, -2.8], vec![4]);

    let result = Floor::forward(&[data], &NodeAttributes::new());

    assert_eq!(result.as_slice(), vec![1.0, 2.0, -2.0, -3.0]);
}
