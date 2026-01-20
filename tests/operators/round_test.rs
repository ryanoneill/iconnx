use iconnx::attributes::NodeAttributes;
use iconnx::operators::round::Round;
/// Test Round operator for Iconnx
use iconnx::tensor::Tensor;

#[test]
fn test_round_basic() {
    let data = Tensor::from_vec(vec![1.4, 1.5, 1.6, 2.5], vec![4]);

    let result = Round::forward(&[data], &NodeAttributes::new());

    assert_eq!(
        result.as_slice(),
        vec![1.0, 2.0, 2.0, 3.0],
        "Round half away from zero"
    );
}
