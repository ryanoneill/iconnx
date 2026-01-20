use iconnx::attributes::NodeAttributes;
use iconnx::operators::pow::Pow;
/// Test Pow operator for Iconnx
use iconnx::tensor::Tensor;

#[test]
fn test_pow_basic() {
    let base = Tensor::from_vec(vec![2.0, 3.0, 4.0], vec![3]);
    let exp = Tensor::from_vec(vec![2.0, 3.0, 0.5], vec![3]);

    let result = Pow::forward(&[base, exp], &NodeAttributes::new());

    let expected = vec![4.0, 27.0, 2.0];
    for (i, (&r, &e)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!((r - e).abs() < 0.001, "Pow at {}: {} vs {}", i, r, e);
    }
}
