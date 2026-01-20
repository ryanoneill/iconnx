use iconnx::attributes::NodeAttributes;
use iconnx::operators::exp::Exp;
/// Test Exp operator for Iconnx
use iconnx::tensor::Tensor;

#[test]
fn test_exp_basic() {
    let data = Tensor::from_vec(vec![0.0, 1.0, 2.0], vec![3]);

    let result = Exp::forward(&[data], &NodeAttributes::new());

    let expected = vec![1.0, 2.7182817, 7.389056];
    for (i, (&r, &e)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!(
            (r - e).abs() < 0.001,
            "Exp mismatch at {}: {} vs {}",
            i,
            r,
            e
        );
    }
}
