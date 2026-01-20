use iconnx::attributes::NodeAttributes;
use iconnx::operators::sigmoid::Sigmoid;
/// Test Sigmoid operator for Iconnx
use iconnx::tensor::Tensor;

#[test]
fn test_sigmoid_basic() {
    let data = Tensor::from_vec(vec![0.0, 1.0, -1.0], vec![3]);

    let result = Sigmoid::forward(&[data], &NodeAttributes::new());

    let expected = vec![
        0.5,        // sigmoid(0) = 0.5
        0.7310586,  // sigmoid(1) ≈ 0.731
        0.26894143, // sigmoid(-1) ≈ 0.269
    ];

    for (i, (&result_val, &expected_val)) in
        result.as_slice().iter().zip(expected.iter()).enumerate()
    {
        assert!(
            (result_val - expected_val).abs() < 0.0001,
            "Sigmoid mismatch at index {}: got {}, expected {}",
            i,
            result_val,
            expected_val
        );
    }
}

#[test]
fn test_sigmoid_large_values() {
    let data = Tensor::from_vec(vec![10.0, -10.0], vec![2]);

    let result = Sigmoid::forward(&[data], &NodeAttributes::new());

    assert!(result.as_slice()[0] > 0.9999, "sigmoid(10) should be ~1.0");
    assert!(result.as_slice()[1] < 0.0001, "sigmoid(-10) should be ~0.0");
}
