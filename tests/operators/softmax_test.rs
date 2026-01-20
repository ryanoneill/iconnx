use iconnx::attributes::NodeAttributes;
/// Test Softmax operator for Iconnx
///
/// TDD: Tests written FIRST
/// Softmax: exp(x) / sum(exp(x)) along an axis
use iconnx::tensor::Tensor;

/// Test 1: Softmax on 1D vector
#[test]
fn test_softmax_1d() {
    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

    let result =
        iconnx::operators::softmax::Softmax::forward(&[data], &NodeAttributes::new());

    assert_eq!(result.shape(), &[3]);

    // exp values: [e^1, e^2, e^3] ≈ [2.718, 7.389, 20.086]
    // sum ≈ 30.193
    // softmax ≈ [0.090, 0.245, 0.665]
    let output = result.as_slice();
    let sum: f32 = output.iter().sum();

    // Sum should be 1.0 (probability distribution)
    assert!((sum - 1.0).abs() < 0.001);

    // Check approximate values
    assert!((output[0] - 0.090).abs() < 0.01);
    assert!((output[1] - 0.245).abs() < 0.01);
    assert!((output[2] - 0.665).abs() < 0.01);
}

/// Test 2: Softmax on 2D (along last axis - default)
#[test]
fn test_softmax_2d() {
    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

    let result =
        iconnx::operators::softmax::Softmax::forward(&[data], &NodeAttributes::new());

    assert_eq!(result.shape(), &[2, 3]);

    // Each row should sum to 1.0
    let output = result.as_slice();
    let row1_sum: f32 = output[0..3].iter().sum();
    let row2_sum: f32 = output[3..6].iter().sum();

    assert!((row1_sum - 1.0).abs() < 0.001);
    assert!((row2_sum - 1.0).abs() < 0.001);
}

/// Test 3: Softmax with explicit axis=1 (same as default for 2D)
#[test]
fn test_softmax_axis_1() {
    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

    let mut attrs = NodeAttributes::new();
    attrs.add_int("axis".to_string(), 1);

    let result = iconnx::operators::softmax::Softmax::forward(&[data], &attrs);

    assert_eq!(result.shape(), &[2, 3]);

    // Each row should sum to 1.0 (softmax along axis 1)
    let output = result.as_slice();
    let row1_sum: f32 = output[0..3].iter().sum();
    let row2_sum: f32 = output[3..6].iter().sum();

    assert!((row1_sum - 1.0).abs() < 0.001);
    assert!((row2_sum - 1.0).abs() < 0.001);
}

/// Test 4: Softmax along axis=0 (columns)
#[test]
fn test_softmax_axis_0() {
    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

    let mut attrs = NodeAttributes::new();
    attrs.add_int("axis".to_string(), 0);

    let result = iconnx::operators::softmax::Softmax::forward(&[data], &attrs);

    assert_eq!(result.shape(), &[2, 3]);

    // Each column should sum to 1.0 (softmax along axis 0)
    let output = result.as_slice();
    // Column sums: [output[0]+output[3], output[1]+output[4], output[2]+output[5]]
    let col1_sum = output[0] + output[3];
    let col2_sum = output[1] + output[4];
    let col3_sum = output[2] + output[5];

    assert!((col1_sum - 1.0).abs() < 0.001, "col1_sum = {}", col1_sum);
    assert!((col2_sum - 1.0).abs() < 0.001, "col2_sum = {}", col2_sum);
    assert!((col3_sum - 1.0).abs() < 0.001, "col3_sum = {}", col3_sum);
}

/// Test 5: Softmax with negative axis (-1 = last)
#[test]
fn test_softmax_negative_axis() {
    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

    let mut attrs = NodeAttributes::new();
    attrs.add_int("axis".to_string(), -1);

    let result = iconnx::operators::softmax::Softmax::forward(&[data], &attrs);

    assert_eq!(result.shape(), &[2, 3]);

    // -1 = last axis, same as axis=1 for 2D
    let output = result.as_slice();
    let row1_sum: f32 = output[0..3].iter().sum();
    let row2_sum: f32 = output[3..6].iter().sum();

    assert!((row1_sum - 1.0).abs() < 0.001);
    assert!((row2_sum - 1.0).abs() < 0.001);
}
