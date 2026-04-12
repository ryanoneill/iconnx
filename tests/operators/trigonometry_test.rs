use iconnx::attributes::NodeAttributes;
/// Test trigonometry operators for Iconnx
///
/// TDD: Tests written FIRST
/// Trig ops: Sin, Cos, Atan
use iconnx::tensor::Tensor;

/// Test Sin: sin(x)
#[test]
fn test_sin_basic() {
    let data = Tensor::from_vec(
        vec![0.0, std::f32::consts::PI / 2.0, std::f32::consts::PI],
        vec![3],
    );

    let result = iconnx::operators::sin::Sin::forward(&[data], &NodeAttributes::new());

    assert_eq!(result.shape(), &[3]);
    // sin(0) = 0, sin(π/2) = 1, sin(π) ≈ 0
    let output = result.as_slice();
    assert!(output[0].abs() < 0.001);
    assert!((output[1] - 1.0).abs() < 0.001);
    assert!(output[2].abs() < 0.001);
}

/// Test Cos: cos(x)
#[test]
fn test_cos_basic() {
    let data = Tensor::from_vec(
        vec![0.0, std::f32::consts::PI / 2.0, std::f32::consts::PI],
        vec![3],
    );

    let result = iconnx::operators::cos::Cos::forward(&[data], &NodeAttributes::new());

    assert_eq!(result.shape(), &[3]);
    // cos(0) = 1, cos(π/2) ≈ 0, cos(π) = -1
    let output = result.as_slice();
    assert!((output[0] - 1.0).abs() < 0.001);
    assert!(output[1].abs() < 0.001);
    assert!((output[2] + 1.0).abs() < 0.001);
}

/// Test Atan: atan(x)
#[test]
fn test_atan_basic() {
    let data = Tensor::from_vec(vec![0.0, 1.0, -1.0], vec![3]);

    let result = iconnx::operators::atan::Atan::forward(&[data], &NodeAttributes::new());

    assert_eq!(result.shape(), &[3]);
    // atan(0) = 0, atan(1) = π/4, atan(-1) = -π/4
    let output = result.as_slice();
    assert!(output[0].abs() < 0.001);
    assert!((output[1] - std::f32::consts::PI / 4.0).abs() < 0.001);
    assert!((output[2] + std::f32::consts::PI / 4.0).abs() < 0.001);
}

/// Test Sin 2D
#[test]
fn test_sin_2d() {
    let data = Tensor::from_vec(vec![0.0, 1.0, 2.0, 3.0], vec![2, 2]);

    let result = iconnx::operators::sin::Sin::forward(&[data], &NodeAttributes::new());

    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.len(), 4);
}

/// Test Cos negative values
#[test]
fn test_cos_negative() {
    let data = Tensor::from_vec(
        vec![-std::f32::consts::PI, 0.0, std::f32::consts::PI],
        vec![3],
    );

    let result = iconnx::operators::cos::Cos::forward(&[data], &NodeAttributes::new());

    assert_eq!(result.shape(), &[3]);
    // cos(-π) = -1, cos(0) = 1, cos(π) = -1
    let output = result.as_slice();
    assert!((output[0] + 1.0).abs() < 0.001);
    assert!((output[1] - 1.0).abs() < 0.001);
    assert!((output[2] + 1.0).abs() < 0.001);
}

/// Test Atan range
#[test]
fn test_atan_range() {
    let data = Tensor::from_vec(vec![0.0, 100.0, -100.0], vec![3]);

    let result = iconnx::operators::atan::Atan::forward(&[data], &NodeAttributes::new());

    assert_eq!(result.shape(), &[3]);
    // atan(x) is always in (-π/2, π/2)
    let output = result.as_slice();
    assert!(output[0].abs() < 0.001);
    assert!(output[1] > 0.0 && output[1] < std::f32::consts::PI / 2.0);
    assert!(output[2] < 0.0 && output[2] > -std::f32::consts::PI / 2.0);
}

/// Test Tanh: tanh(x)
#[test]
fn test_tanh_basic() {
    let data = Tensor::from_vec(vec![0.0, 1.0, -1.0], vec![3]);

    let result = iconnx::operators::tanh::Tanh::forward(&[data], &NodeAttributes::new());

    assert_eq!(result.shape(), &[3]);
    let expected = [0.0, 0.7615942, -0.7615942];
    for (i, (&r, &e)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!((r - e).abs() < 0.0001, "Tanh at {}: {} vs {}", i, r, e);
    }
}

#[test]
fn test_tanh_large_values() {
    let data = Tensor::from_vec(vec![10.0, -10.0], vec![2]);

    let result = iconnx::operators::tanh::Tanh::forward(&[data], &NodeAttributes::new());

    assert!(result.as_slice()[0] > 0.9999, "tanh(10) should be ~1.0");
    assert!(result.as_slice()[1] < -0.9999, "tanh(-10) should be ~-1.0");
}
