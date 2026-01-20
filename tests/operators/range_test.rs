use iconnx::attributes::NodeAttributes;
/// Test Range operator for Iconnx
///
/// TDD: Tests written FIRST
/// Range: Generate sequence [start, start+step, ..., end)
use iconnx::tensor::Tensor;

/// Test 1: Basic range [0, 10) with step 2
#[test]
fn test_range_basic() {
    let start = Tensor::from_vec(vec![0.0], vec![]);
    let limit = Tensor::from_vec(vec![10.0], vec![]);
    let delta = Tensor::from_vec(vec![2.0], vec![]);

    let result = iconnx::operators::range::Range::forward(
        &[start, limit, delta],
        &NodeAttributes::new(),
    );

    assert_eq!(result.shape(), &[5]);
    assert_eq!(result.as_slice(), vec![0.0, 2.0, 4.0, 6.0, 8.0]);
}

/// Test 2: Range with step 1
#[test]
fn test_range_step_one() {
    let start = Tensor::from_vec(vec![1.0], vec![]);
    let limit = Tensor::from_vec(vec![5.0], vec![]);
    let delta = Tensor::from_vec(vec![1.0], vec![]);

    let result = iconnx::operators::range::Range::forward(
        &[start, limit, delta],
        &NodeAttributes::new(),
    );

    assert_eq!(result.shape(), &[4]);
    assert_eq!(result.as_slice(), vec![1.0, 2.0, 3.0, 4.0]);
}

/// Test 3: Negative range (counting down)
#[test]
fn test_range_negative_step() {
    let start = Tensor::from_vec(vec![10.0], vec![]);
    let limit = Tensor::from_vec(vec![0.0], vec![]);
    let delta = Tensor::from_vec(vec![-2.0], vec![]);

    let result = iconnx::operators::range::Range::forward(
        &[start, limit, delta],
        &NodeAttributes::new(),
    );

    assert_eq!(result.shape(), &[5]);
    assert_eq!(result.as_slice(), vec![10.0, 8.0, 6.0, 4.0, 2.0]);
}

/// Test 4: Fractional range with decimal step
#[test]
fn test_range_fractional() {
    let start = Tensor::from_vec(vec![0.0], vec![]);
    let limit = Tensor::from_vec(vec![1.0], vec![]);
    let delta = Tensor::from_vec(vec![0.25], vec![]);

    let result = iconnx::operators::range::Range::forward(
        &[start, limit, delta],
        &NodeAttributes::new(),
    );

    assert_eq!(result.shape(), &[4]);
    let values = result.as_slice();
    assert!((values[0] - 0.0).abs() < 1e-6);
    assert!((values[1] - 0.25).abs() < 1e-6);
    assert!((values[2] - 0.5).abs() < 1e-6);
    assert!((values[3] - 0.75).abs() < 1e-6);
}
