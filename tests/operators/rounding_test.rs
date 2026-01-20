use iconnx::attributes::NodeAttributes;
/// Test rounding operators for Iconnx
///
/// TDD: Tests written FIRST
/// Rounding ops: Floor, Round
use iconnx::tensor::Tensor;

/// Test Floor: floor(x)
#[test]
fn test_floor_basic() {
    let data = Tensor::from_vec(vec![1.2, 3.7, -1.5, -2.9], vec![4]);

    let result =
        iconnx::operators::floor::Floor::forward(&[data], &NodeAttributes::new());

    assert_eq!(result.shape(), &[4]);
    // floor(1.2) = 1, floor(3.7) = 3, floor(-1.5) = -2, floor(-2.9) = -3
    assert_eq!(result.as_slice(), vec![1.0, 3.0, -2.0, -3.0]);
}

/// Test Round: round(x)
#[test]
fn test_round_basic() {
    let data = Tensor::from_vec(vec![1.2, 1.5, 1.7, -1.5], vec![4]);

    let result =
        iconnx::operators::round::Round::forward(&[data], &NodeAttributes::new());

    assert_eq!(result.shape(), &[4]);
    // round(1.2) = 1, round(1.5) = 2, round(1.7) = 2, round(-1.5) = -2
    assert_eq!(result.as_slice(), vec![1.0, 2.0, 2.0, -2.0]);
}

/// Test Floor 2D
#[test]
fn test_floor_2d() {
    let data = Tensor::from_vec(vec![1.9, 2.1, 3.5, 4.8], vec![2, 2]);

    let result =
        iconnx::operators::floor::Floor::forward(&[data], &NodeAttributes::new());

    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.as_slice(), vec![1.0, 2.0, 3.0, 4.0]);
}

/// Test Round edge cases
#[test]
fn test_round_edge_cases() {
    let data = Tensor::from_vec(vec![0.0, 0.5, -0.5], vec![3]);

    let result =
        iconnx::operators::round::Round::forward(&[data], &NodeAttributes::new());

    assert_eq!(result.shape(), &[3]);
    // round(0.0) = 0, round(0.5) = 0 or 1 (banker's rounding), round(-0.5) = 0 or -1
    // Standard round() in Rust rounds away from zero for .5
    assert_eq!(result.as_slice(), vec![0.0, 1.0, -1.0]);
}
