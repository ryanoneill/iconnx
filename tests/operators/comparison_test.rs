use iconnx::attributes::NodeAttributes;
/// Test comparison operators for Iconnx
///
/// TDD: Tests written FIRST
/// Comparison ops: Greater, Less, GreaterOrEqual, Equal
use iconnx::tensor::Tensor;

/// Test Greater: x > y returns 1.0 or 0.0
#[test]
fn test_greater_basic() {
    let x = Tensor::from_vec(vec![1.0, 3.0, 2.0], vec![3]);
    let y = Tensor::from_vec(vec![2.0, 2.0, 2.0], vec![3]);

    let result =
        iconnx::operators::greater::Greater::forward(&[x, y], &NodeAttributes::new());

    assert_eq!(result.shape(), &[3]);
    // 1 > 2 = false (0.0), 3 > 2 = true (1.0), 2 > 2 = false (0.0)
    assert_eq!(result.as_slice(), vec![0.0, 1.0, 0.0]);
}

/// Test Less: x < y returns 1.0 or 0.0
#[test]
fn test_less_basic() {
    let x = Tensor::from_vec(vec![1.0, 3.0, 2.0], vec![3]);
    let y = Tensor::from_vec(vec![2.0, 2.0, 2.0], vec![3]);

    let result = iconnx::operators::less::Less::forward(&[x, y], &NodeAttributes::new());

    assert_eq!(result.shape(), &[3]);
    // 1 < 2 = true (1.0), 3 < 2 = false (0.0), 2 < 2 = false (0.0)
    assert_eq!(result.as_slice(), vec![1.0, 0.0, 0.0]);
}

/// Test GreaterOrEqual: x >= y returns 1.0 or 0.0
#[test]
fn test_greater_or_equal_basic() {
    let x = Tensor::from_vec(vec![1.0, 3.0, 2.0], vec![3]);
    let y = Tensor::from_vec(vec![2.0, 2.0, 2.0], vec![3]);

    let result = iconnx::operators::greater_or_equal::GreaterOrEqual::forward(
        &[x, y],
        &NodeAttributes::new(),
    );

    assert_eq!(result.shape(), &[3]);
    // 1 >= 2 = false (0.0), 3 >= 2 = true (1.0), 2 >= 2 = true (1.0)
    assert_eq!(result.as_slice(), vec![0.0, 1.0, 1.0]);
}

/// Test Equal: x == y returns 1.0 or 0.0
#[test]
fn test_equal_basic() {
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
    let y = Tensor::from_vec(vec![2.0, 2.0, 2.0], vec![3]);

    let result =
        iconnx::operators::equal::Equal::forward(&[x, y], &NodeAttributes::new());

    assert_eq!(result.shape(), &[3]);
    // 1 == 2 = false (0.0), 2 == 2 = true (1.0), 3 == 2 = false (0.0)
    assert_eq!(result.as_slice(), vec![0.0, 1.0, 0.0]);
}

/// Test Greater 2D
#[test]
fn test_greater_2d() {
    let x = Tensor::from_vec(vec![1.0, 5.0, 3.0, 2.0], vec![2, 2]);
    let y = Tensor::from_vec(vec![2.0, 4.0, 3.0, 1.0], vec![2, 2]);

    let result =
        iconnx::operators::greater::Greater::forward(&[x, y], &NodeAttributes::new());

    assert_eq!(result.shape(), &[2, 2]);
    // [1>2, 5>4, 3>3, 2>1] = [0, 1, 0, 1]
    assert_eq!(result.as_slice(), vec![0.0, 1.0, 0.0, 1.0]);
}

/// Test Equal with floats (exact comparison)
#[test]
fn test_equal_floats() {
    let x = Tensor::from_vec(vec![1.0, 2.0, 2.0], vec![3]);
    let y = Tensor::from_vec(vec![1.0, 2.0, 2.001], vec![3]);

    let result =
        iconnx::operators::equal::Equal::forward(&[x, y], &NodeAttributes::new());

    assert_eq!(result.shape(), &[3]);
    // Exact comparison: only first two are equal
    assert_eq!(result.as_slice(), vec![1.0, 1.0, 0.0]);
}
