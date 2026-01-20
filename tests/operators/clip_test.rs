use iconnx::attributes::NodeAttributes;
/// Test Clip operator for Iconnx
///
/// TDD: Tests written FIRST
/// Clip clamps values: min(max(x, min_val), max_val)
use iconnx::tensor::Tensor;

/// Test 1: Basic clipping with min and max
#[test]
fn test_clip_basic() {
    let data = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0], vec![6]);
    let min = Tensor::from_vec(vec![-1.0], vec![1]);
    let max = Tensor::from_vec(vec![2.0], vec![1]);

    let result =
        iconnx::operators::clip::Clip::forward(&[data, min, max], &NodeAttributes::new());

    assert_eq!(result.shape(), &[6]);
    assert_eq!(result.as_slice(), vec![-1.0, -1.0, 0.0, 1.0, 2.0, 2.0]);
}

/// Test 2: Clip 2D tensor
#[test]
fn test_clip_2d() {
    let data = Tensor::from_vec(vec![0.0, 5.0, 10.0, 15.0], vec![2, 2]);
    let min = Tensor::from_vec(vec![2.0], vec![1]);
    let max = Tensor::from_vec(vec![12.0], vec![1]);

    let result =
        iconnx::operators::clip::Clip::forward(&[data, min, max], &NodeAttributes::new());

    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.as_slice(), vec![2.0, 5.0, 10.0, 12.0]);
}

/// Test 3: Clip with only min (no max)
#[test]
fn test_clip_min_only() {
    let data = Tensor::from_vec(vec![-5.0, -1.0, 0.0, 1.0, 5.0], vec![5]);
    let min = Tensor::from_vec(vec![0.0], vec![1]);
    let max = Tensor::from_vec(vec![f32::INFINITY], vec![1]);

    let result =
        iconnx::operators::clip::Clip::forward(&[data, min, max], &NodeAttributes::new());

    assert_eq!(result.as_slice(), vec![0.0, 0.0, 0.0, 1.0, 5.0]);
}

/// Test 4: Clip with only max (no min)
#[test]
fn test_clip_max_only() {
    let data = Tensor::from_vec(vec![-5.0, -1.0, 0.0, 1.0, 5.0], vec![5]);
    let min = Tensor::from_vec(vec![f32::NEG_INFINITY], vec![1]);
    let max = Tensor::from_vec(vec![1.0], vec![1]);

    let result =
        iconnx::operators::clip::Clip::forward(&[data, min, max], &NodeAttributes::new());

    assert_eq!(result.as_slice(), vec![-5.0, -1.0, 0.0, 1.0, 1.0]);
}
