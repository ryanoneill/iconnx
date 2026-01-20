use iconnx::attributes::NodeAttributes;
/// Test Slice operator for Iconnx
///
/// TDD: Tests written FIRST
/// Slice: Extract range [start:end:step] along axes
use iconnx::tensor::Tensor;

/// Test 1: Basic slice of 1D tensor
#[test]
fn test_slice_1d_basic() {
    let data = Tensor::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], vec![6]);
    let starts = Tensor::from_vec(vec![1.0], vec![1]);
    let ends = Tensor::from_vec(vec![4.0], vec![1]);
    let axes = Tensor::from_vec(vec![0.0], vec![1]);
    let steps = Tensor::from_vec(vec![1.0], vec![1]);

    let result = iconnx::operators::slice::Slice::forward(
        &[data, starts, ends, axes, steps],
        &NodeAttributes::new(),
    );

    assert_eq!(result.shape(), &[3]);
    assert_eq!(result.as_slice(), vec![1.0, 2.0, 3.0]);
}

/// Test 2: Slice with step
#[test]
fn test_slice_with_step() {
    let data = Tensor::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], vec![6]);
    let starts = Tensor::from_vec(vec![0.0], vec![1]);
    let ends = Tensor::from_vec(vec![6.0], vec![1]);
    let axes = Tensor::from_vec(vec![0.0], vec![1]);
    let steps = Tensor::from_vec(vec![2.0], vec![1]);

    let result = iconnx::operators::slice::Slice::forward(
        &[data, starts, ends, axes, steps],
        &NodeAttributes::new(),
    );

    assert_eq!(result.shape(), &[3]);
    assert_eq!(result.as_slice(), vec![0.0, 2.0, 4.0]);
}

/// Test 3: Slice 2D along axis 0
#[test]
fn test_slice_2d_axis_0() {
    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
    let starts = Tensor::from_vec(vec![0.0], vec![1]);
    let ends = Tensor::from_vec(vec![2.0], vec![1]);
    let axes = Tensor::from_vec(vec![0.0], vec![1]);
    let steps = Tensor::from_vec(vec![1.0], vec![1]);

    let result = iconnx::operators::slice::Slice::forward(
        &[data, starts, ends, axes, steps],
        &NodeAttributes::new(),
    );

    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.as_slice(), vec![1.0, 2.0, 3.0, 4.0]);
}

/// Test 4: Slice 2D along axis 1
#[test]
fn test_slice_2d_axis_1() {
    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 4]);
    let starts = Tensor::from_vec(vec![1.0], vec![1]);
    let ends = Tensor::from_vec(vec![3.0], vec![1]);
    let axes = Tensor::from_vec(vec![1.0], vec![1]);
    let steps = Tensor::from_vec(vec![1.0], vec![1]);

    let result = iconnx::operators::slice::Slice::forward(
        &[data, starts, ends, axes, steps],
        &NodeAttributes::new(),
    );

    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.as_slice(), vec![2.0, 3.0, 6.0, 7.0]);
}

/// Test 5: Slice with negative axis (-1 = last axis)
#[test]
fn test_slice_negative_axis() {
    // 3D tensor [1, 1, 15620] - slice along last axis
    let data = Tensor::from_vec((0..100).map(|x| x as f32).collect(), vec![1, 1, 100]);
    let starts = Tensor::from_vec_i64(vec![10], vec![1]);
    let ends = Tensor::from_vec_i64(vec![20], vec![1]);
    let axes = Tensor::from_vec_i64(vec![-1], vec![1]); // -1 = axis 2
    let steps = Tensor::from_vec_i64(vec![1], vec![1]);

    let result = iconnx::operators::slice::Slice::forward(
        &[data, starts, ends, axes, steps],
        &NodeAttributes::new(),
    );

    // Should slice [1, 1, 100] along axis 2 from 10 to 20 -> [1, 1, 10]
    assert_eq!(result.shape(), &[1, 1, 10]);
    assert_eq!(
        result.as_slice(),
        (10..20).map(|x| x as f32).collect::<Vec<_>>()
    );
}

/// Test 6: Slice with negative axis (-2)
#[test]
fn test_slice_negative_axis_minus_two() {
    // 3D tensor [2, 4, 3]
    let data = Tensor::from_vec((0..24).map(|x| x as f32).collect(), vec![2, 4, 3]);
    let starts = Tensor::from_vec_i64(vec![1], vec![1]);
    let ends = Tensor::from_vec_i64(vec![3], vec![1]);
    let axes = Tensor::from_vec_i64(vec![-2], vec![1]); // -2 = axis 1
    let steps = Tensor::from_vec_i64(vec![1], vec![1]);

    let result = iconnx::operators::slice::Slice::forward(
        &[data, starts, ends, axes, steps],
        &NodeAttributes::new(),
    );

    // Should slice [2, 4, 3] along axis 1 from 1 to 3 -> [2, 2, 3]
    assert_eq!(result.shape(), &[2, 2, 3]);
}
