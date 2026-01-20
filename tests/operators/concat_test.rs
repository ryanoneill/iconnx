use iconnx::attributes::NodeAttributes;
/// Test Concat operator for Iconnx
///
/// TDD: Tests written FIRST
/// Concat concatenates tensors along an axis
use iconnx::tensor::Tensor;

/// Test 1: Concat two 1D tensors
#[test]
fn test_concat_1d() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
    let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], vec![3]);

    // Set axis via attributes
    let mut attrs = NodeAttributes::new();
    attrs.add_int("axis".to_string(), 0);

    let result = iconnx::operators::concat::Concat::forward(&[a, b], &attrs);

    assert_eq!(result.shape(), &[6]);
    assert_eq!(result.as_slice(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

/// Test 2: Concat along rows (axis 0)
#[test]
fn test_concat_2d_rows() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
    let mut attrs = NodeAttributes::new();
    attrs.add_int("axis".to_string(), 0);

    let result = iconnx::operators::concat::Concat::forward(&[a, b], &attrs);

    // Stack vertically: [[1,2], [3,4], [5,6], [7,8]]
    assert_eq!(result.shape(), &[4, 2]);
}

/// Test 3: Concat along columns (axis 1)
#[test]
fn test_concat_2d_cols() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
    let mut attrs = NodeAttributes::new();
    attrs.add_int("axis".to_string(), 1);

    let result = iconnx::operators::concat::Concat::forward(&[a, b], &attrs);

    // Stack horizontally: [[1,2,5,6], [3,4,7,8]]
    assert_eq!(result.shape(), &[2, 4]);
}

/// Test 4: Concat three tensors
#[test]
fn test_concat_three_tensors() {
    let a = Tensor::from_vec(vec![1.0], vec![1]);
    let b = Tensor::from_vec(vec![2.0], vec![1]);
    let c = Tensor::from_vec(vec![3.0], vec![1]);
    let mut attrs = NodeAttributes::new();
    attrs.add_int("axis".to_string(), 0);

    let result = iconnx::operators::concat::Concat::forward(&[a, b, c], &attrs);

    assert_eq!(result.shape(), &[3]);
    assert_eq!(result.as_slice(), vec![1.0, 2.0, 3.0]);
}

/// Test 5: Concat with negative axis (-1 = last axis)
#[test]
fn test_concat_negative_axis() {
    // Two 3D tensors: [19, 1, 1] each
    let a = Tensor::from_vec(vec![1.0; 19], vec![19, 1, 1]);
    let b = Tensor::from_vec(vec![2.0; 19], vec![19, 1, 1]);

    // axis=-1 means last axis (axis 2 for 3D tensor)
    let mut attrs = NodeAttributes::new();
    attrs.add_int("axis".to_string(), -1);

    let result = iconnx::operators::concat::Concat::forward(&[a, b], &attrs);

    // Concatenate along last axis: [19, 1, 2]
    assert_eq!(result.shape(), &[19, 1, 2]);
}

/// Test 6: Concat with negative axis (-2)
#[test]
fn test_concat_negative_axis_minus_two() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

    // axis=-2 means second-to-last axis (axis 0 for 2D tensor)
    let mut attrs = NodeAttributes::new();
    attrs.add_int("axis".to_string(), -2);

    let result = iconnx::operators::concat::Concat::forward(&[a, b], &attrs);

    // Same as axis=0: stack vertically [4, 2]
    assert_eq!(result.shape(), &[4, 2]);
}
