use iconnx::attributes::NodeAttributes;
/// Test Pad operator for Iconnx
///
/// TDD: Tests written FIRST
/// Pad: Add padding to tensor edges
use iconnx::tensor::Tensor;

/// Test 1: Basic constant padding on 1D tensor
#[test]
fn test_pad_1d_constant() {
    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
    let pads = Tensor::from_vec(vec![1.0, 1.0], vec![2]); // [left, right]
    let constant_value = Tensor::from_vec(vec![0.0], vec![]);

    let result = iconnx::operators::pad::Pad::forward(
        &[data, pads, constant_value],
        &NodeAttributes::new(),
    );

    assert_eq!(result.shape(), &[5]);
    assert_eq!(result.as_slice(), vec![0.0, 1.0, 2.0, 3.0, 0.0]);
}

/// Test 2: Pad 2D tensor on both axes
#[test]
fn test_pad_2d_constant() {
    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    // pads: [top, left, bottom, right]
    let pads = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![4]);
    let constant_value = Tensor::from_vec(vec![0.0], vec![]);

    let result = iconnx::operators::pad::Pad::forward(
        &[data, pads, constant_value],
        &NodeAttributes::new(),
    );

    assert_eq!(result.shape(), &[4, 4]);
    assert_eq!(
        result.as_slice(),
        vec![
            0.0, 0.0, 0.0, 0.0, // top row
            0.0, 1.0, 2.0, 0.0, // first data row with left/right padding
            0.0, 3.0, 4.0, 0.0, // second data row with left/right padding
            0.0, 0.0, 0.0, 0.0 // bottom row
        ]
    );
}

/// Test 3: Asymmetric padding
#[test]
fn test_pad_asymmetric() {
    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
    let pads = Tensor::from_vec(vec![2.0, 1.0], vec![2]); // [left=2, right=1]
    let constant_value = Tensor::from_vec(vec![9.0], vec![]);

    let result = iconnx::operators::pad::Pad::forward(
        &[data, pads, constant_value],
        &NodeAttributes::new(),
    );

    assert_eq!(result.shape(), &[6]);
    assert_eq!(result.as_slice(), vec![9.0, 9.0, 1.0, 2.0, 3.0, 9.0]);
}

/// Test 4: No padding (identity)
#[test]
fn test_pad_no_padding() {
    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let pads = Tensor::from_vec(vec![0.0, 0.0, 0.0, 0.0], vec![4]);
    let constant_value = Tensor::from_vec(vec![0.0], vec![]);

    let result = iconnx::operators::pad::Pad::forward(
        &[data, pads, constant_value],
        &NodeAttributes::new(),
    );

    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.as_slice(), vec![1.0, 2.0, 3.0, 4.0]);
}

/// Test 5: Int64 data type
#[test]
fn test_pad_int64() {
    let data = Tensor::from_vec_i64(vec![1, 2, 3], vec![3]);
    let pads = Tensor::from_vec_i64(vec![1, 1], vec![2]);
    let constant_value = Tensor::from_vec_i64(vec![0], vec![]);

    let result = iconnx::operators::pad::Pad::forward(
        &[data, pads, constant_value],
        &NodeAttributes::new(),
    );

    assert_eq!(result.shape(), &[5]);
    assert!(result.is_int64(), "Result should preserve Int64 dtype");
    assert_eq!(result.as_slice_i64(), vec![0, 1, 2, 3, 0]);
}

/// Test 6: Int32 data type
#[test]
fn test_pad_int32() {
    let data = Tensor::from_vec_i32(vec![1, 2, 3], vec![3]);
    let pads = Tensor::from_vec_i64(vec![1, 1], vec![2]);
    let constant_value = Tensor::from_vec_i32(vec![0], vec![]);

    let result = iconnx::operators::pad::Pad::forward(
        &[data, pads, constant_value],
        &NodeAttributes::new(),
    );

    assert_eq!(result.shape(), &[5]);
    assert!(result.is_int32(), "Result should preserve Int32 dtype");
    assert_eq!(result.as_slice_i32(), vec![0, 1, 2, 3, 0]);
}

/// Test 7: Explicit mode="constant" attribute
#[test]
fn test_pad_mode_constant_explicit() {
    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
    let pads = Tensor::from_vec_i64(vec![1, 1], vec![2]);
    let constant_value = Tensor::from_vec(vec![0.0], vec![]);

    let mut attrs = NodeAttributes::new();
    attrs.add_string("mode".to_string(), "constant".to_string());

    let result =
        iconnx::operators::pad::Pad::forward(&[data, pads, constant_value], &attrs);

    assert_eq!(result.shape(), &[5]);
    assert_eq!(result.as_slice(), vec![0.0, 1.0, 2.0, 3.0, 0.0]);
}

/// Test 8: Edge mode - pad with edge values
#[test]
fn test_pad_mode_edge() {
    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
    let pads = Tensor::from_vec_i64(vec![2, 2], vec![2]);

    let mut attrs = NodeAttributes::new();
    attrs.add_string("mode".to_string(), "edge".to_string());

    let result = iconnx::operators::pad::Pad::forward(&[data, pads], &attrs);

    assert_eq!(result.shape(), &[7]);
    // Edge padding: [1, 1, 1, 2, 3, 3, 3]
    assert_eq!(result.as_slice(), vec![1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0]);
}

/// Test 9: Reflect mode - pad with reflected values
#[test]
fn test_pad_mode_reflect() {
    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    let pads = Tensor::from_vec_i64(vec![2, 2], vec![2]);

    let mut attrs = NodeAttributes::new();
    attrs.add_string("mode".to_string(), "reflect".to_string());

    let result = iconnx::operators::pad::Pad::forward(&[data, pads], &attrs);

    assert_eq!(result.shape(), &[8]);
    // Reflect padding: [3, 2, 1, 2, 3, 4, 3, 2]
    assert_eq!(
        result.as_slice(),
        vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]
    );
}

/// Test 10: Wrap mode - wrap-around padding
#[test]
fn test_pad_mode_wrap() {
    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    let pads = Tensor::from_vec_i64(vec![2, 2], vec![2]);

    let mut attrs = NodeAttributes::new();
    attrs.add_string("mode".to_string(), "wrap".to_string());

    let result = iconnx::operators::pad::Pad::forward(&[data, pads], &attrs);

    assert_eq!(result.shape(), &[8]);
    // Wrap padding: [3, 4, 1, 2, 3, 4, 1, 2]
    assert_eq!(
        result.as_slice(),
        vec![3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0]
    );
}

/// Test 11: Edge mode 2D
#[test]
fn test_pad_mode_edge_2d() {
    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    // pads: [top, left, bottom, right]
    let pads = Tensor::from_vec_i64(vec![1, 1, 1, 1], vec![4]);

    let mut attrs = NodeAttributes::new();
    attrs.add_string("mode".to_string(), "edge".to_string());

    let result = iconnx::operators::pad::Pad::forward(&[data, pads], &attrs);

    assert_eq!(result.shape(), &[4, 4]);
    // Edge padding extends border values
    assert_eq!(
        result.as_slice(),
        vec![
            1.0, 1.0, 2.0, 2.0, // top row (edge of first row)
            1.0, 1.0, 2.0, 2.0, // first data row with left/right edge
            3.0, 3.0, 4.0, 4.0, // second data row with left/right edge
            3.0, 3.0, 4.0, 4.0 // bottom row (edge of last row)
        ]
    );
}

/// Test 12: 3D padding with edge mode
#[test]
fn test_pad_mode_edge_3d() {
    // 3D tensor [2, 2, 2]
    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2]);
    // pads: [d0_begin, d1_begin, d2_begin, d0_end, d1_end, d2_end]
    let pads = Tensor::from_vec_i64(vec![1, 0, 0, 1, 0, 0], vec![6]);

    let mut attrs = NodeAttributes::new();
    attrs.add_string("mode".to_string(), "edge".to_string());

    let result = iconnx::operators::pad::Pad::forward(&[data, pads], &attrs);

    assert_eq!(result.shape(), &[4, 2, 2]);
}
