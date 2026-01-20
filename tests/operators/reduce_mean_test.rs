use iconnx::attributes::NodeAttributes;
/// Comprehensive ReduceMean operator tests for Iconnx
///
/// Tests all branches and regression test for keepdims bug
/// ONNX spec: https://onnx.ai/onnx/operators/onnx__ReduceMean.html
use iconnx::operators::reduce_mean::ReduceMean;
use iconnx::tensor::Tensor;

#[test]
fn test_reduce_mean_keepdims_true() {
    // REGRESSION TEST: keepdims=1 should keep reduced dimension as size 1
    // Was removing dimension before the fix

    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]); // [2, 3]

    let axes = Tensor::from_vec_i64(vec![1], vec![1]); // Reduce along axis 1

    let mut attrs = NodeAttributes::new();
    attrs.add_int("keepdims".to_string(), 1);

    let result = ReduceMean::forward(&[data, axes], &attrs);

    // With keepdims=1, shape should be [2, 1]
    assert_eq!(result.shape(), &[2, 1]);

    // Values: mean of [1,2,3] = 2.0, mean of [4,5,6] = 5.0
    let vals = result.as_slice();
    assert!((vals[0] - 2.0).abs() < 1e-6);
    assert!((vals[1] - 5.0).abs() < 1e-6);
}

#[test]
fn test_reduce_mean_keepdims_false() {
    // When keepdims=0, reduced dimension should be removed

    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

    let axes = Tensor::from_vec_i64(vec![1], vec![1]);

    let mut attrs = NodeAttributes::new();
    attrs.add_int("keepdims".to_string(), 0);

    let result = ReduceMean::forward(&[data, axes], &attrs);

    // With keepdims=0, shape should be [2]
    assert_eq!(result.shape(), &[2]);

    let vals = result.as_slice();
    assert!((vals[0] - 2.0).abs() < 1e-6);
    assert!((vals[1] - 5.0).abs() < 1e-6);
}

#[test]
fn test_reduce_mean_default_keepdims() {
    // ONNX default for keepdims is 1

    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let axes = Tensor::from_vec_i64(vec![0], vec![1]);

    // No keepdims attribute - should default to 1
    let result = ReduceMean::forward(&[data, axes], &NodeAttributes::new());

    // Default keepdims=1, shape should be [1, 2]
    assert_eq!(result.shape(), &[1, 2]);
}

#[test]
fn test_reduce_mean_3d_tensor() {
    // Test with 3D tensor reducing last axis

    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![1, 2, 4]); // [1, 2, 4]

    let axes = Tensor::from_vec_i64(vec![2], vec![1]); // Reduce axis 2

    let mut attrs = NodeAttributes::new();
    attrs.add_int("keepdims".to_string(), 1);

    let result = ReduceMean::forward(&[data, axes], &attrs);

    // Shape should be [1, 2, 1]
    assert_eq!(result.shape(), &[1, 2, 1]);

    // Values: mean of [1,2,3,4] = 2.5, mean of [5,6,7,8] = 6.5
    let vals = result.as_slice();
    assert!((vals[0] - 2.5).abs() < 1e-6);
    assert!((vals[1] - 6.5).abs() < 1e-6);
}

#[test]
fn test_reduce_mean_negative_axis() {
    // Test negative axis indexing (-1 = last axis)

    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let axes = Tensor::from_vec_i64(vec![-1], vec![1]); // -1 = last axis

    let mut attrs = NodeAttributes::new();
    attrs.add_int("keepdims".to_string(), 1);

    let result = ReduceMean::forward(&[data, axes], &attrs);

    // Shape should be [2, 1]
    assert_eq!(result.shape(), &[2, 1]);

    // Values: mean of [1,2] = 1.5, mean of [3,4] = 3.5
    let vals = result.as_slice();
    assert!((vals[0] - 1.5).abs() < 1e-6);
    assert!((vals[1] - 3.5).abs() < 1e-6);
}

#[test]
fn test_reduce_mean_int64() {
    // Test with int64 data type

    let data = Tensor::from_vec_i64(vec![10, 20, 30, 40], vec![2, 2]);
    let axes = Tensor::from_vec_i64(vec![1], vec![1]);

    let mut attrs = NodeAttributes::new();
    attrs.add_int("keepdims".to_string(), 1);

    let result = ReduceMean::forward(&[data, axes], &attrs);

    // Shape should be [2, 1]
    assert_eq!(result.shape(), &[2, 1]);
    assert!(result.is_int64());

    // Values: mean of [10,20] = 15, mean of [30,40] = 35
    let vals = result.as_slice_i64();
    assert_eq!(vals[0], 15);
    assert_eq!(vals[1], 35);
}

#[test]
fn test_reduce_mean_single_element() {
    // Test reducing to a single element

    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let axes = Tensor::from_vec_i64(vec![0], vec![1]);

    let mut attrs = NodeAttributes::new();
    attrs.add_int("keepdims".to_string(), 1);

    let result = ReduceMean::forward(&[data, axes], &attrs);

    // Shape should be [1, 3]
    assert_eq!(result.shape(), &[1, 3]);

    // Values: mean of [1,4] = 2.5, mean of [2,5] = 3.5, mean of [3,6] = 4.5
    let vals = result.as_slice();
    assert!((vals[0] - 2.5).abs() < 1e-6);
    assert!((vals[1] - 3.5).abs() < 1e-6);
    assert!((vals[2] - 4.5).abs() < 1e-6);
}

#[test]
fn test_reduce_mean_axis_0() {
    // Test reducing along axis 0

    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    let axes = Tensor::from_vec_i64(vec![0], vec![1]);

    let mut attrs = NodeAttributes::new();
    attrs.add_int("keepdims".to_string(), 1);

    let result = ReduceMean::forward(&[data, axes], &attrs);

    // Shape should be [1, 2]
    assert_eq!(result.shape(), &[1, 2]);

    // Values: mean of [1,3] = 2.0, mean of [2,4] = 3.0
    let vals = result.as_slice();
    assert!((vals[0] - 2.0).abs() < 1e-6);
    assert!((vals[1] - 3.0).abs() < 1e-6);
}

#[test]
fn test_reduce_mean_reproduces_node_1344() {
    // Reproduce the exact scenario from node 1344
    // Input: [1, 512, 26], reduce along axis 2, keepdims=1
    // Expected output: [1, 512, 1]

    // Create a small representative case
    let data = Tensor::from_vec(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        vec![1, 3, 4],
    ); // [1, 3, 4] similar structure

    let axes = Tensor::from_vec_i64(vec![2], vec![1]); // Reduce last axis

    let mut attrs = NodeAttributes::new();
    attrs.add_int("keepdims".to_string(), 1);

    let result = ReduceMean::forward(&[data, axes], &attrs);

    // Critical: shape must preserve all dimensions
    assert_eq!(result.shape(), &[1, 3, 1]);

    // Verify values
    let vals = result.as_slice();
    assert!((vals[0] - 2.5).abs() < 1e-6); // mean of [1,2,3,4]
    assert!((vals[1] - 6.5).abs() < 1e-6); // mean of [5,6,7,8]
    assert!((vals[2] - 10.5).abs() < 1e-6); // mean of [9,10,11,12]
}
