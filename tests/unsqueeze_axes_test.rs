use iconnx::attributes::NodeAttributes;
use iconnx::operators::unsqueeze::Unsqueeze;
/// Unit tests for Unsqueeze operator with multiple axes
///
/// Testing the ONNX spec: axes specify positions in OUTPUT shape
use iconnx::tensor::Tensor;

#[test]
fn test_unsqueeze_single_axis_at_start() {
    // Input: [2, 3], axes: [0]
    // Output: [1, 2, 3]
    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let axes = Tensor::from_vec_i64(vec![0], vec![1]);

    let result = Unsqueeze::forward(&[data, axes], &NodeAttributes::new());

    assert_eq!(result.shape(), &[1, 2, 3]);
}

#[test]
fn test_unsqueeze_single_axis_at_end() {
    // Input: [2, 3], axes: [2]
    // Output: [2, 3, 1]
    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let axes = Tensor::from_vec_i64(vec![2], vec![1]);

    let result = Unsqueeze::forward(&[data, axes], &NodeAttributes::new());

    assert_eq!(result.shape(), &[2, 3, 1]);
}

#[test]
fn test_unsqueeze_two_axes_at_boundaries() {
    // Input: [2, 3], axes: [0, 3]
    // Output: [1, 2, 3, 1]
    // Position 0: insert at start -> [1, 2, 3]
    // Position 3: insert at end -> [1, 2, 3, 1]
    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let axes = Tensor::from_vec_i64(vec![0, 3], vec![2]);

    let result = Unsqueeze::forward(&[data, axes], &NodeAttributes::new());

    assert_eq!(result.shape(), &[1, 2, 3, 1]);
}

#[test]
fn test_unsqueeze_two_axes_in_middle() {
    // Input: [2, 3], axes: [1, 2]
    // Output: [2, 1, 1, 3]
    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let axes = Tensor::from_vec_i64(vec![1, 2], vec![2]);

    let result = Unsqueeze::forward(&[data, axes], &NodeAttributes::new());

    assert_eq!(result.shape(), &[2, 1, 1, 3]);
}

#[test]
fn test_unsqueeze_from_1d_to_4d() {
    // Input: [3], axes: [0, 2]
    // Output: [1, 3, 1]
    // Wait, that's only 3D. Let me recalculate...
    // Input: [3] (1D), axes: [0, 2]
    // Output should be: [1, 3, 1] (3D)
    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
    let axes = Tensor::from_vec_i64(vec![0, 2], vec![2]);

    let result = Unsqueeze::forward(&[data, axes], &NodeAttributes::new());

    assert_eq!(result.shape(), &[1, 3, 1]);
}

#[test]
fn test_unsqueeze_unsorted_axes() {
    // Input: [2, 3], axes: [3, 0] (unsorted)
    // Should be same as sorted [0, 3]
    // Output: [1, 2, 3, 1]
    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let axes = Tensor::from_vec_i64(vec![3, 0], vec![2]);

    let result = Unsqueeze::forward(&[data, axes], &NodeAttributes::new());

    assert_eq!(result.shape(), &[1, 2, 3, 1]);
}

#[test]
fn test_unsqueeze_node_2453_scenario() {
    // Reproduce the failing scenario from node 2453
    // Let's guess it's trying to insert at invalid positions
    // Input: [19, 1], axes: [?]
    // We'll need to check the actual debug output to know
    // For now, test a plausible case
    let data = Tensor::from_vec(vec![1.0; 19], vec![19, 1]);
    let axes = Tensor::from_vec_i64(vec![0], vec![1]);

    let result = Unsqueeze::forward(&[data, axes], &NodeAttributes::new());

    assert_eq!(result.shape(), &[1, 19, 1]);
}

#[test]
fn test_unsqueeze_negative_axis() {
    // Reproduce the actual failing scenario from node 2453
    // Input: [19, 1], axes: [-1]
    // Output rank: 2 + 1 = 3
    // Axis -1 -> 3 + (-1) = 2
    // Expected output: [19, 1, 1]
    let data = Tensor::from_vec(vec![1.0; 19], vec![19, 1]);
    let axes = Tensor::from_vec_i64(vec![-1], vec![1]);

    let result = Unsqueeze::forward(&[data, axes], &NodeAttributes::new());

    assert_eq!(result.shape(), &[19, 1, 1]);
}

#[test]
fn test_unsqueeze_negative_axis_minus_two() {
    // Input: [2, 3], axes: [-2]
    // Output rank: 2 + 1 = 3
    // Axis -2 -> 3 + (-2) = 1
    // Expected output: [2, 1, 3]
    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let axes = Tensor::from_vec_i64(vec![-2], vec![1]);

    let result = Unsqueeze::forward(&[data, axes], &NodeAttributes::new());

    assert_eq!(result.shape(), &[2, 1, 3]);
}

#[test]
fn test_unsqueeze_mixed_positive_negative_axes() {
    // Input: [2, 3], axes: [0, -1]
    // Output rank: 2 + 2 = 4
    // Axis 0 stays 0
    // Axis -1 -> 4 + (-1) = 3
    // Expected output: [1, 2, 3, 1]
    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let axes = Tensor::from_vec_i64(vec![0, -1], vec![2]);

    let result = Unsqueeze::forward(&[data, axes], &NodeAttributes::new());

    assert_eq!(result.shape(), &[1, 2, 3, 1]);
}
