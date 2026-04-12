use iconnx::attributes::NodeAttributes;
use iconnx::operators::gather::Gather;
/// Regression test for Gather operator with axis attribute
///
/// Bug: Gather was ignoring axis attribute and defaulting to 0
/// Fix: Read axis from attributes when not provided as input
use iconnx::tensor::Tensor;

#[test]
fn test_gather_axis_from_attribute() {
    // This reproduces node 1577 from the Kokoro model
    // Input: [1, 11, 3121, 2], index=0 (scalar), axis=3
    // Expected output: [1, 11, 3121]

    let data = Tensor::from_vec(vec![1.0; 11 * 3121 * 2], vec![1, 11, 3121, 2]);
    let indices = Tensor::from_vec_i64(vec![0], vec![]); // Scalar index

    let mut attrs = NodeAttributes::new();
    attrs.add_int("axis".to_string(), 3);

    let result = Gather::forward(&[data, indices], &attrs);

    assert_eq!(result.shape(), &[1, 11, 3121]);
}

#[test]
fn test_gather_axis_negative_from_attribute() {
    // Test negative axis in attributes
    // Input: [1, 11, 3121, 2], index=0, axis=-1 (same as axis=3)

    let data = Tensor::from_vec(vec![2.0; 11 * 3121 * 2], vec![1, 11, 3121, 2]);
    let indices = Tensor::from_vec_i64(vec![0], vec![]);

    let mut attrs = NodeAttributes::new();
    attrs.add_int("axis".to_string(), -1); // Last axis

    let result = Gather::forward(&[data, indices], &attrs);

    assert_eq!(result.shape(), &[1, 11, 3121]);
}

#[test]
fn test_gather_axis_as_input_still_works() {
    // Ensure backward compatibility: axis as third input still works

    let data = Tensor::from_vec(vec![3.0; 2 * 3 * 4], vec![2, 3, 4]);
    let indices = Tensor::from_vec_i64(vec![0, 1], vec![2]);
    let axis = Tensor::from_vec(vec![1.0], vec![]); // Scalar axis=1

    let attrs = NodeAttributes::new();

    let result = Gather::forward(&[data, indices, axis], &attrs);

    // output.shape = [2] + [2] + [4] = [2, 2, 4]
    assert_eq!(result.shape(), &[2, 2, 4]);
}

#[test]
fn test_gather_attributes_override_input() {
    // If both attribute and input present, attribute should take precedence

    let data = Tensor::from_vec(vec![1.0; 2 * 3 * 4], vec![2, 3, 4]);
    let indices = Tensor::from_vec_i64(vec![0], vec![]);
    let axis_input = Tensor::from_vec(vec![0.0], vec![]); // axis=0 as input

    let mut attrs = NodeAttributes::new();
    attrs.add_int("axis".to_string(), 2); // axis=2 as attribute (should win)

    let result = Gather::forward(&[data, indices, axis_input], &attrs);

    // If axis=2 is used: [2, 3] + [] + [] = [2, 3]
    // If axis=0 is used: [] + [] + [3, 4] = [3, 4]
    assert_eq!(result.shape(), &[2, 3]);
}

#[test]
fn test_gather_4d_tensor_last_axis() {
    // This reproduces the exact scenario from node 1577 where array may not be contiguous
    // Input: [1, 11, 3121, 2], gather along axis=3 with scalar index 0
    // Should work even if array is not contiguous

    // Create a smaller version: [2, 3, 4, 5] -> gather axis=3, index=0
    let data = Tensor::from_vec((0..120).map(|i| i as f32).collect(), vec![2, 3, 4, 5]);
    let indices = Tensor::from_vec_i64(vec![0], vec![]); // Scalar index

    let mut attrs = NodeAttributes::new();
    attrs.add_int("axis".to_string(), 3);

    let result = Gather::forward(&[data, indices], &attrs);

    // Expected: [2, 3, 4] + [] + [] = [2, 3, 4]
    assert_eq!(result.shape(), &[2, 3, 4]);
}
