use iconnx::attributes::NodeAttributes;
use iconnx::operators::squeeze::Squeeze;
/// Regression test for Squeeze operator with multi-dtype support
///
/// Bug: Squeeze panicked on int64 tensors
/// Fix: Handle Int64, Int32, and other dtypes
use iconnx::tensor::Tensor;

#[test]
fn test_squeeze_int64_tensor() {
    // This reproduces node 2432 from Kokoro model
    // Input: int64 tensor of shape [1]
    // Squeeze axis 0 -> scalar []

    let data = Tensor::from_vec_i64(vec![5], vec![1]);
    let axes = Tensor::from_vec_i64(vec![0], vec![1]);

    let attrs = NodeAttributes::new();
    let result = Squeeze::forward(&[data, axes], &attrs);

    assert_eq!(result.shape(), &[] as &[usize]);
    assert!(result.is_int64());
    assert_eq!(result.as_slice_i64(), &[5]);
}

#[test]
fn test_squeeze_int64_multidim() {
    // Input: [2, 1, 3, 1] -> squeeze axes [1, 3] -> [2, 3]

    let data = Tensor::from_vec_i64(vec![1, 2, 3, 4, 5, 6], vec![2, 1, 3, 1]);
    let axes = Tensor::from_vec_i64(vec![1, 3], vec![2]);

    let attrs = NodeAttributes::new();
    let result = Squeeze::forward(&[data, axes], &attrs);

    assert_eq!(result.shape(), &[2, 3]);
    assert!(result.is_int64());
}

#[test]
fn test_squeeze_int32_tensor() {
    // Test Int32 support

    let data = Tensor::from_vec_i32(vec![10, 20], vec![2, 1]);
    let axes = Tensor::from_vec_i64(vec![1], vec![1]);

    let attrs = NodeAttributes::new();
    let result = Squeeze::forward(&[data, axes], &attrs);

    assert_eq!(result.shape(), &[2]);
    assert!(result.is_int32());
    assert_eq!(result.as_slice_i32(), &[10, 20]);
}

#[test]
fn test_squeeze_float32_still_works() {
    // Ensure Float32 path still works

    let data = Tensor::from_vec(vec![1.5, 2.5], vec![2, 1]);
    let axes = Tensor::from_vec_i64(vec![1], vec![1]);

    let attrs = NodeAttributes::new();
    let result = Squeeze::forward(&[data, axes], &attrs);

    assert_eq!(result.shape(), &[2]);
    assert_eq!(result.as_slice(), &[1.5, 2.5]);
}
