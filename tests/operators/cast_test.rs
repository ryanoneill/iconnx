use iconnx::attributes::NodeAttributes;
use iconnx::operators::cast::Cast;
/// Test Cast operator for Iconnx
///
/// TDD: Tests for dtype conversions
use iconnx::tensor::Tensor;

/// Test Cast Float32 to Int64
#[test]
fn test_cast_float32_to_int64() {
    let data = Tensor::from_vec(vec![13.7, 11.2, 2.9], vec![3]);
    let mut attrs = NodeAttributes::new();
    attrs.add_int("to".to_string(), 7); // ONNX_INT64

    let result = Cast::forward(&[data], &attrs);

    assert!(result.is_int64());
    assert_eq!(result.as_slice_i64(), vec![13, 11, 2]);
}

/// Test Cast Int64 to Float32
#[test]
fn test_cast_int64_to_float32() {
    let data = Tensor::from_vec_i64(vec![13, 11, 2], vec![3]);
    let mut attrs = NodeAttributes::new();
    attrs.add_int("to".to_string(), 1); // ONNX_FLOAT

    let result = Cast::forward(&[data], &attrs);

    assert!(result.is_float32());
    assert_eq!(result.as_slice(), vec![13.0, 11.0, 2.0]);
}

/// Test Cast Int64 to Int32
#[test]
fn test_cast_int64_to_int32() {
    let data = Tensor::from_vec_i64(vec![13, 11, 2], vec![3]);
    let mut attrs = NodeAttributes::new();
    attrs.add_int("to".to_string(), 6); // ONNX_INT32

    let result = Cast::forward(&[data], &attrs);

    assert!(result.is_int32());
    assert_eq!(result.as_slice_i32(), vec![13, 11, 2]);
}

/// Test Cast to Bool
#[test]
fn test_cast_to_bool() {
    let data = Tensor::from_vec(vec![0.0, 1.0, 0.0, 5.0], vec![4]);
    let mut attrs = NodeAttributes::new();
    attrs.add_int("to".to_string(), 9); // ONNX_BOOL

    let result = Cast::forward(&[data], &attrs);

    assert!(result.is_float32());
    assert_eq!(
        result.as_slice(),
        vec![0.0, 1.0, 0.0, 1.0],
        "Non-zero values become 1.0"
    );
}

/// Test Cast same dtype (identity)
#[test]
fn test_cast_identity() {
    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
    let mut attrs = NodeAttributes::new();
    attrs.add_int("to".to_string(), 1); // ONNX_FLOAT

    let result = Cast::forward(&[data], &attrs);

    assert!(result.is_float32());
    assert_eq!(result.as_slice(), vec![1.0, 2.0, 3.0]);
}
