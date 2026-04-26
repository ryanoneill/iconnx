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

// =============================================================================
// WS-3 M3.5 — CPU Cast FP16 boundary
// =============================================================================

/// Test Cast Float32 → Float16 (round-half-to-even).
#[test]
fn test_cast_float32_to_float16() {
    let data = Tensor::from_vec(vec![1.5_f32, -2.0, 0.0, 65504.0], vec![4]);
    let mut attrs = NodeAttributes::new();
    attrs.add_int("to".to_string(), 10); // ONNX_FLOAT16

    let result = Cast::forward(&[data], &attrs);

    assert!(result.is_float16());
    let slice = result.as_slice_f16();
    assert_eq!(slice[0], half::f16::from_f32(1.5));
    assert_eq!(slice[1], half::f16::from_f32(-2.0));
    assert_eq!(slice[2], half::f16::ZERO);
    assert_eq!(slice[3], half::f16::from_f32(65504.0));
}

/// Test Cast Float16 → Float32 round-trip.
#[test]
fn test_cast_float16_to_float32() {
    let host = vec![
        half::f16::from_f32(1.5),
        half::f16::from_f32(-2.0),
        half::f16::ZERO,
    ];
    let data = Tensor::from_vec_f16(host, vec![3]);
    let mut attrs = NodeAttributes::new();
    attrs.add_int("to".to_string(), 1); // ONNX_FLOAT

    let result = Cast::forward(&[data], &attrs);

    assert!(result.is_float32());
    assert_eq!(result.as_slice(), vec![1.5_f32, -2.0, 0.0]);
}

/// Test Cast Float16 → Int32 (truncates toward zero per ONNX spec).
#[test]
fn test_cast_float16_to_int32_truncates() {
    let host = vec![
        half::f16::from_f32(1.5),
        half::f16::from_f32(-2.4),
        half::f16::from_f32(0.9),
        half::f16::from_f32(-0.5),
    ];
    let data = Tensor::from_vec_f16(host, vec![4]);
    let mut attrs = NodeAttributes::new();
    attrs.add_int("to".to_string(), 6); // ONNX_INT32

    let result = Cast::forward(&[data], &attrs);

    assert!(result.is_int32());
    assert_eq!(result.as_slice_i32(), vec![1i32, -2, 0, 0]);
}

/// Test Cast Int32 → Float16.
#[test]
fn test_cast_int32_to_float16() {
    let data = Tensor::from_vec_i32(vec![1i32, -2, 0, 1024], vec![4]);
    let mut attrs = NodeAttributes::new();
    attrs.add_int("to".to_string(), 10); // ONNX_FLOAT16

    let result = Cast::forward(&[data], &attrs);

    assert!(result.is_float16());
    let slice = result.as_slice_f16();
    assert_eq!(slice[0], half::f16::from_f32(1.0));
    assert_eq!(slice[1], half::f16::from_f32(-2.0));
    assert_eq!(slice[2], half::f16::ZERO);
    assert_eq!(slice[3], half::f16::from_f32(1024.0));
}

/// Test Cast Float16 → Float16 (identity).
#[test]
fn test_cast_float16_identity() {
    let host = vec![half::f16::from_f32(3.5), half::f16::from_f32(-1.25)];
    let data = Tensor::from_vec_f16(host.clone(), vec![2]);
    let mut attrs = NodeAttributes::new();
    attrs.add_int("to".to_string(), 10); // ONNX_FLOAT16

    let result = Cast::forward(&[data], &attrs);

    assert!(result.is_float16());
    assert_eq!(result.as_slice_f16(), host);
}
