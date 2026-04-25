/// Tensor implementation for Iconnx
///
/// Multi-dtype support for ONNX compatibility
/// Supports: Float32, Float64, Int64, Int32, Int8, UInt8, Bool
use ndarray::{ArrayD, IxDyn};

/// N-dimensional tensor with multiple data types
///
/// ONNX models use various data types for different purposes:
/// - Float32: Most computations (weights, activations)
/// - Int64: Token IDs, indices
/// - Int32: Shape tensors, smaller indices
/// - Bool: Masks, conditions
#[derive(Clone, Debug)]
pub enum Tensor {
    /// 32-bit floating point (f32)
    Float32(ArrayD<f32>),
    /// 64-bit floating point (f64)
    Float64(ArrayD<f64>),
    /// 64-bit signed integer (i64)
    Int64(ArrayD<i64>),
    /// 32-bit signed integer (i32)
    Int32(ArrayD<i32>),
    /// Boolean (bool)
    Bool(ArrayD<bool>),
    /// 8-bit signed integer (i8) — INT8 quantization weights / inputs.
    Int8(ArrayD<i8>),
    /// 8-bit unsigned integer (u8) — UINT8 quantization (DynamicQuantizeLinear output).
    UInt8(ArrayD<u8>),
}

impl Tensor {
    // ========== Constructors ==========

    /// Create Float32 tensor from flat vector and shape
    pub fn from_vec_f32(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let expected_len: usize = if shape.is_empty() {
            1
        } else {
            shape.iter().product()
        };

        if data.len() != expected_len {
            panic!(
                "Shape mismatch: data has {} elements but shape {:?} requires {}",
                data.len(),
                shape,
                expected_len
            );
        }

        let array = ArrayD::from_shape_vec(IxDyn(&shape), data)
            .expect("Failed to create array from shape and data");

        Tensor::Float32(array)
    }

    /// Create Int64 tensor from flat vector and shape
    pub fn from_vec_i64(data: Vec<i64>, shape: Vec<usize>) -> Self {
        let expected_len: usize = if shape.is_empty() {
            1
        } else {
            shape.iter().product()
        };

        if data.len() != expected_len {
            panic!(
                "Shape mismatch: data has {} elements but shape {:?} requires {}",
                data.len(),
                shape,
                expected_len
            );
        }

        let array = ArrayD::from_shape_vec(IxDyn(&shape), data)
            .expect("Failed to create array from shape and data");

        Tensor::Int64(array)
    }

    /// Create Int32 tensor from flat vector and shape
    pub fn from_vec_i32(data: Vec<i32>, shape: Vec<usize>) -> Self {
        let expected_len: usize = if shape.is_empty() {
            1
        } else {
            shape.iter().product()
        };

        if data.len() != expected_len {
            panic!(
                "Shape mismatch: data has {} elements but shape {:?} requires {}",
                data.len(),
                shape,
                expected_len
            );
        }

        let array = ArrayD::from_shape_vec(IxDyn(&shape), data)
            .expect("Failed to create array from shape and data");

        Tensor::Int32(array)
    }

    /// Create Float64 tensor from flat vector and shape.
    ///
    /// Added in WS-1 M1.2 so `extract_weights` can represent ONNX
    /// FLOAT64(11) initializers rather than silently dropping them.
    pub fn from_vec_f64(data: Vec<f64>, shape: Vec<usize>) -> Self {
        let expected_len: usize = if shape.is_empty() {
            1
        } else {
            shape.iter().product()
        };

        if data.len() != expected_len {
            panic!(
                "Shape mismatch: data has {} elements but shape {:?} requires {}",
                data.len(),
                shape,
                expected_len
            );
        }

        let array = ArrayD::from_shape_vec(IxDyn(&shape), data)
            .expect("Failed to create array from shape and data");

        Tensor::Float64(array)
    }

    /// Create Bool tensor from flat vector and shape.
    ///
    /// Added in WS-1 M1.2 so `extract_weights` can represent ONNX
    /// BOOL(9) initializers rather than silently dropping them.
    pub fn from_vec_bool(data: Vec<bool>, shape: Vec<usize>) -> Self {
        let expected_len: usize = if shape.is_empty() {
            1
        } else {
            shape.iter().product()
        };

        if data.len() != expected_len {
            panic!(
                "Shape mismatch: data has {} elements but shape {:?} requires {}",
                data.len(),
                shape,
                expected_len
            );
        }

        let array = ArrayD::from_shape_vec(IxDyn(&shape), data)
            .expect("Failed to create array from shape and data");

        Tensor::Bool(array)
    }

    /// Create Int8 tensor from flat vector and shape.
    ///
    /// Added in WS-4 M4.1 for INT8 quantization support
    /// (DistilBERT-INT8 weights / activations).
    pub fn from_vec_i8(data: Vec<i8>, shape: Vec<usize>) -> Self {
        let expected_len: usize = if shape.is_empty() {
            1
        } else {
            shape.iter().product()
        };

        if data.len() != expected_len {
            panic!(
                "Shape mismatch: data has {} elements but shape {:?} requires {}",
                data.len(),
                shape,
                expected_len
            );
        }

        let array = ArrayD::from_shape_vec(IxDyn(&shape), data)
            .expect("Failed to create array from shape and data");

        Tensor::Int8(array)
    }

    /// Create UInt8 tensor from flat vector and shape.
    ///
    /// Added in WS-4 M4.1 for UINT8 quantization support
    /// (DynamicQuantizeLinear output / DistilBERT-INT8 activations).
    pub fn from_vec_u8(data: Vec<u8>, shape: Vec<usize>) -> Self {
        let expected_len: usize = if shape.is_empty() {
            1
        } else {
            shape.iter().product()
        };

        if data.len() != expected_len {
            panic!(
                "Shape mismatch: data has {} elements but shape {:?} requires {}",
                data.len(),
                shape,
                expected_len
            );
        }

        let array = ArrayD::from_shape_vec(IxDyn(&shape), data)
            .expect("Failed to create array from shape and data");

        Tensor::UInt8(array)
    }

    /// Legacy: Create Float32 tensor (backward compatibility)
    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self::from_vec_f32(data, shape)
    }

    /// Create tensor from existing Float32 ndarray
    pub fn from_array(array: ArrayD<f32>) -> Self {
        Tensor::Float32(array)
    }

    /// Create tensor from existing Int64 ndarray
    pub fn from_array_i64(array: ArrayD<i64>) -> Self {
        Tensor::Int64(array)
    }

    /// Create tensor from existing Int32 ndarray
    pub fn from_array_i32(array: ArrayD<i32>) -> Self {
        Tensor::Int32(array)
    }

    // ========== Type introspection ==========

    /// Get data type as string
    pub fn dtype(&self) -> &'static str {
        match self {
            Tensor::Float32(_) => "float32",
            Tensor::Float64(_) => "float64",
            Tensor::Int64(_) => "int64",
            Tensor::Int32(_) => "int32",
            Tensor::Bool(_) => "bool",
            Tensor::Int8(_) => "int8",
            Tensor::UInt8(_) => "uint8",
        }
    }

    /// Check if tensor is Float32
    pub fn is_float32(&self) -> bool {
        matches!(self, Tensor::Float32(_))
    }

    /// Check if tensor is Int64
    pub fn is_int64(&self) -> bool {
        matches!(self, Tensor::Int64(_))
    }

    /// Check if tensor is Int32
    pub fn is_int32(&self) -> bool {
        matches!(self, Tensor::Int32(_))
    }

    /// Check if tensor is Int8
    pub fn is_int8(&self) -> bool {
        matches!(self, Tensor::Int8(_))
    }

    /// Check if tensor is UInt8
    pub fn is_uint8(&self) -> bool {
        matches!(self, Tensor::UInt8(_))
    }

    // ========== Shape operations ==========

    /// Get tensor shape
    pub fn shape(&self) -> &[usize] {
        match self {
            Tensor::Float32(a) => a.shape(),
            Tensor::Float64(a) => a.shape(),
            Tensor::Int64(a) => a.shape(),
            Tensor::Int32(a) => a.shape(),
            Tensor::Bool(a) => a.shape(),
            Tensor::Int8(a) => a.shape(),
            Tensor::UInt8(a) => a.shape(),
        }
    }

    /// Get number of dimensions
    pub fn ndim(&self) -> usize {
        match self {
            Tensor::Float32(a) => a.ndim(),
            Tensor::Float64(a) => a.ndim(),
            Tensor::Int64(a) => a.ndim(),
            Tensor::Int32(a) => a.ndim(),
            Tensor::Bool(a) => a.ndim(),
            Tensor::Int8(a) => a.ndim(),
            Tensor::UInt8(a) => a.ndim(),
        }
    }

    /// Get total number of elements
    pub fn len(&self) -> usize {
        match self {
            Tensor::Float32(a) => a.len(),
            Tensor::Float64(a) => a.len(),
            Tensor::Int64(a) => a.len(),
            Tensor::Int32(a) => a.len(),
            Tensor::Bool(a) => a.len(),
            Tensor::Int8(a) => a.len(),
            Tensor::UInt8(a) => a.len(),
        }
    }

    /// Check if tensor is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    // ========== Data access ==========

    /// Get reference to Float32 array (panics if wrong type)
    pub fn to_array(&self) -> &ArrayD<f32> {
        match self {
            Tensor::Float32(a) => a,
            _ => panic!("Tensor is not Float32, it is {}", self.dtype()),
        }
    }

    /// Get Float32 array reference, auto-converting if needed
    ///
    /// Returns owned tensor if conversion was needed (so caller gets &ArrayD).
    /// This is a temporary bridge for operators not yet updated for multi-dtype.
    pub fn as_f32_array(&self) -> (&ArrayD<f32>, Option<Tensor>) {
        match self {
            Tensor::Float32(a) => (a, None),
            _ => {
                // Convert and return owned
                let converted = self.to_float32();
                match &converted {
                    Tensor::Float32(_a) => {
                        // Can't return reference to temporary!
                        // Instead, panic with helpful message
                        panic!(
                            "Operator needs multi-dtype support! Tensor is {} but operator expects Float32. \
                             Use map_preserving_dtype or to_float32() explicitly.",
                            self.dtype()
                        )
                    }
                    _ => unreachable!(),
                }
            }
        }
    }

    /// Get reference to Int64 array (panics if wrong type)
    pub fn to_array_i64(&self) -> &ArrayD<i64> {
        match self {
            Tensor::Int64(a) => a,
            _ => panic!("Tensor is not Int64, it is {}", self.dtype()),
        }
    }

    /// Get data as flat Float32 slice (panics if wrong type)
    ///
    /// If data is not contiguous (e.g., after transpose), creates a contiguous copy
    pub fn as_slice(&self) -> Vec<f32> {
        match self {
            Tensor::Float32(a) => {
                if let Some(slice) = a.as_slice() {
                    slice.to_vec()
                } else {
                    a.iter().copied().collect()
                }
            }
            _ => panic!("as_slice() called on non-Float32 tensor ({})", self.dtype()),
        }
    }

    /// Get data as flat Int32 slice (panics if wrong type)
    pub fn as_slice_i32(&self) -> Vec<i32> {
        match self {
            Tensor::Int32(a) => {
                if let Some(slice) = a.as_slice() {
                    slice.to_vec()
                } else {
                    a.iter().copied().collect()
                }
            }
            _ => panic!(
                "as_slice_i32() called on non-Int32 tensor ({})",
                self.dtype()
            ),
        }
    }

    /// Get data as flat Int64 slice (panics if wrong type)
    pub fn as_slice_i64(&self) -> Vec<i64> {
        match self {
            Tensor::Int64(a) => {
                if let Some(slice) = a.as_slice() {
                    slice.to_vec()
                } else {
                    a.iter().copied().collect()
                }
            }
            _ => panic!(
                "as_slice_i64() called on non-Int64 tensor ({})",
                self.dtype()
            ),
        }
    }

    /// Get data as flat Int8 slice (panics if wrong type)
    pub fn as_slice_i8(&self) -> Vec<i8> {
        match self {
            Tensor::Int8(a) => {
                if let Some(slice) = a.as_slice() {
                    slice.to_vec()
                } else {
                    a.iter().copied().collect()
                }
            }
            _ => panic!(
                "as_slice_i8() called on non-Int8 tensor ({})",
                self.dtype()
            ),
        }
    }

    /// Get data as flat UInt8 slice (panics if wrong type)
    pub fn as_slice_u8(&self) -> Vec<u8> {
        match self {
            Tensor::UInt8(a) => {
                if let Some(slice) = a.as_slice() {
                    slice.to_vec()
                } else {
                    a.iter().copied().collect()
                }
            }
            _ => panic!(
                "as_slice_u8() called on non-UInt8 tensor ({})",
                self.dtype()
            ),
        }
    }

    /// Get reference to Int8 array (panics if wrong type)
    pub fn to_array_i8(&self) -> &ArrayD<i8> {
        match self {
            Tensor::Int8(a) => a,
            _ => panic!("to_array_i8() called on non-Int8 tensor ({})", self.dtype()),
        }
    }

    /// Get reference to UInt8 array (panics if wrong type)
    pub fn to_array_u8(&self) -> &ArrayD<u8> {
        match self {
            Tensor::UInt8(a) => a,
            _ => panic!(
                "to_array_u8() called on non-UInt8 tensor ({})",
                self.dtype()
            ),
        }
    }

    // ========== Type conversions ==========

    /// Convert to Float32 (creates copy if needed)
    pub fn to_float32(&self) -> Tensor {
        match self {
            Tensor::Float32(_) => self.clone(),
            Tensor::Float64(a) => {
                let data: Vec<f32> = a.iter().map(|&x| x as f32).collect();
                Tensor::from_vec_f32(data, a.shape().to_vec())
            }
            Tensor::Int64(a) => {
                let data: Vec<f32> = a.iter().map(|&x| x as f32).collect();
                Tensor::from_vec_f32(data, a.shape().to_vec())
            }
            Tensor::Int32(a) => {
                let data: Vec<f32> = a.iter().map(|&x| x as f32).collect();
                Tensor::from_vec_f32(data, a.shape().to_vec())
            }
            Tensor::Bool(a) => {
                let data: Vec<f32> = a.iter().map(|&x| if x { 1.0 } else { 0.0 }).collect();
                Tensor::from_vec_f32(data, a.shape().to_vec())
            }
            Tensor::Int8(a) => {
                let data: Vec<f32> = a.iter().map(|&x| x as f32).collect();
                Tensor::from_vec_f32(data, a.shape().to_vec())
            }
            Tensor::UInt8(a) => {
                let data: Vec<f32> = a.iter().map(|&x| x as f32).collect();
                Tensor::from_vec_f32(data, a.shape().to_vec())
            }
        }
    }

    /// Convert to Int64 (creates copy if needed)
    pub fn to_int64(&self) -> Tensor {
        match self {
            Tensor::Int64(_) => self.clone(),
            Tensor::Float32(a) => {
                let data: Vec<i64> = a.iter().map(|&x| x as i64).collect();
                Tensor::from_vec_i64(data, a.shape().to_vec())
            }
            Tensor::Float64(a) => {
                let data: Vec<i64> = a.iter().map(|&x| x as i64).collect();
                Tensor::from_vec_i64(data, a.shape().to_vec())
            }
            Tensor::Int32(a) => {
                let data: Vec<i64> = a.iter().map(|&x| x as i64).collect();
                Tensor::from_vec_i64(data, a.shape().to_vec())
            }
            Tensor::Bool(a) => {
                let data: Vec<i64> = a.iter().map(|&x| if x { 1 } else { 0 }).collect();
                Tensor::from_vec_i64(data, a.shape().to_vec())
            }
            Tensor::Int8(a) => {
                let data: Vec<i64> = a.iter().map(|&x| x as i64).collect();
                Tensor::from_vec_i64(data, a.shape().to_vec())
            }
            Tensor::UInt8(a) => {
                let data: Vec<i64> = a.iter().map(|&x| x as i64).collect();
                Tensor::from_vec_i64(data, a.shape().to_vec())
            }
        }
    }

    // ========== Multi-dtype operation helpers ==========

    /// Apply a typed operation that preserves dtype
    ///
    /// Dispatches to the correct dtype and wraps result back in same type.
    /// Useful for operations like Concat, Unsqueeze, Reshape that preserve dtype.
    ///
    /// # Example
    /// ```text
    /// tensor.map_preserving_dtype(|arr_f32| {
    ///     // Operate on Float32 array
    ///     arr_f32.clone()
    /// }, |arr_i64| {
    ///     // Operate on Int64 array
    ///     arr_i64.clone()
    /// })
    /// ```
    pub fn map_preserving_dtype<F32Op, I64Op>(&self, f32_op: F32Op, i64_op: I64Op) -> Tensor
    where
        F32Op: FnOnce(&ArrayD<f32>) -> ArrayD<f32>,
        I64Op: FnOnce(&ArrayD<i64>) -> ArrayD<i64>,
    {
        match self {
            Tensor::Float32(arr) => Tensor::Float32(f32_op(arr)),
            Tensor::Int64(arr) => Tensor::Int64(i64_op(arr)),
            Tensor::Float64(arr) => {
                // Convert to f32, operate, convert back
                let as_f32 = arr.mapv(|x| x as f32);
                let result_f32 = f32_op(&as_f32);
                Tensor::Float64(result_f32.mapv(|x| x as f64))
            }
            Tensor::Int32(arr) => {
                // Convert to i64, operate, convert back
                let as_i64 = arr.mapv(|x| x as i64);
                let result_i64 = i64_op(&as_i64);
                Tensor::Int32(result_i64.mapv(|x| x as i32))
            }
            Tensor::Bool(arr) => {
                // Convert to i64, operate, convert back to bool
                let as_i64 = arr.mapv(|x| if x { 1i64 } else { 0i64 });
                let result_i64 = i64_op(&as_i64);
                Tensor::Bool(result_i64.mapv(|x| x != 0))
            }
            Tensor::Int8(arr) => {
                // Convert to i64, operate, convert back to i8
                let as_i64 = arr.mapv(|x| x as i64);
                let result_i64 = i64_op(&as_i64);
                Tensor::Int8(result_i64.mapv(|x| x as i8))
            }
            Tensor::UInt8(arr) => {
                // Convert to i64, operate, convert back to u8
                let as_i64 = arr.mapv(|x| x as i64);
                let result_i64 = i64_op(&as_i64);
                Tensor::UInt8(result_i64.mapv(|x| x as u8))
            }
        }
    }

    /// Get underlying array for any dtype (as dynamic reference)
    ///
    /// Returns a trait object that can be used for shape operations, indexing, etc.
    /// Does NOT give access to data - use map_preserving_dtype for that.
    pub fn array_shape(&self) -> &[usize] {
        self.shape()
    }

    /// Dispatch multi-tensor operations by dtype
    ///
    /// Handles mixed dtypes by converting all to common type.
    /// Prefers to preserve dtype when all tensors match, otherwise converts to Float32.
    pub fn dispatch_multi<F32Op, I64Op>(tensors: &[Tensor], f32_op: F32Op, i64_op: I64Op) -> Tensor
    where
        F32Op: FnOnce(&[&ArrayD<f32>]) -> ArrayD<f32>,
        I64Op: FnOnce(&[&ArrayD<i64>]) -> ArrayD<i64>,
    {
        assert!(
            !tensors.is_empty(),
            "dispatch_multi requires at least 1 tensor"
        );

        // Check if all tensors have same dtype
        let all_float32 = tensors.iter().all(|t| t.is_float32());
        let all_int64 = tensors.iter().all(|t| t.is_int64());

        if all_float32 {
            let arrays: Vec<&ArrayD<f32>> = tensors.iter().map(|t| t.to_array()).collect();
            Tensor::Float32(f32_op(&arrays))
        } else if all_int64 {
            let arrays: Vec<&ArrayD<i64>> = tensors.iter().map(|t| t.to_array_i64()).collect();
            Tensor::Int64(i64_op(&arrays))
        } else {
            // Mixed dtypes - convert all to Float32
            let as_f32: Vec<Tensor> = tensors.iter().map(|t| t.to_float32()).collect();

            // Convert to Float32 and apply operation
            // Let ndarray handle broadcasting naturally
            let arrays: Vec<&ArrayD<f32>> = as_f32.iter().map(|t| t.to_array()).collect();
            Tensor::Float32(f32_op(&arrays))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_int8_constructor_and_accessors() {
        let t = Tensor::from_vec_i8(vec![-128_i8, 0, 127], vec![3]);
        assert_eq!(t.dtype(), "int8");
        assert!(t.is_int8());
        assert_eq!(t.shape(), &[3]);
        assert_eq!(t.len(), 3);
        assert_eq!(t.as_slice_i8(), vec![-128_i8, 0, 127]);
    }

    #[test]
    fn tensor_uint8_constructor_and_accessors() {
        let t = Tensor::from_vec_u8(vec![0_u8, 128, 255], vec![3]);
        assert_eq!(t.dtype(), "uint8");
        assert!(t.is_uint8());
        assert_eq!(t.as_slice_u8(), vec![0_u8, 128, 255]);
    }

    #[test]
    fn tensor_int8_uint8_panic_on_wrong_accessor() {
        let i8t = Tensor::from_vec_i8(vec![0_i8], vec![1]);
        let result = std::panic::catch_unwind(|| i8t.as_slice_u8());
        assert!(result.is_err(), "as_slice_u8 on Int8 must panic");
    }

    #[test]
    fn tensor_int8_uint8_to_array_roundtrip() {
        let i8t = Tensor::from_vec_i8(vec![1, -2, 3], vec![3]);
        let arr = i8t.to_array_i8();
        assert_eq!(arr.shape(), &[3]);
        assert_eq!(arr[[0]], 1_i8);
        assert_eq!(arr[[1]], -2_i8);
    }
}
