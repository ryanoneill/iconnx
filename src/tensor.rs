/// Tensor implementation for Iconnx
///
/// Multi-dtype support for ONNX compatibility
/// Supports: Float32, Float64, Float16, BFloat16, Int64, Int32, Int8, UInt8, Bool
use half::{bf16, f16};
use ndarray::{ArrayD, IxDyn};

/// Re-export of the GPU-side [`DType`](crate::cuda::DType) tag enum so that
/// CPU-only consumers (e.g. the WS-5 `validate` module) can refer to a
/// canonical `crate::tensor::DType` without depending on the `cuda::tensor`
/// module path. Only available with the `cuda` feature, which is enabled by
/// default; if iconnx ever ships a non-cuda build target the validate module
/// will need a dedicated CPU-side dtype tag.
#[cfg(feature = "cuda")]
pub use crate::cuda::DType;

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
    /// IEEE binary16 — Float16 / FP16 (Whisper FP16, BERT-FP16, etc.).
    /// Added in WS-3 M3.1; uses the `half` crate's `f16` type for full
    /// IEEE-754 binary16 semantics including round-half-to-even when
    /// quantizing from f32 via `f16::from_f32`.
    Float16(ArrayD<f16>),
    /// IEEE bfloat16 — Float16-alternative with 8-bit exponent (FP32-compatible
    /// range) and 7-bit mantissa. Common in transformer training and inference
    /// (Google TPU origin; A100/H100 native). Bit-compatible with CUDA
    /// `__nv_bfloat16`. Added in WS-3.5.
    BFloat16(ArrayD<bf16>),
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

    /// Create Float16 tensor from flat vector and shape.
    ///
    /// Added in WS-3 M3.1 for FP16 model support (Whisper-Tiny encoder
    /// FP16, BERT-FP16). `f16` is from the `half` crate; round-half-to-
    /// even quantization happens at `f16::from_f32` call sites.
    pub fn from_vec_f16(data: Vec<f16>, shape: Vec<usize>) -> Self {
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

        Tensor::Float16(array)
    }

    /// Create BFloat16 tensor from flat vector and shape.
    ///
    /// Added in WS-3.5 for BF16 model support (BERT-base BF16, Whisper-Tiny
    /// BF16). `bf16` is from the `half` crate; round-half-to-even quantization
    /// happens at `bf16::from_f32` call sites. Bit-compatible with CUDA
    /// `__nv_bfloat16` for zero-copy GPU upload.
    pub fn from_vec_bf16(data: Vec<bf16>, shape: Vec<usize>) -> Self {
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

        Tensor::BFloat16(array)
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

    /// Create tensor from existing Float16 ndarray
    pub fn from_array_f16(array: ArrayD<f16>) -> Self {
        Tensor::Float16(array)
    }

    /// Create tensor from existing BFloat16 ndarray
    pub fn from_array_bf16(array: ArrayD<bf16>) -> Self {
        Tensor::BFloat16(array)
    }

    // ========== Type introspection ==========

    /// Get data type as string
    pub fn dtype(&self) -> &'static str {
        match self {
            Tensor::Float32(_) => "float32",
            Tensor::Float64(_) => "float64",
            Tensor::Float16(_) => "float16",
            Tensor::BFloat16(_) => "bfloat16",
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

    /// Check if tensor is Float64
    pub fn is_float64(&self) -> bool {
        matches!(self, Tensor::Float64(_))
    }

    /// Check if tensor is Float16
    pub fn is_float16(&self) -> bool {
        matches!(self, Tensor::Float16(_))
    }

    /// Check if tensor is BFloat16
    pub fn is_bfloat16(&self) -> bool {
        matches!(self, Tensor::BFloat16(_))
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

    /// Check if tensor is Bool
    pub fn is_bool(&self) -> bool {
        matches!(self, Tensor::Bool(_))
    }

    // ========== Shape operations ==========

    /// Get tensor shape
    pub fn shape(&self) -> &[usize] {
        match self {
            Tensor::Float32(a) => a.shape(),
            Tensor::Float64(a) => a.shape(),
            Tensor::Float16(a) => a.shape(),
            Tensor::BFloat16(a) => a.shape(),
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
            Tensor::Float16(a) => a.ndim(),
            Tensor::BFloat16(a) => a.ndim(),
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
            Tensor::Float16(a) => a.len(),
            Tensor::BFloat16(a) => a.len(),
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

    /// Get data as flat Float16 slice (panics if wrong type)
    pub fn as_slice_f16(&self) -> Vec<f16> {
        match self {
            Tensor::Float16(a) => {
                if let Some(slice) = a.as_slice() {
                    slice.to_vec()
                } else {
                    a.iter().copied().collect()
                }
            }
            _ => panic!(
                "as_slice_f16() called on non-Float16 tensor ({})",
                self.dtype()
            ),
        }
    }

    /// Get reference to Float16 array (panics if wrong type)
    pub fn to_array_f16(&self) -> &ArrayD<f16> {
        match self {
            Tensor::Float16(a) => a,
            _ => panic!(
                "to_array_f16() called on non-Float16 tensor ({})",
                self.dtype()
            ),
        }
    }

    /// Get data as flat BFloat16 slice (panics if wrong type)
    ///
    /// Added in WS-3.5 for BF16 model support. BF16 and FP16 are NOT
    /// interchangeable formats; calling `as_slice_f16()` on a BFloat16
    /// tensor panics, and vice versa.
    pub fn as_slice_bf16(&self) -> Vec<bf16> {
        match self {
            Tensor::BFloat16(a) => {
                if let Some(slice) = a.as_slice() {
                    slice.to_vec()
                } else {
                    a.iter().copied().collect()
                }
            }
            _ => panic!(
                "as_slice_bf16() called on non-BFloat16 tensor ({})",
                self.dtype()
            ),
        }
    }

    /// Get reference to BFloat16 array (panics if wrong type)
    pub fn to_array_bf16(&self) -> &ArrayD<bf16> {
        match self {
            Tensor::BFloat16(a) => a,
            _ => panic!(
                "to_array_bf16() called on non-BFloat16 tensor ({})",
                self.dtype()
            ),
        }
    }

    /// Get data as flat Bool slice (panics if wrong type)
    ///
    /// Added in WS-3 M3.1's Bool surface audit. Closes the L2.1/L3.5
    /// asymmetry: Bool match arms existed throughout but the dedicated
    /// accessor was missing, forcing callers to either pattern-match the
    /// Tensor variant directly or convert through a different dtype.
    pub fn as_slice_bool(&self) -> Vec<bool> {
        match self {
            Tensor::Bool(a) => {
                if let Some(slice) = a.as_slice() {
                    slice.to_vec()
                } else {
                    a.iter().copied().collect()
                }
            }
            _ => panic!(
                "as_slice_bool() called on non-Bool tensor ({})",
                self.dtype()
            ),
        }
    }

    /// Get reference to Bool array (panics if wrong type)
    pub fn to_array_bool(&self) -> &ArrayD<bool> {
        match self {
            Tensor::Bool(a) => a,
            _ => panic!(
                "to_array_bool() called on non-Bool tensor ({})",
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
            Tensor::Float16(a) => {
                let data: Vec<f32> = a.iter().map(|x| x.to_f32()).collect();
                Tensor::from_vec_f32(data, a.shape().to_vec())
            }
            Tensor::BFloat16(a) => {
                let data: Vec<f32> = a.iter().map(|x| x.to_f32()).collect();
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
            Tensor::Float16(a) => {
                let data: Vec<i64> = a.iter().map(|x| x.to_f32() as i64).collect();
                Tensor::from_vec_i64(data, a.shape().to_vec())
            }
            Tensor::BFloat16(a) => {
                let data: Vec<i64> = a.iter().map(|x| x.to_f32() as i64).collect();
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
            Tensor::Float16(arr) => {
                // Convert to f32, operate, convert back to f16. Float16
                // ops in iconnx today funnel through f32 for shape-class
                // operations (Concat / Reshape / Transpose / etc.) and
                // through dedicated FP16 NVRTC kernels for math (M3.4);
                // this CPU path is the constant-folding fallback.
                let as_f32 = arr.mapv(|x| x.to_f32());
                let result_f32 = f32_op(&as_f32);
                Tensor::Float16(result_f32.mapv(f16::from_f32))
            }
            Tensor::BFloat16(arr) => {
                // Convert to f32, operate, convert back to bf16. BFloat16
                // ops follow the FP16 pattern — CPU-side shape-class ops
                // (Concat / Reshape / Transpose / etc.) funnel through f32;
                // GPU dispatch lives in dedicated BF16 NVRTC kernels (Y(2)).
                // This CPU path is the constant-folding fallback.
                let as_f32 = arr.mapv(|x| x.to_f32());
                let result_f32 = f32_op(&as_f32);
                Tensor::BFloat16(result_f32.mapv(bf16::from_f32))
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

// ===== WS-5 helpers =====
//
// `dtype_for_tensor` and `tensor_byte_size` give the WS-5 `validate` module
// a uniform way to walk a `HashMap<String, Tensor>` from
// `OnnxModel::extract_weights()` without re-implementing the variant match
// at every call site. Defined here (next to the `Tensor` enum) so any
// future variant added to `Tensor` is a compile error if the helpers
// aren't updated.

/// Return the [`DType`] tag corresponding to a `Tensor` variant.
#[cfg(feature = "cuda")]
pub fn dtype_for_tensor(t: &Tensor) -> DType {
    match t {
        Tensor::Float32(_) => DType::Float32,
        Tensor::Float16(_) => DType::Float16,
        Tensor::BFloat16(_) => DType::BFloat16,
        Tensor::Int32(_) => DType::Int32,
        Tensor::Int64(_) => DType::Int64,
        Tensor::Int8(_) => DType::Int8,
        Tensor::UInt8(_) => DType::UInt8,
        Tensor::Bool(_) => DType::Bool,
        // FLOAT64 has no `DType` tag in the GPU-side enum (the GPU path
        // doesn't accept f64 today). For capability discovery purposes,
        // treat f64 as Float32 — `validate_model` reports it via
        // `used_dtypes` only and never dispatches kernels.
        Tensor::Float64(_) => DType::Float32,
    }
}

/// Total byte size of the `Tensor`'s underlying storage (element count ×
/// element-byte-size).
pub fn tensor_byte_size(t: &Tensor) -> usize {
    let element_size = match t {
        Tensor::Float32(_) | Tensor::Int32(_) => 4,
        Tensor::Float16(_) | Tensor::BFloat16(_) => 2,
        Tensor::Int64(_) | Tensor::Float64(_) => 8,
        Tensor::Int8(_) | Tensor::UInt8(_) | Tensor::Bool(_) => 1,
    };
    element_size * tensor_element_count(t)
}

fn tensor_element_count(t: &Tensor) -> usize {
    match t {
        Tensor::Float32(a) => a.len(),
        Tensor::Float16(a) => a.len(),
        Tensor::BFloat16(a) => a.len(),
        Tensor::Int32(a) => a.len(),
        Tensor::Int64(a) => a.len(),
        Tensor::Int8(a) => a.len(),
        Tensor::UInt8(a) => a.len(),
        Tensor::Bool(a) => a.len(),
        Tensor::Float64(a) => a.len(),
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

    // ========== WS-3 M3.1: Float16 + Bool surface audit ==========

    #[test]
    fn tensor_float16_constructor_and_accessors() {
        use half::f16;
        let v = vec![f16::from_f32(1.5), f16::from_f32(-2.0), f16::from_f32(0.0)];
        let t = Tensor::from_vec_f16(v.clone(), vec![3]);
        assert_eq!(t.dtype(), "float16");
        assert!(t.is_float16());
        assert_eq!(t.shape(), &[3]);
        assert_eq!(t.len(), 3);
        assert_eq!(t.as_slice_f16(), v);
    }

    #[test]
    fn tensor_float16_round_half_to_even_parity() {
        use half::f16;
        // 1.0 + 2^-12 (~1.000244) is below the f16 ULP at 1.0 (which is 2^-10);
        // f16 quantizes it back to exactly 1.0, exercising IEEE-754 round-to-
        // nearest-even on a tied f16 boundary.
        let v = vec![f16::from_f32(1.0_f32 + (1.0_f32 / 4096.0))];
        let t = Tensor::from_vec_f16(v, vec![1]);
        let back: f32 = t.as_slice_f16()[0].to_f32();
        assert!(
            (back - 1.0).abs() < 1e-6,
            "expected even round to 1.0, got {back}"
        );
    }

    #[test]
    fn tensor_float16_panic_on_wrong_accessor() {
        use half::f16;
        let t = Tensor::from_vec_f16(vec![f16::from_f32(0.0)], vec![1]);
        let result = std::panic::catch_unwind(|| t.as_slice());
        assert!(
            result.is_err(),
            "f32-strict accessor on Float16 must panic"
        );
    }

    #[test]
    fn tensor_float16_to_array_roundtrip() {
        use half::f16;
        let v = vec![f16::from_f32(0.5), f16::from_f32(-1.5), f16::from_f32(2.0)];
        let t = Tensor::from_vec_f16(v, vec![3]);
        let arr = t.to_array_f16();
        assert_eq!(arr.shape(), &[3]);
        assert_eq!(arr[[0]].to_f32(), 0.5);
        assert_eq!(arr[[1]].to_f32(), -1.5);
    }

    /// L2.1 / L3.5 audit closure: confirm Bool has the same accessor
    /// surface as the other dtype variants. Pre-WS-3 the Bool path lacked
    /// `is_bool`, `as_slice_bool`, `to_array_bool` — only the match-arm
    /// sites (`dtype`, `shape`, `len`, `to_float32`, `to_int64`,
    /// `map_preserving_dtype`) had Bool coverage. M3.3's GPU upload path
    /// needs the dedicated accessors.
    #[test]
    fn tensor_bool_accessor_symmetry() {
        let t = Tensor::from_vec_bool(vec![true, false, true], vec![3]);
        assert_eq!(t.dtype(), "bool");
        assert!(t.is_bool());
        assert_eq!(t.shape(), &[3]);
        assert_eq!(t.len(), 3);
        assert_eq!(t.as_slice_bool(), vec![true, false, true]);
        let arr = t.to_array_bool();
        assert_eq!(arr.shape(), &[3]);
        assert!(arr[[0]]);
        assert!(!arr[[1]]);
    }

    /// Float64 also lacked `is_float64`. Add and verify alongside the
    /// Float16 / Bool work — small symmetry win on the tensor introspection
    /// surface.
    #[test]
    fn tensor_float64_is_float64() {
        let t = Tensor::from_vec_f64(vec![1.0, 2.0], vec![2]);
        assert_eq!(t.dtype(), "float64");
        assert!(t.is_float64());
        assert!(!t.is_float32());
    }

    // ========== WS-3.5: BFloat16 surface audit ==========

    #[test]
    fn tensor_bfloat16_constructor_and_accessors() {
        use half::bf16;
        let v = vec![bf16::from_f32(1.5), bf16::from_f32(-2.0), bf16::from_f32(0.0)];
        let t = Tensor::from_vec_bf16(v.clone(), vec![3]);
        assert_eq!(t.dtype(), "bfloat16");
        assert!(t.is_bfloat16());
        assert_eq!(t.shape(), &[3]);
        assert_eq!(t.len(), 3);
        assert_eq!(t.as_slice_bf16(), v);
    }

    #[test]
    fn tensor_bfloat16_round_half_to_even_parity() {
        use half::bf16;
        // 1.0 + 1/256 = 1.00390625 sits exactly halfway between BF16's 1.0
        // (mantissa LSB = 0, even) and 1.0 + 2^-7 = 1.0078125 (mantissa LSB
        // = 1, odd). Round-half-to-even resolves to the even-LSB value, so
        // bf16::from_f32 must return 1.0 — not 1.0078125, and certainly
        // never 1.003_906_3 (which isn't representable in BF16).
        let v = vec![bf16::from_f32(1.0_f32 + (1.0_f32 / 256.0))];
        let t = Tensor::from_vec_bf16(v.clone(), vec![1]);
        let back: f32 = t.as_slice_bf16()[0].to_f32();
        assert!(
            (back - 1.0).abs() < 1e-9,
            "round-half-to-even should resolve 1.00390625 to 1.0 (even LSB), got {back}"
        );
    }

    #[test]
    fn tensor_bfloat16_panic_on_wrong_accessor() {
        use half::bf16;
        let t = Tensor::from_vec_bf16(vec![bf16::from_f32(0.0)], vec![1]);
        let result = std::panic::catch_unwind(|| t.as_slice());
        assert!(result.is_err(), "f32-strict accessor on BFloat16 must panic");
    }

    #[test]
    fn tensor_bfloat16_panic_on_f16_accessor() {
        use half::bf16;
        let t = Tensor::from_vec_bf16(vec![bf16::from_f32(0.0)], vec![1]);
        let result = std::panic::catch_unwind(|| t.as_slice_f16());
        assert!(
            result.is_err(),
            "f16 accessor on BFloat16 must panic — formats are not interchangeable"
        );
    }
}
