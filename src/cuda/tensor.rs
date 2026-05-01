//! GPU Tensor type for Iconnx.
//!
//! Multi-dtype support for ONNX compatibility on GPU. Mirrors the CPU
//! Tensor enum design.
//!
//! Storage is a garboard `DeviceSlice<'static, T>` wrapped in `Arc` for
//! cheap clone/reshape operations without GPU memory copies. The
//! `'static` lifetime matches the leaked garboard `Device` owned by
//! [`IconnxCudaContext`].

use std::sync::Arc;

use garboard::DeviceSlice;

use super::context::{CudaError, IconnxCudaContext};

/// Data type enum matching ONNX types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DType {
    Float32,
    Float16,
    /// IEEE bfloat16 — 8-bit exponent (FP32-compatible range), 7-bit
    /// mantissa. Bit-compatible with CUDA `__nv_bfloat16`. Native
    /// compute requires sm_80+ (Ampere); see
    /// `GpuTensor::check_bf16_hardware`. Added in WS-3.5.
    BFloat16,
    Int64,
    Int32,
    Int8,
    UInt8,
    Bool,
}

impl DType {
    /// Size of one element in bytes.
    pub fn size_bytes(&self) -> usize {
        match self {
            DType::Float32 => 4,
            DType::Float16 => 2,
            DType::BFloat16 => 2,
            DType::Int64 => 8,
            DType::Int32 => 4,
            DType::Int8 => 1,
            DType::UInt8 => 1,
            DType::Bool => 1,
        }
    }

    /// Name as string (matches CPU Tensor).
    pub fn name(&self) -> &'static str {
        match self {
            DType::Float32 => "float32",
            DType::Float16 => "float16",
            DType::BFloat16 => "bfloat16",
            DType::Int64 => "int64",
            DType::Int32 => "int32",
            DType::Int8 => "int8",
            DType::UInt8 => "uint8",
            DType::Bool => "bool",
        }
    }
}

/// Typed slice enum for memory pool extraction.
///
/// Used when extracting the inner `DeviceSlice` from a `GpuTensor`
/// for returning to the memory pool.
pub enum TypedSlice {
    Float32(DeviceSlice<'static, f32>),
    Float16(DeviceSlice<'static, half::f16>),
    /// IEEE bfloat16 — element type is `half::bf16`. Added in WS-3.5.
    BFloat16(DeviceSlice<'static, half::bf16>),
    Int64(DeviceSlice<'static, i64>),
    Int32(DeviceSlice<'static, i32>),
    Int8(DeviceSlice<'static, i8>),
    UInt8(DeviceSlice<'static, u8>),
    Bool(DeviceSlice<'static, u8>),
}

/// GPU-resident tensor with multiple data type support.
///
/// Each variant holds GPU memory and shape information for its element
/// type. The underlying `DeviceSlice` is wrapped in `Arc` for cheap
/// clone/reshape operations without GPU memory copies.
#[derive(Clone)]
pub enum GpuTensor {
    Float32 {
        data: Arc<DeviceSlice<'static, f32>>,
        shape: Vec<usize>,
    },
    /// IEEE binary16 — element type is `half::f16`. Added in WS-3 M3.3
    /// for FP16 model support (Whisper-Tiny encoder FP16, BERT-FP16).
    Float16 {
        data: Arc<DeviceSlice<'static, half::f16>>,
        shape: Vec<usize>,
    },
    /// IEEE bfloat16 — element type is `half::bf16`. Added in WS-3.5
    /// for BF16 model support (BERT-base BF16, Whisper-Tiny BF16). Native
    /// compute requires sm_80+ (Ampere); checked at construction in
    /// `zeros_bf16` / `from_host_bf16`. Bit-compatible with CUDA
    /// `__nv_bfloat16` for kernel pointer-cast.
    BFloat16 {
        data: Arc<DeviceSlice<'static, half::bf16>>,
        shape: Vec<usize>,
    },
    Int64 {
        data: Arc<DeviceSlice<'static, i64>>,
        shape: Vec<usize>,
    },
    Int32 {
        data: Arc<DeviceSlice<'static, i32>>,
        shape: Vec<usize>,
    },
    Int8 {
        data: Arc<DeviceSlice<'static, i8>>,
        shape: Vec<usize>,
    },
    UInt8 {
        data: Arc<DeviceSlice<'static, u8>>,
        shape: Vec<usize>,
    },
    /// Bool — CUDA has no native bool type; conventional choice is one
    /// byte per element (`u8`) where 0 = false, nonzero = true. Host
    /// `&[bool]` is converted at upload via `b as u8`. WS-3 M3.3
    /// closes the L2.1/L3.5 audit: pre-WS-3 the `Tensor::Bool` upload
    /// path cast to `f32` to fit a single supported GPU dtype.
    Bool {
        data: Arc<DeviceSlice<'static, u8>>,
        shape: Vec<usize>,
    },
}

impl GpuTensor {
    // ==================== Type-agnostic methods ====================

    /// Get the tensor shape.
    pub fn shape(&self) -> &[usize] {
        match self {
            GpuTensor::Float32 { shape, .. } => shape,
            GpuTensor::Float16 { shape, .. } => shape,
            GpuTensor::BFloat16 { shape, .. } => shape,
            GpuTensor::Int64 { shape, .. } => shape,
            GpuTensor::Int32 { shape, .. } => shape,
            GpuTensor::Int8 { shape, .. } => shape,
            GpuTensor::UInt8 { shape, .. } => shape,
            GpuTensor::Bool { shape, .. } => shape,
        }
    }

    /// Get the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape().len()
    }

    /// Get the total number of elements.
    pub fn len(&self) -> usize {
        self.shape().iter().product()
    }

    /// Check if tensor is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the data type.
    pub fn dtype(&self) -> DType {
        match self {
            GpuTensor::Float32 { .. } => DType::Float32,
            GpuTensor::Float16 { .. } => DType::Float16,
            GpuTensor::BFloat16 { .. } => DType::BFloat16,
            GpuTensor::Int64 { .. } => DType::Int64,
            GpuTensor::Int32 { .. } => DType::Int32,
            GpuTensor::Int8 { .. } => DType::Int8,
            GpuTensor::UInt8 { .. } => DType::UInt8,
            GpuTensor::Bool { .. } => DType::Bool,
        }
    }

    /// Get memory size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.len() * self.dtype().size_bytes()
    }

    /// Get the raw CUDA device pointer (for cache keying and direct GPU
    /// operations). Returns the `u64` `CUdeviceptr` value straight from the
    /// underlying garboard `DeviceSlice`. The `ctx` argument is accepted
    /// for API parity with older call sites that required a stream handle
    /// for read/write tracking; garboard needs no stream here.
    pub fn device_ptr(&self, _ctx: &IconnxCudaContext) -> u64 {
        match self {
            GpuTensor::Float32 { data, .. } => data.as_raw_device_ptr(),
            GpuTensor::Float16 { data, .. } => data.as_raw_device_ptr(),
            GpuTensor::BFloat16 { data, .. } => data.as_raw_device_ptr(),
            GpuTensor::Int64 { data, .. } => data.as_raw_device_ptr(),
            GpuTensor::Int32 { data, .. } => data.as_raw_device_ptr(),
            GpuTensor::Int8 { data, .. } => data.as_raw_device_ptr(),
            GpuTensor::UInt8 { data, .. } => data.as_raw_device_ptr(),
            GpuTensor::Bool { data, .. } => data.as_raw_device_ptr(),
        }
    }

    // ==================== Float32 methods ====================

    /// Create a new Float32 GPU tensor from shape and GPU data.
    pub fn new_f32(data: DeviceSlice<'static, f32>, shape: Vec<usize>) -> Self {
        GpuTensor::Float32 {
            data: Arc::new(data),
            shape,
        }
    }

    /// Create a Float32 GPU tensor from host data.
    pub fn from_host_f32(
        ctx: &IconnxCudaContext,
        data: &[f32],
        shape: Vec<usize>,
    ) -> Result<Self, CudaError> {
        let expected_len: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected_len,
            "Data length {} doesn't match shape {:?} (expected {})",
            data.len(),
            shape,
            expected_len
        );

        let gpu_data = ctx.htod(data)?;
        Ok(GpuTensor::Float32 {
            data: Arc::new(gpu_data),
            shape,
        })
    }

    /// Create a zeroed Float32 GPU tensor with given shape.
    pub fn zeros_f32(ctx: &IconnxCudaContext, shape: Vec<usize>) -> Result<Self, CudaError> {
        let len: usize = shape.iter().product();
        let gpu_data = ctx.alloc_zeros::<f32>(len)?;
        Ok(GpuTensor::Float32 {
            data: Arc::new(gpu_data),
            shape,
        })
    }

    /// Copy Float32 tensor data back to host.
    ///
    /// Note: Returns only the elements defined by the tensor's shape,
    /// even if the underlying slice has more capacity (due to memory
    /// pool sizing).
    pub fn to_host_f32(&self, ctx: &IconnxCudaContext) -> Result<Vec<f32>, CudaError> {
        match self {
            GpuTensor::Float32 { data, shape } => {
                let mut full = ctx.dtoh(data)?;
                let len: usize = shape.iter().product();
                full.truncate(len);
                Ok(full)
            }
            _ => Err(CudaError::Kernel(format!(
                "to_host_f32 called on {} tensor",
                self.dtype().name()
            ))),
        }
    }

    /// Get a reference to the underlying Float32 `DeviceSlice`.
    pub fn data_f32(&self) -> Result<&DeviceSlice<'static, f32>, CudaError> {
        match self {
            GpuTensor::Float32 { data, .. } => Ok(data),
            _ => Err(CudaError::Kernel(format!(
                "data_f32 called on {} tensor",
                self.dtype().name()
            ))),
        }
    }

    /// Get a mutable reference to the underlying Float32 `DeviceSlice`.
    ///
    /// Fails if the `Arc` has multiple references: garboard's
    /// `DeviceSlice` does not implement `Clone`, so a copy-on-write
    /// fallback is not possible here. Shared-`Arc` mutation would have
    /// been a bug anyway (two aliases would see different post-mutation
    /// values).
    pub fn data_f32_mut(&mut self) -> Result<&mut DeviceSlice<'static, f32>, CudaError> {
        match self {
            GpuTensor::Float32 { data, .. } => Arc::get_mut(data).ok_or_else(|| {
                CudaError::Kernel(
                    "data_f32_mut: tensor is shared (Arc refcount > 1); \
                     cannot obtain mutable access without aliasing"
                        .to_string(),
                )
            }),
            _ => Err(CudaError::Kernel(format!(
                "data_f32_mut called on {} tensor",
                self.dtype().name()
            ))),
        }
    }

    /// Check if this is a Float32 tensor.
    pub fn is_f32(&self) -> bool {
        matches!(self, GpuTensor::Float32 { .. })
    }

    // ==================== Int64 methods ====================

    /// Create a new Int64 GPU tensor from shape and GPU data.
    pub fn new_i64(data: DeviceSlice<'static, i64>, shape: Vec<usize>) -> Self {
        GpuTensor::Int64 {
            data: Arc::new(data),
            shape,
        }
    }

    /// Create an Int64 GPU tensor from host data.
    pub fn from_host_i64(
        ctx: &IconnxCudaContext,
        data: &[i64],
        shape: Vec<usize>,
    ) -> Result<Self, CudaError> {
        let expected_len: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected_len,
            "Data length {} doesn't match shape {:?} (expected {})",
            data.len(),
            shape,
            expected_len
        );

        let gpu_data = ctx.htod_i64(data)?;
        Ok(GpuTensor::Int64 {
            data: Arc::new(gpu_data),
            shape,
        })
    }

    /// Create a zeroed Int64 GPU tensor with given shape.
    pub fn zeros_i64(ctx: &IconnxCudaContext, shape: Vec<usize>) -> Result<Self, CudaError> {
        let len: usize = shape.iter().product();
        let gpu_data = ctx.alloc_zeros::<i64>(len)?;
        Ok(GpuTensor::Int64 {
            data: Arc::new(gpu_data),
            shape,
        })
    }

    /// Copy Int64 tensor data back to host.
    ///
    /// Note: Returns only the elements defined by the tensor's shape,
    /// even if the underlying slice has more capacity (due to memory
    /// pool sizing).
    pub fn to_host_i64(&self, ctx: &IconnxCudaContext) -> Result<Vec<i64>, CudaError> {
        match self {
            GpuTensor::Int64 { data, shape } => {
                let mut full = ctx.dtoh_i64(data)?;
                let len: usize = shape.iter().product();
                full.truncate(len);
                Ok(full)
            }
            _ => Err(CudaError::Kernel(format!(
                "to_host_i64 called on {} tensor",
                self.dtype().name()
            ))),
        }
    }

    /// Get a reference to the underlying Int64 `DeviceSlice`.
    pub fn data_i64(&self) -> Result<&DeviceSlice<'static, i64>, CudaError> {
        match self {
            GpuTensor::Int64 { data, .. } => Ok(data),
            _ => Err(CudaError::Kernel(format!(
                "data_i64 called on {} tensor",
                self.dtype().name()
            ))),
        }
    }

    /// Get a mutable reference to the underlying Int64 `DeviceSlice`.
    /// See [`GpuTensor::data_f32_mut`] for the shared-Arc note.
    pub fn data_i64_mut(&mut self) -> Result<&mut DeviceSlice<'static, i64>, CudaError> {
        match self {
            GpuTensor::Int64 { data, .. } => Arc::get_mut(data).ok_or_else(|| {
                CudaError::Kernel(
                    "data_i64_mut: tensor is shared (Arc refcount > 1); \
                     cannot obtain mutable access without aliasing"
                        .to_string(),
                )
            }),
            _ => Err(CudaError::Kernel(format!(
                "data_i64_mut called on {} tensor",
                self.dtype().name()
            ))),
        }
    }

    /// Check if this is an Int64 tensor.
    pub fn is_i64(&self) -> bool {
        matches!(self, GpuTensor::Int64 { .. })
    }

    // ==================== Int32 methods ====================

    /// Create a new Int32 GPU tensor from shape and GPU data.
    pub fn new_i32(data: DeviceSlice<'static, i32>, shape: Vec<usize>) -> Self {
        GpuTensor::Int32 {
            data: Arc::new(data),
            shape,
        }
    }

    /// Create an Int32 GPU tensor from host data.
    pub fn from_host_i32(
        ctx: &IconnxCudaContext,
        data: &[i32],
        shape: Vec<usize>,
    ) -> Result<Self, CudaError> {
        let expected_len: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected_len,
            "Data length {} doesn't match shape {:?} (expected {})",
            data.len(),
            shape,
            expected_len
        );

        let gpu_data = ctx.htod_i32(data)?;
        Ok(GpuTensor::Int32 {
            data: Arc::new(gpu_data),
            shape,
        })
    }

    /// Create a zeroed Int32 GPU tensor with given shape.
    pub fn zeros_i32(ctx: &IconnxCudaContext, shape: Vec<usize>) -> Result<Self, CudaError> {
        let len: usize = shape.iter().product();
        let gpu_data = ctx.alloc_zeros::<i32>(len)?;
        Ok(GpuTensor::Int32 {
            data: Arc::new(gpu_data),
            shape,
        })
    }

    /// Copy Int32 tensor data back to host.
    ///
    /// Note: Returns only the elements defined by the tensor's shape,
    /// even if the underlying slice has more capacity (due to memory
    /// pool sizing).
    pub fn to_host_i32(&self, ctx: &IconnxCudaContext) -> Result<Vec<i32>, CudaError> {
        match self {
            GpuTensor::Int32 { data, shape } => {
                let mut full = ctx.dtoh_i32(data)?;
                let len: usize = shape.iter().product();
                full.truncate(len);
                Ok(full)
            }
            _ => Err(CudaError::Kernel(format!(
                "to_host_i32 called on {} tensor",
                self.dtype().name()
            ))),
        }
    }

    /// Get a reference to the underlying Int32 `DeviceSlice`.
    pub fn data_i32(&self) -> Result<&DeviceSlice<'static, i32>, CudaError> {
        match self {
            GpuTensor::Int32 { data, .. } => Ok(data),
            _ => Err(CudaError::Kernel(format!(
                "data_i32 called on {} tensor",
                self.dtype().name()
            ))),
        }
    }

    /// Get a mutable reference to the underlying Int32 `DeviceSlice`.
    /// See [`GpuTensor::data_f32_mut`] for the shared-Arc note.
    pub fn data_i32_mut(&mut self) -> Result<&mut DeviceSlice<'static, i32>, CudaError> {
        match self {
            GpuTensor::Int32 { data, .. } => Arc::get_mut(data).ok_or_else(|| {
                CudaError::Kernel(
                    "data_i32_mut: tensor is shared (Arc refcount > 1); \
                     cannot obtain mutable access without aliasing"
                        .to_string(),
                )
            }),
            _ => Err(CudaError::Kernel(format!(
                "data_i32_mut called on {} tensor",
                self.dtype().name()
            ))),
        }
    }

    /// Check if this is an Int32 tensor.
    pub fn is_i32(&self) -> bool {
        matches!(self, GpuTensor::Int32 { .. })
    }

    // ==================== Int8 methods ====================

    /// Create a new Int8 GPU tensor from shape and GPU data.
    pub fn new_i8(data: DeviceSlice<'static, i8>, shape: Vec<usize>) -> Self {
        GpuTensor::Int8 {
            data: Arc::new(data),
            shape,
        }
    }

    /// Create an Int8 GPU tensor from host data.
    pub fn from_host_i8(
        ctx: &IconnxCudaContext,
        data: &[i8],
        shape: Vec<usize>,
    ) -> Result<Self, CudaError> {
        let expected_len: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected_len,
            "Data length {} doesn't match shape {:?} (expected {})",
            data.len(),
            shape,
            expected_len
        );

        let gpu_data = ctx.htod(data)?;
        Ok(GpuTensor::Int8 {
            data: Arc::new(gpu_data),
            shape,
        })
    }

    /// Create a zeroed Int8 GPU tensor with given shape.
    pub fn zeros_i8(ctx: &IconnxCudaContext, shape: Vec<usize>) -> Result<Self, CudaError> {
        let len: usize = shape.iter().product();
        let gpu_data = ctx.alloc_zeros::<i8>(len)?;
        Ok(GpuTensor::Int8 {
            data: Arc::new(gpu_data),
            shape,
        })
    }

    /// Copy Int8 tensor data back to host.
    ///
    /// Note: Returns only the elements defined by the tensor's shape,
    /// even if the underlying slice has more capacity (due to memory
    /// pool sizing).
    pub fn to_host_i8(&self, ctx: &IconnxCudaContext) -> Result<Vec<i8>, CudaError> {
        match self {
            GpuTensor::Int8 { data, shape } => {
                let mut full = ctx.dtoh(data)?;
                let len: usize = shape.iter().product();
                full.truncate(len);
                Ok(full)
            }
            _ => Err(CudaError::Kernel(format!(
                "to_host_i8 called on {} tensor",
                self.dtype().name()
            ))),
        }
    }

    /// Get a reference to the underlying Int8 `DeviceSlice`.
    pub fn data_i8(&self) -> Result<&DeviceSlice<'static, i8>, CudaError> {
        match self {
            GpuTensor::Int8 { data, .. } => Ok(data),
            _ => Err(CudaError::Kernel(format!(
                "data_i8 called on {} tensor",
                self.dtype().name()
            ))),
        }
    }

    /// Get a mutable reference to the underlying Int8 `DeviceSlice`.
    /// See [`GpuTensor::data_f32_mut`] for the shared-Arc note.
    pub fn data_i8_mut(&mut self) -> Result<&mut DeviceSlice<'static, i8>, CudaError> {
        match self {
            GpuTensor::Int8 { data, .. } => Arc::get_mut(data).ok_or_else(|| {
                CudaError::Kernel(
                    "data_i8_mut: tensor is shared (Arc refcount > 1); \
                     cannot obtain mutable access without aliasing"
                        .to_string(),
                )
            }),
            _ => Err(CudaError::Kernel(format!(
                "data_i8_mut called on {} tensor",
                self.dtype().name()
            ))),
        }
    }

    /// Check if this is an Int8 tensor.
    pub fn is_i8(&self) -> bool {
        matches!(self, GpuTensor::Int8 { .. })
    }

    // ==================== UInt8 methods ====================

    /// Create a new UInt8 GPU tensor from shape and GPU data.
    pub fn new_u8(data: DeviceSlice<'static, u8>, shape: Vec<usize>) -> Self {
        GpuTensor::UInt8 {
            data: Arc::new(data),
            shape,
        }
    }

    /// Create a UInt8 GPU tensor from host data.
    pub fn from_host_u8(
        ctx: &IconnxCudaContext,
        data: &[u8],
        shape: Vec<usize>,
    ) -> Result<Self, CudaError> {
        let expected_len: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected_len,
            "Data length {} doesn't match shape {:?} (expected {})",
            data.len(),
            shape,
            expected_len
        );

        let gpu_data = ctx.htod(data)?;
        Ok(GpuTensor::UInt8 {
            data: Arc::new(gpu_data),
            shape,
        })
    }

    /// Create a zeroed UInt8 GPU tensor with given shape.
    pub fn zeros_u8(ctx: &IconnxCudaContext, shape: Vec<usize>) -> Result<Self, CudaError> {
        let len: usize = shape.iter().product();
        let gpu_data = ctx.alloc_zeros::<u8>(len)?;
        Ok(GpuTensor::UInt8 {
            data: Arc::new(gpu_data),
            shape,
        })
    }

    /// Copy UInt8 tensor data back to host.
    ///
    /// Note: Returns only the elements defined by the tensor's shape,
    /// even if the underlying slice has more capacity (due to memory
    /// pool sizing).
    pub fn to_host_u8(&self, ctx: &IconnxCudaContext) -> Result<Vec<u8>, CudaError> {
        match self {
            GpuTensor::UInt8 { data, shape } => {
                let mut full = ctx.dtoh(data)?;
                let len: usize = shape.iter().product();
                full.truncate(len);
                Ok(full)
            }
            _ => Err(CudaError::Kernel(format!(
                "to_host_u8 called on {} tensor",
                self.dtype().name()
            ))),
        }
    }

    /// Get a reference to the underlying UInt8 `DeviceSlice`.
    pub fn data_u8(&self) -> Result<&DeviceSlice<'static, u8>, CudaError> {
        match self {
            GpuTensor::UInt8 { data, .. } => Ok(data),
            _ => Err(CudaError::Kernel(format!(
                "data_u8 called on {} tensor",
                self.dtype().name()
            ))),
        }
    }

    /// Get a mutable reference to the underlying UInt8 `DeviceSlice`.
    /// See [`GpuTensor::data_f32_mut`] for the shared-Arc note.
    pub fn data_u8_mut(&mut self) -> Result<&mut DeviceSlice<'static, u8>, CudaError> {
        match self {
            GpuTensor::UInt8 { data, .. } => Arc::get_mut(data).ok_or_else(|| {
                CudaError::Kernel(
                    "data_u8_mut: tensor is shared (Arc refcount > 1); \
                     cannot obtain mutable access without aliasing"
                        .to_string(),
                )
            }),
            _ => Err(CudaError::Kernel(format!(
                "data_u8_mut called on {} tensor",
                self.dtype().name()
            ))),
        }
    }

    /// Check if this is a UInt8 tensor.
    pub fn is_u8(&self) -> bool {
        matches!(self, GpuTensor::UInt8 { .. })
    }

    // ==================== Float16 methods ====================

    /// Create a new Float16 GPU tensor from shape and GPU data.
    pub fn new_f16(data: DeviceSlice<'static, half::f16>, shape: Vec<usize>) -> Self {
        GpuTensor::Float16 {
            data: Arc::new(data),
            shape,
        }
    }

    /// Create a Float16 GPU tensor from host data.
    pub fn from_host_f16(
        ctx: &IconnxCudaContext,
        data: &[half::f16],
        shape: Vec<usize>,
    ) -> Result<Self, CudaError> {
        let expected_len: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected_len,
            "Data length {} doesn't match shape {:?} (expected {})",
            data.len(),
            shape,
            expected_len
        );

        let gpu_data = ctx.htod(data)?;
        Ok(GpuTensor::Float16 {
            data: Arc::new(gpu_data),
            shape,
        })
    }

    /// Create a zeroed Float16 GPU tensor with given shape.
    pub fn zeros_f16(ctx: &IconnxCudaContext, shape: Vec<usize>) -> Result<Self, CudaError> {
        let len: usize = shape.iter().product();
        let gpu_data = ctx.alloc_zeros::<half::f16>(len)?;
        Ok(GpuTensor::Float16 {
            data: Arc::new(gpu_data),
            shape,
        })
    }

    /// Copy Float16 tensor data back to host.
    pub fn to_host_f16(&self, ctx: &IconnxCudaContext) -> Result<Vec<half::f16>, CudaError> {
        match self {
            GpuTensor::Float16 { data, shape } => {
                let mut full = ctx.dtoh(data)?;
                let len: usize = shape.iter().product();
                full.truncate(len);
                Ok(full)
            }
            _ => Err(CudaError::Kernel(format!(
                "to_host_f16 called on {} tensor",
                self.dtype().name()
            ))),
        }
    }

    /// Get a reference to the underlying Float16 `DeviceSlice`.
    pub fn data_f16(&self) -> Result<&DeviceSlice<'static, half::f16>, CudaError> {
        match self {
            GpuTensor::Float16 { data, .. } => Ok(data),
            _ => Err(CudaError::Kernel(format!(
                "data_f16 called on {} tensor",
                self.dtype().name()
            ))),
        }
    }

    /// Get a mutable reference to the underlying Float16 `DeviceSlice`.
    /// See [`GpuTensor::data_f32_mut`] for the shared-Arc note.
    pub fn data_f16_mut(&mut self) -> Result<&mut DeviceSlice<'static, half::f16>, CudaError> {
        match self {
            GpuTensor::Float16 { data, .. } => Arc::get_mut(data).ok_or_else(|| {
                CudaError::Kernel(
                    "data_f16_mut: tensor is shared (Arc refcount > 1); \
                     cannot obtain mutable access without aliasing"
                        .to_string(),
                )
            }),
            _ => Err(CudaError::Kernel(format!(
                "data_f16_mut called on {} tensor",
                self.dtype().name()
            ))),
        }
    }

    /// Check if this is a Float16 tensor.
    pub fn is_f16(&self) -> bool {
        matches!(self, GpuTensor::Float16 { .. })
    }

    // ==================== BFloat16 methods ====================
    //
    // WS-3.5 Y(1). Mirror of the Float16 methods section above with
    // `half::f16` → `half::bf16` substitution. Two semantic differences
    // from the FP16 surface:
    //
    //  - Hardware-capability gate: `zeros_bf16` and `from_host_bf16` call
    //    `check_bf16_hardware` to enforce the sm_80+ floor at the single
    //    ctor entry. Downstream dispatch sites can assume any
    //    `GpuTensor::BFloat16` they receive came through this gate.
    //
    //  - Typed `CudaError::ShapeMismatch` instead of the older
    //    `assert_eq!`-on-shape panic — newer-pattern error contract for
    //    the BF16 surface, parallel to the WS-3 M3.6 typed-error stance.

    /// Create a new BFloat16 GPU tensor from shape and GPU data.
    ///
    /// Caller is responsible for ensuring the device data is sized
    /// appropriately for the shape; this constructor performs no GPU
    /// allocation, so no hardware check (the slice was already created
    /// elsewhere — typically by the memory pool, which gates BF16
    /// allocation paths through `check_bf16_hardware`).
    pub fn new_bf16(data: DeviceSlice<'static, half::bf16>, shape: Vec<usize>) -> Self {
        GpuTensor::BFloat16 {
            data: Arc::new(data),
            shape,
        }
    }

    /// Create a BFloat16 GPU tensor from host data.
    ///
    /// Hardware-capability gate: BF16 native compute requires sm_80+.
    /// Returns `CudaError::UnsupportedHardware` on older devices.
    pub fn from_host_bf16(
        ctx: &IconnxCudaContext,
        data: &[half::bf16],
        shape: Vec<usize>,
    ) -> Result<Self, CudaError> {
        Self::check_bf16_hardware(ctx)?;
        let expected_len: usize = shape.iter().product();
        if data.len() != expected_len {
            return Err(CudaError::ShapeMismatch {
                data_len: data.len(),
                expected: expected_len,
            });
        }
        let gpu_data = ctx.htod(data)?;
        Ok(GpuTensor::BFloat16 {
            data: Arc::new(gpu_data),
            shape,
        })
    }

    /// Create a zeroed BFloat16 GPU tensor with given shape.
    ///
    /// Hardware-capability gate: see `from_host_bf16`.
    pub fn zeros_bf16(
        ctx: &IconnxCudaContext,
        shape: Vec<usize>,
    ) -> Result<Self, CudaError> {
        Self::check_bf16_hardware(ctx)?;
        let len: usize = shape.iter().product();
        let gpu_data = ctx.alloc_zeros::<half::bf16>(len)?;
        Ok(GpuTensor::BFloat16 {
            data: Arc::new(gpu_data),
            shape,
        })
    }

    /// Copy BFloat16 tensor data back to host.
    pub fn to_host_bf16(&self, ctx: &IconnxCudaContext) -> Result<Vec<half::bf16>, CudaError> {
        match self {
            GpuTensor::BFloat16 { data, shape } => {
                let mut full = ctx.dtoh(data)?;
                let len: usize = shape.iter().product();
                full.truncate(len);
                Ok(full)
            }
            _ => Err(CudaError::DtypeMismatch {
                expected: "bfloat16",
                actual: self.dtype().name(),
            }),
        }
    }

    /// Get a reference to the underlying BFloat16 `DeviceSlice`.
    pub fn data_bf16(&self) -> Result<&DeviceSlice<'static, half::bf16>, CudaError> {
        match self {
            GpuTensor::BFloat16 { data, .. } => Ok(data),
            _ => Err(CudaError::DtypeMismatch {
                expected: "bfloat16",
                actual: self.dtype().name(),
            }),
        }
    }

    /// Get a mutable reference to the underlying BFloat16 `DeviceSlice`.
    /// See [`GpuTensor::data_f32_mut`] for the shared-Arc note.
    pub fn data_bf16_mut(&mut self) -> Result<&mut DeviceSlice<'static, half::bf16>, CudaError> {
        match self {
            GpuTensor::BFloat16 { data, .. } => Arc::get_mut(data).ok_or_else(|| {
                CudaError::Kernel(
                    "data_bf16_mut: tensor is shared (Arc refcount > 1); \
                     cannot obtain mutable access without aliasing"
                        .to_string(),
                )
            }),
            _ => Err(CudaError::DtypeMismatch {
                expected: "bfloat16",
                actual: self.dtype().name(),
            }),
        }
    }

    /// Check if this is a BFloat16 tensor.
    pub fn is_bf16(&self) -> bool {
        matches!(self, GpuTensor::BFloat16 { .. })
    }

    /// Hardware-capability check for BF16 native compute.
    ///
    /// BF16 native compute requires sm_80+ (Ampere and later: A100, H100,
    /// RTX 30/40-series). Older Turing (sm_75) compiles bf16 NVRTC but
    /// emulates with significant perf loss; some intrinsics
    /// (`__bfloat162float`, `__hadd` on `__nv_bfloat16`) require sm_80+
    /// outright. Volta (sm_70, V100) has no BF16 support at all.
    ///
    /// One-time-per-graph-load check (called from `zeros_bf16` /
    /// `from_host_bf16`). Downstream dispatch sites can assume any
    /// `GpuTensor::BFloat16` they receive is hardware-capable.
    fn check_bf16_hardware(ctx: &IconnxCudaContext) -> Result<(), CudaError> {
        let (major, minor) = ctx.compute_capability();
        if major < 8 {
            return Err(CudaError::UnsupportedHardware {
                dtype: "bfloat16",
                required_compute_capability: "sm_80",
                device_compute_capability: format!("sm_{major}{minor}"),
            });
        }
        Ok(())
    }

    // ==================== Bool methods ====================

    /// Create a new Bool GPU tensor from shape and GPU data.
    /// The underlying device storage is `u8` — 0 = false, nonzero = true.
    pub fn new_bool(data: DeviceSlice<'static, u8>, shape: Vec<usize>) -> Self {
        GpuTensor::Bool {
            data: Arc::new(data),
            shape,
        }
    }

    /// Create a Bool GPU tensor from host `&[bool]`.
    ///
    /// Bool's GPU representation is `u8` (CUDA has no native bool); host
    /// `bool` values are converted via `b as u8` (which the Rust spec
    /// guarantees is 0 or 1) before upload. Reading back via
    /// [`to_host_bool`] inverts the same mapping (`b != 0`).
    pub fn from_host_bool(
        ctx: &IconnxCudaContext,
        data: &[bool],
        shape: Vec<usize>,
    ) -> Result<Self, CudaError> {
        let expected_len: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected_len,
            "Data length {} doesn't match shape {:?} (expected {})",
            data.len(),
            shape,
            expected_len
        );

        let bytes: Vec<u8> = data.iter().map(|&b| b as u8).collect();
        let gpu_data = ctx.htod(&bytes)?;
        Ok(GpuTensor::Bool {
            data: Arc::new(gpu_data),
            shape,
        })
    }

    /// Create a zeroed Bool GPU tensor (all false) with given shape.
    pub fn zeros_bool(ctx: &IconnxCudaContext, shape: Vec<usize>) -> Result<Self, CudaError> {
        let len: usize = shape.iter().product();
        let gpu_data = ctx.alloc_zeros::<u8>(len)?;
        Ok(GpuTensor::Bool {
            data: Arc::new(gpu_data),
            shape,
        })
    }

    /// Copy Bool tensor data back to host as `Vec<bool>`. The byte-level
    /// `u8` storage is reinterpreted via `b != 0` so values written by
    /// kernels that use any nonzero "true" sentinel decode correctly.
    pub fn to_host_bool(&self, ctx: &IconnxCudaContext) -> Result<Vec<bool>, CudaError> {
        match self {
            GpuTensor::Bool { data, shape } => {
                let mut full: Vec<u8> = ctx.dtoh(data)?;
                let len: usize = shape.iter().product();
                full.truncate(len);
                Ok(full.into_iter().map(|b| b != 0).collect())
            }
            _ => Err(CudaError::Kernel(format!(
                "to_host_bool called on {} tensor",
                self.dtype().name()
            ))),
        }
    }

    /// Get a reference to the underlying Bool `DeviceSlice<u8>`.
    pub fn data_bool(&self) -> Result<&DeviceSlice<'static, u8>, CudaError> {
        match self {
            GpuTensor::Bool { data, .. } => Ok(data),
            _ => Err(CudaError::Kernel(format!(
                "data_bool called on {} tensor",
                self.dtype().name()
            ))),
        }
    }

    /// Get a mutable reference to the underlying Bool `DeviceSlice<u8>`.
    /// See [`GpuTensor::data_f32_mut`] for the shared-Arc note.
    pub fn data_bool_mut(&mut self) -> Result<&mut DeviceSlice<'static, u8>, CudaError> {
        match self {
            GpuTensor::Bool { data, .. } => Arc::get_mut(data).ok_or_else(|| {
                CudaError::Kernel(
                    "data_bool_mut: tensor is shared (Arc refcount > 1); \
                     cannot obtain mutable access without aliasing"
                        .to_string(),
                )
            }),
            _ => Err(CudaError::Kernel(format!(
                "data_bool_mut called on {} tensor",
                self.dtype().name()
            ))),
        }
    }

    /// Check if this is a Bool tensor.
    pub fn is_bool(&self) -> bool {
        matches!(self, GpuTensor::Bool { .. })
    }

    // ==================== Reshape (type-preserving) ====================

    /// Reshape the tensor (no data copy, just shape change). Returns
    /// `None` if the total element count doesn't match.
    ///
    /// Consumes `self` and moves the inner `Arc` into the new tensor —
    /// so the caller's `Arc` refcount goes away rather than being
    /// cloned. This matters because post-garboard the storage type
    /// (`DeviceSlice`) is not `Clone`, so `Arc::make_mut` is no longer
    /// available; downstream mutation via `data_*_mut()` requires
    /// refcount == 1. For borrowed inputs, clone explicitly:
    /// `tensor.clone().reshape(new_shape)`.
    pub fn reshape(self, new_shape: Vec<usize>) -> Option<Self> {
        let old_len: usize = self.shape().iter().product();
        let new_len: usize = new_shape.iter().product();

        if old_len != new_len {
            return None;
        }

        match self {
            GpuTensor::Float32 { data, .. } => Some(GpuTensor::Float32 {
                data,
                shape: new_shape,
            }),
            GpuTensor::Float16 { data, .. } => Some(GpuTensor::Float16 {
                data,
                shape: new_shape,
            }),
            GpuTensor::BFloat16 { data, .. } => Some(GpuTensor::BFloat16 {
                data,
                shape: new_shape,
            }),
            GpuTensor::Int64 { data, .. } => Some(GpuTensor::Int64 {
                data,
                shape: new_shape,
            }),
            GpuTensor::Int32 { data, .. } => Some(GpuTensor::Int32 {
                data,
                shape: new_shape,
            }),
            GpuTensor::Int8 { data, .. } => Some(GpuTensor::Int8 {
                data,
                shape: new_shape,
            }),
            GpuTensor::UInt8 { data, .. } => Some(GpuTensor::UInt8 {
                data,
                shape: new_shape,
            }),
            GpuTensor::Bool { data, .. } => Some(GpuTensor::Bool {
                data,
                shape: new_shape,
            }),
        }
    }

    // ==================== Memory pool support ====================

    /// Consume tensor and extract inner `DeviceSlice` if we have sole
    /// ownership. Returns `None` if the `Arc` is shared (refcount > 1),
    /// meaning we can't safely extract the slice for pooling.
    pub fn into_slice(self) -> Option<TypedSlice> {
        match self {
            GpuTensor::Float32 { data, .. } => {
                Arc::try_unwrap(data).ok().map(TypedSlice::Float32)
            }
            GpuTensor::Float16 { data, .. } => {
                Arc::try_unwrap(data).ok().map(TypedSlice::Float16)
            }
            GpuTensor::BFloat16 { data, .. } => {
                Arc::try_unwrap(data).ok().map(TypedSlice::BFloat16)
            }
            GpuTensor::Int64 { data, .. } => {
                Arc::try_unwrap(data).ok().map(TypedSlice::Int64)
            }
            GpuTensor::Int32 { data, .. } => {
                Arc::try_unwrap(data).ok().map(TypedSlice::Int32)
            }
            GpuTensor::Int8 { data, .. } => {
                Arc::try_unwrap(data).ok().map(TypedSlice::Int8)
            }
            GpuTensor::UInt8 { data, .. } => {
                Arc::try_unwrap(data).ok().map(TypedSlice::UInt8)
            }
            GpuTensor::Bool { data, .. } => {
                Arc::try_unwrap(data).ok().map(TypedSlice::Bool)
            }
        }
    }

    // ==================== Backward compatibility ====================
    // These methods maintain API compatibility with code expecting f32-only tensors.

    /// Create a GPU tensor from host data (Float32, backward compatible).
    #[deprecated(note = "Use from_host_f32 for explicit typing")]
    pub fn from_host(
        ctx: &IconnxCudaContext,
        data: &[f32],
        shape: Vec<usize>,
    ) -> Result<Self, CudaError> {
        Self::from_host_f32(ctx, data, shape)
    }

    /// Create a zeroed GPU tensor (Float32, backward compatible).
    #[deprecated(note = "Use zeros_f32 for explicit typing")]
    pub fn zeros(ctx: &IconnxCudaContext, shape: Vec<usize>) -> Result<Self, CudaError> {
        Self::zeros_f32(ctx, shape)
    }

    /// Copy tensor data back to host (Float32, backward compatible).
    #[deprecated(note = "Use to_host_f32 for explicit typing")]
    pub fn to_host(&self, ctx: &IconnxCudaContext) -> Result<Vec<f32>, CudaError> {
        self.to_host_f32(ctx)
    }

    /// Get a reference to the underlying `DeviceSlice` (Float32,
    /// backward compatible). Panics if not Float32.
    #[deprecated(note = "Use data_f32 for explicit typing")]
    pub fn data(&self) -> &DeviceSlice<'static, f32> {
        match self {
            GpuTensor::Float32 { data, .. } => data,
            _ => panic!(
                "data() called on {} tensor, use data_f32()",
                self.dtype().name()
            ),
        }
    }

    /// Get a mutable reference to the underlying `DeviceSlice` (Float32,
    /// backward compatible). Panics if not Float32 or if shared.
    #[deprecated(note = "Use data_f32_mut for explicit typing")]
    pub fn data_mut(&mut self) -> &mut DeviceSlice<'static, f32> {
        match self {
            GpuTensor::Float32 { data, .. } => Arc::get_mut(data).expect(
                "data_mut: tensor is shared (Arc refcount > 1); cannot obtain \
                 mutable access without aliasing",
            ),
            _ => panic!(
                "data_mut() called on {} tensor, use data_f32_mut()",
                self.dtype().name()
            ),
        }
    }

    /// Create a new Float32 tensor (backward compatible).
    #[deprecated(note = "Use new_f32 for explicit typing")]
    pub fn new(data: DeviceSlice<'static, f32>, shape: Vec<usize>) -> Self {
        Self::new_f32(data, shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires CUDA GPU
    fn test_gpu_tensor_f32_creation() {
        let ctx = IconnxCudaContext::new().expect("Failed to create CUDA context");

        // Create a 2x3 Float32 tensor.
        let host_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = GpuTensor::from_host_f32(&ctx, &host_data, vec![2, 3])
            .expect("Failed to create GPU tensor");

        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.ndim(), 2);
        assert_eq!(tensor.len(), 6);
        assert_eq!(tensor.dtype(), DType::Float32);
        assert!(tensor.is_f32());

        // Verify data round-trip.
        let result = tensor.to_host_f32(&ctx).expect("Failed to copy to host");
        assert_eq!(result, host_data);
    }

    #[test]
    #[ignore] // Requires CUDA GPU
    fn test_gpu_tensor_i64_creation() {
        let ctx = IconnxCudaContext::new().expect("Failed to create CUDA context");

        // Create a 1D Int64 tensor (like Shape output).
        let host_data: Vec<i64> = vec![2, 3, 4];
        let tensor = GpuTensor::from_host_i64(&ctx, &host_data, vec![3])
            .expect("Failed to create GPU tensor");

        assert_eq!(tensor.shape(), &[3]);
        assert_eq!(tensor.ndim(), 1);
        assert_eq!(tensor.len(), 3);
        assert_eq!(tensor.dtype(), DType::Int64);
        assert!(tensor.is_i64());

        // Verify data round-trip.
        let result = tensor.to_host_i64(&ctx).expect("Failed to copy to host");
        assert_eq!(result, host_data);
    }

    #[test]
    #[ignore] // Requires CUDA GPU
    fn test_gpu_tensor_i32_creation() {
        let ctx = IconnxCudaContext::new().expect("Failed to create CUDA context");

        // Create a 1D Int32 tensor.
        let host_data: Vec<i32> = vec![10, 20, 30, 40];
        let tensor = GpuTensor::from_host_i32(&ctx, &host_data, vec![4])
            .expect("Failed to create GPU tensor");

        assert_eq!(tensor.shape(), &[4]);
        assert_eq!(tensor.dtype(), DType::Int32);
        assert!(tensor.is_i32());

        // Verify data round-trip.
        let result = tensor.to_host_i32(&ctx).expect("Failed to copy to host");
        assert_eq!(result, host_data);
    }

    #[test]
    #[ignore] // Requires CUDA GPU
    fn test_gpu_tensor_zeros() {
        let ctx = IconnxCudaContext::new().expect("Failed to create CUDA context");

        // Float32 zeros.
        let tensor_f32 =
            GpuTensor::zeros_f32(&ctx, vec![10, 10]).expect("Failed to create zeros tensor");
        assert_eq!(tensor_f32.dtype(), DType::Float32);
        let result_f32 = tensor_f32.to_host_f32(&ctx).unwrap();
        assert!(result_f32.iter().all(|&x| x == 0.0));

        // Int64 zeros.
        let tensor_i64 =
            GpuTensor::zeros_i64(&ctx, vec![5]).expect("Failed to create zeros tensor");
        assert_eq!(tensor_i64.dtype(), DType::Int64);
        let result_i64 = tensor_i64.to_host_i64(&ctx).unwrap();
        assert!(result_i64.iter().all(|&x| x == 0));
    }

    #[test]
    #[ignore] // Requires CUDA GPU
    fn test_gpu_tensor_reshape() {
        let ctx = IconnxCudaContext::new().expect("Failed to create CUDA context");

        let host_data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let tensor = GpuTensor::from_host_f32(&ctx, &host_data, vec![2, 3, 4])
            .expect("Failed to create GPU tensor");

        // Reshape to [4, 6].
        let reshaped = tensor.clone().reshape(vec![4, 6]).expect("Reshape failed");
        assert_eq!(reshaped.shape(), &[4, 6]);
        assert_eq!(reshaped.len(), 24);
        assert_eq!(reshaped.dtype(), DType::Float32); // Type preserved.

        // Invalid reshape should fail.
        assert!(tensor.reshape(vec![5, 5]).is_none());
    }

    #[test]
    #[ignore] // Requires CUDA GPU
    fn test_type_mismatch_errors() {
        let ctx = IconnxCudaContext::new().expect("Failed to create CUDA context");

        // Create an Int64 tensor.
        let tensor =
            GpuTensor::from_host_i64(&ctx, &[1, 2, 3], vec![3]).expect("Failed to create tensor");

        // Trying to get f32 data should fail.
        assert!(tensor.data_f32().is_err());
        assert!(tensor.to_host_f32(&ctx).is_err());

        // Create a Float32 tensor.
        let tensor_f32 =
            GpuTensor::from_host_f32(&ctx, &[1.0, 2.0], vec![2]).expect("Failed to create tensor");

        // Trying to get i64 data should fail.
        assert!(tensor_f32.data_i64().is_err());
        assert!(tensor_f32.to_host_i64(&ctx).is_err());

        // INT8 / UINT8 (WS-4 M4.3): same wrong-variant contract.
        let tensor_i8 = GpuTensor::from_host_i8(&ctx, &[-1_i8, 0, 1], vec![3])
            .expect("Failed to create i8 tensor");
        assert!(tensor_i8.data_u8().is_err());
        assert!(tensor_i8.to_host_u8(&ctx).is_err());
        assert!(tensor_i8.data_f32().is_err());

        let tensor_u8 = GpuTensor::from_host_u8(&ctx, &[0_u8, 128, 255], vec![3])
            .expect("Failed to create u8 tensor");
        assert!(tensor_u8.data_i8().is_err());
        assert!(tensor_u8.to_host_i8(&ctx).is_err());
        assert!(tensor_u8.data_i32().is_err());
    }

    #[test]
    #[ignore] // Requires CUDA GPU
    fn test_size_bytes() {
        let ctx = IconnxCudaContext::new().expect("Failed to create CUDA context");

        let tensor_f32 = GpuTensor::zeros_f32(&ctx, vec![100]).unwrap();
        assert_eq!(tensor_f32.size_bytes(), 400); // 100 * 4 bytes.

        let tensor_i64 = GpuTensor::zeros_i64(&ctx, vec![100]).unwrap();
        assert_eq!(tensor_i64.size_bytes(), 800); // 100 * 8 bytes.

        let tensor_i32 = GpuTensor::zeros_i32(&ctx, vec![100]).unwrap();
        assert_eq!(tensor_i32.size_bytes(), 400); // 100 * 4 bytes.

        // INT8 / UINT8 (WS-4 M4.3): 1 byte per element.
        let tensor_i8 = GpuTensor::zeros_i8(&ctx, vec![100]).unwrap();
        assert_eq!(tensor_i8.size_bytes(), 100);

        let tensor_u8 = GpuTensor::zeros_u8(&ctx, vec![100]).unwrap();
        assert_eq!(tensor_u8.size_bytes(), 100);
    }
}
