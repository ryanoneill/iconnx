//! CUDA context management for Iconnx
//!
//! Wraps cudarc's CudaContext, CudaStream, and CudaBlas for ergonomic GPU operations.

use cudarc::cublas::CudaBlas;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use std::sync::Arc;

/// Iconnx CUDA context - manages GPU device, stream, and cuBLAS
pub struct IconnxCudaContext {
    /// The CUDA context
    context: Arc<CudaContext>,
    /// Default stream for operations
    stream: Arc<CudaStream>,
    /// cuBLAS handle for matrix operations
    cublas: CudaBlas,
}

impl IconnxCudaContext {
    /// Create a new CUDA context for device 0 (default GPU)
    pub fn new() -> Result<Self, CudaError> {
        Self::new_for_device(0)
    }

    /// Create a new CUDA context for a specific device
    pub fn new_for_device(device_ordinal: usize) -> Result<Self, CudaError> {
        let context =
            CudaContext::new(device_ordinal).map_err(|e| CudaError::DeviceInit(e.to_string()))?;
        let stream = context.default_stream();
        let cublas = CudaBlas::new(stream.clone()).map_err(|e| CudaError::Cublas(e.to_string()))?;

        Ok(Self {
            context,
            stream,
            cublas,
        })
    }

    /// Get the underlying CUDA context
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// Get the default stream
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Get the cuBLAS handle for matrix operations
    pub fn cublas(&self) -> &CudaBlas {
        &self.cublas
    }

    /// Allocate zeroed GPU memory
    pub fn alloc_zeros<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(
        &self,
        len: usize,
    ) -> Result<CudaSlice<T>, CudaError> {
        self.stream
            .alloc_zeros::<T>(len)
            .map_err(|e| CudaError::Alloc(e.to_string()))
    }

    /// Copy data from host to device
    pub fn htod<T: cudarc::driver::DeviceRepr>(
        &self,
        data: &[T],
    ) -> Result<CudaSlice<T>, CudaError> {
        self.stream
            .memcpy_stod(data)
            .map_err(|e| CudaError::Transfer(e.to_string()))
    }

    /// Copy i64 data from host to device
    pub fn htod_i64(&self, data: &[i64]) -> Result<CudaSlice<i64>, CudaError> {
        self.stream
            .memcpy_stod(data)
            .map_err(|e| CudaError::Transfer(e.to_string()))
    }

    /// Copy i64 data from device to host
    pub fn dtoh_i64(&self, gpu_data: &CudaSlice<i64>) -> Result<Vec<i64>, CudaError> {
        self.stream
            .memcpy_dtov(gpu_data)
            .map_err(|e| CudaError::Transfer(e.to_string()))
    }

    /// Copy i32 data from host to device
    pub fn htod_i32(&self, data: &[i32]) -> Result<CudaSlice<i32>, CudaError> {
        self.stream
            .memcpy_stod(data)
            .map_err(|e| CudaError::Transfer(e.to_string()))
    }

    /// Copy i32 data from device to host
    pub fn dtoh_i32(&self, gpu_data: &CudaSlice<i32>) -> Result<Vec<i32>, CudaError> {
        self.stream
            .memcpy_dtov(gpu_data)
            .map_err(|e| CudaError::Transfer(e.to_string()))
    }

    /// Copy usize data from host to device (as u64 for CUDA size_t compatibility)
    pub fn htod_usize(&self, data: &[usize]) -> Result<CudaSlice<u64>, CudaError> {
        // Convert usize to u64 for CUDA size_t (which is 64-bit on modern GPUs)
        let data_u64: Vec<u64> = data.iter().map(|&x| x as u64).collect();
        self.stream
            .memcpy_stod(&data_u64)
            .map_err(|e| CudaError::Transfer(e.to_string()))
    }

    /// Copy data from device to host
    pub fn dtoh<T: cudarc::driver::DeviceRepr>(
        &self,
        gpu_data: &CudaSlice<T>,
    ) -> Result<Vec<T>, CudaError> {
        self.stream
            .memcpy_dtov(gpu_data)
            .map_err(|e| CudaError::Transfer(e.to_string()))
    }

    /// Synchronize the stream (wait for all operations to complete)
    pub fn sync(&self) -> Result<(), CudaError> {
        self.stream
            .synchronize()
            .map_err(|e| CudaError::Sync(e.to_string()))
    }

    /// Zero out an i64 slice (sets all elements to 0)
    pub fn zero_i64(&self, slice: &mut CudaSlice<i64>) -> Result<(), CudaError> {
        self.stream
            .memset_zeros(slice)
            .map_err(|e| CudaError::Memory(e.to_string()))
    }

    /// Zero out an i32 slice (sets all elements to 0)
    pub fn zero_i32(&self, slice: &mut CudaSlice<i32>) -> Result<(), CudaError> {
        self.stream
            .memset_zeros(slice)
            .map_err(|e| CudaError::Memory(e.to_string()))
    }

    /// Zero out an f32 slice (sets all elements to 0.0)
    pub fn zero_f32(&self, slice: &mut CudaSlice<f32>) -> Result<(), CudaError> {
        self.stream
            .memset_zeros(slice)
            .map_err(|e| CudaError::Memory(e.to_string()))
    }
}

/// CUDA errors for Iconnx
#[derive(Debug, Clone)]
pub enum CudaError {
    /// Failed to initialize device
    DeviceInit(String),
    /// Failed to create stream
    StreamCreate(String),
    /// Failed to allocate memory
    Alloc(String),
    /// Failed to transfer data
    Transfer(String),
    /// Failed to synchronize
    Sync(String),
    /// Kernel execution failed
    Kernel(String),
    /// cuBLAS operation failed
    Cublas(String),
    /// cuDNN operation failed
    Cudnn(String),
    /// Memory operation failed
    Memory(String),
    /// Fusion shapes are incompatible — fall back to unfused execution.
    /// This is a sentinel, not a hard error: the executor catches it and
    /// runs the individual ops instead.
    FusionShapeMismatch(String),
}

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CudaError::DeviceInit(s) => write!(f, "CUDA device init failed: {}", s),
            CudaError::StreamCreate(s) => write!(f, "CUDA stream creation failed: {}", s),
            CudaError::Alloc(s) => write!(f, "CUDA allocation failed: {}", s),
            CudaError::Transfer(s) => write!(f, "CUDA transfer failed: {}", s),
            CudaError::Sync(s) => write!(f, "CUDA sync failed: {}", s),
            CudaError::Kernel(s) => write!(f, "CUDA kernel failed: {}", s),
            CudaError::Cublas(s) => write!(f, "cuBLAS operation failed: {}", s),
            CudaError::Cudnn(s) => write!(f, "cuDNN operation failed: {}", s),
            CudaError::Memory(s) => write!(f, "CUDA memory operation failed: {}", s),
            CudaError::FusionShapeMismatch(s) => {
                write!(f, "Fusion shape mismatch (fallback): {}", s)
            }
        }
    }
}

impl std::error::Error for CudaError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires CUDA GPU
    fn test_context_creation() {
        let ctx = IconnxCudaContext::new().expect("Failed to create CUDA context");
        ctx.sync().expect("Failed to sync");
    }

    #[test]
    #[ignore] // Requires CUDA GPU
    fn test_memory_allocation() {
        let ctx = IconnxCudaContext::new().expect("Failed to create CUDA context");

        // Allocate 1000 floats on GPU
        let gpu_data = ctx.alloc_zeros::<f32>(1000).expect("Failed to allocate");

        // Transfer to host and verify
        let host_data = ctx.dtoh(&gpu_data).expect("Failed to transfer");
        assert_eq!(host_data.len(), 1000);
        assert!(host_data.iter().all(|&x| x == 0.0));
    }

    #[test]
    #[ignore] // Requires CUDA GPU
    fn test_host_to_device_transfer() {
        let ctx = IconnxCudaContext::new().expect("Failed to create CUDA context");

        // Create host data
        let host_data: Vec<f32> = (0..100).map(|i| i as f32).collect();

        // Transfer to GPU
        let gpu_data = ctx.htod(&host_data).expect("Failed to transfer to GPU");

        // Transfer back
        let result = ctx.dtoh(&gpu_data).expect("Failed to transfer from GPU");

        // Verify
        assert_eq!(result, host_data);
    }
}
