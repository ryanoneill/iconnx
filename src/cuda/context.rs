//! CUDA context management for Iconnx.
//!
//! Holds both cudarc and garboard handles during the Phase 1 migration:
//!
//! - **cudarc** (`CudaContext`, `CudaStream`) — a handful of remaining
//!   launch sites (14+-arg layout kernels, cuDNN conv/RNN) still route
//!   through cudarc via the `GbKernelArg` bridge or raw FFI. They migrate
//!   in the final Phase 1 PR.
//! - **garboard** (`Device`, `Stream`, `BlasContext`) — memory allocation,
//!   host/device transfers, device zero-fill, and cuBLAS GEMMs all run
//!   through garboard now. The garboard `Device` is leaked to `'static`
//!   so `DeviceSlice<'static, T>` can flow through `Arc`-backed
//!   `GpuTensor` fields and the memory pool without self-borrow issues.
//!
//! Cross-backend synchronization is handled by `sync()` which drains
//! both streams.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream};
use garboard::{
    BlasContext, Device as GbDevice, DeviceSlice as GbDeviceSlice, Stream as GbStream,
};

use super::bridge::CudarcView;

/// Iconnx CUDA context — manages GPU device, stream, BLAS, etc.
pub struct IconnxCudaContext {
    // ---- cudarc (residual: layout bridge kernels, cuDNN) ----
    /// The cudarc CUDA context.
    context: Arc<CudaContext>,
    /// Default stream for cudarc-driven operations (stream 0, CUDA's
    /// legacy null stream — implicitly synchronizes with garboard's
    /// user stream below).
    stream: Arc<CudaStream>,

    // ---- garboard (memory allocation, transfers, BLAS) ----
    /// Garboard device, leaked to `'static` so `DeviceSlice<'static, T>`
    /// can flow through long-lived structures without self-borrow issues.
    /// One Device is leaked per distinct ordinal per process; this is
    /// consistent with the singleton nature of a GPU context.
    garboard_device: &'static GbDevice,
    /// Default stream on the garboard device, used for host/device
    /// transfers, all typed-kernel launches, and cuBLAS GEMMs.
    garboard_stream: GbStream<'static>,
    /// cuBLAS handle (via garboard), bound to `garboard_stream` on each
    /// BLAS call.
    blas: BlasContext<'static>,
}

impl IconnxCudaContext {
    /// Create a new CUDA context for device 0 (default GPU).
    pub fn new() -> Result<Self, CudaError> {
        Self::new_for_device(0)
    }

    /// Create a new CUDA context for a specific device.
    ///
    /// The garboard `Device` is leaked to `'static` — this matches the
    /// singleton lifecycle of a process's CUDA context. Memory usage is
    /// bounded by the number of distinct ordinals seen during the
    /// process's lifetime.
    pub fn new_for_device(device_ordinal: usize) -> Result<Self, CudaError> {
        // cudarc side (residual: for kernels still on the bridge and cuDNN).
        let context =
            CudaContext::new(device_ordinal).map_err(|e| CudaError::DeviceInit(e.to_string()))?;
        let stream = context.default_stream();

        // garboard side.
        let ordinal_i32: i32 = device_ordinal.try_into().map_err(|_| {
            CudaError::DeviceInit(format!(
                "garboard device ordinal out of i32 range: {}",
                device_ordinal
            ))
        })?;
        let garboard_device: &'static GbDevice = Box::leak(Box::new(
            GbDevice::new_cuda(ordinal_i32)
                .map_err(|e| CudaError::DeviceInit(e.to_string()))?,
        ));
        let garboard_stream = garboard_device
            .create_stream()
            .map_err(|e| CudaError::StreamCreate(e.to_string()))?;
        let blas = garboard_device
            .create_blas_context()
            .map_err(|e| CudaError::Cublas(e.to_string()))?;

        Ok(Self {
            context,
            stream,
            garboard_device,
            garboard_stream,
            blas,
        })
    }

    // ==================== Accessors ====================

    /// Get the underlying cudarc CUDA context.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// Get the default cudarc stream (used for kernel launches, cuBLAS, cuDNN).
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Get the garboard cuBLAS context for matrix operations.
    /// Each BLAS call takes `&Stream` at its call site; this accessor
    /// returns the handle to pair with `garboard_stream()`.
    pub fn blas(&self) -> &BlasContext<'static> {
        &self.blas
    }

    /// Get the garboard device (leaked to `'static`). Use this for
    /// direct garboard operations that need a `&Device` receiver.
    pub fn garboard_device(&self) -> &'static GbDevice {
        self.garboard_device
    }

    /// Get the garboard stream associated with this context.
    /// Used for async transfers and (in future PRs) TypedKernel launches.
    pub fn garboard_stream(&self) -> &GbStream<'static> {
        &self.garboard_stream
    }

    // ==================== Memory API (garboard) ====================

    /// Allocate a zero-initialized GPU buffer.
    pub fn alloc_zeros<T: bytemuck::Pod>(
        &self,
        len: usize,
    ) -> Result<GbDeviceSlice<'static, T>, CudaError> {
        self.garboard_device
            .alloc_zeros::<T>(len)
            .map_err(|e| CudaError::Alloc(e.to_string()))
    }

    /// Copy data from host to device.
    pub fn htod<T: bytemuck::Pod>(
        &self,
        data: &[T],
    ) -> Result<GbDeviceSlice<'static, T>, CudaError> {
        let mut slice = self
            .garboard_device
            .alloc_uninit::<T>(data.len())
            .map_err(|e| CudaError::Alloc(e.to_string()))?;
        self.garboard_device
            .copy_host_to_device(data, &mut slice)
            .map_err(|e| CudaError::Transfer(e.to_string()))?;
        Ok(slice)
    }

    /// Copy i64 data from host to device.
    pub fn htod_i64(&self, data: &[i64]) -> Result<GbDeviceSlice<'static, i64>, CudaError> {
        self.htod(data)
    }

    /// Copy i64 data from device to host.
    pub fn dtoh_i64(
        &self,
        gpu_data: &GbDeviceSlice<'static, i64>,
    ) -> Result<Vec<i64>, CudaError> {
        self.dtoh(gpu_data)
    }

    /// Copy i32 data from host to device.
    pub fn htod_i32(&self, data: &[i32]) -> Result<GbDeviceSlice<'static, i32>, CudaError> {
        self.htod(data)
    }

    /// Copy i32 data from device to host.
    pub fn dtoh_i32(
        &self,
        gpu_data: &GbDeviceSlice<'static, i32>,
    ) -> Result<Vec<i32>, CudaError> {
        self.dtoh(gpu_data)
    }

    /// Copy usize data from host to device as u64 (matches CUDA's size_t).
    pub fn htod_usize(&self, data: &[usize]) -> Result<GbDeviceSlice<'static, u64>, CudaError> {
        let data_u64: Vec<u64> = data.iter().map(|&x| x as u64).collect();
        self.htod(&data_u64)
    }

    /// Copy data from device to host.
    pub fn dtoh<T: bytemuck::Pod>(
        &self,
        gpu_data: &GbDeviceSlice<'static, T>,
    ) -> Result<Vec<T>, CudaError> {
        let mut out: Vec<T> = vec![T::zeroed(); gpu_data.len()];
        self.garboard_device
            .copy_device_to_host(gpu_data, &mut out)
            .map_err(|e| CudaError::Transfer(e.to_string()))?;
        Ok(out)
    }

    /// Synchronize — block until all enqueued GPU work is complete.
    ///
    /// Drains both the cudarc stream (kernel launches, cuBLAS, cuDNN) and
    /// the garboard stream (async transfers, TypedKernel launches once
    /// they come online).
    pub fn sync(&self) -> Result<(), CudaError> {
        self.stream
            .synchronize()
            .map_err(|e| CudaError::Sync(e.to_string()))?;
        self.garboard_stream
            .synchronize()
            .map_err(|e| CudaError::Sync(e.to_string()))?;
        Ok(())
    }

    /// Zero out an i64 slice in place.
    pub fn zero_i64(&self, slice: &mut GbDeviceSlice<'static, i64>) -> Result<(), CudaError> {
        self.zero_slice(slice)
    }

    /// Zero out an i32 slice in place.
    pub fn zero_i32(&self, slice: &mut GbDeviceSlice<'static, i32>) -> Result<(), CudaError> {
        self.zero_slice(slice)
    }

    /// Zero out an f32 slice in place.
    pub fn zero_f32(&self, slice: &mut GbDeviceSlice<'static, f32>) -> Result<(), CudaError> {
        self.zero_slice(slice)
    }

    /// Internal helper: zero a garboard slice via cudarc's `memset_zeros`.
    /// Garboard has no in-place memset for existing slices, so during the
    /// migration we bridge to cudarc. Future PRs will replace this with a
    /// garboard-native API (or add one).
    fn zero_slice<T>(&self, slice: &mut GbDeviceSlice<'static, T>) -> Result<(), CudaError>
    where
        T: bytemuck::Pod + cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits,
    {
        // SAFETY: the slice is currently owned exclusively by us (&mut),
        // the cudarc stream is associated with the same CUDA context as
        // the garboard device (same ordinal), and `memset_zeros` only
        // writes bytes — the target type does not need to be valid for T
        // after the operation.
        let mut view = unsafe { CudarcView::borrow_mut(&self.stream, slice) };
        self.stream
            .memset_zeros(view.as_mut_slice())
            .map_err(|e| CudaError::Memory(e.to_string()))
    }
}

/// CUDA errors for Iconnx.
#[derive(Debug, Clone)]
pub enum CudaError {
    /// Failed to initialize device.
    DeviceInit(String),
    /// Failed to create stream.
    StreamCreate(String),
    /// Failed to allocate memory.
    Alloc(String),
    /// Failed to transfer data.
    Transfer(String),
    /// Failed to synchronize.
    Sync(String),
    /// Kernel execution failed.
    Kernel(String),
    /// cuBLAS operation failed.
    Cublas(String),
    /// cuDNN operation failed.
    Cudnn(String),
    /// Memory operation failed.
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

        // Allocate 1000 floats on GPU.
        let gpu_data = ctx.alloc_zeros::<f32>(1000).expect("Failed to allocate");

        // Transfer to host and verify.
        let host_data = ctx.dtoh(&gpu_data).expect("Failed to transfer");
        assert_eq!(host_data.len(), 1000);
        assert!(host_data.iter().all(|&x| x == 0.0));
    }

    #[test]
    #[ignore] // Requires CUDA GPU
    fn test_host_to_device_transfer() {
        let ctx = IconnxCudaContext::new().expect("Failed to create CUDA context");

        // Create host data.
        let host_data: Vec<f32> = (0..100).map(|i| i as f32).collect();

        // Transfer to GPU.
        let gpu_data = ctx.htod(&host_data).expect("Failed to transfer to GPU");

        // Transfer back.
        let result = ctx.dtoh(&gpu_data).expect("Failed to transfer from GPU");

        // Verify.
        assert_eq!(result, host_data);
    }
}
