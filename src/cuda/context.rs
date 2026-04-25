//! CUDA context management for Iconnx.
//!
//! Garboard owns the device, stream, cuBLAS handle, and cuDNN handle;
//! iconnx's [`IconnxCudaContext`] simply holds them together so operator
//! implementations can borrow what they need. The garboard `Device` is
//! leaked to `'static` so `DeviceSlice<'static, T>` can flow through
//! `Arc`-backed `GpuTensor` fields and the memory pool without
//! self-borrow issues.
//!
//! All GPU work — memory allocation, host/device transfers, device
//! zero-fill, NVRTC kernel launches, cuBLAS GEMMs, and cuDNN conv/RNN —
//! is scheduled on the single garboard user stream owned by this
//! context.

use std::cell::RefCell;

use garboard::{
    BlasContext, Device as GbDevice, DeviceSlice as GbDeviceSlice, DnnContext, Stream as GbStream,
};

/// Iconnx CUDA context — manages GPU device, stream, BLAS, etc.
pub struct IconnxCudaContext {
    /// Garboard device, leaked to `'static` so `DeviceSlice<'static, T>`
    /// can flow through long-lived structures without self-borrow issues.
    /// One Device is leaked per distinct ordinal per process; this is
    /// consistent with the singleton nature of a GPU context.
    garboard_device: &'static GbDevice,
    /// Default stream on the garboard device, used for host/device
    /// transfers, all typed-kernel launches, cuBLAS GEMMs, and cuDNN
    /// conv/RNN calls.
    garboard_stream: GbStream<'static>,
    /// cuBLAS handle (via garboard), bound to `garboard_stream` on each
    /// BLAS call.
    blas: BlasContext<'static>,
    /// cuDNN handle (via garboard). `RefCell` because `DnnContext`
    /// operations take `&mut self` (workspace auto-grows), but
    /// `IconnxCudaContext` is passed as `&ctx` at inference sites.
    dnn: RefCell<DnnContext<'static>>,
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
        let dnn = garboard_device
            .create_dnn_context()
            .map_err(|e| CudaError::Cudnn(e.to_string()))?;

        Ok(Self {
            garboard_device,
            garboard_stream,
            blas,
            dnn: RefCell::new(dnn),
        })
    }

    // ==================== Accessors ====================

    /// Get the garboard cuBLAS context for matrix operations.
    /// Each BLAS call takes `&Stream` at its call site; this accessor
    /// returns the handle to pair with `garboard_stream()`.
    pub fn blas(&self) -> &BlasContext<'static> {
        &self.blas
    }

    /// Mutable handle on the garboard cuDNN context. The workspace may
    /// auto-grow during `conv_*` / `rnn_*` calls, hence `&mut self` on
    /// `DnnContext` methods. We hand out a `RefMut` via `RefCell`.
    pub fn dnn(&self) -> std::cell::RefMut<'_, DnnContext<'static>> {
        self.dnn.borrow_mut()
    }

    /// Get the garboard device (leaked to `'static`). Use this for
    /// direct garboard operations that need a `&Device` receiver.
    pub fn garboard_device(&self) -> &'static GbDevice {
        self.garboard_device
    }

    /// Get the garboard stream associated with this context.
    /// Used for async transfers and TypedKernel launches.
    pub fn garboard_stream(&self) -> &GbStream<'static> {
        &self.garboard_stream
    }

    // ==================== Memory API (garboard) ====================

    /// Allocate a zero-initialized GPU buffer.
    ///
    /// Uses `alloc_uninit` + `Stream::memset_zero` rather than garboard's
    /// `Device::alloc_zeros`. The latter performs a synchronous
    /// `cuMemsetD8_v2`, which blocks the host AND induces an implicit
    /// synchronization with any active CUDA stream — at ~25 us per call,
    /// that overhead compounded across the memory-pool's miss path and
    /// other allocation sites was a major contributor to the post-garboard
    /// regression. Routing the zero-fill through garboard's user stream
    /// keeps it on the async path and avoids the cross-stream sync.
    pub fn alloc_zeros<T: bytemuck::Pod>(
        &self,
        len: usize,
    ) -> Result<GbDeviceSlice<'static, T>, CudaError> {
        let mut slice = self
            .garboard_device
            .alloc_uninit::<T>(len)
            .map_err(|e| CudaError::Alloc(e.to_string()))?;
        self.garboard_stream
            .memset_zero(&mut slice)
            .map_err(|e| CudaError::Memory(e.to_string()))?;
        Ok(slice)
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
        // Use the stream-aware copy (`cuMemcpyHtoDAsync`) rather than
        // `Device::copy_host_to_device` (`cuMemcpyHtoD_v2`). The sync
        // `_v2` form not only blocks the host but also induces an
        // implicit cross-stream synchronization with every other
        // in-flight stream — at ~25 us per call that overhead
        // compounded across every layout op's shape/stride upload into
        // a measurable per-inference regression.
        self.garboard_stream
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
    /// Drains the garboard user stream on which all kernels, transfers,
    /// cuBLAS, and cuDNN work is scheduled.
    pub fn sync(&self) -> Result<(), CudaError> {
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

    /// Zero out an i8 slice in place.
    pub fn zero_i8(&self, slice: &mut GbDeviceSlice<'static, i8>) -> Result<(), CudaError> {
        self.zero_slice(slice)
    }

    /// Zero out a u8 slice in place.
    pub fn zero_u8(&self, slice: &mut GbDeviceSlice<'static, u8>) -> Result<(), CudaError> {
        self.zero_slice(slice)
    }

    /// Internal helper: zero a garboard slice on the garboard user stream.
    ///
    /// Uses garboard's `Stream::memset_zero`, which enqueues
    /// `cuMemsetD8Async` on the garboard user stream — no null-stream
    /// involvement, so no implicit cross-stream synchronization with
    /// subsequent kernel launches.
    fn zero_slice<T: bytemuck::Pod>(
        &self,
        slice: &mut GbDeviceSlice<'static, T>,
    ) -> Result<(), CudaError> {
        self.garboard_stream
            .memset_zero(slice)
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
