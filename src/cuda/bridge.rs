//! cudarc ↔ garboard interop bridge for the Phase 1 migration.
//!
//! During Phase 1, memory allocation has moved to garboard while kernel
//! launches, cuBLAS, and cuDNN still go through cudarc. This module provides
//! the glue that lets cudarc operate on garboard-allocated buffers without
//! double-freeing the underlying GPU memory.
//!
//! Two helpers, one per cudarc API surface:
//!
//! - [`GbKernelArg`] — for cudarc kernel launches (`.arg(...)`). A
//!   `#[repr(C)]` wrapper around the `CUdeviceptr` that implements
//!   cudarc's `DeviceRepr` marker trait, so `LaunchArgs::arg(&GbArg)` works
//!   directly via cudarc's generic `PushKernelArg<&T: DeviceRepr>` impl.
//!   Zero runtime overhead, no `leak()` dance — construction is just a
//!   `u64` copy.
//!
//! - [`CudarcView`] — for cuBLAS and cuDNN APIs (and cudarc's
//!   `memset_zeros`) that take a full `&CudaSlice<T>` / `&mut CudaSlice<T>`.
//!   Wraps the garboard slice as a temporary cudarc view via
//!   `CudaStream::upgrade_device_ptr` and calls `.leak()` on drop so
//!   cudarc's `cuMemFree_v2` is not run against garboard-owned memory.
//!
//! Subsequent PRs will migrate kernel launches, cuBLAS, and cuDNN to
//! garboard's own `TypedKernel` / `BlasContext` / `DnnContext`, at which
//! point this bridge goes away.

use std::marker::PhantomData;
use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr};
use garboard::DeviceSlice;

// =========================================================================
// GbKernelArg — zero-overhead cudarc kernel launch argument
// =========================================================================

/// Kernel-launch argument wrapper for a garboard [`DeviceSlice`].
///
/// Cudarc's `LaunchArgs::arg(&T)` accepts any `T: DeviceRepr`; the default
/// pushing semantics do `&arg as *const _ as *mut c_void` and read the
/// first machine word as a CUDA device pointer. Because `GbKernelArg` is
/// `#[repr(C)]` with a `u64` (`CUdeviceptr`) as its only runtime field,
/// this works: the kernel sees the device pointer for the wrapped garboard
/// slice.
///
/// This is the preferred helper for kernel-launch sites during the
/// migration — it has no runtime overhead beyond a single `u64` copy, and
/// it drops trivially (no `leak()` contract like [`CudarcView`]).
///
/// # Example
///
/// ```ignore
/// let out_arg = GbKernelArg::new(output.data_f32_mut()?);
/// let in_arg = GbKernelArg::new(input.data_f32()?);
/// unsafe {
///     stream.launch_builder(kernel)
///         .arg(&out_arg)
///         .arg(&in_arg)
///         .arg(&n)
///         .launch(cfg)?;
/// }
/// ```
///
/// # Safety of this type
///
/// Constructing a `GbKernelArg` is safe: it only copies a `u64`. The
/// unsafety is concentrated in the kernel launch, which is already
/// `unsafe` in cudarc. The caller of `stream.launch_builder(...).launch(cfg)`
/// must ensure:
///
/// - the wrapped slice is alive for the duration of the kernel (i.e. until
///   the stream drains past the launch); and
/// - any mutable aliasing constraint holds (for `borrow_mut` construction,
///   no other kernel may read or write the slice until this one completes).
#[repr(C)]
pub struct GbKernelArg<T> {
    /// First field — read by the CUDA driver as an 8-byte `CUdeviceptr`.
    ptr: u64,
    _marker: PhantomData<T>,
}

impl<T: bytemuck::Pod> GbKernelArg<T> {
    /// Construct a kernel arg for read-only use. For mutable kernel
    /// parameters, use [`GbKernelArg::new_mut`] — the `&mut` receiver
    /// documents exclusive access.
    #[inline]
    pub fn new(slice: &DeviceSlice<'static, T>) -> Self {
        Self {
            ptr: slice.as_raw_device_ptr(),
            _marker: PhantomData,
        }
    }

    /// Construct a kernel arg for mutable use. Identical byte-for-byte to
    /// [`new`]; the separate method exists so the Rust borrow checker
    /// enforces non-aliasing at the call site.
    #[inline]
    pub fn new_mut(slice: &mut DeviceSlice<'static, T>) -> Self {
        Self {
            ptr: slice.as_raw_device_ptr(),
            _marker: PhantomData,
        }
    }
}

// SAFETY: `GbKernelArg<T>` is `#[repr(C)]` with a `u64` (CUdeviceptr) as
// its first and only runtime field. Cudarc's default `PushKernelArg<&T>`
// impl casts `&self` to `*mut c_void` and forwards that to
// `cuLaunchKernel`, which reads the first 8 bytes as a device pointer.
// The `u64` we store is a valid CUdeviceptr obtained from the wrapped
// garboard slice. `PhantomData<T>` is zero-sized and does not change the
// layout. The caller's launch is already `unsafe` and must maintain the
// underlying slice's lifetime and aliasing discipline.
unsafe impl<T: bytemuck::Pod> DeviceRepr for GbKernelArg<T> {}

// =========================================================================
// CudarcView — RAII cuBLAS/cuDNN/memset bridge
// =========================================================================

/// RAII view of a `garboard::DeviceSlice<'static, T>` as a
/// `cudarc::CudaSlice<T>`, for APIs that require a full `&CudaSlice<T>` —
/// specifically cuBLAS, cuDNN, and `CudaStream::memset_zeros`. For kernel
/// launches use [`GbKernelArg`] instead.
///
/// The view does NOT own the underlying GPU allocation — garboard does.
/// On drop, the view is leaked via `CudaSlice::leak` so cudarc's Drop does
/// not call `cuMemFree_v2` on memory still owned by garboard.
pub struct CudarcView<T> {
    /// Wrapped cudarc view. `Option` lets us `take()` in Drop.
    slice: Option<CudaSlice<T>>,
}

impl<T> Drop for CudarcView<T> {
    fn drop(&mut self) {
        if let Some(s) = self.slice.take() {
            // `leak` consumes the CudaSlice, cleans up its bookkeeping
            // (read/write events, stream refcount), and `mem::forget`s
            // the slice so `cuMemFree_v2` is NOT called on the pointer.
            // Garboard's DeviceSlice remains responsible for the
            // underlying allocation.
            let _ = s.leak();
        }
    }
}

impl<T: bytemuck::Pod> CudarcView<T> {
    /// Construct a read-only cudarc view of a garboard slice.
    ///
    /// # Safety
    /// - `slice` must outlive the returned `CudarcView`.
    /// - `stream` must be associated with the same CUDA device as `slice`.
    /// - While the view is alive, no code may mutate `slice` through any
    ///   other reference. Cudarc work scheduled against the view runs on
    ///   `stream`; callers must ensure it completes before the underlying
    ///   garboard slice is freed or reused.
    /// - `slice` must not be allocated on garboard's host-simulation
    ///   backend — `as_raw_device_ptr` returns 0 there, which is not a
    ///   valid CUDA device pointer.
    pub unsafe fn borrow(
        stream: &Arc<CudaStream>,
        slice: &DeviceSlice<'static, T>,
    ) -> Self {
        // SAFETY: upgrade_device_ptr wraps the pointer; the caller's safety
        // contract (above) ensures the pointer is valid, the stream matches,
        // and no aliasing occurs for the view's lifetime.
        let s = unsafe {
            stream.upgrade_device_ptr::<T>(slice.as_raw_device_ptr(), slice.len())
        };
        Self { slice: Some(s) }
    }

    /// Construct a mutable cudarc view of a garboard slice.
    ///
    /// The `&mut DeviceSlice` receiver documents exclusive access for the
    /// view's lifetime. Semantically identical to [`borrow`] on the
    /// cudarc side; the separate method exists so the Rust borrow checker
    /// enforces non-aliasing at the call site.
    ///
    /// # Safety
    /// Same as [`borrow`]. Additionally, the garboard slice is accessible
    /// for write through cudarc API calls for the view's lifetime.
    pub unsafe fn borrow_mut(
        stream: &Arc<CudaStream>,
        slice: &mut DeviceSlice<'static, T>,
    ) -> Self {
        // SAFETY: see `borrow` — plus the &mut receiver on `slice` documents
        // exclusive access.
        let s = unsafe {
            stream.upgrade_device_ptr::<T>(slice.as_raw_device_ptr(), slice.len())
        };
        Self { slice: Some(s) }
    }

    /// Access the cudarc view as `&CudaSlice<T>`.
    pub fn as_slice(&self) -> &CudaSlice<T> {
        self.slice
            .as_ref()
            .expect("CudarcView used after drop (internal bug)")
    }

    /// Access the cudarc view as `&mut CudaSlice<T>`.
    pub fn as_mut_slice(&mut self) -> &mut CudaSlice<T> {
        self.slice
            .as_mut()
            .expect("CudarcView used after drop (internal bug)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::context::IconnxCudaContext;

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn cudarc_view_roundtrip() {
        let ctx = IconnxCudaContext::new().expect("ctx");

        // Allocate on garboard; view as cudarc; memset to zero via cudarc;
        // read back via garboard. Verifies the bridge does not corrupt
        // data and does not double-free.
        let mut gb_slice = ctx.alloc_zeros::<u32>(16).expect("alloc");

        // Write non-zero via host-to-device.
        let ones = vec![1u32; 16];
        ctx.garboard_device()
            .copy_host_to_device(&ones, &mut gb_slice)
            .expect("h2d");

        // Zero via cudarc bridge.
        {
            let mut view =
                unsafe { CudarcView::borrow_mut(ctx.stream(), &mut gb_slice) };
            ctx.stream()
                .memset_zeros(view.as_mut_slice())
                .expect("memset");
        }
        ctx.sync().expect("sync");

        // Read back via garboard.
        let mut host = vec![0u32; 16];
        ctx.garboard_device()
            .copy_device_to_host(&gb_slice, &mut host)
            .expect("d2h");
        assert!(host.iter().all(|&x| x == 0));
    }

    #[test]
    fn gb_kernel_arg_layout() {
        // Structural check: GbKernelArg<T> must be #[repr(C)] with a u64
        // as the first field, so cudarc's generic PushKernelArg<&T: DeviceRepr>
        // impl sees the CUdeviceptr when it reinterprets `&arg`.
        assert_eq!(
            std::mem::size_of::<GbKernelArg<f32>>(),
            std::mem::size_of::<u64>()
        );
        assert_eq!(
            std::mem::align_of::<GbKernelArg<f32>>(),
            std::mem::align_of::<u64>()
        );
    }
}
