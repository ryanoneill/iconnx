//! cuDNN bindings for high-performance deep learning operations
//!
//! This module provides safe Rust wrappers around cuDNN for operations like:
//! - Convolution (forward)
//! - ConvTranspose (backward data)
//!
//! These operations are significantly faster than our custom CUDA kernels
//! because cuDNN uses optimized algorithms and tensor cores.

#![allow(clippy::too_many_arguments)] // cuDNN APIs require many parameters

mod sys;
pub mod rnn;
pub use rnn::*;

use cudarc::driver::CudaStream;
use garboard::DeviceSlice;
use std::collections::HashMap;
use std::os::raw::c_int;
use std::ptr;

pub use sys::{
    cudnnConvolutionBwdDataAlgo_t, cudnnConvolutionFwdAlgo_t, cudnnConvolutionMode_t,
    cudnnDataType_t, cudnnStatus_t, cudnnTensorFormat_t,
};

use super::context::CudaError;
use super::context::IconnxCudaContext;
use super::tensor::GpuTensor;

// =============================================================================
// Error Handling
// =============================================================================

/// Convert cuDNN status to Result
fn check_cudnn(status: cudnnStatus_t) -> Result<(), CudaError> {
    if status.is_success() {
        Ok(())
    } else {
        Err(CudaError::Cudnn(format!("{}", status)))
    }
}

// =============================================================================
// CudnnHandle - Safe wrapper around cuDNN context
// =============================================================================

/// Safe wrapper around cuDNN handle
pub struct CudnnHandle {
    handle: sys::cudnnHandle_t,
}

impl CudnnHandle {
    /// Create a new cuDNN handle
    pub fn new() -> Result<Self, CudaError> {
        let mut handle: sys::cudnnHandle_t = ptr::null_mut();
        unsafe {
            check_cudnn(sys::cudnnCreate(&mut handle))?;
        }
        Ok(Self { handle })
    }

    /// Set the CUDA stream for this handle
    pub fn set_stream(&self, stream: &CudaStream) -> Result<(), CudaError> {
        // Get the raw stream pointer from cudarc
        let stream_ptr = stream.cu_stream() as sys::cudaStream_t;
        unsafe {
            check_cudnn(sys::cudnnSetStream(self.handle, stream_ptr))?;
        }
        Ok(())
    }

    /// Get cuDNN version
    pub fn version() -> usize {
        unsafe { sys::cudnnGetVersion() }
    }

    /// Get raw handle (for FFI calls)
    pub(crate) fn raw(&self) -> sys::cudnnHandle_t {
        self.handle
    }
}

impl Drop for CudnnHandle {
    fn drop(&mut self) {
        unsafe {
            sys::cudnnDestroy(self.handle);
        }
    }
}

// Safety: cuDNN handles can be sent between threads
unsafe impl Send for CudnnHandle {}
unsafe impl Sync for CudnnHandle {}

// =============================================================================
// TensorDescriptor - Describes tensor layout
// =============================================================================

/// Safe wrapper around cuDNN tensor descriptor
pub struct TensorDescriptor {
    desc: sys::cudnnTensorDescriptor_t,
}

impl TensorDescriptor {
    /// Create a new tensor descriptor
    pub fn new() -> Result<Self, CudaError> {
        let mut desc: sys::cudnnTensorDescriptor_t = ptr::null_mut();
        unsafe {
            check_cudnn(sys::cudnnCreateTensorDescriptor(&mut desc))?;
        }
        Ok(Self { desc })
    }

    /// Set 4D tensor descriptor (NCHW format)
    pub fn set_4d(
        &mut self,
        n: usize,
        c: usize,
        h: usize,
        w: usize,
        data_type: cudnnDataType_t,
    ) -> Result<(), CudaError> {
        unsafe {
            check_cudnn(sys::cudnnSetTensor4dDescriptor(
                self.desc,
                cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
                data_type,
                n as c_int,
                c as c_int,
                h as c_int,
                w as c_int,
            ))?;
        }
        Ok(())
    }

    /// Set ND tensor descriptor with custom strides
    pub fn set_nd(
        &mut self,
        dims: &[usize],
        strides: &[usize],
        data_type: cudnnDataType_t,
    ) -> Result<(), CudaError> {
        let dims_i32: Vec<c_int> = dims.iter().map(|&x| x as c_int).collect();
        let strides_i32: Vec<c_int> = strides.iter().map(|&x| x as c_int).collect();

        unsafe {
            check_cudnn(sys::cudnnSetTensorNdDescriptor(
                self.desc,
                data_type,
                dims.len() as c_int,
                dims_i32.as_ptr(),
                strides_i32.as_ptr(),
            ))?;
        }
        Ok(())
    }

    /// Get raw descriptor (for FFI calls)
    pub(crate) fn raw(&self) -> sys::cudnnTensorDescriptor_t {
        self.desc
    }
}

impl Drop for TensorDescriptor {
    fn drop(&mut self) {
        unsafe {
            sys::cudnnDestroyTensorDescriptor(self.desc);
        }
    }
}

// =============================================================================
// FilterDescriptor - Describes convolution filter layout
// =============================================================================

/// Safe wrapper around cuDNN filter descriptor
pub struct FilterDescriptor {
    desc: sys::cudnnFilterDescriptor_t,
}

impl FilterDescriptor {
    /// Create a new filter descriptor
    pub fn new() -> Result<Self, CudaError> {
        let mut desc: sys::cudnnFilterDescriptor_t = ptr::null_mut();
        unsafe {
            check_cudnn(sys::cudnnCreateFilterDescriptor(&mut desc))?;
        }
        Ok(Self { desc })
    }

    /// Set 4D filter descriptor (KCHW format: out_channels, in_channels, height, width)
    pub fn set_4d(
        &mut self,
        k: usize, // output channels
        c: usize, // input channels per group
        h: usize, // height
        w: usize, // width
        data_type: cudnnDataType_t,
    ) -> Result<(), CudaError> {
        unsafe {
            check_cudnn(sys::cudnnSetFilter4dDescriptor(
                self.desc,
                data_type,
                cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
                k as c_int,
                c as c_int,
                h as c_int,
                w as c_int,
            ))?;
        }
        Ok(())
    }

    /// Set ND filter descriptor
    pub fn set_nd(&mut self, dims: &[usize], data_type: cudnnDataType_t) -> Result<(), CudaError> {
        let dims_i32: Vec<c_int> = dims.iter().map(|&x| x as c_int).collect();

        unsafe {
            check_cudnn(sys::cudnnSetFilterNdDescriptor(
                self.desc,
                data_type,
                cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
                dims.len() as c_int,
                dims_i32.as_ptr(),
            ))?;
        }
        Ok(())
    }

    /// Get raw descriptor (for FFI calls)
    pub(crate) fn raw(&self) -> sys::cudnnFilterDescriptor_t {
        self.desc
    }
}

impl Drop for FilterDescriptor {
    fn drop(&mut self) {
        unsafe {
            sys::cudnnDestroyFilterDescriptor(self.desc);
        }
    }
}

// =============================================================================
// ConvolutionDescriptor - Describes convolution parameters
// =============================================================================

/// Safe wrapper around cuDNN convolution descriptor
pub struct ConvolutionDescriptor {
    desc: sys::cudnnConvolutionDescriptor_t,
}

impl ConvolutionDescriptor {
    /// Create a new convolution descriptor
    pub fn new() -> Result<Self, CudaError> {
        let mut desc: sys::cudnnConvolutionDescriptor_t = ptr::null_mut();
        unsafe {
            check_cudnn(sys::cudnnCreateConvolutionDescriptor(&mut desc))?;
        }
        Ok(Self { desc })
    }

    /// Set 2D convolution parameters
    pub fn set_2d(
        &mut self,
        pad_h: usize,
        pad_w: usize,
        stride_h: usize,
        stride_w: usize,
        dilation_h: usize,
        dilation_w: usize,
        mode: cudnnConvolutionMode_t,
    ) -> Result<(), CudaError> {
        unsafe {
            check_cudnn(sys::cudnnSetConvolution2dDescriptor(
                self.desc,
                pad_h as c_int,
                pad_w as c_int,
                stride_h as c_int,
                stride_w as c_int,
                dilation_h as c_int,
                dilation_w as c_int,
                mode,
                cudnnDataType_t::CUDNN_DATA_FLOAT,
            ))?;
        }
        Ok(())
    }

    /// Set ND convolution parameters
    pub fn set_nd(
        &mut self,
        pads: &[usize],
        strides: &[usize],
        dilations: &[usize],
        mode: cudnnConvolutionMode_t,
    ) -> Result<(), CudaError> {
        let pads_i32: Vec<c_int> = pads.iter().map(|&x| x as c_int).collect();
        let strides_i32: Vec<c_int> = strides.iter().map(|&x| x as c_int).collect();
        let dilations_i32: Vec<c_int> = dilations.iter().map(|&x| x as c_int).collect();

        unsafe {
            check_cudnn(sys::cudnnSetConvolutionNdDescriptor(
                self.desc,
                pads.len() as c_int,
                pads_i32.as_ptr(),
                strides_i32.as_ptr(),
                dilations_i32.as_ptr(),
                mode,
                cudnnDataType_t::CUDNN_DATA_FLOAT,
            ))?;
        }
        Ok(())
    }

    /// Set group count for grouped convolution
    pub fn set_group_count(&mut self, groups: usize) -> Result<(), CudaError> {
        unsafe {
            check_cudnn(sys::cudnnSetConvolutionGroupCount(
                self.desc,
                groups as c_int,
            ))?;
        }
        Ok(())
    }

    /// Enable tensor core math
    pub fn set_math_type(&mut self, use_tensor_cores: bool) -> Result<(), CudaError> {
        let math_type = if use_tensor_cores {
            sys::cudnnMathType_t::CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION
        } else {
            sys::cudnnMathType_t::CUDNN_DEFAULT_MATH
        };
        unsafe {
            check_cudnn(sys::cudnnSetConvolutionMathType(self.desc, math_type))?;
        }
        Ok(())
    }

    /// Get raw descriptor (for FFI calls)
    pub(crate) fn raw(&self) -> sys::cudnnConvolutionDescriptor_t {
        self.desc
    }
}

impl Drop for ConvolutionDescriptor {
    fn drop(&mut self) {
        unsafe {
            sys::cudnnDestroyConvolutionDescriptor(self.desc);
        }
    }
}

// =============================================================================
// Workspace - GPU memory for algorithm scratch space
// =============================================================================

/// Workspace buffer for cuDNN operations
pub struct Workspace {
    buffer: Option<DeviceSlice<'static, u8>>,
    size: usize,
}

impl Workspace {
    /// Create an empty workspace
    pub fn new() -> Self {
        Self {
            buffer: None,
            size: 0,
        }
    }

    /// Ensure workspace has at least `size` bytes
    pub fn ensure_size(&mut self, ctx: &IconnxCudaContext, size: usize) -> Result<(), CudaError> {
        if size > self.size {
            let buffer = ctx
                .alloc_zeros::<u8>(size)
                .map_err(|e| CudaError::Memory(format!("Workspace alloc failed: {}", e)))?;
            self.buffer = Some(buffer);
            self.size = size;
        }
        Ok(())
    }

    /// Get raw pointer (null if empty). Post-garboard migration this does
    /// not need a stream argument — garboard's `as_raw_device_ptr` returns
    /// the plain `CUdeviceptr` without cudarc's per-slice sync tracking.
    pub fn ptr(&self) -> *mut std::ffi::c_void {
        match &self.buffer {
            Some(buf) => buf.as_raw_device_ptr() as *mut std::ffi::c_void,
            None => ptr::null_mut(),
        }
    }

    /// Get current size
    pub fn size(&self) -> usize {
        self.size
    }
}

impl Default for Workspace {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Algorithm Selection Cache
// =============================================================================

/// Key for caching cuDNN algorithm selections.
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct ConvAlgoKey {
    pub input_shape: Vec<usize>,
    pub kernel_shape: Vec<usize>,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub groups: usize,
    pub dtype: &'static str,
}

/// Cache for cuDNN algorithm selections, keyed by conv shape signature.
pub struct ConvAlgoCache {
    forward: HashMap<ConvAlgoKey, sys::cudnnConvolutionFwdAlgo_t>,
    backward_data: HashMap<ConvAlgoKey, sys::cudnnConvolutionBwdDataAlgo_t>,
}

impl ConvAlgoCache {
    pub fn new() -> Self {
        Self {
            forward: HashMap::new(),
            backward_data: HashMap::new(),
        }
    }
}

impl Default for ConvAlgoCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Query cuDNN for the best forward convolution algorithm, caching by shape key.
fn find_best_fwd_algo(
    handle: &CudnnHandle,
    input_desc: &TensorDescriptor,
    filter_desc: &FilterDescriptor,
    conv_desc: &ConvolutionDescriptor,
    output_desc: &TensorDescriptor,
    cache: &mut ConvAlgoCache,
    key: &ConvAlgoKey,
) -> Result<sys::cudnnConvolutionFwdAlgo_t, CudaError> {
    if let Some(&algo) = cache.forward.get(key) {
        return Ok(algo);
    }

    const MAX_ALGOS: usize = 8;
    let mut perf_results = [sys::cudnnConvolutionFwdAlgoPerf_t {
        algo: sys::cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
        status: sys::cudnnStatus_t::CUDNN_STATUS_NOT_INITIALIZED,
        time: 0.0,
        memory: 0,
        determinism: sys::cudnnDeterminism_t::CUDNN_NON_DETERMINISTIC,
        mathType: sys::cudnnMathType_t::CUDNN_DEFAULT_MATH,
        reserved: [0; 3],
    }; MAX_ALGOS];
    let mut returned_count: c_int = 0;

    unsafe {
        check_cudnn(sys::cudnnGetConvolutionForwardAlgorithm_v7(
            handle.raw(),
            input_desc.raw(),
            filter_desc.raw(),
            conv_desc.raw(),
            output_desc.raw(),
            MAX_ALGOS as c_int,
            &mut returned_count,
            perf_results.as_mut_ptr(),
        ))?;
    }

    let best = perf_results[..returned_count as usize]
        .iter()
        .find(|r| r.status == sys::cudnnStatus_t::CUDNN_STATUS_SUCCESS)
        .ok_or_else(|| {
            CudaError::Cudnn(
                "cudnnGetConvolutionForwardAlgorithm_v7: no successful algorithms".into(),
            )
        })?;

    let algo = best.algo;
    cache.forward.insert(key.clone(), algo);
    Ok(algo)
}

/// Query cuDNN for the best backward-data convolution algorithm, caching by shape key.
fn find_best_bwd_data_algo(
    handle: &CudnnHandle,
    filter_desc: &FilterDescriptor,
    input_desc: &TensorDescriptor,
    conv_desc: &ConvolutionDescriptor,
    output_desc: &TensorDescriptor,
    cache: &mut ConvAlgoCache,
    key: &ConvAlgoKey,
) -> Result<sys::cudnnConvolutionBwdDataAlgo_t, CudaError> {
    if let Some(&algo) = cache.backward_data.get(key) {
        return Ok(algo);
    }

    const MAX_ALGOS: usize = 6;
    let mut perf_results = [sys::cudnnConvolutionBwdDataAlgoPerf_t {
        algo: sys::cudnnConvolutionBwdDataAlgo_t::CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
        status: sys::cudnnStatus_t::CUDNN_STATUS_NOT_INITIALIZED,
        time: 0.0,
        memory: 0,
        determinism: sys::cudnnDeterminism_t::CUDNN_NON_DETERMINISTIC,
        mathType: sys::cudnnMathType_t::CUDNN_DEFAULT_MATH,
        reserved: [0; 3],
    }; MAX_ALGOS];
    let mut returned_count: c_int = 0;

    unsafe {
        check_cudnn(sys::cudnnGetConvolutionBackwardDataAlgorithm_v7(
            handle.raw(),
            filter_desc.raw(),
            input_desc.raw(),
            conv_desc.raw(),
            output_desc.raw(),
            MAX_ALGOS as c_int,
            &mut returned_count,
            perf_results.as_mut_ptr(),
        ))?;
    }

    let best = perf_results[..returned_count as usize]
        .iter()
        .find(|r| r.status == sys::cudnnStatus_t::CUDNN_STATUS_SUCCESS)
        .ok_or_else(|| {
            CudaError::Cudnn(
                "cudnnGetConvolutionBackwardDataAlgorithm_v7: no successful algorithms".into(),
            )
        })?;

    let algo = best.algo;
    cache.backward_data.insert(key.clone(), algo);
    Ok(algo)
}

// =============================================================================
// High-Level Operations
// =============================================================================

/// Perform 1D ConvTranspose using cuDNN
///
/// This uses cudnnConvolutionBackwardData which implements transposed convolution.
/// Input shape: [batch, in_channels, length]
/// Kernel shape: [in_channels, out_channels/groups, kernel_size]
/// Output shape: [batch, out_channels, output_length]
pub fn cudnn_conv_transpose_1d(
    handle: &CudnnHandle,
    ctx: &IconnxCudaContext,
    workspace: &mut Workspace,
    cache: &mut ConvAlgoCache,
    input: &GpuTensor,
    kernel: &GpuTensor,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<GpuTensor, CudaError> {
    // Validate input
    let input_shape = input.shape();
    let kernel_shape = kernel.shape();

    if input_shape.len() != 3 {
        return Err(CudaError::Cudnn(format!(
            "Expected 3D input [N,C,L], got {:?}",
            input_shape
        )));
    }
    if kernel_shape.len() != 3 {
        return Err(CudaError::Cudnn(format!(
            "Expected 3D kernel [C_in,C_out/g,K], got {:?}",
            kernel_shape
        )));
    }

    let batch = input_shape[0];
    let in_channels = input_shape[1];
    let in_length = input_shape[2];
    let kernel_size = kernel_shape[2];
    let out_channels_per_group = kernel_shape[1];
    let out_channels = out_channels_per_group * groups;

    // Calculate output length
    let out_length =
        (in_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;

    // Create descriptors - treat 1D as 2D with height=1
    let mut input_desc = TensorDescriptor::new()?;
    input_desc.set_4d(
        batch,
        in_channels,
        1,
        in_length,
        cudnnDataType_t::CUDNN_DATA_FLOAT,
    )?;

    let mut output_desc = TensorDescriptor::new()?;
    output_desc.set_4d(
        batch,
        out_channels,
        1,
        out_length,
        cudnnDataType_t::CUDNN_DATA_FLOAT,
    )?;

    // Filter: [in_channels, out_channels/groups, 1, kernel_size]
    let mut filter_desc = FilterDescriptor::new()?;
    filter_desc.set_4d(
        in_channels,
        out_channels_per_group,
        1,
        kernel_size,
        cudnnDataType_t::CUDNN_DATA_FLOAT,
    )?;

    // Convolution descriptor
    let mut conv_desc = ConvolutionDescriptor::new()?;
    conv_desc.set_2d(
        0,
        padding,
        1,
        stride,
        1,
        dilation,
        cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
    )?;
    conv_desc.set_group_count(groups)?;
    conv_desc.set_math_type(true)?; // Enable tensor cores

    // Select best algorithm (cached per shape)
    let key = ConvAlgoKey {
        input_shape: input_shape.to_vec(),
        kernel_shape: kernel_shape.to_vec(),
        stride,
        padding,
        dilation,
        groups,
        dtype: "f32",
    };
    let algo = find_best_bwd_data_algo(
        handle,
        &filter_desc,
        &input_desc,
        &conv_desc,
        &output_desc,
        cache,
        &key,
    )?;

    // Get workspace size
    let mut workspace_size: usize = 0;
    unsafe {
        check_cudnn(sys::cudnnGetConvolutionBackwardDataWorkspaceSize(
            handle.raw(),
            filter_desc.raw(),
            input_desc.raw(),
            conv_desc.raw(),
            output_desc.raw(),
            algo,
            &mut workspace_size,
        ))?;
    }

    // Allocate workspace
    workspace.ensure_size(ctx, workspace_size)?;

    // Allocate output
    let mut output = GpuTensor::zeros_f32(ctx, vec![batch, out_channels, out_length])?;

    // Perform convolution backward data (transposed convolution).
    // Post-garboard: device pointers come straight from garboard slices
    // (plain `CUdeviceptr`); lifetime of the underlying allocations is
    // the function's scope, which covers the synchronous cuDNN call.
    {
        let kernel_ptr = kernel.data_f32()?.as_raw_device_ptr();
        let input_ptr = input.data_f32()?.as_raw_device_ptr();
        let output_ptr = output.data_f32_mut()?.as_raw_device_ptr();
        let workspace_ptr = workspace.ptr();

        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;

        unsafe {
            check_cudnn(sys::cudnnConvolutionBackwardData(
                handle.raw(),
                &alpha as *const f32 as *const std::ffi::c_void,
                filter_desc.raw(),
                kernel_ptr as *const std::ffi::c_void,
                input_desc.raw(),
                input_ptr as *const std::ffi::c_void,
                conv_desc.raw(),
                algo,
                workspace_ptr,
                workspace.size(),
                &beta as *const f32 as *const std::ffi::c_void,
                output_desc.raw(),
                output_ptr as *mut std::ffi::c_void,
            ))?;
        }
    }

    Ok(output)
}

/// Perform 1D Convolution using cuDNN
pub fn cudnn_conv_1d(
    handle: &CudnnHandle,
    ctx: &IconnxCudaContext,
    workspace: &mut Workspace,
    cache: &mut ConvAlgoCache,
    input: &GpuTensor,
    kernel: &GpuTensor,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<GpuTensor, CudaError> {
    // Validate input
    let input_shape = input.shape();
    let kernel_shape = kernel.shape();

    if input_shape.len() != 3 {
        return Err(CudaError::Cudnn(format!(
            "Expected 3D input [N,C,L], got {:?}",
            input_shape
        )));
    }
    if kernel_shape.len() != 3 {
        return Err(CudaError::Cudnn(format!(
            "Expected 3D kernel [C_out,C_in/g,K], got {:?}",
            kernel_shape
        )));
    }

    let batch = input_shape[0];
    let in_channels = input_shape[1];
    let in_length = input_shape[2];
    let out_channels = kernel_shape[0];
    let in_channels_per_group = kernel_shape[1];
    let kernel_size = kernel_shape[2];

    // Calculate output length
    let out_length = (in_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    // Create descriptors - treat 1D as 2D with height=1
    let mut input_desc = TensorDescriptor::new()?;
    input_desc.set_4d(
        batch,
        in_channels,
        1,
        in_length,
        cudnnDataType_t::CUDNN_DATA_FLOAT,
    )?;

    let mut output_desc = TensorDescriptor::new()?;
    output_desc.set_4d(
        batch,
        out_channels,
        1,
        out_length,
        cudnnDataType_t::CUDNN_DATA_FLOAT,
    )?;

    // Filter: [out_channels, in_channels/groups, 1, kernel_size]
    let mut filter_desc = FilterDescriptor::new()?;
    filter_desc.set_4d(
        out_channels,
        in_channels_per_group,
        1,
        kernel_size,
        cudnnDataType_t::CUDNN_DATA_FLOAT,
    )?;

    // Convolution descriptor
    let mut conv_desc = ConvolutionDescriptor::new()?;
    conv_desc.set_2d(
        0,
        padding,
        1,
        stride,
        1,
        dilation,
        cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
    )?;
    conv_desc.set_group_count(groups)?;
    conv_desc.set_math_type(true)?; // Enable tensor cores

    // Select best algorithm (cached per shape)
    let key = ConvAlgoKey {
        input_shape: input_shape.to_vec(),
        kernel_shape: kernel_shape.to_vec(),
        stride,
        padding,
        dilation,
        groups,
        dtype: "f32",
    };
    let algo = find_best_fwd_algo(
        handle,
        &input_desc,
        &filter_desc,
        &conv_desc,
        &output_desc,
        cache,
        &key,
    )?;

    // Get workspace size
    let mut workspace_size: usize = 0;
    unsafe {
        check_cudnn(sys::cudnnGetConvolutionForwardWorkspaceSize(
            handle.raw(),
            input_desc.raw(),
            filter_desc.raw(),
            conv_desc.raw(),
            output_desc.raw(),
            algo,
            &mut workspace_size,
        ))?;
    }

    // Allocate workspace
    workspace.ensure_size(ctx, workspace_size)?;

    // Allocate output
    let mut output = GpuTensor::zeros_f32(ctx, vec![batch, out_channels, out_length])?;

    // Perform convolution forward. Device pointers come from garboard
    // slices; the synchronous cuDNN call completes within this scope.
    {
        let input_ptr = input.data_f32()?.as_raw_device_ptr();
        let kernel_ptr = kernel.data_f32()?.as_raw_device_ptr();
        let output_ptr = output.data_f32_mut()?.as_raw_device_ptr();
        let workspace_ptr = workspace.ptr();

        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;

        unsafe {
            check_cudnn(sys::cudnnConvolutionForward(
                handle.raw(),
                &alpha as *const f32 as *const std::ffi::c_void,
                input_desc.raw(),
                input_ptr as *const std::ffi::c_void,
                filter_desc.raw(),
                kernel_ptr as *const std::ffi::c_void,
                conv_desc.raw(),
                algo,
                workspace_ptr,
                workspace.size(),
                &beta as *const f32 as *const std::ffi::c_void,
                output_desc.raw(),
                output_ptr as *mut std::ffi::c_void,
            ))?;
        }
    }

    Ok(output)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires CUDA GPU with cuDNN"]
    fn test_cudnn_version() {
        let version = CudnnHandle::version();
        println!("cuDNN version: {}", version);
        assert!(version >= 9000, "Expected cuDNN 9.x or later");
    }

    #[test]
    #[ignore = "requires CUDA GPU with cuDNN"]
    fn test_cudnn_handle_creation() {
        let handle = CudnnHandle::new().expect("Failed to create cuDNN handle");
        drop(handle);
        println!("cuDNN handle created and destroyed successfully");
    }

    #[test]
    #[ignore = "requires CUDA GPU with cuDNN"]
    fn test_cudnn_conv_1d() {
        let ctx = IconnxCudaContext::new().expect("Failed to create CUDA context");
        let handle = CudnnHandle::new().expect("Failed to create cuDNN handle");
        handle
            .set_stream(ctx.stream())
            .expect("Failed to set stream");

        let mut workspace = Workspace::new();
        let mut cache = ConvAlgoCache::new();

        // Create test input: [batch=1, channels=4, length=16]
        let input_data: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        let input = GpuTensor::from_host_f32(&ctx, &input_data, vec![1, 4, 16]).unwrap();

        // Create test kernel: [out_channels=8, in_channels=4, kernel_size=3]
        let kernel_data: Vec<f32> = (0..96).map(|i| i as f32 * 0.01).collect();
        let kernel = GpuTensor::from_host_f32(&ctx, &kernel_data, vec![8, 4, 3]).unwrap();

        // Perform convolution
        let output = cudnn_conv_1d(
            &handle,
            &ctx,
            &mut workspace,
            &mut cache,
            &input,
            &kernel,
            1, // stride
            1, // padding
            1, // dilation
            1, // groups
        )
        .expect("Conv1D failed");

        assert_eq!(output.shape(), &[1, 8, 16]);
        println!("Conv1D test passed! Output shape: {:?}", output.shape());
    }

    #[test]
    #[ignore = "requires CUDA GPU with cuDNN"]
    fn test_cudnn_conv_transpose_1d() {
        let ctx = IconnxCudaContext::new().expect("Failed to create CUDA context");
        let handle = CudnnHandle::new().expect("Failed to create cuDNN handle");
        handle
            .set_stream(ctx.stream())
            .expect("Failed to set stream");

        let mut workspace = Workspace::new();
        let mut cache = ConvAlgoCache::new();

        // Create test input: [batch=1, channels=8, length=8]
        let input_data: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        let input = GpuTensor::from_host_f32(&ctx, &input_data, vec![1, 8, 8]).unwrap();

        // Create test kernel: [in_channels=8, out_channels=4, kernel_size=3]
        let kernel_data: Vec<f32> = (0..96).map(|i| i as f32 * 0.01).collect();
        let kernel = GpuTensor::from_host_f32(&ctx, &kernel_data, vec![8, 4, 3]).unwrap();

        // Perform transposed convolution
        let output = cudnn_conv_transpose_1d(
            &handle,
            &ctx,
            &mut workspace,
            &mut cache,
            &input,
            &kernel,
            2, // stride
            1, // padding
            0, // output_padding
            1, // dilation
            1, // groups
        )
        .expect("ConvTranspose1D failed");

        // Output length = (8 - 1) * 2 - 2 * 1 + 1 * (3 - 1) + 0 + 1 = 15
        assert_eq!(output.shape(), &[1, 4, 15]);
        println!(
            "ConvTranspose1D test passed! Output shape: {:?}",
            output.shape()
        );
    }
}
