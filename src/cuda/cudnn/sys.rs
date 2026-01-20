//! Low-level FFI bindings to cuDNN
//!
//! These bindings cover the subset of cuDNN we need for Kokoro TTS inference:
//! - Convolution (forward and backward data for ConvTranspose)
//! - Tensor operations
//!
//! Based on cuDNN 9.x API.

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

use std::ffi::c_void;
use std::os::raw::c_int;

// =============================================================================
// Types
// =============================================================================

/// cuDNN status codes
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cudnnStatus_t {
    CUDNN_STATUS_SUCCESS = 0,
    CUDNN_STATUS_NOT_INITIALIZED = 1,
    CUDNN_STATUS_ALLOC_FAILED = 2,
    CUDNN_STATUS_BAD_PARAM = 3,
    CUDNN_STATUS_INTERNAL_ERROR = 4,
    CUDNN_STATUS_INVALID_VALUE = 5,
    CUDNN_STATUS_ARCH_MISMATCH = 6,
    CUDNN_STATUS_MAPPING_ERROR = 7,
    CUDNN_STATUS_EXECUTION_FAILED = 8,
    CUDNN_STATUS_NOT_SUPPORTED = 9,
    CUDNN_STATUS_LICENSE_ERROR = 10,
    CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING = 11,
    CUDNN_STATUS_RUNTIME_IN_PROGRESS = 12,
    CUDNN_STATUS_RUNTIME_FP_OVERFLOW = 13,
}

impl cudnnStatus_t {
    pub fn is_success(self) -> bool {
        self == cudnnStatus_t::CUDNN_STATUS_SUCCESS
    }
}

/// cuDNN data types
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cudnnDataType_t {
    CUDNN_DATA_FLOAT = 0,
    CUDNN_DATA_DOUBLE = 1,
    CUDNN_DATA_HALF = 2,
    CUDNN_DATA_INT8 = 3,
    CUDNN_DATA_INT32 = 4,
    CUDNN_DATA_INT8x4 = 5,
    CUDNN_DATA_UINT8 = 6,
    CUDNN_DATA_UINT8x4 = 7,
    CUDNN_DATA_INT8x32 = 8,
    CUDNN_DATA_BFLOAT16 = 9,
    CUDNN_DATA_INT64 = 10,
    CUDNN_DATA_BOOLEAN = 11,
}

/// Tensor format
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cudnnTensorFormat_t {
    CUDNN_TENSOR_NCHW = 0,
    CUDNN_TENSOR_NHWC = 1,
    CUDNN_TENSOR_NCHW_VECT_C = 2,
}

/// Convolution mode
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cudnnConvolutionMode_t {
    CUDNN_CONVOLUTION = 0,
    CUDNN_CROSS_CORRELATION = 1,
}

/// Convolution forward algorithm
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cudnnConvolutionFwdAlgo_t {
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = 0,
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = 1,
    CUDNN_CONVOLUTION_FWD_ALGO_GEMM = 2,
    CUDNN_CONVOLUTION_FWD_ALGO_DIRECT = 3,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT = 4,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING = 5,
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD = 6,
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED = 7,
}

/// Convolution backward data algorithm (used for ConvTranspose)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cudnnConvolutionBwdDataAlgo_t {
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 = 0,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_1 = 1,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT = 2,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING = 3,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD = 4,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED = 5,
}

/// Math type for tensor cores
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cudnnMathType_t {
    CUDNN_DEFAULT_MATH = 0,
    CUDNN_TENSOR_OP_MATH = 1,
    CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION = 2,
    CUDNN_FMA_MATH = 3,
}

/// NaN propagation
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cudnnNanPropagation_t {
    CUDNN_NOT_PROPAGATE_NAN = 0,
    CUDNN_PROPAGATE_NAN = 1,
}

/// Activation mode
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cudnnActivationMode_t {
    CUDNN_ACTIVATION_SIGMOID = 0,
    CUDNN_ACTIVATION_RELU = 1,
    CUDNN_ACTIVATION_TANH = 2,
    CUDNN_ACTIVATION_CLIPPED_RELU = 3,
    CUDNN_ACTIVATION_ELU = 4,
    CUDNN_ACTIVATION_IDENTITY = 5,
    CUDNN_ACTIVATION_SWISH = 6,
}

// =============================================================================
// Opaque Handle Types
// =============================================================================

/// cuDNN library context handle
#[repr(C)]
pub struct cudnnContext {
    _private: [u8; 0],
}
pub type cudnnHandle_t = *mut cudnnContext;

/// Tensor descriptor
#[repr(C)]
pub struct cudnnTensorStruct {
    _private: [u8; 0],
}
pub type cudnnTensorDescriptor_t = *mut cudnnTensorStruct;

/// Filter descriptor
#[repr(C)]
pub struct cudnnFilterStruct {
    _private: [u8; 0],
}
pub type cudnnFilterDescriptor_t = *mut cudnnFilterStruct;

/// Convolution descriptor
#[repr(C)]
pub struct cudnnConvolutionStruct {
    _private: [u8; 0],
}
pub type cudnnConvolutionDescriptor_t = *mut cudnnConvolutionStruct;

/// Activation descriptor
#[repr(C)]
pub struct cudnnActivationStruct {
    _private: [u8; 0],
}
pub type cudnnActivationDescriptor_t = *mut cudnnActivationStruct;

// CUDA stream type (from CUDA driver API)
pub type cudaStream_t = *mut c_void;

// =============================================================================
// External Functions
// =============================================================================

#[link(name = "cudnn")]
extern "C" {
    // -------------------------------------------------------------------------
    // Handle Management
    // -------------------------------------------------------------------------

    /// Create a cuDNN handle
    pub fn cudnnCreate(handle: *mut cudnnHandle_t) -> cudnnStatus_t;

    /// Destroy a cuDNN handle
    pub fn cudnnDestroy(handle: cudnnHandle_t) -> cudnnStatus_t;

    /// Set the stream for a cuDNN handle
    pub fn cudnnSetStream(handle: cudnnHandle_t, stream: cudaStream_t) -> cudnnStatus_t;

    /// Get the stream for a cuDNN handle
    pub fn cudnnGetStream(handle: cudnnHandle_t, stream: *mut cudaStream_t) -> cudnnStatus_t;

    /// Get cuDNN version
    pub fn cudnnGetVersion() -> usize;

    // -------------------------------------------------------------------------
    // Tensor Descriptor
    // -------------------------------------------------------------------------

    /// Create a tensor descriptor
    pub fn cudnnCreateTensorDescriptor(tensorDesc: *mut cudnnTensorDescriptor_t) -> cudnnStatus_t;

    /// Destroy a tensor descriptor
    pub fn cudnnDestroyTensorDescriptor(tensorDesc: cudnnTensorDescriptor_t) -> cudnnStatus_t;

    /// Set 4D tensor descriptor
    pub fn cudnnSetTensor4dDescriptor(
        tensorDesc: cudnnTensorDescriptor_t,
        format: cudnnTensorFormat_t,
        dataType: cudnnDataType_t,
        n: c_int, // batch
        c: c_int, // channels
        h: c_int, // height
        w: c_int, // width
    ) -> cudnnStatus_t;

    /// Set ND tensor descriptor with strides
    pub fn cudnnSetTensorNdDescriptor(
        tensorDesc: cudnnTensorDescriptor_t,
        dataType: cudnnDataType_t,
        nbDims: c_int,
        dimA: *const c_int,
        strideA: *const c_int,
    ) -> cudnnStatus_t;

    // -------------------------------------------------------------------------
    // Filter Descriptor
    // -------------------------------------------------------------------------

    /// Create a filter descriptor
    pub fn cudnnCreateFilterDescriptor(filterDesc: *mut cudnnFilterDescriptor_t) -> cudnnStatus_t;

    /// Destroy a filter descriptor
    pub fn cudnnDestroyFilterDescriptor(filterDesc: cudnnFilterDescriptor_t) -> cudnnStatus_t;

    /// Set 4D filter descriptor
    pub fn cudnnSetFilter4dDescriptor(
        filterDesc: cudnnFilterDescriptor_t,
        dataType: cudnnDataType_t,
        format: cudnnTensorFormat_t,
        k: c_int, // output channels
        c: c_int, // input channels
        h: c_int, // height
        w: c_int, // width
    ) -> cudnnStatus_t;

    /// Set ND filter descriptor
    pub fn cudnnSetFilterNdDescriptor(
        filterDesc: cudnnFilterDescriptor_t,
        dataType: cudnnDataType_t,
        format: cudnnTensorFormat_t,
        nbDims: c_int,
        filterDimA: *const c_int,
    ) -> cudnnStatus_t;

    // -------------------------------------------------------------------------
    // Convolution Descriptor
    // -------------------------------------------------------------------------

    /// Create a convolution descriptor
    pub fn cudnnCreateConvolutionDescriptor(
        convDesc: *mut cudnnConvolutionDescriptor_t,
    ) -> cudnnStatus_t;

    /// Destroy a convolution descriptor
    pub fn cudnnDestroyConvolutionDescriptor(
        convDesc: cudnnConvolutionDescriptor_t,
    ) -> cudnnStatus_t;

    /// Set 2D convolution descriptor
    pub fn cudnnSetConvolution2dDescriptor(
        convDesc: cudnnConvolutionDescriptor_t,
        pad_h: c_int,
        pad_w: c_int,
        u: c_int, // vertical stride
        v: c_int, // horizontal stride
        dilation_h: c_int,
        dilation_w: c_int,
        mode: cudnnConvolutionMode_t,
        computeType: cudnnDataType_t,
    ) -> cudnnStatus_t;

    /// Set ND convolution descriptor
    pub fn cudnnSetConvolutionNdDescriptor(
        convDesc: cudnnConvolutionDescriptor_t,
        arrayLength: c_int,
        padA: *const c_int,
        filterStrideA: *const c_int,
        dilationA: *const c_int,
        mode: cudnnConvolutionMode_t,
        computeType: cudnnDataType_t,
    ) -> cudnnStatus_t;

    /// Set convolution group count
    pub fn cudnnSetConvolutionGroupCount(
        convDesc: cudnnConvolutionDescriptor_t,
        groupCount: c_int,
    ) -> cudnnStatus_t;

    /// Set convolution math type (for tensor cores)
    pub fn cudnnSetConvolutionMathType(
        convDesc: cudnnConvolutionDescriptor_t,
        mathType: cudnnMathType_t,
    ) -> cudnnStatus_t;

    // -------------------------------------------------------------------------
    // Convolution Forward
    // -------------------------------------------------------------------------

    /// Get output dimensions for convolution forward
    pub fn cudnnGetConvolutionNdForwardOutputDim(
        convDesc: cudnnConvolutionDescriptor_t,
        inputTensorDesc: cudnnTensorDescriptor_t,
        filterDesc: cudnnFilterDescriptor_t,
        nbDims: c_int,
        tensorOuputDimA: *mut c_int,
    ) -> cudnnStatus_t;

    /// Get workspace size for convolution forward
    pub fn cudnnGetConvolutionForwardWorkspaceSize(
        handle: cudnnHandle_t,
        xDesc: cudnnTensorDescriptor_t,
        wDesc: cudnnFilterDescriptor_t,
        convDesc: cudnnConvolutionDescriptor_t,
        yDesc: cudnnTensorDescriptor_t,
        algo: cudnnConvolutionFwdAlgo_t,
        sizeInBytes: *mut usize,
    ) -> cudnnStatus_t;

    /// Perform convolution forward
    pub fn cudnnConvolutionForward(
        handle: cudnnHandle_t,
        alpha: *const c_void,
        xDesc: cudnnTensorDescriptor_t,
        x: *const c_void,
        wDesc: cudnnFilterDescriptor_t,
        w: *const c_void,
        convDesc: cudnnConvolutionDescriptor_t,
        algo: cudnnConvolutionFwdAlgo_t,
        workSpace: *mut c_void,
        workSpaceSizeInBytes: usize,
        beta: *const c_void,
        yDesc: cudnnTensorDescriptor_t,
        y: *mut c_void,
    ) -> cudnnStatus_t;

    // -------------------------------------------------------------------------
    // Convolution Backward Data (used for ConvTranspose)
    // -------------------------------------------------------------------------

    /// Get workspace size for backward data
    pub fn cudnnGetConvolutionBackwardDataWorkspaceSize(
        handle: cudnnHandle_t,
        wDesc: cudnnFilterDescriptor_t,
        dyDesc: cudnnTensorDescriptor_t,
        convDesc: cudnnConvolutionDescriptor_t,
        dxDesc: cudnnTensorDescriptor_t,
        algo: cudnnConvolutionBwdDataAlgo_t,
        sizeInBytes: *mut usize,
    ) -> cudnnStatus_t;

    /// Perform convolution backward data (ConvTranspose)
    pub fn cudnnConvolutionBackwardData(
        handle: cudnnHandle_t,
        alpha: *const c_void,
        wDesc: cudnnFilterDescriptor_t,
        w: *const c_void,
        dyDesc: cudnnTensorDescriptor_t,
        dy: *const c_void,
        convDesc: cudnnConvolutionDescriptor_t,
        algo: cudnnConvolutionBwdDataAlgo_t,
        workSpace: *mut c_void,
        workSpaceSizeInBytes: usize,
        beta: *const c_void,
        dxDesc: cudnnTensorDescriptor_t,
        dx: *mut c_void,
    ) -> cudnnStatus_t;

    // -------------------------------------------------------------------------
    // Tensor Operations
    // -------------------------------------------------------------------------

    /// Add tensors: y = alpha * x + beta * y
    pub fn cudnnAddTensor(
        handle: cudnnHandle_t,
        alpha: *const c_void,
        aDesc: cudnnTensorDescriptor_t,
        A: *const c_void,
        beta: *const c_void,
        cDesc: cudnnTensorDescriptor_t,
        C: *mut c_void,
    ) -> cudnnStatus_t;

    // -------------------------------------------------------------------------
    // Activation
    // -------------------------------------------------------------------------

    /// Create activation descriptor
    pub fn cudnnCreateActivationDescriptor(
        activationDesc: *mut cudnnActivationDescriptor_t,
    ) -> cudnnStatus_t;

    /// Destroy activation descriptor
    pub fn cudnnDestroyActivationDescriptor(
        activationDesc: cudnnActivationDescriptor_t,
    ) -> cudnnStatus_t;

    /// Set activation descriptor
    pub fn cudnnSetActivationDescriptor(
        activationDesc: cudnnActivationDescriptor_t,
        mode: cudnnActivationMode_t,
        reluNanOpt: cudnnNanPropagation_t,
        coef: f64,
    ) -> cudnnStatus_t;

    /// Activation forward
    pub fn cudnnActivationForward(
        handle: cudnnHandle_t,
        activationDesc: cudnnActivationDescriptor_t,
        alpha: *const c_void,
        xDesc: cudnnTensorDescriptor_t,
        x: *const c_void,
        beta: *const c_void,
        yDesc: cudnnTensorDescriptor_t,
        y: *mut c_void,
    ) -> cudnnStatus_t;
}

// =============================================================================
// Helper Functions
// =============================================================================

impl std::fmt::Display for cudnnStatus_t {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let msg = match self {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => "Success",
            cudnnStatus_t::CUDNN_STATUS_NOT_INITIALIZED => "cuDNN not initialized",
            cudnnStatus_t::CUDNN_STATUS_ALLOC_FAILED => "Allocation failed",
            cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => "Bad parameter",
            cudnnStatus_t::CUDNN_STATUS_INTERNAL_ERROR => "Internal error",
            cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE => "Invalid value",
            cudnnStatus_t::CUDNN_STATUS_ARCH_MISMATCH => "Architecture mismatch",
            cudnnStatus_t::CUDNN_STATUS_MAPPING_ERROR => "Mapping error",
            cudnnStatus_t::CUDNN_STATUS_EXECUTION_FAILED => "Execution failed",
            cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED => "Not supported",
            cudnnStatus_t::CUDNN_STATUS_LICENSE_ERROR => "License error",
            cudnnStatus_t::CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING => {
                "Runtime prerequisite missing"
            }
            cudnnStatus_t::CUDNN_STATUS_RUNTIME_IN_PROGRESS => "Runtime in progress",
            cudnnStatus_t::CUDNN_STATUS_RUNTIME_FP_OVERFLOW => "FP overflow",
        };
        write!(f, "{}", msg)
    }
}
