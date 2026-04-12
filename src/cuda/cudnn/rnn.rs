//! cuDNN RNN (LSTM) wrappers for fused sequence-level forward pass
//!
//! Provides safe Rust wrappers around cuDNN's legacy RNN API (v8 descriptor)
//! for LSTM inference. Instead of per-timestep cuBLAS GEMMs + custom CUDA
//! kernels, this module uses a single cudnnRNNForward call per LSTM node.

#![allow(clippy::too_many_arguments)]

use std::ffi::c_void;
use std::ptr;

use cudarc::driver::{CudaSlice, DevicePtrMut};

use super::sys;
use super::{check_cudnn, CudnnHandle};
use crate::cuda::context::{CudaError, IconnxCudaContext};

// =============================================================================
// DropoutDescriptor
// =============================================================================

/// RAII wrapper around cudnnDropoutDescriptor_t.
/// Always configured with dropout rate 0.0 for inference.
pub struct DropoutDescriptor {
    desc: sys::cudnnDropoutDescriptor_t,
    _states: CudaSlice<u8>,
}

impl DropoutDescriptor {
    pub fn new(handle: &CudnnHandle, ctx: &IconnxCudaContext) -> Result<Self, CudaError> {
        let mut desc: sys::cudnnDropoutDescriptor_t = ptr::null_mut();
        unsafe {
            check_cudnn(sys::cudnnCreateDropoutDescriptor(&mut desc))?;
        }

        let mut state_size: usize = 0;
        unsafe {
            check_cudnn(sys::cudnnDropoutGetStatesSize(
                handle.raw(),
                &mut state_size,
            ))?;
        }

        let mut states = ctx.alloc_zeros::<u8>(state_size.max(1))?;

        {
            let stream = ctx.stream();
            let (states_ptr, _guard) = states.device_ptr_mut(stream);
            unsafe {
                check_cudnn(sys::cudnnSetDropoutDescriptor(
                    desc,
                    handle.raw(),
                    0.0,
                    states_ptr as *mut c_void,
                    state_size,
                    0,
                ))?;
            }
        }

        Ok(Self { desc, _states: states })
    }

    pub(crate) fn raw(&self) -> sys::cudnnDropoutDescriptor_t {
        self.desc
    }
}

impl Drop for DropoutDescriptor {
    fn drop(&mut self) {
        unsafe {
            sys::cudnnDestroyDropoutDescriptor(self.desc);
        }
    }
}

// =============================================================================
// RnnDescriptor
// =============================================================================

/// RAII wrapper around cudnnRNNDescriptor_t, configured for LSTM inference.
pub struct RnnDescriptor {
    desc: sys::cudnnRNNDescriptor_t,
    _dropout: DropoutDescriptor,
}

impl RnnDescriptor {
    pub fn new_lstm(
        handle: &CudnnHandle,
        ctx: &IconnxCudaContext,
        hidden_size: usize,
        input_size: usize,
        num_directions: usize,
    ) -> Result<Self, CudaError> {
        let mut desc: sys::cudnnRNNDescriptor_t = ptr::null_mut();
        unsafe {
            check_cudnn(sys::cudnnCreateRNNDescriptor(&mut desc))?;
        }

        let dropout = DropoutDescriptor::new(handle, ctx)?;

        let dir_mode = if num_directions == 2 {
            sys::cudnnDirectionMode_t::CUDNN_BIDIRECTIONAL
        } else {
            sys::cudnnDirectionMode_t::CUDNN_UNIDIRECTIONAL
        };

        unsafe {
            check_cudnn(sys::cudnnSetRNNDescriptor_v8(
                desc,
                sys::cudnnRNNAlgo_t::CUDNN_RNN_ALGO_STANDARD,
                sys::cudnnRNNMode_t::CUDNN_LSTM,
                sys::cudnnRNNBiasMode_t::CUDNN_RNN_DOUBLE_BIAS,
                dir_mode,
                sys::cudnnRNNInputMode_t::CUDNN_LINEAR_INPUT,
                sys::cudnnDataType_t::CUDNN_DATA_FLOAT,
                sys::cudnnDataType_t::CUDNN_DATA_FLOAT,
                sys::cudnnMathType_t::CUDNN_DEFAULT_MATH,
                input_size as i32,
                hidden_size as i32,
                0,  // projSize (no projection)
                1,  // numLayers
                dropout.raw(),
                1,  // auxFlags: CUDNN_RNN_PADDED_IO_ENABLED
            ))?;
        }

        Ok(Self { desc, _dropout: dropout })
    }

    pub(crate) fn raw(&self) -> sys::cudnnRNNDescriptor_t {
        self.desc
    }
}

impl Drop for RnnDescriptor {
    fn drop(&mut self) {
        unsafe {
            sys::cudnnDestroyRNNDescriptor(self.desc);
        }
    }
}

// =============================================================================
// RnnDataDescriptor
// =============================================================================

/// RAII wrapper around cudnnRNNDataDescriptor_t.
pub struct RnnDataDescriptor {
    desc: sys::cudnnRNNDataDescriptor_t,
}

impl RnnDataDescriptor {
    pub fn new(seq_len: usize, batch_size: usize, vector_size: usize) -> Result<Self, CudaError> {
        let mut desc: sys::cudnnRNNDataDescriptor_t = ptr::null_mut();
        unsafe {
            check_cudnn(sys::cudnnCreateRNNDataDescriptor(&mut desc))?;
        }

        let seq_lengths: Vec<i32> = vec![seq_len as i32; batch_size];

        unsafe {
            check_cudnn(sys::cudnnSetRNNDataDescriptor(
                desc,
                sys::cudnnDataType_t::CUDNN_DATA_FLOAT,
                sys::cudnnRNNDataLayout_t::CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED,
                seq_len as i32,
                batch_size as i32,
                vector_size as i32,
                seq_lengths.as_ptr(),
                ptr::null_mut(),
            ))?;
        }

        Ok(Self { desc })
    }

    pub(crate) fn raw(&self) -> sys::cudnnRNNDataDescriptor_t {
        self.desc
    }
}

impl Drop for RnnDataDescriptor {
    fn drop(&mut self) {
        unsafe {
            sys::cudnnDestroyRNNDataDescriptor(self.desc);
        }
    }
}
