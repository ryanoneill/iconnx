//! cuDNN RNN (LSTM) wrappers for fused sequence-level forward pass
//!
//! Provides safe Rust wrappers around cuDNN's legacy RNN API (v8 descriptor)
//! for LSTM inference. Instead of per-timestep cuBLAS GEMMs + custom CUDA
//! kernels, this module uses a single cudnnRNNForward call per LSTM node.

#![allow(clippy::too_many_arguments)]

use std::ffi::c_void;
use std::ptr;

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use super::sys;
use super::{check_cudnn, CudnnHandle, TensorDescriptor};
use crate::cuda::context::{CudaError, IconnxCudaContext};
use crate::cuda::tensor::GpuTensor;

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

// =============================================================================
// Packed LSTM Weights
// =============================================================================

/// GPU buffer containing LSTM weights repacked from ONNX gate order (IOFC) to
/// cuDNN gate order (IFCO).
///
/// Created by `pack_lstm_weights_for_cudnn` and consumed by `cudnn_lstm_forward`.
/// The buffer layout is opaque -- only cuDNN knows the internal offsets. We
/// query offsets via `cudnnGetRNNWeightParams` during packing.
///
/// **Cache pointer invariant:** This struct is valid only while the original
/// ONNX W/R tensors (whose data was copied into this buffer) remain alive.
/// In practice the executor owns both the weight tensors and the cache, so
/// this invariant is trivially satisfied.
pub struct PackedLstmWeights {
    /// Packed weight buffer on GPU
    buffer: CudaSlice<u8>,
    /// Size in bytes
    size: usize,
    /// RNN descriptor that was used to create this packing
    /// (must outlive any cudnnRNNForward call using this buffer)
    rnn_desc: RnnDescriptor,
}

impl PackedLstmWeights {
    /// Size of the packed weight buffer in bytes
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get the RNN descriptor
    pub(crate) fn rnn_desc(&self) -> &RnnDescriptor {
        &self.rnn_desc
    }
}

/// Maps ONNX LSTM gate index to cuDNN linLayerID for input weights (W).
///
/// ONNX gate order: I=0, O=1, F=2, C=3
/// cuDNN linLayerID for W: I=0, F=1, G(=C)=2, O=3
///
/// So: ONNX_gate[i] -> cuDNN linLayerID ONNX_TO_CUDNN_GATE[i]
const ONNX_TO_CUDNN_GATE: [i32; 4] = [
    0, // ONNX I -> cuDNN 0 (W_i)
    3, // ONNX O -> cuDNN 3 (W_o)
    1, // ONNX F -> cuDNN 1 (W_f)
    2, // ONNX C -> cuDNN 2 (W_g)
];

/// Repack ONNX LSTM weights into cuDNN's packed weight buffer format.
///
/// ONNX stores weights in IOFC gate order; cuDNN expects IFCO. This function:
/// 1. Creates an RNN descriptor for the given LSTM configuration
/// 2. Queries cuDNN for the packed weight buffer size
/// 3. Allocates the buffer on GPU
/// 4. For each direction and each gate, queries cuDNN for the offset within
///    the buffer and copies the corresponding ONNX weight slice there
///
/// The returned `PackedLstmWeights` owns both the GPU buffer and the RNN
/// descriptor. Both must outlive any `cudnn_lstm_forward` call.
pub fn pack_lstm_weights_for_cudnn(
    handle: &CudnnHandle,
    ctx: &IconnxCudaContext,
    w: &GpuTensor,         // [num_directions, 4*hidden, input_size]
    r: &GpuTensor,         // [num_directions, 4*hidden, hidden_size]
    b: Option<&GpuTensor>, // [num_directions, 8*hidden]
    hidden_size: usize,
    input_size: usize,
    num_directions: usize,
) -> Result<PackedLstmWeights, CudaError> {
    // Create RNN descriptor
    let rnn_desc = RnnDescriptor::new_lstm(handle, ctx, hidden_size, input_size, num_directions)?;

    // Query packed weight buffer size
    let mut weight_space_size: usize = 0;
    unsafe {
        check_cudnn(sys::cudnnGetRNNWeightSpaceSize(
            handle.raw(),
            rnn_desc.raw(),
            &mut weight_space_size,
        ))?;
    }

    // Allocate packed weight buffer (zero-initialized for safety)
    let mut weight_buffer = ctx.alloc_zeros::<u8>(weight_space_size)?;

    // Create temporary tensor descriptors for cudnnGetRNNWeightParams output
    let m_desc = TensorDescriptor::new()?;
    let b_desc = TensorDescriptor::new()?;

    let stream = ctx.stream();

    for dir in 0..num_directions {
        // pseudoLayer: 0 for forward direction, 1 for backward direction
        let pseudo_layer = dir as i32;

        // Process input weights (W): linLayerID 0-3
        for (onnx_gate, &cudnn_lin_id) in ONNX_TO_CUDNN_GATE.iter().enumerate() {

            // Query cuDNN for the destination offset within packed buffer
            let mut m_addr: *mut c_void = ptr::null_mut();
            let mut b_addr: *mut c_void = ptr::null_mut();

            let (wb_ptr, _wb_guard) = weight_buffer.device_ptr_mut(stream);
            unsafe {
                check_cudnn(sys::cudnnGetRNNWeightParams(
                    handle.raw(),
                    rnn_desc.raw(),
                    pseudo_layer,
                    weight_space_size,
                    wb_ptr as *const c_void,
                    cudnn_lin_id,
                    m_desc.raw(),
                    &mut m_addr,
                    b_desc.raw(),
                    &mut b_addr,
                ))?;
            }
            // Drop the mutable guard before borrowing source tensors
            drop(_wb_guard);

            // Copy input weight matrix for this gate
            // Source: W[dir, onnx_gate*hidden..(onnx_gate+1)*hidden, 0..input_size]
            // This is a contiguous block of hidden_size * input_size floats
            let w_offset_floats = dir * 4 * hidden_size * input_size
                + onnx_gate * hidden_size * input_size;
            let w_size_bytes = hidden_size * input_size * std::mem::size_of::<f32>();

            let (w_ptr, _w_guard) = w.data_f32()?.device_ptr(stream);
            let src_ptr = (w_ptr as usize + w_offset_floats * std::mem::size_of::<f32>())
                as cudarc::driver::sys::CUdeviceptr;

            unsafe {
                let status = cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                    m_addr as cudarc::driver::sys::CUdeviceptr,
                    src_ptr,
                    w_size_bytes,
                    stream.cu_stream(),
                );
                if status != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                    return Err(CudaError::Transfer(format!(
                        "cuMemcpyDtoDAsync failed for W gate {}: {:?}",
                        onnx_gate, status
                    )));
                }
            }
        }

        // Process recurrence weights (R): linLayerID 4-7
        for (onnx_gate, &cudnn_base) in ONNX_TO_CUDNN_GATE.iter().enumerate() {
            let cudnn_lin_id = cudnn_base + 4;

            let mut m_addr: *mut c_void = ptr::null_mut();
            let mut b_addr: *mut c_void = ptr::null_mut();

            let (wb_ptr, _wb_guard) = weight_buffer.device_ptr_mut(stream);
            unsafe {
                check_cudnn(sys::cudnnGetRNNWeightParams(
                    handle.raw(),
                    rnn_desc.raw(),
                    pseudo_layer,
                    weight_space_size,
                    wb_ptr as *const c_void,
                    cudnn_lin_id,
                    m_desc.raw(),
                    &mut m_addr,
                    b_desc.raw(),
                    &mut b_addr,
                ))?;
            }
            drop(_wb_guard);

            // Copy recurrence weight matrix for this gate
            // Source: R[dir, onnx_gate*hidden..(onnx_gate+1)*hidden, 0..hidden_size]
            let r_offset_floats = dir * 4 * hidden_size * hidden_size
                + onnx_gate * hidden_size * hidden_size;
            let r_size_bytes = hidden_size * hidden_size * std::mem::size_of::<f32>();

            let (r_ptr, _r_guard) = r.data_f32()?.device_ptr(stream);
            let src_ptr = (r_ptr as usize + r_offset_floats * std::mem::size_of::<f32>())
                as cudarc::driver::sys::CUdeviceptr;

            unsafe {
                let status = cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                    m_addr as cudarc::driver::sys::CUdeviceptr,
                    src_ptr,
                    r_size_bytes,
                    stream.cu_stream(),
                );
                if status != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                    return Err(CudaError::Transfer(format!(
                        "cuMemcpyDtoDAsync failed for R gate {}: {:?}",
                        onnx_gate, status
                    )));
                }
            }
        }

        // Process biases if provided
        // ONNX B layout: [Wb_i, Wb_o, Wb_f, Wb_c, Rb_i, Rb_o, Rb_f, Rb_c]
        // = 8*hidden values per direction
        // cuDNN: each linLayerID 0-7 has its own bias vector of size hidden
        if let Some(b_tensor) = b {
            for (onnx_gate, &cudnn_base) in ONNX_TO_CUDNN_GATE.iter().enumerate() {
                let bias_size_bytes = hidden_size * std::mem::size_of::<f32>();

                // Input bias (Wb): linLayerID 0-3
                {
                    let cudnn_lin_id = cudnn_base;

                    let mut m_addr: *mut c_void = ptr::null_mut();
                    let mut b_addr: *mut c_void = ptr::null_mut();

                    let (wb_ptr, _wb_guard) = weight_buffer.device_ptr_mut(stream);
                    unsafe {
                        check_cudnn(sys::cudnnGetRNNWeightParams(
                            handle.raw(),
                            rnn_desc.raw(),
                            pseudo_layer,
                            weight_space_size,
                            wb_ptr as *const c_void,
                            cudnn_lin_id,
                            m_desc.raw(),
                            &mut m_addr,
                            b_desc.raw(),
                            &mut b_addr,
                        ))?;
                    }
                    drop(_wb_guard);

                    // Wb[gate]: offset = dir * 8*hidden + onnx_gate * hidden
                    let wb_offset_floats = dir * 8 * hidden_size + onnx_gate * hidden_size;

                    let (b_ptr, _b_guard) = b_tensor.data_f32()?.device_ptr(stream);
                    let src_ptr = (b_ptr as usize
                        + wb_offset_floats * std::mem::size_of::<f32>())
                        as cudarc::driver::sys::CUdeviceptr;

                    unsafe {
                        let status = cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                            b_addr as cudarc::driver::sys::CUdeviceptr,
                            src_ptr,
                            bias_size_bytes,
                            stream.cu_stream(),
                        );
                        if status != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                            return Err(CudaError::Transfer(format!(
                                "cuMemcpyDtoDAsync failed for Wb gate {}: {:?}",
                                onnx_gate, status
                            )));
                        }
                    }
                }

                // Recurrence bias (Rb): linLayerID 4-7
                {
                    let cudnn_lin_id_r = cudnn_base + 4;

                    let mut m_addr_r: *mut c_void = ptr::null_mut();
                    let mut b_addr_r: *mut c_void = ptr::null_mut();

                    let (wb_ptr, _wb_guard) = weight_buffer.device_ptr_mut(stream);
                    unsafe {
                        check_cudnn(sys::cudnnGetRNNWeightParams(
                            handle.raw(),
                            rnn_desc.raw(),
                            pseudo_layer,
                            weight_space_size,
                            wb_ptr as *const c_void,
                            cudnn_lin_id_r,
                            m_desc.raw(),
                            &mut m_addr_r,
                            b_desc.raw(),
                            &mut b_addr_r,
                        ))?;
                    }
                    drop(_wb_guard);

                    // Rb[gate]: offset = dir * 8*hidden + (4 + onnx_gate) * hidden
                    let rb_offset_floats =
                        dir * 8 * hidden_size + (4 + onnx_gate) * hidden_size;

                    let (b_ptr, _b_guard) = b_tensor.data_f32()?.device_ptr(stream);
                    let src_ptr_r = (b_ptr as usize
                        + rb_offset_floats * std::mem::size_of::<f32>())
                        as cudarc::driver::sys::CUdeviceptr;

                    unsafe {
                        let status = cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                            b_addr_r as cudarc::driver::sys::CUdeviceptr,
                            src_ptr_r,
                            bias_size_bytes,
                            stream.cu_stream(),
                        );
                        if status != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                            return Err(CudaError::Transfer(format!(
                                "cuMemcpyDtoDAsync failed for Rb gate {}: {:?}",
                                onnx_gate, status
                            )));
                        }
                    }
                }
            }
        }
    }

    Ok(PackedLstmWeights {
        buffer: weight_buffer,
        size: weight_space_size,
        rnn_desc,
    })
}
