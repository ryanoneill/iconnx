//! cuDNN RNN (LSTM) wrappers for fused sequence-level forward pass
//!
//! Provides safe Rust wrappers around cuDNN's legacy RNN API (v8 descriptor)
//! for LSTM inference. Instead of per-timestep cuBLAS GEMMs + custom CUDA
//! kernels, this module uses a single cudnnRNNForward call per LSTM node.

#![allow(clippy::too_many_arguments)]

use std::ffi::c_void;
use std::ptr;

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};

use super::sys;
use super::{check_cudnn, CudnnHandle, TensorDescriptor, Workspace};
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

// cuDNN descriptors contain raw pointers that are not auto-Send/Sync.
// cuDNN handles are thread-safe when used with proper stream synchronization,
// which IconnxCudaContext enforces. Same reasoning as CudnnHandle's impls in mod.rs.
unsafe impl Send for DropoutDescriptor {}
unsafe impl Sync for DropoutDescriptor {}

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
                hidden_size as i32, // projSize: cuDNN 9 requires hiddenSize when no projection
                1,                  // numLayers
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

unsafe impl Send for RnnDescriptor {}
unsafe impl Sync for RnnDescriptor {}

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

unsafe impl Send for RnnDataDescriptor {}
unsafe impl Sync for RnnDataDescriptor {}

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

    /// Get the device pointer and guard for the packed weight buffer
    pub(crate) fn buffer_device_ptr<'a>(
        &'a self,
        stream: &'a CudaStream,
    ) -> (cudarc::driver::sys::CUdeviceptr, impl Drop + 'a) {
        self.buffer.device_ptr(stream)
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

// =============================================================================
// Forward Pass
// =============================================================================

/// Execute a fused LSTM forward pass using cuDNN.
///
/// Processes the entire sequence in a single cuDNN call, replacing per-timestep
/// cuBLAS GEMMs + custom CUDA kernels. For Kokoro's bidirectional LSTM with
/// seq_len=95, this reduces ~570 kernel launches to 1 call.
///
/// Returns Y with shape [seq_len, num_directions, batch, hidden_size].
/// The output layout matches ONNX LSTM's default seq_major format.
pub fn cudnn_lstm_forward(
    handle: &CudnnHandle,
    ctx: &IconnxCudaContext,
    workspace: &mut Workspace,
    packed_weights: &PackedLstmWeights,
    x: &GpuTensor,             // [seq_len, batch, input_size]
    h_init: Option<&GpuTensor>, // [num_directions, batch, hidden_size]
    c_init: Option<&GpuTensor>, // [num_directions, batch, hidden_size]
    hidden_size: usize,
    num_directions: usize,
) -> Result<GpuTensor, CudaError> {
    let x_shape = x.shape();

    // Handle 2D input [seq_len, input_size] by treating as batch=1
    let (seq_len, batch_size, input_size) = if x_shape.len() == 2 {
        (x_shape[0], 1, x_shape[1])
    } else if x_shape.len() == 3 {
        (x_shape[0], x_shape[1], x_shape[2])
    } else {
        return Err(CudaError::Cudnn(format!(
            "LSTM input must be 2D or 3D, got {:?}",
            x_shape
        )));
    };

    let output_size = num_directions * hidden_size;

    // Create data descriptors for input X and output Y
    let x_desc = RnnDataDescriptor::new(seq_len, batch_size, input_size)?;
    let y_desc = RnnDataDescriptor::new(seq_len, batch_size, output_size)?;

    // Create tensor descriptors for hidden/cell state
    // cuDNN expects h/c descriptors as 3D: [numLayers * numDirs, batch, hidden]
    let mut h_desc = TensorDescriptor::new()?;
    let h_dims = [num_directions, batch_size, hidden_size];
    let h_strides = [batch_size * hidden_size, hidden_size, 1];
    h_desc.set_nd(
        &h_dims,
        &h_strides,
        sys::cudnnDataType_t::CUDNN_DATA_FLOAT,
    )?;

    let mut c_desc = TensorDescriptor::new()?;
    c_desc.set_nd(
        &h_dims,
        &h_strides,
        sys::cudnnDataType_t::CUDNN_DATA_FLOAT,
    )?;

    // Query workspace size
    let mut work_space_size: usize = 0;
    let mut _reserve_space_size: usize = 0;
    unsafe {
        check_cudnn(sys::cudnnGetRNNTempSpaceSizes(
            handle.raw(),
            packed_weights.rnn_desc().raw(),
            sys::cudnnForwardMode_t::CUDNN_FWD_MODE_INFERENCE,
            x_desc.raw(),
            &mut work_space_size,
            &mut _reserve_space_size,
        ))?;
    }

    // Ensure workspace is large enough
    workspace.ensure_size(ctx, work_space_size)?;

    // Allocate output Y: [seq_len, batch, num_directions * hidden_size]
    let mut y = GpuTensor::zeros_f32(ctx, vec![seq_len, batch_size, output_size])?;

    // Create seq_lengths array on device (all same length)
    let seq_lengths_host: Vec<i32> = vec![seq_len as i32; batch_size];
    let seq_lengths_dev: CudaSlice<i32> = ctx
        .htod_i32(&seq_lengths_host)
        .map_err(|e| CudaError::Transfer(format!("Failed to copy seq_lengths: {}", e)))?;

    // Execute forward pass in a scoped block so device pointer guards are
    // dropped before we reshape the output tensor.
    {
        let stream = ctx.stream();

        let (x_ptr, _x_guard) = x.data_f32()?.device_ptr(stream);
        let (y_ptr, _y_guard) = y.data_f32_mut()?.device_ptr_mut(stream);
        let (w_ptr, _w_guard) = packed_weights.buffer_device_ptr(stream);
        let workspace_ptr = workspace.ptr(stream);
        let (seq_dev_ptr, _seq_guard) = seq_lengths_dev.device_ptr(stream);

        // Get h_init and c_init raw pointers (null if not provided)
        let (hx_ptr, _hx_guard) = if let Some(h) = h_init {
            let (ptr, guard) = h.data_f32()?.device_ptr(stream);
            (ptr as *const c_void, Some(guard))
        } else {
            (ptr::null(), None)
        };

        let (cx_ptr, _cx_guard) = if let Some(c) = c_init {
            let (ptr, guard) = c.data_f32()?.device_ptr(stream);
            (ptr as *const c_void, Some(guard))
        } else {
            (ptr::null(), None)
        };

        unsafe {
            check_cudnn(sys::cudnnRNNForward(
                handle.raw(),
                packed_weights.rnn_desc().raw(),
                sys::cudnnForwardMode_t::CUDNN_FWD_MODE_INFERENCE,
                seq_dev_ptr as *const i32,
                x_desc.raw(),
                x_ptr as *const c_void,
                y_desc.raw(),
                y_ptr as *mut c_void,
                h_desc.raw(),
                hx_ptr,
                ptr::null_mut(), // hy: don't need final hidden state
                c_desc.raw(),
                cx_ptr,
                ptr::null_mut(), // cy: don't need final cell state
                packed_weights.size(),
                w_ptr as *const c_void,
                work_space_size,
                workspace_ptr,
                0,               // reserveSpaceSize: 0 for inference
                ptr::null_mut(), // reserveSpace: null for inference
            ))?;
        }
    }

    // cuDNN output Y is [seq_len, batch, num_directions * hidden_size]
    // but the existing gpu_lstm returns [seq_len, num_directions, batch, hidden_size]
    // Reshape to match the expected output format
    let y_reshaped = y
        .reshape(vec![seq_len, batch_size, num_directions, hidden_size])
        .ok_or_else(|| CudaError::Cudnn("Failed to reshape Y for direction split".into()))?;

    // Transpose from [seq, batch, dirs, hidden] to [seq, dirs, batch, hidden]
    // This matches the ONNX LSTM output layout that gpu_lstm produces.
    //
    // For batch=1 (Kokoro's case), [seq, 1, dirs, hidden] == [seq, dirs, 1, hidden]
    // so reshape alone suffices with no data movement. For batch>1 a transpose
    // kernel would be needed.
    if batch_size == 1 {
        y_reshaped
            .reshape(vec![seq_len, num_directions, batch_size, hidden_size])
            .ok_or_else(|| {
                CudaError::Cudnn("Failed to reshape Y to [seq, dirs, batch, hidden]".into())
            })
    } else {
        // General case: need actual data movement for transposition
        // [seq, batch, dirs, hidden] -> [seq, dirs, batch, hidden]
        // For Kokoro all LSTMs have batch=1, so this path is not exercised.
        Err(CudaError::Cudnn(format!(
            "cuDNN LSTM with batch_size={} > 1 not yet supported (needs transpose kernel)",
            batch_size
        )))
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::lstm::LstmKernelCache;
    use crate::cuda::memory_pool::GpuMemoryPool;
    use crate::cuda::ops::OpsKernelCache;

    // Compile-time assertion: cuDNN RNN types must be Send+Sync so
    // GpuGraphExecutor (which contains them) can be used across threads.
    const _: () = {
        fn assert_send_sync<T: Send + Sync>() {}
        fn check() {
            assert_send_sync::<DropoutDescriptor>();
            assert_send_sync::<RnnDescriptor>();
            assert_send_sync::<RnnDataDescriptor>();
            assert_send_sync::<PackedLstmWeights>();
        }
    };

    /// Helper: create a GpuTensor from a Vec<f32> with given shape
    fn gpu_from_vec(
        ctx: &IconnxCudaContext,
        data: Vec<f32>,
        shape: Vec<usize>,
    ) -> GpuTensor {
        GpuTensor::from_host_f32(ctx, &data, shape).expect("Failed to create GPU tensor")
    }

    /// Helper: download GPU tensor to host
    fn to_host(ctx: &IconnxCudaContext, t: &GpuTensor) -> Vec<f32> {
        t.to_host_f32(ctx).expect("Failed to download tensor")
    }

    #[test]
    #[ignore = "requires CUDA GPU with cuDNN"]
    fn test_cudnn_lstm_matches_gpu_lstm() {
        // Small synthetic LSTM: seq_len=3, batch=1, input_size=4, hidden=4, bidirectional
        let ctx = IconnxCudaContext::new().expect("Failed to create CUDA context");
        let handle = CudnnHandle::new().expect("Failed to create cuDNN handle");
        handle
            .set_stream(ctx.stream())
            .expect("Failed to set stream");

        let seq_len = 3;
        let batch = 1;
        let input_size = 4;
        let hidden = 4;
        let num_dirs = 2;

        // Create deterministic inputs
        let x_data: Vec<f32> = (0..seq_len * batch * input_size)
            .map(|i| (i as f32) * 0.1 - 0.5)
            .collect();
        let x = gpu_from_vec(&ctx, x_data, vec![seq_len, batch, input_size]);

        // Weights: W [num_dirs, 4*hidden, input_size]
        let w_data: Vec<f32> = (0..num_dirs * 4 * hidden * input_size)
            .map(|i| ((i as f32) * 0.02 - 0.5).sin() * 0.1)
            .collect();
        let w = gpu_from_vec(
            &ctx,
            w_data,
            vec![num_dirs, 4 * hidden, input_size],
        );

        // Recurrence: R [num_dirs, 4*hidden, hidden]
        let r_data: Vec<f32> = (0..num_dirs * 4 * hidden * hidden)
            .map(|i| ((i as f32) * 0.03 + 0.7).cos() * 0.1)
            .collect();
        let r = gpu_from_vec(
            &ctx,
            r_data,
            vec![num_dirs, 4 * hidden, hidden],
        );

        // Bias: B [num_dirs, 8*hidden]
        let b_data: Vec<f32> = (0..num_dirs * 8 * hidden)
            .map(|i| (i as f32) * 0.01 - 0.3)
            .collect();
        let b = gpu_from_vec(&ctx, b_data, vec![num_dirs, 8 * hidden]);

        // Initial hidden state: h_init [num_dirs, batch, hidden]
        let h_data: Vec<f32> = vec![0.0; num_dirs * batch * hidden];
        let h_init = gpu_from_vec(&ctx, h_data, vec![num_dirs, batch, hidden]);

        // Run gpu_lstm (reference implementation)
        let lstm_cache = LstmKernelCache::new(&ctx).expect("Failed to create LSTM cache");
        let ops_cache = OpsKernelCache::new(&ctx).expect("Failed to create ops cache");
        let mut pool = GpuMemoryPool::new();

        let reference_output = crate::cuda::lstm::gpu_lstm(
            &ctx,
            &lstm_cache,
            &ops_cache,
            &mut pool,
            &x,
            &w,
            &r,
            Some(&b),
            Some(&h_init),
            None, // c_init
        )
        .expect("gpu_lstm failed");

        // Run cudnn_lstm_forward
        let mut workspace = Workspace::new();
        let packed = pack_lstm_weights_for_cudnn(
            &handle,
            &ctx,
            &w,
            &r,
            Some(&b),
            hidden,
            input_size,
            num_dirs,
        )
        .expect("Weight packing failed");

        let cudnn_output = cudnn_lstm_forward(
            &handle,
            &ctx,
            &mut workspace,
            &packed,
            &x,
            Some(&h_init),
            None, // c_init
            hidden,
            num_dirs,
        )
        .expect("cudnn_lstm_forward failed");

        // Compare shapes
        assert_eq!(
            reference_output.shape(),
            cudnn_output.shape(),
            "Output shapes must match: ref={:?}, cudnn={:?}",
            reference_output.shape(),
            cudnn_output.shape()
        );

        // Compare values
        let ref_data = to_host(&ctx, &reference_output);
        let cudnn_data = to_host(&ctx, &cudnn_output);

        // Track max difference for diagnostics
        let mut max_diff: f32 = 0.0;

        // Tolerance: 1e-4 because float accumulation order differs between
        // per-timestep cuBLAS GEMMs and cuDNN's fused implementation.
        // On long sequences the divergence grows, but for seq_len=3 it
        // should be well within this bound.
        let tolerance = 1e-4;
        for (i, (r_val, c_val)) in ref_data.iter().zip(cudnn_data.iter()).enumerate() {
            let diff = (r_val - c_val).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            assert!(
                diff < tolerance,
                "Mismatch at index {}: ref={}, cudnn={}, diff={} (tolerance={})",
                i, r_val, c_val, diff, tolerance
            );
        }

        println!(
            "cuDNN LSTM matches gpu_lstm within {} tolerance (max_diff={}). Shape: {:?}",
            tolerance,
            max_diff,
            cudnn_output.shape()
        );
    }

    #[test]
    #[ignore = "requires CUDA GPU with cuDNN"]
    fn test_cudnn_lstm_unidirectional() {
        // Verify unidirectional LSTM works (num_directions=1)
        let ctx = IconnxCudaContext::new().expect("Failed to create CUDA context");
        let handle = CudnnHandle::new().expect("Failed to create cuDNN handle");
        handle
            .set_stream(ctx.stream())
            .expect("Failed to set stream");

        let seq_len = 3;
        let batch = 1;
        let input_size = 4;
        let hidden = 4;
        let num_dirs = 1;

        let x_data: Vec<f32> = (0..seq_len * batch * input_size)
            .map(|i| (i as f32) * 0.1)
            .collect();
        let x = gpu_from_vec(&ctx, x_data, vec![seq_len, batch, input_size]);

        let w_data: Vec<f32> = (0..num_dirs * 4 * hidden * input_size)
            .map(|i| ((i as f32) * 0.02).sin() * 0.1)
            .collect();
        let w = gpu_from_vec(&ctx, w_data, vec![num_dirs, 4 * hidden, input_size]);

        let r_data: Vec<f32> = (0..num_dirs * 4 * hidden * hidden)
            .map(|i| ((i as f32) * 0.03).cos() * 0.1)
            .collect();
        let r = gpu_from_vec(&ctx, r_data, vec![num_dirs, 4 * hidden, hidden]);

        let mut workspace = Workspace::new();
        let packed = pack_lstm_weights_for_cudnn(
            &handle, &ctx, &w, &r, None, hidden, input_size, num_dirs,
        )
        .expect("Weight packing failed");

        let output = cudnn_lstm_forward(
            &handle, &ctx, &mut workspace, &packed, &x, None, None,
            hidden, num_dirs,
        )
        .expect("cudnn_lstm_forward failed");

        assert_eq!(
            output.shape(),
            &[seq_len, num_dirs, batch, hidden],
            "Unidirectional output shape mismatch"
        );

        // Verify output is not all zeros (sanity check that computation ran)
        let data = to_host(&ctx, &output);
        let max_val = data.iter().cloned().fold(0.0f32, f32::max);
        assert!(
            max_val > 1e-6,
            "Output appears to be all zeros -- computation may not have run"
        );

        println!(
            "Unidirectional cuDNN LSTM OK. Shape: {:?}, max: {}",
            output.shape(),
            max_val
        );
    }
}
