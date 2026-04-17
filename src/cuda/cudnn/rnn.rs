//! cuDNN RNN (LSTM) wrappers backed by garboard's `DnnContext`.
//!
//! This module uses garboard's cached-descriptor RNN API
//! (`DnnContext::rnn_prepare` + `rnn_forward_with_plan`). cuDNN descriptors
//! are created once per (weights, batch_size, seq_length) configuration and
//! reused across forward calls — crucial because the descriptor creation
//! cost dominates short-sequence LSTM latency.
//!
//! Responsibilities live in two places:
//!
//! - This module owns ONNX-to-cuDNN weight packing
//!   ([`pack_lstm_weights_for_cudnn`]) and the thin forward call
//!   ([`lstm_forward`]) that dispatches to garboard.
//! - The executor in `cuda::inference` owns the caches: one for packed
//!   weights keyed by `(w_ptr, r_ptr)`, and one for plans keyed by
//!   `(w_ptr, r_ptr, seq_length)` (since plans bake seq_length into the
//!   RNN data descriptors).
//!
//! The weight-packing D2D copies use `garboard_sys::cuMemcpyDtoD_v2`
//! directly — the synchronous driver-level memcpy — because garboard's
//! safe surface does not yet expose sub-slicing of `DeviceSlice<u8>`
//! (needed to target individual cuDNN linLayerID offsets within the
//! packed weight buffer). Packing happens at most once per unique
//! weight pair per process lifetime (~72 small copies for a Kokoro
//! BiLSTM), so the sync-copy cost is well under 1 ms at model-load
//! time and never on the inference hot path. Follow-up: replace with
//! `Stream::copy_device_to_device` once garboard adds a sub-slice API.

#![allow(clippy::too_many_arguments)]

use garboard::{DeviceSlice, RnnForwardConfig, RnnPlan};

use crate::cuda::context::{CudaError, IconnxCudaContext};
use crate::cuda::tensor::GpuTensor;

// =============================================================================
// Packed LSTM Weights
// =============================================================================

/// GPU buffer containing LSTM weights repacked from ONNX gate order (IOFC) to
/// cuDNN gate order (IFCO).
///
/// Created by [`pack_lstm_weights_for_cudnn`] once per unique `(W, R)` pair
/// and cached by the executor. Consumed by [`lstm_forward`] alongside a
/// per-`(weights, seq_length)` [`RnnPlan`] owned by the executor's plan
/// cache.
///
/// **Cache pointer invariant:** This struct is valid only while the
/// original ONNX `W`/`R` tensors (whose data was copied into this buffer)
/// remain alive. In practice the executor owns both the weight tensors
/// and the cache, so this invariant is trivially satisfied.
pub struct PackedLstmWeights {
    /// Packed weight buffer on GPU (cuDNN's opaque internal layout).
    buffer: DeviceSlice<'static, u8>,
    /// Input feature size. Retained so the plan cache can rebuild an
    /// `RnnForwardConfig` for a given seq_length without the caller
    /// re-deriving it.
    input_size: usize,
    /// Hidden state size.
    hidden_size: usize,
    /// 1 for unidirectional, 2 for bidirectional.
    num_directions: usize,
}

impl PackedLstmWeights {
    /// Size of the packed weight buffer in bytes.
    pub fn size(&self) -> usize {
        self.buffer.size_bytes()
    }

    /// Input feature size.
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Hidden state size.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Number of directions (1 or 2).
    pub fn num_directions(&self) -> usize {
        self.num_directions
    }

    /// Borrow the packed weight buffer as a garboard `DeviceSlice<u8>`.
    pub fn buffer(&self) -> &DeviceSlice<'static, u8> {
        &self.buffer
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

/// Build an `RnnForwardConfig` for an LSTM.
///
/// `batch_size` and `seq_length` are used by the plan-build path only —
/// for the packing path, the weight space is batch/seq-invariant, so the
/// caller passes 1 for both.
fn make_rnn_config(
    input_size: i32,
    hidden_size: i32,
    num_directions: i32,
    batch_size: i32,
    seq_length: i32,
) -> RnnForwardConfig {
    RnnForwardConfig {
        input_size,
        hidden_size,
        num_layers: 1,
        bidirectional: num_directions == 2,
        batch_size,
        seq_length,
        dropout: 0.0,
    }
}

/// Repack ONNX LSTM weights into cuDNN's packed weight buffer format.
///
/// ONNX stores weights in IOFC gate order; cuDNN expects IFCO. This function:
/// 1. Queries garboard for the packed weight buffer size.
/// 2. Allocates the buffer on GPU.
/// 3. For each direction and each gate, queries garboard for the
///    offset within the buffer and copies the corresponding ONNX weight
///    slice there via `cuMemcpyDtoDAsync_v2`.
///
/// Packing happens at most once per `(W, R)` pair per executor lifetime;
/// the executor caches [`PackedLstmWeights`] by device-pointer pair.
pub fn pack_lstm_weights_for_cudnn(
    ctx: &IconnxCudaContext,
    w: &GpuTensor,         // [num_directions, 4*hidden, input_size]
    r: &GpuTensor,         // [num_directions, 4*hidden, hidden_size]
    b: Option<&GpuTensor>, // [num_directions, 8*hidden]
    hidden_size: usize,
    input_size: usize,
    num_directions: usize,
) -> Result<PackedLstmWeights, CudaError> {
    // Weight space size is batch/seq-invariant — safe to use placeholders.
    let config = make_rnn_config(
        input_size as i32,
        hidden_size as i32,
        num_directions as i32,
        1,
        1,
    );

    let weight_space_size = ctx
        .dnn()
        .rnn_weight_space_size(&config)
        .map_err(|e| CudaError::Cudnn(format!("rnn_weight_space_size failed: {}", e)))?;

    // Allocate packed weight buffer (zero-initialized for safety).
    let weight_buffer = ctx.alloc_zeros::<u8>(weight_space_size)?;

    // The alloc_zeros above queued a `cuMemsetD8Async` on the garboard
    // user stream; drain before issuing the sync D2D copies so the
    // zero-fill is observed.
    ctx.sync()?;

    for dir in 0..num_directions {
        let pseudo_layer = dir as i32;

        // Input weights (W): cuDNN linLayerID 0..3.
        for (onnx_gate, &cudnn_lin_id) in ONNX_TO_CUDNN_GATE.iter().enumerate() {
            let loc = ctx
                .dnn()
                .rnn_weight_params(&config, &weight_buffer, pseudo_layer, cudnn_lin_id)
                .map_err(|e| {
                    CudaError::Cudnn(format!("rnn_weight_params(W, dir={}, gate={}) failed: {}", dir, onnx_gate, e))
                })?;

            let wb_base_ptr = weight_buffer.as_raw_device_ptr();
            let dst_ptr = wb_base_ptr + loc.matrix_offset as u64;

            let w_offset_floats =
                dir * 4 * hidden_size * input_size + onnx_gate * hidden_size * input_size;
            let w_size_bytes = hidden_size * input_size * std::mem::size_of::<f32>();

            let w_base_ptr = w.data_f32()?.as_raw_device_ptr();
            let src_ptr =
                (w_base_ptr as usize + w_offset_floats * std::mem::size_of::<f32>()) as u64;

            // SAFETY: dst_ptr is a valid device pointer inside
            // `weight_buffer` (offset returned by cuDNN itself), src_ptr
            // is a valid device pointer inside `w.data_f32()`, and
            // `w_size_bytes` is the per-gate weight tile size that cuDNN
            // expects. The call is synchronous; no stream handle required.
            unsafe {
                let status = garboard_sys::cuMemcpyDtoD_v2(
                    dst_ptr,
                    src_ptr,
                    w_size_bytes,
                );
                if status != garboard_sys::cudaError_enum::CUDA_SUCCESS {
                    return Err(CudaError::Transfer(format!(
                        "cuMemcpyDtoD failed for W gate {}: {:?}",
                        onnx_gate, status
                    )));
                }
            }
        }

        // Recurrence weights (R): cuDNN linLayerID 4..7.
        for (onnx_gate, &cudnn_base) in ONNX_TO_CUDNN_GATE.iter().enumerate() {
            let cudnn_lin_id = cudnn_base + 4;

            let loc = ctx
                .dnn()
                .rnn_weight_params(&config, &weight_buffer, pseudo_layer, cudnn_lin_id)
                .map_err(|e| {
                    CudaError::Cudnn(format!("rnn_weight_params(R, dir={}, gate={}) failed: {}", dir, onnx_gate, e))
                })?;

            let wb_base_ptr = weight_buffer.as_raw_device_ptr();
            let dst_ptr = wb_base_ptr + loc.matrix_offset as u64;

            let r_offset_floats =
                dir * 4 * hidden_size * hidden_size + onnx_gate * hidden_size * hidden_size;
            let r_size_bytes = hidden_size * hidden_size * std::mem::size_of::<f32>();

            let r_base_ptr = r.data_f32()?.as_raw_device_ptr();
            let src_ptr =
                (r_base_ptr as usize + r_offset_floats * std::mem::size_of::<f32>()) as u64;

            // SAFETY: see W-weight copy above — dst is inside
            // weight_buffer at the cuDNN-reported offset, src is inside
            // `r.data_f32()`, and `r_size_bytes` is the per-gate
            // recurrence-weight tile. Synchronous copy.
            unsafe {
                let status = garboard_sys::cuMemcpyDtoD_v2(
                    dst_ptr,
                    src_ptr,
                    r_size_bytes,
                );
                if status != garboard_sys::cudaError_enum::CUDA_SUCCESS {
                    return Err(CudaError::Transfer(format!(
                        "cuMemcpyDtoD failed for R gate {}: {:?}",
                        onnx_gate, status
                    )));
                }
            }
        }

        // Biases (optional). ONNX B layout: [Wb_i, Wb_o, Wb_f, Wb_c,
        // Rb_i, Rb_o, Rb_f, Rb_c] = 8*hidden values per direction.
        // cuDNN keeps a separate bias vector per linLayerID 0..7.
        if let Some(b_tensor) = b {
            for (onnx_gate, &cudnn_base) in ONNX_TO_CUDNN_GATE.iter().enumerate() {
                let bias_size_bytes = hidden_size * std::mem::size_of::<f32>();

                // Input bias Wb: cuDNN linLayerID 0..3.
                {
                    let cudnn_lin_id = cudnn_base;
                    let loc = ctx
                        .dnn()
                        .rnn_weight_params(&config, &weight_buffer, pseudo_layer, cudnn_lin_id)
                        .map_err(|e| {
                            CudaError::Cudnn(format!(
                                "rnn_weight_params(Wb, dir={}, gate={}) failed: {}",
                                dir, onnx_gate, e
                            ))
                        })?;

                    if loc.bias_size == 0 {
                        return Err(CudaError::Cudnn(format!(
                            "cuDNN reports no bias slot at (pseudo_layer={}, lin_layer_id={}) but ONNX provides biases",
                            pseudo_layer, cudnn_lin_id
                        )));
                    }

                    let wb_base_ptr = weight_buffer.as_raw_device_ptr();
                    let dst_ptr = wb_base_ptr + loc.bias_offset as u64;

                    let wb_offset_floats = dir * 8 * hidden_size + onnx_gate * hidden_size;
                    let b_base_ptr = b_tensor.data_f32()?.as_raw_device_ptr();
                    let src_ptr = (b_base_ptr as usize
                        + wb_offset_floats * std::mem::size_of::<f32>())
                        as u64;

                    // SAFETY: dst is the cuDNN-reported bias offset
                    // within weight_buffer, src is inside the ONNX
                    // bias tensor's device storage; `bias_size_bytes`
                    // is the per-gate bias size. Synchronous copy.
                    unsafe {
                        let status = garboard_sys::cuMemcpyDtoD_v2(
                            dst_ptr,
                            src_ptr,
                            bias_size_bytes,
                        );
                        if status != garboard_sys::cudaError_enum::CUDA_SUCCESS {
                            return Err(CudaError::Transfer(format!(
                                "cuMemcpyDtoD failed for Wb gate {}: {:?}",
                                onnx_gate, status
                            )));
                        }
                    }
                }

                // Recurrence bias Rb: cuDNN linLayerID 4..7.
                {
                    let cudnn_lin_id_r = cudnn_base + 4;
                    let loc = ctx
                        .dnn()
                        .rnn_weight_params(&config, &weight_buffer, pseudo_layer, cudnn_lin_id_r)
                        .map_err(|e| {
                            CudaError::Cudnn(format!(
                                "rnn_weight_params(Rb, dir={}, gate={}) failed: {}",
                                dir, onnx_gate, e
                            ))
                        })?;

                    if loc.bias_size == 0 {
                        return Err(CudaError::Cudnn(format!(
                            "cuDNN reports no bias slot at (pseudo_layer={}, lin_layer_id={}) but ONNX provides biases",
                            pseudo_layer, cudnn_lin_id_r
                        )));
                    }

                    let wb_base_ptr = weight_buffer.as_raw_device_ptr();
                    let dst_ptr = wb_base_ptr + loc.bias_offset as u64;

                    let rb_offset_floats =
                        dir * 8 * hidden_size + (4 + onnx_gate) * hidden_size;
                    let b_base_ptr = b_tensor.data_f32()?.as_raw_device_ptr();
                    let src_ptr = (b_base_ptr as usize
                        + rb_offset_floats * std::mem::size_of::<f32>())
                        as u64;

                    // SAFETY: recurrence-bias variant of the Wb copy
                    // above; same device-pointer validity argument.
                    // Synchronous copy.
                    unsafe {
                        let status = garboard_sys::cuMemcpyDtoD_v2(
                            dst_ptr,
                            src_ptr,
                            bias_size_bytes,
                        );
                        if status != garboard_sys::cudaError_enum::CUDA_SUCCESS {
                            return Err(CudaError::Transfer(format!(
                                "cuMemcpyDtoD failed for Rb gate {}: {:?}",
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
        input_size,
        hidden_size,
        num_directions,
    })
}

/// Build an `RnnPlan` for the given packed weights + seq_length.
///
/// Plans bake the RNN data descriptors (which encode seq_length and
/// batch_size) into cuDNN descriptors. Different seq_lengths need
/// different plans. The executor caches plans keyed by
/// `(w_ptr, r_ptr, seq_length)` alongside the weight cache.
pub fn prepare_lstm_plan(
    ctx: &IconnxCudaContext,
    packed_weights: &PackedLstmWeights,
    batch_size: usize,
    seq_length: usize,
) -> Result<RnnPlan<'static>, CudaError> {
    let config = make_rnn_config(
        packed_weights.input_size() as i32,
        packed_weights.hidden_size() as i32,
        packed_weights.num_directions() as i32,
        batch_size as i32,
        seq_length as i32,
    );

    ctx.dnn()
        .rnn_prepare(ctx.garboard_stream(), &config)
        .map_err(|e| CudaError::Cudnn(format!("rnn_prepare failed: {}", e)))
}

// =============================================================================
// Forward Pass
// =============================================================================

/// Execute a fused LSTM forward pass using garboard's `rnn_forward_with_plan`.
///
/// Processes the entire sequence in a single cuDNN call. The plan caches
/// all cuDNN descriptors — no per-call descriptor creation. For Kokoro's
/// 6 LSTM calls per inference this lifts ~162 ms of descriptor-setup
/// overhead off the critical path.
///
/// Returns `Y` with shape `[seq_len, num_directions, batch, hidden_size]`
/// to match the output layout produced by [`crate::cuda::lstm::gpu_lstm`].
pub fn lstm_forward(
    ctx: &IconnxCudaContext,
    packed_weights: &PackedLstmWeights,
    plan: &RnnPlan<'static>,
    x: &GpuTensor,               // [seq_len, batch, input_size]
    h_init: Option<&GpuTensor>,  // [num_directions, batch, hidden_size]
    c_init: Option<&GpuTensor>,  // [num_directions, batch, hidden_size]
    hidden_size: usize,
    num_directions: usize,
) -> Result<GpuTensor, CudaError> {
    let x_shape = x.shape();

    // Handle 2D input [seq_len, input_size] by treating as batch=1.
    let (seq_len, batch_size, _input_size) = if x_shape.len() == 2 {
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

    // Allocate output Y: [seq_len, batch, num_directions * hidden_size].
    let mut y = GpuTensor::zeros_f32(ctx, vec![seq_len, batch_size, output_size])?;

    {
        let x_slice = x.data_f32()?;
        let y_slice = y.data_f32_mut()?;
        let hx_slice = match h_init {
            Some(h) => Some(h.data_f32()?),
            None => None,
        };
        let cx_slice = match c_init {
            Some(c) => Some(c.data_f32()?),
            None => None,
        };

        ctx.dnn()
            .rnn_forward_with_plan(
                ctx.garboard_stream(),
                plan,
                x_slice,
                y_slice,
                hx_slice,
                None, // hy: final hidden state not needed
                cx_slice,
                None, // cy: final cell state not needed
                packed_weights.buffer(),
            )
            .map_err(|e| CudaError::Cudnn(format!("rnn_forward_with_plan failed: {}", e)))?;
    }

    // cuDNN's output is [seq_len, batch, num_directions * hidden_size].
    // The reference `gpu_lstm` produces [seq_len, num_directions, batch,
    // hidden_size]. For batch==1 these are bitwise identical — just
    // reshape with no data movement. batch>1 would need a transpose
    // kernel; Kokoro never exercises that path.
    let y_reshaped = y
        .reshape(vec![seq_len, batch_size, num_directions, hidden_size])
        .ok_or_else(|| CudaError::Cudnn("Failed to reshape Y for direction split".into()))?;

    if batch_size == 1 {
        y_reshaped
            .reshape(vec![seq_len, num_directions, batch_size, hidden_size])
            .ok_or_else(|| {
                CudaError::Cudnn("Failed to reshape Y to [seq, dirs, batch, hidden]".into())
            })
    } else {
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

    // Compile-time assertion: the packed weights and plan types must be
    // `Send` so the executor can be moved across threads. They are
    // stored inside `RefCell` on the executor, which is already
    // non-`Sync`, so we do *not* require `Sync` here — requiring it
    // would break compilation because `RnnPlan` is deliberately
    // `!Sync` upstream.
    const _: () = {
        fn assert_send<T: Send>() {}
        fn check() {
            assert_send::<PackedLstmWeights>();
            assert_send::<RnnPlan<'static>>();
        }
    };

    /// Helper: create a GpuTensor from a Vec<f32> with given shape.
    fn gpu_from_vec(
        ctx: &IconnxCudaContext,
        data: Vec<f32>,
        shape: Vec<usize>,
    ) -> GpuTensor {
        GpuTensor::from_host_f32(ctx, &data, shape).expect("Failed to create GPU tensor")
    }

    /// Helper: download GPU tensor to host.
    fn to_host(ctx: &IconnxCudaContext, t: &GpuTensor) -> Vec<f32> {
        t.to_host_f32(ctx).expect("Failed to download tensor")
    }

    #[test]
    #[ignore = "requires CUDA GPU with cuDNN"]
    fn test_cudnn_lstm_matches_gpu_lstm() {
        // Bidirectional LSTM with Some(&h_init) + Some(&c_init), matching
        // Kokoro's production path (where ConstantOfShape(0) supplies
        // zero-filled initial states). Passing None here would narrow
        // the test contract and hide bugs in the Some path, so this
        // test deliberately exercises both state arguments.
        let ctx = IconnxCudaContext::new().expect("Failed to create CUDA context");

        let seq_len = 3;
        let batch = 1;
        let input_size = 4;
        let hidden = 4;
        let num_dirs = 2;

        let x_data: Vec<f32> = (0..seq_len * batch * input_size)
            .map(|i| (i as f32) * 0.1 - 0.5)
            .collect();
        let x = gpu_from_vec(&ctx, x_data, vec![seq_len, batch, input_size]);

        let w_data: Vec<f32> = (0..num_dirs * 4 * hidden * input_size)
            .map(|i| ((i as f32) * 0.02 - 0.5).sin() * 0.1)
            .collect();
        let w = gpu_from_vec(&ctx, w_data, vec![num_dirs, 4 * hidden, input_size]);

        let r_data: Vec<f32> = (0..num_dirs * 4 * hidden * hidden)
            .map(|i| ((i as f32) * 0.03 + 0.7).cos() * 0.1)
            .collect();
        let r = gpu_from_vec(&ctx, r_data, vec![num_dirs, 4 * hidden, hidden]);

        let b_data: Vec<f32> = (0..num_dirs * 8 * hidden)
            .map(|i| (i as f32) * 0.01 - 0.3)
            .collect();
        let b = gpu_from_vec(&ctx, b_data, vec![num_dirs, 8 * hidden]);

        // Zero-filled initial states — the production invariant for
        // Kokoro's LSTMs and what `ConstantOfShape(value=0)` produces.
        let h_data: Vec<f32> = vec![0.0; num_dirs * batch * hidden];
        let h_init = gpu_from_vec(&ctx, h_data, vec![num_dirs, batch, hidden]);
        let c_data: Vec<f32> = vec![0.0; num_dirs * batch * hidden];
        let c_init = gpu_from_vec(&ctx, c_data, vec![num_dirs, batch, hidden]);

        // Reference: legacy per-timestep LSTM.
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
            Some(&c_init),
        )
        .expect("gpu_lstm failed");

        // Pack weights + build plan via the cached-descriptor API.
        let packed = pack_lstm_weights_for_cudnn(
            &ctx, &w, &r, Some(&b), hidden, input_size, num_dirs,
        )
        .expect("Weight packing failed");

        let plan = prepare_lstm_plan(&ctx, &packed, batch, seq_len)
            .expect("prepare_lstm_plan failed");

        let cudnn_output = lstm_forward(
            &ctx,
            &packed,
            &plan,
            &x,
            Some(&h_init),
            Some(&c_init),
            hidden,
            num_dirs,
        )
        .expect("lstm_forward failed");

        assert_eq!(
            reference_output.shape(),
            cudnn_output.shape(),
            "Output shapes must match: ref={:?}, cudnn={:?}",
            reference_output.shape(),
            cudnn_output.shape()
        );

        let ref_data = to_host(&ctx, &reference_output);
        let cudnn_data = to_host(&ctx, &cudnn_output);

        let mut max_diff: f32 = 0.0;
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
        // Smoke test: unidirectional LSTM with no initial states, using
        // the cached-descriptor path end-to-end.
        let ctx = IconnxCudaContext::new().expect("Failed to create CUDA context");

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

        let packed =
            pack_lstm_weights_for_cudnn(&ctx, &w, &r, None, hidden, input_size, num_dirs)
                .expect("Weight packing failed");

        let plan = prepare_lstm_plan(&ctx, &packed, batch, seq_len)
            .expect("prepare_lstm_plan failed");

        let output = lstm_forward(&ctx, &packed, &plan, &x, None, None, hidden, num_dirs)
            .expect("lstm_forward failed");

        assert_eq!(
            output.shape(),
            &[seq_len, num_dirs, batch, hidden],
            "Unidirectional output shape mismatch"
        );

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
