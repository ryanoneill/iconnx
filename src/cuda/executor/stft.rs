//! STFT dispatch.
//!
//! Ports the `"STFT"` arm of today's
//! `execute_gpu_operator_with_precomputed` in `src/cuda/inference/mod.rs`
//! verbatim: ONNX STFT (opset 17) with frame_step, optional window,
//! optional frame_length, onesided attribute. The FFT itself runs on the
//! CPU via FFTW for numerical precision; only the surrounding
//! reshape/upload plumbing runs on the GPU today. The `stft_kernels`
//! cache is plumbed through the `Executor` but not yet consumed — once a
//! true cuFFT path lands, the dispatch body will switch over without the
//! caller (`dispatch_op`) noticing.

use std::collections::HashMap;

use crate::cuda::{CudaError, GpuTensor};
use crate::ir::PlannedOp;
use crate::operators::fftw_stft::fftw_stft;

use super::{resolve_inputs, Executor};

impl Executor {
    /// Dispatch the ONNX `STFT` op.
    ///
    /// Inputs:
    /// 0. `signal`      — `[length]`, `[batch, length]`, or `[batch, length, 1]`
    /// 1. `frame_step`  — scalar int64, hop length
    /// 2. `window`      — optional, float32
    /// 3. `frame_length` — optional int64, inferred from window if absent
    ///
    /// Attribute `onesided` (default 1) controls whether only the
    /// non-redundant half-spectrum is returned. The signal must be
    /// batch-1 today; batched STFT isn't implemented.
    pub(crate) fn dispatch_stft(
        &self,
        op: &PlannedOp,
        values: &HashMap<String, GpuTensor>,
        weights: &HashMap<String, GpuTensor>,
    ) -> Result<GpuTensor, CudaError> {
        let inputs = resolve_inputs(&op.node.inputs, values, weights)?;
        let attributes = &op.node.attributes;

        if inputs.len() < 2 {
            return Err(CudaError::Kernel(
                "STFT requires at least signal and frame_step inputs".to_string(),
            ));
        }

        // Hop length from input 1 (frame_step) — scalar int64.
        let hop_length = {
            let data = inputs[1].to_host_i64(&self.ctx)?;
            data.first().copied().unwrap_or(512) as usize
        };

        // Window from input 2 (optional).
        let window = if inputs.len() > 2 { Some(inputs[2]) } else { None };

        // Frame length from input 3, else inferred from window.
        let frame_length = if inputs.len() > 3 {
            let data = inputs[3].to_host_i64(&self.ctx)?;
            data.first().copied().unwrap_or(2048) as usize
        } else if let Some(w) = window {
            w.shape().iter().product::<usize>()
        } else {
            return Err(CudaError::Kernel(
                "STFT needs frame_length input or window to infer frame size".to_string(),
            ));
        };

        let onesided = attributes.get_int("onesided").unwrap_or(1) != 0;

        // Flatten batched input shapes to 1D; only batch=1 is supported.
        let signal_shape = inputs[0].shape();
        let (signal_1d, batch_size) = if signal_shape.len() == 2 {
            if signal_shape[0] != 1 {
                return Err(CudaError::Kernel(format!(
                    "STFT only supports batch=1, got shape {:?}",
                    signal_shape
                )));
            }
            let signal_reshaped = inputs[0]
                .clone()
                .reshape(vec![signal_shape[1]])
                .ok_or_else(|| CudaError::Kernel("Failed to reshape signal".to_string()))?;
            (signal_reshaped, 1)
        } else if signal_shape.len() == 3 && signal_shape[2] == 1 {
            if signal_shape[0] != 1 {
                return Err(CudaError::Kernel(format!(
                    "STFT only supports batch=1, got shape {:?}",
                    signal_shape
                )));
            }
            let signal_reshaped = inputs[0]
                .clone()
                .reshape(vec![signal_shape[1]])
                .ok_or_else(|| CudaError::Kernel("Failed to reshape signal".to_string()))?;
            (signal_reshaped, 1)
        } else if signal_shape.len() == 1 {
            (inputs[0].clone(), 1)
        } else {
            return Err(CudaError::Kernel(format!(
                "STFT signal must be 1D, 2D [batch, len] or 3D [batch, len, 1], got {:?}",
                signal_shape
            )));
        };

        // FFTW-based STFT (pure Rust, high-precision FFT). Requires a window.
        if let Some(w) = window {
            let signal_cpu = signal_1d.to_host_f32(&self.ctx)?;
            let window_cpu = w.to_host_f32(&self.ctx)?;

            let (stft_data, stft_shape) =
                fftw_stft(&signal_cpu, &window_cpu, frame_length, hop_length, onesided)
                    .map_err(|e| CudaError::Kernel(format!("FFTW STFT failed: {}", e)))?;

            // FFTW returns [1, num_frames, fft_bins, 2] or [num_frames, fft_bins, 2].
            // Normalize to [batch, num_frames, fft_bins, 2].
            let new_shape = if stft_shape.len() == 4 {
                vec![batch_size, stft_shape[1], stft_shape[2], stft_shape[3]]
            } else {
                vec![batch_size, stft_shape[0], stft_shape[1], stft_shape[2]]
            };

            GpuTensor::from_host_f32(&self.ctx, &stft_data, new_shape)
        } else {
            Err(CudaError::Kernel(
                "STFT requires window input for FFTW implementation".to_string(),
            ))
        }
    }
}
