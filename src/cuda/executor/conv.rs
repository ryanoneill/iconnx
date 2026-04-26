//! Conv / ConvTranspose dispatch.
//!
//! Routes 1D inputs to the garboard cuDNN wrappers
//! (`garboard_conv_1d` / `garboard_conv_transpose_1d`) and 2D inputs to the
//! in-tree `gpu_conv2d` / `gpu_conv_transpose_2d` kernels. Post-Phase-1 the
//! 1D path is garboard-native; cuDNN does not include bias in conv so
//! `add_bias` is applied on top when a bias input is present.
//!
//! Dispatch target parity with today's `execute_gpu_operator_with_precomputed`
//! in `src/cuda/inference/mod.rs`: same kernel calls, same attribute
//! defaults (`stride=1`, `padding=0`, `dilation=1`, `group=1`,
//! `output_padding=0`), same 1D/2D branching by input rank. Kernel wrappers
//! stay in their source modules; this file owns only the dispatch match.

use std::collections::HashMap;

use crate::cuda::conv::{
    add_bias, gpu_conv1d, gpu_conv2d, gpu_conv_transpose_2d, Conv1dParams, Conv2dParams,
};
use crate::cuda::cudnn::{garboard_conv_1d, garboard_conv_transpose_1d};
use crate::cuda::tensor::DType;
use crate::cuda::{CudaError, GpuTensor};
use crate::ir::PlannedOp;

use super::{resolve_inputs, Executor};

impl Executor {
    /// Dispatch a Conv or ConvTranspose op to its backing kernel.
    ///
    /// 1D inputs (rank 3: `[batch, channels, length]`) route through the
    /// garboard cuDNN wrappers; cuDNN omits bias from conv itself so we
    /// follow up with `add_bias` when a bias input is provided. 2D inputs
    /// (rank 4: `[batch, channels, height, width]`) go through the in-tree
    /// `gpu_conv2d` / `gpu_conv_transpose_2d`, which accept bias inline.
    pub(crate) fn dispatch_conv(
        &self,
        op: &PlannedOp,
        values: &HashMap<String, GpuTensor>,
        weights: &HashMap<String, GpuTensor>,
    ) -> Result<GpuTensor, CudaError> {
        let inputs = resolve_inputs(&op.node.inputs, values, weights)?;
        let attributes = &op.node.attributes;
        let mut pool = self.memory_pool.borrow_mut();

        match op.node.op_type.as_str() {
            "Conv" => {
                let shape = inputs[0].shape();
                match shape.len() {
                    3 => {
                        // 1D conv: [batch, channels, length].
                        //
                        // Float32 routes through garboard's cuDNN binding
                        // (`conv_forward`, sealed to f32 — see garboard
                        // dnn.rs:48). Float16 routes through the in-tree
                        // `gpu_conv1d` (im2col + cuBLAS gemm_fp16_acc32 lit
                        // up at WS-3 M3.6); cuDNN supports FP16 in
                        // principle but the garboard binding is f32-only
                        // today. Queued upstream ask alongside the FP16
                        // batched-MatMul gap.
                        let strides = attributes.get_ints("strides").unwrap_or(&[1]);
                        let pads = attributes.get_ints("pads").unwrap_or(&[0, 0]);
                        let dilations = attributes.get_ints("dilations").unwrap_or(&[1]);
                        let group = attributes.get_int("group").unwrap_or(1) as usize;

                        let stride = strides.first().copied().unwrap_or(1) as usize;
                        let padding = pads.first().copied().unwrap_or(0) as usize;
                        let dilation = dilations.first().copied().unwrap_or(1) as usize;

                        let mut output = match inputs[0].dtype() {
                            DType::Float32 => garboard_conv_1d(
                                &self.ctx,
                                inputs[0],
                                inputs[1],
                                stride,
                                padding,
                                dilation,
                                group,
                            )?,
                            DType::Float16 => {
                                if group != 1 {
                                    return Err(CudaError::Kernel(format!(
                                        "Float16 1D Conv with groups={} not supported \
                                         (garboard cuDNN binding is f32-only; in-tree \
                                         gpu_conv1d does not support groups)",
                                        group
                                    )));
                                }
                                let kernel_size = inputs[1].shape()[2];
                                let params = Conv1dParams {
                                    kernel_size,
                                    stride,
                                    padding,
                                    dilation,
                                };
                                let bias =
                                    if inputs.len() > 2 { Some(inputs[2]) } else { None };
                                let out = gpu_conv1d(
                                    &self.ctx,
                                    &self.conv_kernels,
                                    &mut pool,
                                    inputs[0],
                                    inputs[1],
                                    bias,
                                    &params,
                                )?;
                                // gpu_conv1d folds bias inline; skip the
                                // explicit add_bias step below by returning
                                // early.
                                return Ok(out);
                            }
                            dt => {
                                return Err(CudaError::Kernel(format!(
                                    "1D Conv: unsupported input dtype {}",
                                    dt.name()
                                )))
                            }
                        };

                        // cuDNN doesn't include bias in conv; apply it here
                        // (Float32 path only — Float16 path already folded
                        // bias inside gpu_conv1d).
                        if inputs.len() > 2 {
                            let bias = inputs[2];
                            let out_shape = output.shape();
                            let batch_size = out_shape[0];
                            let out_channels = out_shape[1];
                            let spatial_size = out_shape[2];
                            add_bias(
                                &self.ctx,
                                &self.conv_kernels,
                                &mut output,
                                bias,
                                batch_size,
                                out_channels,
                                spatial_size,
                            )?;
                        }

                        Ok(output)
                    }
                    4 => {
                        // 2D conv: [batch, channels, height, width].
                        let kernel_shape = attributes.get_ints("kernel_shape").unwrap_or(&[3, 3]);
                        let strides = attributes.get_ints("strides").unwrap_or(&[1, 1]);
                        let pads = attributes.get_ints("pads").unwrap_or(&[0, 0, 0, 0]);
                        let params = Conv2dParams {
                            kernel_h: kernel_shape.first().copied().unwrap_or(3) as usize,
                            kernel_w: kernel_shape.get(1).copied().unwrap_or(3) as usize,
                            stride_h: strides.first().copied().unwrap_or(1) as usize,
                            stride_w: strides.get(1).copied().unwrap_or(1) as usize,
                            pad_h: pads.first().copied().unwrap_or(0) as usize,
                            pad_w: pads.get(1).copied().unwrap_or(0) as usize,
                        };
                        let bias = if inputs.len() > 2 { Some(inputs[2]) } else { None };
                        gpu_conv2d(
                            &self.ctx,
                            &self.conv_kernels,
                            &mut pool,
                            inputs[0],
                            inputs[1],
                            bias,
                            &params,
                        )
                    }
                    ndim => Err(CudaError::Kernel(format!(
                        "Conv expects 3D (1D conv) or 4D (2D conv) input, got {}D with shape {:?}",
                        ndim, shape
                    ))),
                }
            }

            "ConvTranspose" => {
                let shape = inputs[0].shape();
                let group = attributes.get_int("group").unwrap_or(1) as usize;
                if shape.len() == 3 {
                    // 1D ConvTranspose - use cuDNN via garboard.
                    let strides = attributes.get_ints("strides").unwrap_or(&[1]);
                    let pads = attributes.get_ints("pads").unwrap_or(&[0, 0]);
                    let output_padding = attributes.get_ints("output_padding").unwrap_or(&[0]);
                    let dilations = attributes.get_ints("dilations").unwrap_or(&[1]);

                    let stride = strides.first().copied().unwrap_or(1) as usize;
                    let padding = pads.first().copied().unwrap_or(0) as usize;
                    let output_pad = output_padding.first().copied().unwrap_or(0) as usize;
                    let dilation = dilations.first().copied().unwrap_or(1) as usize;

                    // Debug: Print input stats for ConvTranspose
                    if std::env::var("DEBUG_CONV_TRANSPOSE").is_ok() {
                        let inp = inputs[0];
                        let kern = inputs[1];
                        eprintln!("DEBUG ConvTranspose input:");
                        eprintln!("  input shape: {:?}, kernel shape: {:?}", inp.shape(), kern.shape());
                        eprintln!("  stride={}, padding={}, output_pad={}, dilation={}, group={}",
                            stride, padding, output_pad, dilation, group);
                        if let Ok(data) = inp.to_host_f32(&self.ctx) {
                            let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
                            let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                            let sum: f32 = data.iter().sum();
                            eprintln!("  input range=[{:.6}, {:.6}], sum={:.6}", min, max, sum);
                        }
                        if let Ok(data) = kern.to_host_f32(&self.ctx) {
                            let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
                            let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                            let sum: f32 = data.iter().sum();
                            eprintln!("  kernel range=[{:.6}, {:.6}], sum={:.6}", min, max, sum);
                        }
                    }

                    let mut output = garboard_conv_transpose_1d(
                        &self.ctx,
                        inputs[0],
                        inputs[1],
                        stride,
                        padding,
                        output_pad,
                        dilation,
                        group,
                    )?;

                    // Debug: Print output stats for ConvTranspose
                    if std::env::var("DEBUG_CONV_TRANSPOSE").is_ok() {
                        if let Ok(data) = output.to_host_f32(&self.ctx) {
                            let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
                            let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                            let sum: f32 = data.iter().sum();
                            eprintln!("DEBUG ConvTranspose output:");
                            eprintln!("  output shape: {:?}", output.shape());
                            eprintln!("  output range=[{:.6}, {:.6}], sum={:.6}", min, max, sum);
                        }
                    }

                    // cuDNN doesn't include bias in conv; apply it here.
                    if inputs.len() > 2 {
                        let bias = inputs[2];
                        let out_shape = output.shape();
                        let batch_size = out_shape[0];
                        let out_channels = out_shape[1];
                        let spatial_size = out_shape[2];
                        add_bias(
                            &self.ctx,
                            &self.conv_kernels,
                            &mut output,
                            bias,
                            batch_size,
                            out_channels,
                            spatial_size,
                        )?;
                    }

                    Ok(output)
                } else {
                    // 2D ConvTranspose.
                    let kernel_shape = attributes.get_ints("kernel_shape").unwrap_or(&[3, 3]);
                    let strides = attributes.get_ints("strides").unwrap_or(&[1, 1]);
                    let pads = attributes.get_ints("pads").unwrap_or(&[0, 0, 0, 0]);
                    let output_padding =
                        attributes.get_ints("output_padding").unwrap_or(&[0, 0]);
                    let params = Conv2dParams {
                        kernel_h: kernel_shape.first().copied().unwrap_or(3) as usize,
                        kernel_w: kernel_shape.get(1).copied().unwrap_or(3) as usize,
                        stride_h: strides.first().copied().unwrap_or(1) as usize,
                        stride_w: strides.get(1).copied().unwrap_or(1) as usize,
                        pad_h: pads.first().copied().unwrap_or(0) as usize,
                        pad_w: pads.get(1).copied().unwrap_or(0) as usize,
                    };
                    let bias = if inputs.len() > 2 { Some(inputs[2]) } else { None };
                    let output_pad_h = output_padding.first().copied().unwrap_or(0) as usize;
                    let output_pad_w = output_padding.get(1).copied().unwrap_or(0) as usize;
                    gpu_conv_transpose_2d(
                        &self.ctx,
                        &self.conv_kernels,
                        &mut pool,
                        inputs[0],
                        inputs[1],
                        bias,
                        &params,
                        output_pad_h,
                        output_pad_w,
                    )
                }
            }

            other => Err(CudaError::Kernel(format!(
                "dispatch_conv called with non-conv op '{}'",
                other
            ))),
        }
    }
}
