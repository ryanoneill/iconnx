//! cuDNN-backed operations (convolution, RNN) via garboard's `DnnContext`.
//!
//! All cuDNN descriptors, workspaces, and algorithm selection are managed
//! inside garboard. This module provides thin wrappers that translate
//! iconnx's `GpuTensor` / shape conventions into garboard's
//! `ConvForwardConfig` / `ConvBwdDataConfig` / RNN APIs.
//!
//! - `garboard_conv_1d` / `garboard_conv_transpose_1d` — 1D convolution
//!   and its transpose (2D with H=1 under the hood).
//! - `rnn` submodule — LSTM weight packing, plan preparation, and
//!   forward pass.

#![allow(clippy::too_many_arguments)] // cuDNN APIs require many parameters

pub mod rnn;
pub use rnn::*;

use super::context::{CudaError, IconnxCudaContext};
use super::conv::Conv2dParams;
use super::tensor::GpuTensor;

// =============================================================================
// Garboard-based conv
// =============================================================================

/// 1D Convolution via garboard's DnnContext.
///
/// All descriptor management, workspace sizing, and algorithm selection
/// happen inside garboard.
pub fn garboard_conv_1d(
    ctx: &IconnxCudaContext,
    input: &GpuTensor,
    kernel: &GpuTensor,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<GpuTensor, CudaError> {
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

    let out_length = (in_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    let config = garboard::ConvForwardConfig {
        input_dims: [batch as i32, in_channels as i32, 1, in_length as i32],
        filter_dims: [
            out_channels as i32,
            in_channels_per_group as i32,
            1,
            kernel_size as i32,
        ],
        padding: [0, padding as i32],
        stride: [1, stride as i32],
        dilation: [1, dilation as i32],
        groups: groups as i32,
        algorithm: None, // cuDNN heuristic picks per-call.
    };

    let mut output = GpuTensor::zeros_f32(ctx, vec![batch, out_channels, out_length])?;

    ctx.dnn()
        .conv_forward(
            ctx.garboard_stream(),
            &config,
            input.data_f32()?,
            kernel.data_f32()?,
            output.data_f32_mut()?,
        )
        .map_err(|e| CudaError::Cudnn(e.to_string()))?;

    Ok(output)
}

/// 2D Convolution via garboard's DnnContext (NCHW + groups + dilation).
///
/// Mirrors [`garboard_conv_1d`] but for rank-4 `[N, C, H, W]` input. This is
/// the path for grouped/depthwise (`group > 1`) and dilated 2D convs, which
/// the in-tree im2col `gpu_conv2d` does not support — cuDNN handles groups and
/// dilation natively. The `group == 1` non-dilated case stays on `gpu_conv2d`.
///
/// cuDNN's `conv_forward` does NOT fold bias; the caller applies `add_bias`.
pub fn garboard_conv_2d(
    ctx: &IconnxCudaContext,
    input: &GpuTensor,
    kernel: &GpuTensor,
    params: &Conv2dParams,
) -> Result<GpuTensor, CudaError> {
    let input_shape = input.shape();
    let kernel_shape = kernel.shape();

    if input_shape.len() != 4 {
        return Err(CudaError::Cudnn(format!(
            "Expected 4D input [N,C,H,W], got {:?}",
            input_shape
        )));
    }
    if kernel_shape.len() != 4 {
        return Err(CudaError::Cudnn(format!(
            "Expected 4D kernel [C_out,C_in/g,kH,kW], got {:?}",
            kernel_shape
        )));
    }

    let batch = input_shape[0];
    let in_channels = input_shape[1];
    let in_h = input_shape[2];
    let in_w = input_shape[3];

    let out_channels = kernel_shape[0];
    let in_channels_per_group = kernel_shape[1];
    let kernel_h = kernel_shape[2];
    let kernel_w = kernel_shape[3];

    let out_h =
        (in_h + 2 * params.pad_h - params.dilation_h * (kernel_h - 1) - 1) / params.stride_h + 1;
    let out_w =
        (in_w + 2 * params.pad_w - params.dilation_w * (kernel_w - 1) - 1) / params.stride_w + 1;

    let config = garboard::ConvForwardConfig {
        input_dims: [
            batch as i32,
            in_channels as i32,
            in_h as i32,
            in_w as i32,
        ],
        filter_dims: [
            out_channels as i32,
            in_channels_per_group as i32,
            kernel_h as i32,
            kernel_w as i32,
        ],
        padding: [params.pad_h as i32, params.pad_w as i32],
        stride: [params.stride_h as i32, params.stride_w as i32],
        dilation: [params.dilation_h as i32, params.dilation_w as i32],
        groups: params.group as i32,
        algorithm: None, // cuDNN heuristic picks per-call.
    };

    let mut output = GpuTensor::zeros_f32(ctx, vec![batch, out_channels, out_h, out_w])?;

    ctx.dnn()
        .conv_forward(
            ctx.garboard_stream(),
            &config,
            input.data_f32()?,
            kernel.data_f32()?,
            output.data_f32_mut()?,
        )
        .map_err(|e| CudaError::Cudnn(e.to_string()))?;

    Ok(output)
}

/// 1D ConvTranspose via garboard's DnnContext.
///
/// Garboard manages descriptors and workspace internally.
pub fn garboard_conv_transpose_1d(
    ctx: &IconnxCudaContext,
    input: &GpuTensor,
    kernel: &GpuTensor,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<GpuTensor, CudaError> {
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

    let out_length =
        (in_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;

    let config = garboard::ConvBwdDataConfig {
        input_dims: [batch as i32, in_channels as i32, 1, in_length as i32],
        filter_dims: [
            in_channels as i32,
            out_channels_per_group as i32,
            1,
            kernel_size as i32,
        ],
        output_dims: [batch as i32, out_channels as i32, 1, out_length as i32],
        padding: [0, padding as i32],
        stride: [1, stride as i32],
        dilation: [1, dilation as i32],
        groups: groups as i32,
        algorithm: None,
    };

    let mut output = GpuTensor::zeros_f32(ctx, vec![batch, out_channels, out_length])?;

    ctx.dnn()
        .conv_backward_data(
            ctx.garboard_stream(),
            &config,
            kernel.data_f32()?,
            input.data_f32()?,
            output.data_f32_mut()?,
        )
        .map_err(|e| CudaError::Cudnn(e.to_string()))?;

    Ok(output)
}
