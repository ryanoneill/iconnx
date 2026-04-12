//! Transposed convolution (deconvolution) operations using col2im

#![allow(clippy::too_many_arguments)] // CUDA kernels require many parameters

use super::cache::ConvKernelCache;
use super::forward::add_bias;
use super::params::{Conv1dParams, Conv2dParams};
use crate::cuda::context::{CudaError, IconnxCudaContext};
use crate::cuda::matmul::gpu_gemm;
use crate::cuda::memory_pool::GpuMemoryPool;
use crate::cuda::ops::{gpu_concat, gpu_slice_nd, OpsKernelCache};
use crate::cuda::tensor::GpuTensor;
use cudarc::driver::{LaunchConfig, PushKernelArg};

/// GPU col2im transformation for 2D ConvTranspose
///
/// Transforms [C*kH*kW, N*out_H*out_W] -> [N, C, H, W]
/// This is the inverse of im2col, used by ConvTranspose
pub fn gpu_col2im(
    ctx: &IconnxCudaContext,
    cache: &ConvKernelCache,
    pool: &mut GpuMemoryPool,
    col_matrix: &GpuTensor,
    batch_size: usize,
    channels: usize,
    height: usize,
    width: usize,
    params: &Conv2dParams,
    out_h: usize,
    out_w: usize,
) -> Result<GpuTensor, CudaError> {
    let total_output = batch_size * channels * height * width;
    let mut output = pool.get_tensor_f32(ctx, vec![batch_size, channels, height, width])?;

    let kernel = cache
        .get("col2im_kernel")
        .ok_or_else(|| CudaError::Kernel("col2im_kernel not found".to_string()))?;

    let config = LaunchConfig::for_num_elems(total_output as u32);

    unsafe {
        ctx.stream()
            .launch_builder(kernel)
            .arg(output.data_f32_mut()?)
            .arg(col_matrix.data_f32()?)
            .arg(&batch_size)
            .arg(&channels)
            .arg(&height)
            .arg(&width)
            .arg(&params.kernel_h)
            .arg(&params.kernel_w)
            .arg(&params.stride_h)
            .arg(&params.stride_w)
            .arg(&params.pad_h)
            .arg(&params.pad_w)
            .arg(&out_h)
            .arg(&out_w)
            .launch(config)
            .map_err(|e| CudaError::Kernel(format!("col2im launch failed: {}", e)))?;
    }

    Ok(output)
}

/// GPU col2im transformation for 1D ConvTranspose
///
/// Transforms [C*K, N*out_L] -> [N, C, L]
pub fn gpu_col2im_1d(
    ctx: &IconnxCudaContext,
    cache: &ConvKernelCache,
    pool: &mut GpuMemoryPool,
    col_matrix: &GpuTensor,
    batch_size: usize,
    channels: usize,
    length: usize,
    params: &Conv1dParams,
    out_length: usize,
) -> Result<GpuTensor, CudaError> {
    let total_output = batch_size * channels * length;
    let mut output = pool.get_tensor_f32(ctx, vec![batch_size, channels, length])?;

    let kernel = cache
        .get("col2im_1d_kernel")
        .ok_or_else(|| CudaError::Kernel("col2im_1d_kernel not found".to_string()))?;

    let config = LaunchConfig::for_num_elems(total_output as u32);

    unsafe {
        ctx.stream()
            .launch_builder(kernel)
            .arg(output.data_f32_mut()?)
            .arg(col_matrix.data_f32()?)
            .arg(&batch_size)
            .arg(&channels)
            .arg(&length)
            .arg(&params.kernel_size)
            .arg(&params.stride)
            .arg(&params.padding)
            .arg(&out_length)
            .launch(config)
            .map_err(|e| CudaError::Kernel(format!("col2im_1d launch failed: {}", e)))?;
    }

    Ok(output)
}

/// GPU 2D Transposed Convolution (Deconvolution)
///
/// ConvTranspose is the gradient of Conv with respect to the input.
/// Algorithm: W^T @ input -> col_matrix -> col2im -> output
///
/// Input: [N, C_in, H, W]
/// Kernel: [C_in, C_out, kH, kW] (note: reversed from Conv2D)
/// Bias (optional): [C_out]
/// Output: [N, C_out, out_H, out_W]
///
/// Output dimensions:
/// out_H = (H - 1) * stride_h - 2 * pad_h + kernel_h + output_padding_h
/// out_W = (W - 1) * stride_w - 2 * pad_w + kernel_w + output_padding_w
pub fn gpu_conv_transpose_2d(
    ctx: &IconnxCudaContext,
    cache: &ConvKernelCache,
    pool: &mut GpuMemoryPool,
    input: &GpuTensor,
    kernel: &GpuTensor,
    bias: Option<&GpuTensor>,
    params: &Conv2dParams,
    output_padding_h: usize,
    output_padding_w: usize,
) -> Result<GpuTensor, CudaError> {
    let input_shape = input.shape();
    let kernel_shape = kernel.shape();

    assert_eq!(
        input_shape.len(),
        4,
        "ConvTranspose2D input must be 4D [N, C, H, W]"
    );
    assert_eq!(
        kernel_shape.len(),
        4,
        "ConvTranspose2D kernel must be 4D [C_in, C_out, kH, kW]"
    );

    let batch_size = input_shape[0];
    let in_channels = input_shape[1];
    let height = input_shape[2];
    let width = input_shape[3];

    let kernel_in_channels = kernel_shape[0];
    let out_channels = kernel_shape[1];

    assert_eq!(
        in_channels, kernel_in_channels,
        "Input channels ({}) must match kernel input channels ({})",
        in_channels, kernel_in_channels
    );
    assert_eq!(kernel_shape[2], params.kernel_h, "Kernel height mismatch");
    assert_eq!(kernel_shape[3], params.kernel_w, "Kernel width mismatch");

    // Calculate output dimensions
    let out_h =
        (height - 1) * params.stride_h - 2 * params.pad_h + params.kernel_h + output_padding_h;
    let out_w =
        (width - 1) * params.stride_w - 2 * params.pad_w + params.kernel_w + output_padding_w;

    // Step 1: Reshape input to [in_channels, batch_size * height * width]
    let input_cols = batch_size * height * width;
    let input_2d = input
        .reshape(vec![in_channels, input_cols])
        .ok_or_else(|| CudaError::Kernel("Failed to reshape input".to_string()))?;

    // Step 2: Reshape kernel to [in_channels, out_channels * kH * kW]
    let kernel_cols = out_channels * params.kernel_h * params.kernel_w;
    let kernel_2d = kernel
        .reshape(vec![in_channels, kernel_cols])
        .ok_or_else(|| CudaError::Kernel("Failed to reshape kernel".to_string()))?;

    // Step 3: GEMM: kernel_2d^T @ input_2d
    let col_matrix = gpu_gemm(
        ctx, pool, &kernel_2d, &input_2d, None, 1.0, 0.0, true, // Transpose kernel
        false,
    )?;

    // Step 4: col2im to get output
    let mut output = gpu_col2im(
        ctx,
        cache,
        pool,
        &col_matrix,
        batch_size,
        out_channels,
        out_h,
        out_w,
        params,
        height,
        width,
    )?;

    // Step 5: Add bias if provided
    if let Some(b) = bias {
        let spatial_size = out_h * out_w;
        add_bias(
            ctx,
            cache,
            &mut output,
            b,
            batch_size,
            out_channels,
            spatial_size,
        )?;
    }

    Ok(output)
}

/// GPU 1D Transposed Convolution (Deconvolution)
///
/// Input: [N, C_in, L]
/// Kernel: [C_in, C_out, K] (note: reversed from Conv1D)
/// Bias (optional): [C_out]
/// Output: [N, C_out, out_L]
///
/// Output length: out_L = (L - 1) * stride - 2 * padding + kernel_size + output_padding
pub fn gpu_conv_transpose_1d(
    ctx: &IconnxCudaContext,
    cache: &ConvKernelCache,
    pool: &mut GpuMemoryPool,
    input: &GpuTensor,
    kernel: &GpuTensor,
    bias: Option<&GpuTensor>,
    params: &Conv1dParams,
    output_padding: usize,
) -> Result<GpuTensor, CudaError> {
    let input_shape = input.shape();
    let kernel_shape = kernel.shape();

    assert_eq!(
        input_shape.len(),
        3,
        "ConvTranspose1D input must be 3D [N, C, L]"
    );
    assert_eq!(
        kernel_shape.len(),
        3,
        "ConvTranspose1D kernel must be 3D [C_in, C_out, K]"
    );

    let batch_size = input_shape[0];
    let in_channels = input_shape[1];
    let length = input_shape[2];

    let kernel_in_channels = kernel_shape[0];
    let out_channels = kernel_shape[1];
    let kernel_size = kernel_shape[2];

    assert_eq!(
        in_channels, kernel_in_channels,
        "Input channels ({}) must match kernel input channels ({})",
        in_channels, kernel_in_channels
    );
    assert_eq!(kernel_size, params.kernel_size, "Kernel size mismatch");

    // Calculate output length
    let out_length =
        (length - 1) * params.stride - 2 * params.padding + params.kernel_size + output_padding;

    // Step 1: Reshape input to [in_channels, batch_size * length]
    let input_cols = batch_size * length;
    let input_2d = input
        .reshape(vec![in_channels, input_cols])
        .ok_or_else(|| CudaError::Kernel("Failed to reshape input".to_string()))?;

    // Step 2: Reshape kernel to [in_channels, out_channels * kernel_size]
    let kernel_cols = out_channels * params.kernel_size;
    let kernel_2d = kernel
        .reshape(vec![in_channels, kernel_cols])
        .ok_or_else(|| CudaError::Kernel("Failed to reshape kernel".to_string()))?;

    // Step 3: GEMM: kernel_2d^T @ input_2d
    let col_matrix = gpu_gemm(
        ctx, pool, &kernel_2d, &input_2d, None, 1.0, 0.0, true, // Transpose kernel
        false,
    )?;

    // Step 4: col2im to get output
    let mut output = gpu_col2im_1d(
        ctx,
        cache,
        pool,
        &col_matrix,
        batch_size,
        out_channels,
        out_length,
        params,
        length,
    )?;

    // Step 5: Add bias if provided
    if let Some(b) = bias {
        add_bias(
            ctx,
            cache,
            &mut output,
            b,
            batch_size,
            out_channels,
            out_length,
        )?;
    }

    Ok(output)
}

/// GPU 1D Transposed Convolution with group support
///
/// Input: [N, C_in, L]
/// Kernel: [C_in, C_out/group, K]
/// Bias (optional): [C_out]
/// Output: [N, C_out, out_L]
///
/// For depthwise ConvTranspose (group == C_in):
///   - Each input channel gets its own kernel
///   - Kernel shape: [C_in, 1, K] -> produces [N, C_in, out_L]
///
/// This implementation uses GPU-side slicing and concatenation to avoid
/// expensive device-to-host transfers.
#[allow(dead_code)] // Kept as fallback; cuDNN is primary implementation
pub fn gpu_conv_transpose_1d_grouped(
    ctx: &IconnxCudaContext,
    conv_cache: &ConvKernelCache,
    ops_cache: &OpsKernelCache,
    pool: &mut GpuMemoryPool,
    input: &GpuTensor,
    kernel: &GpuTensor,
    bias: Option<&GpuTensor>,
    params: &Conv1dParams,
    output_padding: usize,
    group: usize,
) -> Result<GpuTensor, CudaError> {
    // If group=1, use standard implementation
    if group == 1 {
        return gpu_conv_transpose_1d(ctx, conv_cache, pool, input, kernel, bias, params, output_padding);
    }

    let input_shape = input.shape();
    let kernel_shape = kernel.shape();

    assert_eq!(
        input_shape.len(),
        3,
        "ConvTranspose1D input must be 3D [N, C, L]"
    );
    assert_eq!(
        kernel_shape.len(),
        3,
        "ConvTranspose1D kernel must be 3D [C_in, C_out/group, K]"
    );

    let batch_size = input_shape[0];
    let in_channels = input_shape[1];
    let length = input_shape[2];

    let kernel_in_channels = kernel_shape[0];
    let out_channels_per_group = kernel_shape[1];
    let kernel_size = kernel_shape[2];

    // For grouped conv, kernel has [C_in, C_out/group, K]
    // Total output channels = out_channels_per_group * group
    let out_channels = out_channels_per_group * group;
    let channels_per_group = in_channels / group;

    assert_eq!(
        in_channels, kernel_in_channels,
        "Input channels ({}) must match kernel input channels ({})",
        in_channels, kernel_in_channels
    );
    assert_eq!(
        in_channels % group,
        0,
        "Input channels ({}) must be divisible by group ({})",
        in_channels,
        group
    );
    assert_eq!(kernel_size, params.kernel_size, "Kernel size mismatch");

    // Process each group on GPU and collect outputs for concatenation
    let mut group_outputs: Vec<GpuTensor> = Vec::with_capacity(group);

    for g in 0..group {
        let in_start = (g * channels_per_group) as i64;
        let in_end = ((g + 1) * channels_per_group) as i64;

        // Extract input slice for this group using GPU: [N, channels_per_group, L]
        // Slice along axis 1 (channel dimension)
        let group_input = gpu_slice_nd(
            ctx,
            ops_cache,
            pool,
            input,
            &[0, in_start, 0],
            &[batch_size as i64, in_end, length as i64],
            Some(&[0, 1, 2]),
            None,
        )?;

        // Extract kernel slice for this group using GPU: [channels_per_group, out_channels_per_group, K]
        // Slice along axis 0 (input channel dimension)
        let group_kernel = gpu_slice_nd(
            ctx,
            ops_cache,
            pool,
            kernel,
            &[in_start, 0, 0],
            &[in_end, out_channels_per_group as i64, kernel_size as i64],
            Some(&[0, 1, 2]),
            None,
        )?;

        // Run standard ConvTranspose on this group (without bias - we'll add it at the end)
        let group_output = gpu_conv_transpose_1d(
            ctx,
            conv_cache,
            pool,
            &group_input,
            &group_kernel,
            None,
            params,
            output_padding,
        )?;

        group_outputs.push(group_output);
    }

    // Concatenate all group outputs along channel axis (axis 1) on GPU
    let output_refs: Vec<&GpuTensor> = group_outputs.iter().collect();
    let mut output = gpu_concat(ctx, ops_cache, pool, &output_refs, 1)?;

    // Add bias if provided
    if let Some(b) = bias {
        let out_length = output.shape()[2];
        add_bias(
            ctx,
            conv_cache,
            &mut output,
            b,
            batch_size,
            out_channels,
            out_length,
        )?;
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (IconnxCudaContext, ConvKernelCache, GpuMemoryPool) {
        let ctx = IconnxCudaContext::new().expect("Failed to create CUDA context");
        let cache = ConvKernelCache::new(&ctx).expect("Failed to compile kernels");
        let pool = GpuMemoryPool::new();
        (ctx, cache, pool)
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_conv_transpose_1d_simple() {
        let (ctx, cache, mut pool) = setup();

        let input_data = vec![1.0f32, 2.0, 3.0];
        let input_tensor = GpuTensor::from_host_f32(&ctx, &input_data, vec![1, 1, 3]).unwrap();

        let kernel_data = vec![1.0f32, 1.0, 1.0];
        let kernel_tensor = GpuTensor::from_host_f32(&ctx, &kernel_data, vec![1, 1, 3]).unwrap();

        let params = Conv1dParams {
            kernel_size: 3,
            stride: 1,
            padding: 0,
            dilation: 1,
        };

        let output = gpu_conv_transpose_1d(
            &ctx,
            &cache,
            &mut pool,
            &kernel_tensor,
            &input_tensor,
            None,
            &params,
            0,
        )
        .unwrap();
        let result = output.to_host_f32(&ctx).unwrap();

        assert_eq!(output.shape(), &[1, 1, 5]);
        let expected = [1.0, 3.0, 6.0, 5.0, 3.0];

        for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!((r - e).abs() < 1e-5, "Mismatch at {}: {} vs {}", i, r, e);
        }

        println!("ConvTranspose1D simple test passed!");
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_conv_transpose_1d_with_stride() {
        let (ctx, cache, mut pool) = setup();

        let input_data = vec![1.0f32, 2.0, 3.0];
        let input_tensor = GpuTensor::from_host_f32(&ctx, &input_data, vec![1, 1, 3]).unwrap();

        let kernel_data = vec![1.0f32, 1.0, 1.0];
        let kernel_tensor = GpuTensor::from_host_f32(&ctx, &kernel_data, vec![1, 1, 3]).unwrap();

        let params = Conv1dParams {
            kernel_size: 3,
            stride: 2,
            padding: 0,
            dilation: 1,
        };

        let output = gpu_conv_transpose_1d(
            &ctx,
            &cache,
            &mut pool,
            &kernel_tensor,
            &input_tensor,
            None,
            &params,
            0,
        )
        .unwrap();
        let result = output.to_host_f32(&ctx).unwrap();

        assert_eq!(output.shape(), &[1, 1, 7]);
        let expected = [1.0, 1.0, 3.0, 2.0, 5.0, 3.0, 3.0];

        for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!((r - e).abs() < 1e-5, "Mismatch at {}: {} vs {}", i, r, e);
        }

        println!("ConvTranspose1D with stride test passed!");
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_conv_transpose_2d_simple() {
        let (ctx, cache, mut pool) = setup();

        let input_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let input_tensor = GpuTensor::from_host_f32(&ctx, &input_data, vec![1, 1, 2, 2]).unwrap();

        let kernel_data = vec![1.0f32, 1.0, 1.0, 1.0];
        let kernel_tensor = GpuTensor::from_host_f32(&ctx, &kernel_data, vec![1, 1, 2, 2]).unwrap();

        let params = Conv2dParams {
            kernel_h: 2,
            kernel_w: 2,
            stride_h: 1,
            stride_w: 1,
            pad_h: 0,
            pad_w: 0,
        };

        let output = gpu_conv_transpose_2d(
            &ctx,
            &cache,
            &mut pool,
            &kernel_tensor,
            &input_tensor,
            None,
            &params,
            0,
            0,
        )
        .unwrap();
        let result = output.to_host_f32(&ctx).unwrap();

        assert_eq!(output.shape(), &[1, 1, 3, 3]);
        let expected = vec![1.0, 3.0, 2.0, 4.0, 10.0, 6.0, 3.0, 7.0, 4.0];

        for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!((r - e).abs() < 1e-5, "Mismatch at {}: {} vs {}", i, r, e);
        }

        println!("ConvTranspose2D simple test passed!");
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_conv_transpose_1d_with_bias() {
        let (ctx, cache, mut pool) = setup();

        let input_data = vec![1.0f32, 2.0, 3.0];
        let input_tensor = GpuTensor::from_host_f32(&ctx, &input_data, vec![1, 1, 3]).unwrap();

        let kernel_data = vec![1.0f32; 6];
        let kernel_tensor = GpuTensor::from_host_f32(&ctx, &kernel_data, vec![1, 2, 3]).unwrap();

        let bias_data = vec![10.0f32, 20.0];
        let bias_tensor = GpuTensor::from_host_f32(&ctx, &bias_data, vec![2]).unwrap();

        let params = Conv1dParams {
            kernel_size: 3,
            stride: 1,
            padding: 0,
            dilation: 1,
        };

        let output = gpu_conv_transpose_1d(
            &ctx,
            &cache,
            &mut pool,
            &kernel_tensor,
            &input_tensor,
            Some(&bias_tensor),
            &params,
            0,
        )
        .unwrap();
        let result = output.to_host_f32(&ctx).unwrap();

        assert_eq!(output.shape(), &[1, 2, 5]);

        let expected_ch0 = [11.0, 13.0, 16.0, 15.0, 13.0];
        let expected_ch1 = [21.0, 23.0, 26.0, 25.0, 23.0];

        for (i, (r, e)) in result[0..5].iter().zip(expected_ch0.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-5,
                "Channel 0 mismatch at {}: {} vs {}",
                i,
                r,
                e
            );
        }
        for (i, (r, e)) in result[5..10].iter().zip(expected_ch1.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-5,
                "Channel 1 mismatch at {}: {} vs {}",
                i,
                r,
                e
            );
        }

        println!("ConvTranspose1D with bias test passed!");
    }
}
