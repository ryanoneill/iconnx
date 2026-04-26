//! Forward convolution operations (im2col + GEMM)

use super::cache::ConvKernelCache;
use super::params::{Conv1dParams, Conv2dParams};
use crate::cuda::context::{CudaError, IconnxCudaContext};
use crate::cuda::matmul::gpu_gemm;
use crate::cuda::memory_pool::GpuMemoryPool;
use crate::cuda::tensor::{DType, GpuTensor};
use garboard::{DeviceSlice, LaunchConfig};

/// GPU im2col transformation for 2D convolution
///
/// Transforms [N, C, H, W] into [C*kH*kW, N*out_H*out_W]. Float32 and
/// Float16 supported (M3.6 light-up: gpu_conv2d → gpu_im2col → gpu_gemm
/// only stays FP16 if both seams accept it).
pub fn gpu_im2col(
    ctx: &IconnxCudaContext,
    cache: &ConvKernelCache,
    pool: &mut GpuMemoryPool,
    input: &GpuTensor,
    params: &Conv2dParams,
) -> Result<GpuTensor, CudaError> {
    let shape = input.shape();
    assert_eq!(shape.len(), 4, "im2col input must be 4D [N, C, H, W]");

    let batch_size = shape[0];
    let channels = shape[1];
    let height = shape[2];
    let width = shape[3];

    let out_h = (height + 2 * params.pad_h - params.kernel_h) / params.stride_h + 1;
    let out_w = (width + 2 * params.pad_w - params.kernel_w) / params.stride_w + 1;

    let col_height = channels * params.kernel_h * params.kernel_w;
    let col_width = batch_size * out_h * out_w;
    let launch_cfg = LaunchConfig::for_num_elems(col_width as u32);

    match input.dtype() {
        DType::Float32 => {
            let mut output = pool.get_tensor_f32(ctx, vec![col_height, col_width])?;
            // SAFETY: im2col_kernel signature is
            // `(float* out, const float* inp, size_t N, size_t C, size_t H, size_t W,
            //   size_t kH, size_t kW, size_t sH, size_t sW, size_t pH, size_t pW,
            //   size_t out_H, size_t out_W)`.
            let kernel = unsafe {
                cache.module().typed_kernel::<(
                    &mut DeviceSlice<'_, f32>,
                    &DeviceSlice<'_, f32>,
                    usize, usize, usize, usize,
                    usize, usize, usize, usize,
                    usize, usize, usize, usize,
                )>("im2col_kernel")
            }
            .map_err(|e| CudaError::Kernel(format!("im2col_kernel lookup failed: {}", e)))?;

            kernel
                .launch(
                    ctx.garboard_stream(),
                    &launch_cfg,
                    (
                        output.data_f32_mut()?,
                        input.data_f32()?,
                        batch_size, channels, height, width,
                        params.kernel_h, params.kernel_w,
                        params.stride_h, params.stride_w,
                        params.pad_h, params.pad_w,
                        out_h, out_w,
                    ),
                )
                .map_err(|e| CudaError::Kernel(format!("im2col launch failed: {}", e)))?;
            Ok(output)
        }
        DType::Float16 => {
            let mut output = pool.get_tensor_f16(ctx, vec![col_height, col_width])?;
            // SAFETY: im2col_f16_kernel signature mirrors im2col_kernel
            // with `unsigned short*` (2-byte) data pointers — half::f16
            // is `repr(transparent) u16`, byte-identical layout.
            let kernel = unsafe {
                cache.module().typed_kernel::<(
                    &mut DeviceSlice<'_, half::f16>,
                    &DeviceSlice<'_, half::f16>,
                    usize, usize, usize, usize,
                    usize, usize, usize, usize,
                    usize, usize, usize, usize,
                )>("im2col_f16_kernel")
            }
            .map_err(|e| CudaError::Kernel(format!("im2col_f16_kernel lookup failed: {}", e)))?;

            kernel
                .launch(
                    ctx.garboard_stream(),
                    &launch_cfg,
                    (
                        output.data_f16_mut()?,
                        input.data_f16()?,
                        batch_size, channels, height, width,
                        params.kernel_h, params.kernel_w,
                        params.stride_h, params.stride_w,
                        params.pad_h, params.pad_w,
                        out_h, out_w,
                    ),
                )
                .map_err(|e| CudaError::Kernel(format!("im2col_f16 launch failed: {}", e)))?;
            Ok(output)
        }
        dt => Err(CudaError::Kernel(format!(
            "im2col: unsupported dtype {} (Float32 + Float16 supported)",
            dt.name()
        ))),
    }
}

/// GPU im2col transformation for 1D convolution
///
/// Transforms [N, C, L] into [C*K, N*out_L]. Float32 + Float16 supported.
pub fn gpu_im2col_1d(
    ctx: &IconnxCudaContext,
    cache: &ConvKernelCache,
    pool: &mut GpuMemoryPool,
    input: &GpuTensor,
    params: &Conv1dParams,
) -> Result<GpuTensor, CudaError> {
    let shape = input.shape();
    assert_eq!(shape.len(), 3, "im2col_1d input must be 3D [N, C, L]");

    let batch_size = shape[0];
    let channels = shape[1];
    let length = shape[2];

    let effective_k = (params.kernel_size - 1) * params.dilation + 1;
    let out_length = (length + 2 * params.padding - effective_k) / params.stride + 1;

    let col_height = channels * params.kernel_size;
    let col_width = batch_size * out_length;
    let launch_cfg = LaunchConfig::for_num_elems(col_width as u32);

    match input.dtype() {
        DType::Float32 => {
            let mut output = pool.get_tensor_f32(ctx, vec![col_height, col_width])?;
            // SAFETY: im2col_1d_kernel signature is
            // `(float* out, const float* inp, size_t N, size_t C, size_t L,
            //   size_t K, size_t stride, size_t pad, size_t dilation, size_t out_L)`.
            let kernel = unsafe {
                cache.module().typed_kernel::<(
                    &mut DeviceSlice<'_, f32>,
                    &DeviceSlice<'_, f32>,
                    usize, usize, usize,
                    usize, usize, usize, usize, usize,
                )>("im2col_1d_kernel")
            }
            .map_err(|e| CudaError::Kernel(format!("im2col_1d_kernel lookup failed: {}", e)))?;

            kernel
                .launch(
                    ctx.garboard_stream(),
                    &launch_cfg,
                    (
                        output.data_f32_mut()?,
                        input.data_f32()?,
                        batch_size, channels, length,
                        params.kernel_size, params.stride,
                        params.padding, params.dilation, out_length,
                    ),
                )
                .map_err(|e| CudaError::Kernel(format!("im2col_1d launch failed: {}", e)))?;
            Ok(output)
        }
        DType::Float16 => {
            let mut output = pool.get_tensor_f16(ctx, vec![col_height, col_width])?;
            // SAFETY: im2col_1d_f16_kernel signature mirrors im2col_1d_kernel
            // with `unsigned short*` data pointers (FP16 byte-identical to u16).
            let kernel = unsafe {
                cache.module().typed_kernel::<(
                    &mut DeviceSlice<'_, half::f16>,
                    &DeviceSlice<'_, half::f16>,
                    usize, usize, usize,
                    usize, usize, usize, usize, usize,
                )>("im2col_1d_f16_kernel")
            }
            .map_err(|e| {
                CudaError::Kernel(format!("im2col_1d_f16_kernel lookup failed: {}", e))
            })?;

            kernel
                .launch(
                    ctx.garboard_stream(),
                    &launch_cfg,
                    (
                        output.data_f16_mut()?,
                        input.data_f16()?,
                        batch_size, channels, length,
                        params.kernel_size, params.stride,
                        params.padding, params.dilation, out_length,
                    ),
                )
                .map_err(|e| CudaError::Kernel(format!("im2col_1d_f16 launch failed: {}", e)))?;
            Ok(output)
        }
        dt => Err(CudaError::Kernel(format!(
            "im2col_1d: unsupported dtype {}",
            dt.name()
        ))),
    }
}

/// GPU 2D Convolution using im2col + cuBLAS GEMM
///
/// Input: [N, C_in, H, W]
/// Kernel: [C_out, C_in, kH, kW]
/// Bias (optional): [C_out]
/// Output: [N, C_out, out_H, out_W]
pub fn gpu_conv2d(
    ctx: &IconnxCudaContext,
    cache: &ConvKernelCache,
    pool: &mut GpuMemoryPool,
    input: &GpuTensor,
    kernel: &GpuTensor,
    bias: Option<&GpuTensor>,
    params: &Conv2dParams,
) -> Result<GpuTensor, CudaError> {
    let input_shape = input.shape();
    let kernel_shape = kernel.shape();

    assert_eq!(input_shape.len(), 4, "Conv2D input must be 4D [N, C, H, W]");
    assert_eq!(
        kernel_shape.len(),
        4,
        "Conv2D kernel must be 4D [C_out, C_in, kH, kW]"
    );

    let batch_size = input_shape[0];
    let in_channels = input_shape[1];
    let height = input_shape[2];
    let width = input_shape[3];

    let out_channels = kernel_shape[0];
    let kernel_in_channels = kernel_shape[1];

    assert_eq!(
        in_channels, kernel_in_channels,
        "Input channels ({}) must match kernel input channels ({})",
        in_channels, kernel_in_channels
    );
    assert_eq!(kernel_shape[2], params.kernel_h, "Kernel height mismatch");
    assert_eq!(kernel_shape[3], params.kernel_w, "Kernel width mismatch");

    // Calculate output dimensions
    let out_h = (height + 2 * params.pad_h - params.kernel_h) / params.stride_h + 1;
    let out_w = (width + 2 * params.pad_w - params.kernel_w) / params.stride_w + 1;

    // Step 1: im2col transformation
    let col_matrix = gpu_im2col(ctx, cache, pool, input, params)?;

    // Step 2: Reshape kernel to [C_out, C_in * kH * kW]
    let kernel_rows = out_channels;
    let kernel_cols = in_channels * params.kernel_h * params.kernel_w;
    let kernel_2d = kernel
        .clone()
        .reshape(vec![kernel_rows, kernel_cols])
        .ok_or_else(|| CudaError::Kernel("Failed to reshape kernel".to_string()))?;

    // Step 3: GEMM: kernel_2d @ col_matrix
    let conv_result = gpu_gemm(ctx, pool, &kernel_2d, &col_matrix, None, 1.0, 0.0, false, false)?;

    // Step 4: Reshape to [N, C_out, out_H, out_W]
    let spatial_size = out_h * out_w;
    let mut output = conv_result
        .reshape(vec![batch_size, out_channels, out_h, out_w])
        .ok_or_else(|| CudaError::Kernel("Failed to reshape conv output".to_string()))?;

    // Step 5: Add bias if provided
    if let Some(b) = bias {
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

/// GPU 1D Convolution using im2col + cuBLAS GEMM
///
/// Input: [N, C_in, L]
/// Kernel: [C_out, C_in, K]
/// Bias (optional): [C_out]
/// Output: [N, C_out, out_L]
pub fn gpu_conv1d(
    ctx: &IconnxCudaContext,
    cache: &ConvKernelCache,
    pool: &mut GpuMemoryPool,
    input: &GpuTensor,
    kernel: &GpuTensor,
    bias: Option<&GpuTensor>,
    params: &Conv1dParams,
) -> Result<GpuTensor, CudaError> {
    let input_shape = input.shape();
    let kernel_shape = kernel.shape();

    assert_eq!(input_shape.len(), 3, "Conv1D input must be 3D [N, C, L]");
    assert_eq!(
        kernel_shape.len(),
        3,
        "Conv1D kernel must be 3D [C_out, C_in, K]"
    );

    let batch_size = input_shape[0];
    let in_channels = input_shape[1];
    let length = input_shape[2];

    let out_channels = kernel_shape[0];
    let kernel_in_channels = kernel_shape[1];
    let kernel_size = kernel_shape[2];

    assert_eq!(
        in_channels, kernel_in_channels,
        "Input channels ({}) must match kernel input channels ({})",
        in_channels, kernel_in_channels
    );
    assert_eq!(kernel_size, params.kernel_size, "Kernel size mismatch");

    // Calculate output length
    let effective_k = (params.kernel_size - 1) * params.dilation + 1;
    let out_length = (length + 2 * params.padding - effective_k) / params.stride + 1;

    // Step 1: im2col transformation
    let col_matrix = gpu_im2col_1d(ctx, cache, pool, input, params)?;

    // Step 2: Reshape kernel to [C_out, C_in * K]
    let kernel_rows = out_channels;
    let kernel_cols = in_channels * params.kernel_size;
    let kernel_2d = kernel
        .clone()
        .reshape(vec![kernel_rows, kernel_cols])
        .ok_or_else(|| CudaError::Kernel("Failed to reshape kernel".to_string()))?;

    // Step 3: GEMM
    let conv_result = gpu_gemm(ctx, pool, &kernel_2d, &col_matrix, None, 1.0, 0.0, false, false)?;

    // Step 4: Reshape to [N, C_out, out_L]
    let mut output = conv_result
        .reshape(vec![batch_size, out_channels, out_length])
        .ok_or_else(|| CudaError::Kernel("Failed to reshape conv1d output".to_string()))?;

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

/// Add bias to convolution output. Float32 + Float16 supported.
pub fn add_bias(
    ctx: &IconnxCudaContext,
    cache: &ConvKernelCache,
    output: &mut GpuTensor,
    bias: &GpuTensor,
    batch_size: usize,
    out_channels: usize,
    spatial_size: usize,
) -> Result<(), CudaError> {
    let total_elements = batch_size * out_channels * spatial_size;
    let launch_cfg = LaunchConfig::for_num_elems(total_elements as u32);

    if output.dtype() != bias.dtype() {
        return Err(CudaError::Kernel(format!(
            "add_bias: output/bias dtype mismatch ({} vs {})",
            output.dtype().name(),
            bias.dtype().name()
        )));
    }

    match output.dtype() {
        DType::Float32 => {
            // SAFETY: add_bias_kernel signature is
            // `(float* output, const float* bias, size_t batch, size_t C, size_t spatial)`.
            let kernel = unsafe {
                cache.module().typed_kernel::<(
                    &mut DeviceSlice<'_, f32>,
                    &DeviceSlice<'_, f32>,
                    usize, usize, usize,
                )>("add_bias_kernel")
            }
            .map_err(|e| CudaError::Kernel(format!("add_bias_kernel lookup failed: {}", e)))?;

            kernel
                .launch(
                    ctx.garboard_stream(),
                    &launch_cfg,
                    (
                        output.data_f32_mut()?,
                        bias.data_f32()?,
                        batch_size, out_channels, spatial_size,
                    ),
                )
                .map_err(|e| CudaError::Kernel(format!("add_bias launch failed: {}", e)))?;
        }
        DType::Float16 => {
            // SAFETY: add_bias_f16_kernel signature is
            // `(__half* output, const __half* bias, size_t batch, size_t C, size_t spatial)`.
            let kernel = unsafe {
                cache.module().typed_kernel::<(
                    &mut DeviceSlice<'_, half::f16>,
                    &DeviceSlice<'_, half::f16>,
                    usize, usize, usize,
                )>("add_bias_f16_kernel")
            }
            .map_err(|e| {
                CudaError::Kernel(format!("add_bias_f16_kernel lookup failed: {}", e))
            })?;

            kernel
                .launch(
                    ctx.garboard_stream(),
                    &launch_cfg,
                    (
                        output.data_f16_mut()?,
                        bias.data_f16()?,
                        batch_size, out_channels, spatial_size,
                    ),
                )
                .map_err(|e| CudaError::Kernel(format!("add_bias_f16 launch failed: {}", e)))?;
        }
        dt => {
            return Err(CudaError::Kernel(format!(
                "add_bias: unsupported dtype {}",
                dt.name()
            )))
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::memory_pool::GpuMemoryPool;

    fn setup() -> (IconnxCudaContext, ConvKernelCache, GpuMemoryPool) {
        let ctx = IconnxCudaContext::new().expect("Failed to create CUDA context");
        let cache = ConvKernelCache::new(&ctx).expect("Failed to compile kernels");
        let pool = GpuMemoryPool::new();
        (ctx, cache, pool)
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_conv2d_simple() {
        let (ctx, cache, mut pool) = setup();

        // Input: [1, 1, 4, 4] - single batch, single channel, 4x4 image
        let input_data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let input_tensor = GpuTensor::from_host_f32(&ctx, &input_data, vec![1, 1, 4, 4]).unwrap();

        // Kernel: [1, 1, 2, 2] - single output channel, 2x2 kernel
        let kernel_data = vec![1.0f32, 0.0, 0.0, 1.0]; // Diagonal kernel
        let kernel_tensor = GpuTensor::from_host_f32(&ctx, &kernel_data, vec![1, 1, 2, 2]).unwrap();

        let params = Conv2dParams {
            kernel_h: 2,
            kernel_w: 2,
            stride_h: 1,
            stride_w: 1,
            pad_h: 0,
            pad_w: 0,
        };

        let output =
            gpu_conv2d(&ctx, &cache, &mut pool, &input_tensor, &kernel_tensor, None, &params).unwrap();
        let result = output.to_host_f32(&ctx).unwrap();

        assert_eq!(output.shape(), &[1, 1, 3, 3]);
        let expected = vec![7.0, 9.0, 11.0, 15.0, 17.0, 19.0, 23.0, 25.0, 27.0];

        for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!((r - e).abs() < 1e-5, "Mismatch at {}: {} vs {}", i, r, e);
        }

        println!("Conv2D simple test passed!");
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_conv2d_with_bias() {
        let (ctx, cache, mut pool) = setup();

        let input_data = vec![1.0f32; 9];
        let input_tensor = GpuTensor::from_host_f32(&ctx, &input_data, vec![1, 1, 3, 3]).unwrap();

        let kernel_data = vec![1.0f32; 8];
        let kernel_tensor = GpuTensor::from_host_f32(&ctx, &kernel_data, vec![2, 1, 2, 2]).unwrap();

        let bias_data = vec![10.0f32, 20.0];
        let bias_tensor = GpuTensor::from_host_f32(&ctx, &bias_data, vec![2]).unwrap();

        let params = Conv2dParams {
            kernel_h: 2,
            kernel_w: 2,
            stride_h: 1,
            stride_w: 1,
            pad_h: 0,
            pad_w: 0,
        };

        let output = gpu_conv2d(
            &ctx,
            &cache,
            &mut pool,
            &input_tensor,
            &kernel_tensor,
            Some(&bias_tensor),
            &params,
        )
        .unwrap();
        let result = output.to_host_f32(&ctx).unwrap();

        assert_eq!(output.shape(), &[1, 2, 2, 2]);

        for (i, &val) in result.iter().enumerate().take(4) {
            assert!(
                (val - 14.0).abs() < 1e-5,
                "Channel 0 mismatch at {}: {}",
                i,
                val
            );
        }
        for (i, &val) in result.iter().enumerate().take(8).skip(4) {
            assert!(
                (val - 24.0).abs() < 1e-5,
                "Channel 1 mismatch at {}: {}",
                i,
                val
            );
        }

        println!("Conv2D with bias test passed!");
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_conv1d_simple() {
        let (ctx, cache, mut pool) = setup();

        let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let input_tensor = GpuTensor::from_host_f32(&ctx, &input_data, vec![1, 1, 5]).unwrap();

        let kernel_data = vec![1.0f32, 2.0, 1.0];
        let kernel_tensor = GpuTensor::from_host_f32(&ctx, &kernel_data, vec![1, 1, 3]).unwrap();

        let params = Conv1dParams {
            kernel_size: 3,
            stride: 1,
            padding: 0,
            dilation: 1,
        };

        let output =
            gpu_conv1d(&ctx, &cache, &mut pool, &input_tensor, &kernel_tensor, None, &params).unwrap();
        let result = output.to_host_f32(&ctx).unwrap();

        assert_eq!(output.shape(), &[1, 1, 3]);
        let expected = [8.0, 12.0, 16.0];

        for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!((r - e).abs() < 1e-5, "Mismatch at {}: {} vs {}", i, r, e);
        }

        println!("Conv1D simple test passed!");
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_conv1d_with_dilation() {
        let (ctx, cache, mut pool) = setup();

        let input_data: Vec<f32> = (1..=7).map(|x| x as f32).collect();
        let input_tensor = GpuTensor::from_host_f32(&ctx, &input_data, vec![1, 1, 7]).unwrap();

        let kernel_data = vec![1.0f32, 1.0, 1.0];
        let kernel_tensor = GpuTensor::from_host_f32(&ctx, &kernel_data, vec![1, 1, 3]).unwrap();

        let params = Conv1dParams {
            kernel_size: 3,
            stride: 1,
            padding: 0,
            dilation: 2,
        };

        let output =
            gpu_conv1d(&ctx, &cache, &mut pool, &input_tensor, &kernel_tensor, None, &params).unwrap();
        let result = output.to_host_f32(&ctx).unwrap();

        assert_eq!(output.shape(), &[1, 1, 3]);
        let expected = [9.0, 12.0, 15.0];

        for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!((r - e).abs() < 1e-5, "Mismatch at {}: {} vs {}", i, r, e);
        }

        println!("Conv1D with dilation test passed!");
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_conv2d_multi_channel() {
        let (ctx, cache, mut pool) = setup();

        let input_data: Vec<f32> = (1..=18).map(|x| x as f32).collect();
        let input_tensor = GpuTensor::from_host_f32(&ctx, &input_data, vec![1, 2, 3, 3]).unwrap();

        let kernel_data = vec![1.0f32; 8];
        let kernel_tensor = GpuTensor::from_host_f32(&ctx, &kernel_data, vec![1, 2, 2, 2]).unwrap();

        let params = Conv2dParams {
            kernel_h: 2,
            kernel_w: 2,
            stride_h: 1,
            stride_w: 1,
            pad_h: 0,
            pad_w: 0,
        };

        let output =
            gpu_conv2d(&ctx, &cache, &mut pool, &input_tensor, &kernel_tensor, None, &params).unwrap();
        let result = output.to_host_f32(&ctx).unwrap();

        assert_eq!(output.shape(), &[1, 1, 2, 2]);
        let expected = [60.0, 68.0, 84.0, 92.0];

        for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!((r - e).abs() < 1e-5, "Mismatch at {}: {} vs {}", i, r, e);
        }

        println!("Conv2D multi-channel test passed!");
    }
}
