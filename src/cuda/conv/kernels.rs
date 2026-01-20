//! CUDA kernel source code for convolution operations

/// Kernel names for loading from compiled module
pub const KERNEL_NAMES: &[&str] = &[
    "im2col_kernel",
    "col2im_kernel",
    "im2col_1d_kernel",
    "col2im_1d_kernel",
    "add_bias_kernel",
];

/// Convolution CUDA kernel source code
pub const CONV_KERNELS: &str = r#"
// im2col: Transform 2D convolution into matrix multiplication
// Input: [N, C, H, W] -> Output col_matrix: [C*kH*kW, N*out_H*out_W]
//
// Each thread handles one output column (one spatial position)
extern "C" __global__ void im2col_kernel(
    float* col_matrix,       // Output: [col_height, col_width]
    const float* input,      // Input: [N, C, H, W]
    size_t batch_size,
    size_t channels,
    size_t height,
    size_t width,
    size_t kernel_h,
    size_t kernel_w,
    size_t stride_h,
    size_t stride_w,
    size_t pad_h,
    size_t pad_w,
    size_t out_h,
    size_t out_w
) {
    // Each thread handles one output column
    size_t col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_cols = batch_size * out_h * out_w;

    if (col_idx >= total_cols) return;

    // Decompose col_idx into batch, out_y, out_x
    size_t batch = col_idx / (out_h * out_w);
    size_t spatial_idx = col_idx % (out_h * out_w);
    size_t out_y = spatial_idx / out_w;
    size_t out_x = spatial_idx % out_w;

    // Input start position (with stride)
    size_t in_y_start = out_y * stride_h;
    size_t in_x_start = out_x * stride_w;

    // Fill this column of col_matrix
    size_t col_height = channels * kernel_h * kernel_w;
    size_t row_idx = 0;

    for (size_t c = 0; c < channels; c++) {
        for (size_t ky = 0; ky < kernel_h; ky++) {
            for (size_t kx = 0; kx < kernel_w; kx++) {
                size_t in_y = in_y_start + ky;
                size_t in_x = in_x_start + kx;

                float val = 0.0f;

                // Check padding bounds
                if (in_y >= pad_h && in_y < height + pad_h &&
                    in_x >= pad_w && in_x < width + pad_w) {
                    size_t actual_y = in_y - pad_h;
                    size_t actual_x = in_x - pad_w;
                    size_t input_idx = batch * (channels * height * width) +
                                       c * (height * width) +
                                       actual_y * width +
                                       actual_x;
                    val = input[input_idx];
                }

                // Write to col_matrix [row_idx, col_idx]
                col_matrix[row_idx * total_cols + col_idx] = val;
                row_idx++;
            }
        }
    }
}

// col2im: Inverse of im2col for ConvTranspose
// Scatters column values back to spatial positions
extern "C" __global__ void col2im_kernel(
    float* output,           // Output: [N, C, H, W]
    const float* col_matrix, // Input: [col_height, col_width]
    size_t batch_size,
    size_t channels,
    size_t height,
    size_t width,
    size_t kernel_h,
    size_t kernel_w,
    size_t stride_h,
    size_t stride_w,
    size_t pad_h,
    size_t pad_w,
    size_t out_h,       // Input spatial dimensions (before transpose)
    size_t out_w
) {
    // Each thread handles one output pixel
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_output = batch_size * channels * height * width;

    if (idx >= total_output) return;

    // Decompose idx into batch, channel, y, x
    size_t batch = idx / (channels * height * width);
    size_t rem = idx % (channels * height * width);
    size_t c = rem / (height * width);
    rem = rem % (height * width);
    size_t y = rem / width;
    size_t x = rem % width;

    // Sum contributions from all kernel positions that could write to this pixel
    float sum = 0.0f;
    size_t col_width = batch_size * out_h * out_w;

    for (size_t ky = 0; ky < kernel_h; ky++) {
        for (size_t kx = 0; kx < kernel_w; kx++) {
            // Find which output position would write to (y, x) using kernel position (ky, kx)
            // y = out_y * stride_h + ky - pad_h
            // out_y = (y + pad_h - ky) / stride_h

            size_t y_padded = y + pad_h;
            size_t x_padded = x + pad_w;

            if (y_padded < ky || x_padded < kx) continue;

            size_t diff_y = y_padded - ky;
            size_t diff_x = x_padded - kx;

            if (diff_y % stride_h != 0 || diff_x % stride_w != 0) continue;

            size_t out_y = diff_y / stride_h;
            size_t out_x = diff_x / stride_w;

            if (out_y >= out_h || out_x >= out_w) continue;

            // Calculate row index in col_matrix
            size_t row_idx = c * kernel_h * kernel_w + ky * kernel_w + kx;
            size_t col_idx = batch * out_h * out_w + out_y * out_w + out_x;

            sum += col_matrix[row_idx * col_width + col_idx];
        }
    }

    output[idx] = sum;
}

// im2col for 1D convolution
// Input: [N, C, L] -> Output: [C*K, N*out_L]
extern "C" __global__ void im2col_1d_kernel(
    float* col_matrix,
    const float* input,
    size_t batch_size,
    size_t channels,
    size_t length,
    size_t kernel_size,
    size_t stride,
    size_t padding,
    size_t dilation,
    size_t out_length
) {
    size_t col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_cols = batch_size * out_length;

    if (col_idx >= total_cols) return;

    size_t batch = col_idx / out_length;
    size_t out_pos = col_idx % out_length;

    size_t in_pos_start = out_pos * stride;
    size_t col_height = channels * kernel_size;
    size_t row_idx = 0;

    for (size_t c = 0; c < channels; c++) {
        for (size_t k = 0; k < kernel_size; k++) {
            // Apply dilation
            size_t in_pos = in_pos_start + k * dilation;

            float val = 0.0f;
            if (in_pos >= padding && in_pos < length + padding) {
                size_t actual_pos = in_pos - padding;
                size_t input_idx = batch * (channels * length) + c * length + actual_pos;
                val = input[input_idx];
            }

            col_matrix[row_idx * total_cols + col_idx] = val;
            row_idx++;
        }
    }
}

// col2im for 1D ConvTranspose
extern "C" __global__ void col2im_1d_kernel(
    float* output,
    const float* col_matrix,
    size_t batch_size,
    size_t channels,
    size_t length,
    size_t kernel_size,
    size_t stride,
    size_t padding,
    size_t out_length
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_output = batch_size * channels * length;

    if (idx >= total_output) return;

    size_t batch = idx / (channels * length);
    size_t rem = idx % (channels * length);
    size_t c = rem / length;
    size_t pos = rem % length;

    float sum = 0.0f;
    size_t col_width = batch_size * out_length;

    for (size_t k = 0; k < kernel_size; k++) {
        size_t pos_padded = pos + padding;

        if (pos_padded < k) continue;

        size_t diff = pos_padded - k;
        if (diff % stride != 0) continue;

        size_t out_pos = diff / stride;
        if (out_pos >= out_length) continue;

        size_t row_idx = c * kernel_size + k;
        size_t col_idx = batch * out_length + out_pos;

        sum += col_matrix[row_idx * col_width + col_idx];
    }

    output[idx] = sum;
}

// Add bias to conv output: output[n, c, h, w] += bias[c]
extern "C" __global__ void add_bias_kernel(
    float* output,
    const float* bias,
    size_t batch_size,
    size_t channels,
    size_t spatial_size  // H*W or L
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch_size * channels * spatial_size;

    if (idx >= total) return;

    // Determine which channel this element belongs to
    size_t channel = (idx / spatial_size) % channels;
    output[idx] += bias[channel];
}
"#;
