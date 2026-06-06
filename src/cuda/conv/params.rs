//! Convolution parameter structures

/// Parameters for 2D convolution
#[derive(Debug, Clone)]
pub struct Conv2dParams {
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub pad_h: usize,
    pub pad_w: usize,
    /// Number of convolution groups (1 = standard conv). `group == in_channels`
    /// is depthwise. The im2col `gpu_conv2d` path handles only `group == 1`;
    /// `group > 1` routes through `garboard_conv_2d` (cuDNN).
    pub group: usize,
    /// Dilation along H (1 = no dilation).
    pub dilation_h: usize,
    /// Dilation along W (1 = no dilation).
    pub dilation_w: usize,
}

impl Default for Conv2dParams {
    fn default() -> Self {
        Self {
            kernel_h: 1,
            kernel_w: 1,
            stride_h: 1,
            stride_w: 1,
            pad_h: 0,
            pad_w: 0,
            group: 1,
            dilation_h: 1,
            dilation_w: 1,
        }
    }
}

/// Parameters for 1D convolution
#[derive(Debug, Clone)]
pub struct Conv1dParams {
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
}

impl Default for Conv1dParams {
    fn default() -> Self {
        Self {
            kernel_size: 1,
            stride: 1,
            padding: 0,
            dilation: 1,
        }
    }
}
