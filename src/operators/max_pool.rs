//! ONNX MaxPool: slide a window over the spatial dims of an N-D
//! input and emit the elementwise max per window.
//!
//! Scope (matches ResNet-50 / YOLOS-tiny / typical CNN stems):
//!   * 4-D NCHW input.
//!   * `kernel_shape` (required, 2 ints), `strides` (default [1, 1]),
//!     `pads` (default [0, 0, 0, 0] in `[top, left, bottom, right]`
//!     order), `dilations` (default [1, 1]), `auto_pad` only NOTSET
//!     supported, `ceil_mode` 0 or 1, `storage_order` 0 only.
//!   * Optional second `Indices` output is not produced — almost
//!     no production model requests it; iconnx surfaces this via
//!     the executor when (and if) the second output is consumed.
//!
//! ResNet-50 stem MaxPool: `kernel_shape=[3,3]`, `strides=[2,2]`,
//! `pads=[1,1,1,1]`, input `[N, 64, 112, 112]` → output
//! `[N, 64, 56, 56]`. Verified by the `max_pool_resnet_stem`
//! test below.

use crate::attributes::NodeAttributes;
use crate::tensor::Tensor;
use ndarray::{ArrayD, IxDyn};

/// ONNX MaxPool.
pub struct MaxPool;

/// Resolved 2-D pooling parameters.
#[derive(Clone, Copy, Debug)]
pub struct MaxPool2dParams {
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub pad_top: usize,
    pub pad_left: usize,
    pub pad_bottom: usize,
    pub pad_right: usize,
    pub dilation_h: usize,
    pub dilation_w: usize,
    pub ceil_mode: bool,
}

impl MaxPool2dParams {
    /// Output spatial size for one dim, per ONNX:
    /// floor or ceil of `((D + p_lo + p_hi - dilation*(K-1) - 1) / stride + 1)`.
    pub fn out_dim(in_dim: usize, kernel: usize, stride: usize, p_lo: usize, p_hi: usize, dilation: usize, ceil_mode: bool) -> usize {
        let effective_kernel = dilation * (kernel - 1) + 1;
        let numerator = (in_dim + p_lo + p_hi).saturating_sub(effective_kernel);
        if ceil_mode {
            numerator.div_ceil(stride) + 1
        } else {
            numerator / stride + 1
        }
    }
}

impl MaxPool {
    /// Resolve attributes into [`MaxPool2dParams`] for 2-D MaxPool.
    /// Returns an error string if the attribute set isn't supported
    /// (auto_pad != NOTSET, kernel_shape rank != 2, etc.).
    pub fn params_2d(attrs: &NodeAttributes) -> Result<MaxPool2dParams, String> {
        let auto_pad = attrs.get_string("auto_pad").unwrap_or("NOTSET");
        if auto_pad != "NOTSET" {
            return Err(format!(
                "MaxPool: auto_pad={} not supported (only NOTSET)",
                auto_pad
            ));
        }
        let storage_order = attrs.get_int("storage_order").unwrap_or(0);
        if storage_order != 0 {
            return Err(format!(
                "MaxPool: storage_order={} not supported (only 0)",
                storage_order
            ));
        }
        let kernel_shape = attrs.get_ints("kernel_shape").ok_or_else(|| {
            "MaxPool: kernel_shape attribute is required".to_string()
        })?;
        if kernel_shape.len() != 2 {
            return Err(format!(
                "MaxPool: only 2-D kernels supported, got rank {}",
                kernel_shape.len()
            ));
        }
        let strides = attrs.get_ints("strides").map(|s| s.to_vec()).unwrap_or_else(|| vec![1, 1]);
        let pads = attrs.get_ints("pads").map(|p| p.to_vec()).unwrap_or_else(|| vec![0, 0, 0, 0]);
        let dilations = attrs.get_ints("dilations").map(|d| d.to_vec()).unwrap_or_else(|| vec![1, 1]);
        if strides.len() != 2 || dilations.len() != 2 {
            return Err(format!(
                "MaxPool: strides and dilations must have rank 2, got {}/{}",
                strides.len(),
                dilations.len()
            ));
        }
        if pads.len() != 4 {
            return Err(format!(
                "MaxPool: pads must have 4 elements [top, left, bottom, right], got {}",
                pads.len()
            ));
        }
        Ok(MaxPool2dParams {
            kernel_h: kernel_shape[0] as usize,
            kernel_w: kernel_shape[1] as usize,
            stride_h: strides[0] as usize,
            stride_w: strides[1] as usize,
            pad_top: pads[0] as usize,
            pad_left: pads[1] as usize,
            pad_bottom: pads[2] as usize,
            pad_right: pads[3] as usize,
            dilation_h: dilations[0] as usize,
            dilation_w: dilations[1] as usize,
            ceil_mode: attrs.get_int("ceil_mode").unwrap_or(0) != 0,
        })
    }

    pub fn forward(inputs: &[Tensor], attrs: &NodeAttributes) -> Tensor {
        let input = &inputs[0];
        let arr = match input {
            Tensor::Float32(a) => a,
            other => panic!("MaxPool: only Float32 supported, got {}", other.dtype()),
        };
        let p = Self::params_2d(attrs).expect("MaxPool params");

        let shape = arr.shape();
        assert_eq!(shape.len(), 4, "MaxPool: input must be 4-D NCHW");
        let n = shape[0];
        let c = shape[1];
        let ih = shape[2];
        let iw = shape[3];

        let oh = MaxPool2dParams::out_dim(ih, p.kernel_h, p.stride_h, p.pad_top, p.pad_bottom, p.dilation_h, p.ceil_mode);
        let ow = MaxPool2dParams::out_dim(iw, p.kernel_w, p.stride_w, p.pad_left, p.pad_right, p.dilation_w, p.ceil_mode);

        let mut out = vec![f32::NEG_INFINITY; n * c * oh * ow];

        for ni in 0..n {
            for ci in 0..c {
                for ohi in 0..oh {
                    for owi in 0..ow {
                        let mut maxv = f32::NEG_INFINITY;
                        let mut any = false;
                        let ih_start = (ohi * p.stride_h) as isize - p.pad_top as isize;
                        let iw_start = (owi * p.stride_w) as isize - p.pad_left as isize;
                        for ki in 0..p.kernel_h {
                            let ihi = ih_start + (ki * p.dilation_h) as isize;
                            if ihi < 0 || ihi as usize >= ih {
                                continue;
                            }
                            for kj in 0..p.kernel_w {
                                let iwi = iw_start + (kj * p.dilation_w) as isize;
                                if iwi < 0 || iwi as usize >= iw {
                                    continue;
                                }
                                let v = arr[[ni, ci, ihi as usize, iwi as usize]];
                                if !any || v > maxv {
                                    maxv = v;
                                }
                                any = true;
                            }
                        }
                        out[((ni * c + ci) * oh + ohi) * ow + owi] = if any { maxv } else { f32::NEG_INFINITY };
                    }
                }
            }
        }

        Tensor::Float32(
            ArrayD::from_shape_vec(IxDyn(&[n, c, oh, ow]), out).expect("MaxPool reshape output"),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn attrs(kernel: &[i64], strides: Option<&[i64]>, pads: Option<&[i64]>, ceil_mode: bool) -> NodeAttributes {
        let mut a = NodeAttributes::new();
        a.add_ints("kernel_shape".into(), kernel.to_vec());
        if let Some(s) = strides {
            a.add_ints("strides".into(), s.to_vec());
        }
        if let Some(p) = pads {
            a.add_ints("pads".into(), p.to_vec());
        }
        if ceil_mode {
            a.add_int("ceil_mode".into(), 1);
        }
        a
    }

    #[test]
    fn max_pool_1x1_kernel_is_identity() {
        let input = Tensor::from_vec_f32(
            (0..16).map(|i| i as f32).collect(),
            vec![1, 1, 4, 4],
        );
        let out = MaxPool::forward(&[input], &attrs(&[1, 1], None, None, false));
        assert_eq!(out.shape(), &[1, 1, 4, 4]);
        assert_eq!(out.as_slice(), (0..16).map(|i| i as f32).collect::<Vec<_>>());
    }

    #[test]
    fn max_pool_2x2_stride_2_downsample() {
        // [1, 1, 4, 4] → [1, 1, 2, 2]. Each output is the max of a
        // 2x2 non-overlapping block: 5, 7, 13, 15.
        let input = Tensor::from_vec_f32(
            (0..16).map(|i| i as f32).collect(),
            vec![1, 1, 4, 4],
        );
        let out = MaxPool::forward(&[input], &attrs(&[2, 2], Some(&[2, 2]), None, false));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.as_slice(), vec![5.0, 7.0, 13.0, 15.0]);
    }

    #[test]
    fn max_pool_resnet_stem_3x3_stride_2_pad_1() {
        // ResNet-50 stem: 3x3 kernel, stride 2, pads 1 on all sides.
        // Input [1, 1, 4, 4] → output dim = floor((4+2-3)/2 + 1) = 2.
        // Output [1, 1, 2, 2].
        let input = Tensor::from_vec_f32(
            (0..16).map(|i| i as f32).collect(),
            vec![1, 1, 4, 4],
        );
        let out = MaxPool::forward(
            &[input],
            &attrs(&[3, 3], Some(&[2, 2]), Some(&[1, 1, 1, 1]), false),
        );
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        // Top-left window covers (-1..2, -1..2) clamped to (0..2, 0..2):
        //   values [0, 1, 4, 5] → max 5.
        // Top-right window covers (-1..2, 1..4) clamped: [0,1,2,3,4,5,6,7] → max 7.
        // Wait — center of windows are at (1,1), (1,3), (3,1), (3,3) due to
        // stride 2, pad 1: window i-th in [oh*sH - pad, oh*sH - pad + K).
        // For oh=0: -1..2 → [0,1] rows; oh=1: 1..4 → [1,2,3] rows.
        // For ow=0: -1..2 → [0,1] cols; ow=1: 1..4 → [1,2,3] cols.
        // (0, 0): rows[0,1] × cols[0,1] = [0,1,4,5] → 5
        // (0, 1): rows[0,1] × cols[1,2,3] = [1,2,3,5,6,7] → 7
        // (1, 0): rows[1,2,3] × cols[0,1] = [4,5,8,9,12,13] → 13
        // (1, 1): rows[1,2,3] × cols[1,2,3] = [5,6,7,9,10,11,13,14,15] → 15
        assert_eq!(out.as_slice(), vec![5.0, 7.0, 13.0, 15.0]);
    }

    #[test]
    fn max_pool_ceil_mode_changes_output_size() {
        // 5x5 input, 2x2 kernel, stride 2, no pad.
        // floor: (5-2)/2+1 = 2;  ceil: ceil(3/2)+1 = 2+1 = 3. Wait:
        // formula is floor((D + p_lo + p_hi - effective_kernel)/stride + 1)
        // Here effective_kernel = 2, numerator = 5+0+0-2 = 3.
        // floor: 3/2 + 1 = 1 + 1 = 2.
        // ceil: ceil(3/2) + 1 = 2 + 1 = 3.
        let input_floor = Tensor::from_vec_f32(
            (0..25).map(|i| i as f32).collect(),
            vec![1, 1, 5, 5],
        );
        let input_ceil = input_floor.clone();
        let floor_out = MaxPool::forward(&[input_floor], &attrs(&[2, 2], Some(&[2, 2]), None, false));
        let ceil_out = MaxPool::forward(&[input_ceil], &attrs(&[2, 2], Some(&[2, 2]), None, true));
        assert_eq!(floor_out.shape(), &[1, 1, 2, 2]);
        assert_eq!(ceil_out.shape(), &[1, 1, 3, 3]);
    }
}
