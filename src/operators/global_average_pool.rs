//! ONNX GlobalAveragePool: reduce all spatial dimensions of an
//! N-D input via mean, keeping the leading batch and channel dims.
//!
//! Spec: input shape `[N, C, D1, D2, ..., Dk]`, output shape
//! `[N, C, 1, 1, ..., 1]`. No attributes. Numerically equivalent to
//! `ReduceMean(axes=[2, 3, ..., k+1], keepdims=true)`.
//!
//! Implementation note: iconnx's existing `ReduceMean` only reduces a
//! single axis per kernel call (CPU and GPU). GlobalAveragePool
//! therefore peels the last spatial axis ndim-2 times, then restores
//! the spatial slots as size-1 via reshape. The arithmetic is exact
//! up to floating-point rounding regardless of which order the
//! spatial axes are reduced — `mean(mean(x, axis=W), axis=H)` over
//! contiguous data equals `mean(x, axes=(H,W))` to better than ulp
//! per intermediate division.

use crate::attributes::NodeAttributes;
use crate::tensor::Tensor;
use half::f16;
use ndarray::Axis;

/// ONNX GlobalAveragePool.
pub struct GlobalAveragePool;

impl GlobalAveragePool {
    pub fn forward(inputs: &[Tensor], _attrs: &NodeAttributes) -> Tensor {
        let input = &inputs[0];
        let ndim = input.ndim();
        assert!(
            ndim >= 3,
            "GlobalAveragePool requires at least 3 dims (N, C, spatial...), got {}",
            ndim
        );

        // Iteratively reduce the last axis (always the last *current*
        // axis after each step) until only [N, C] remains.
        let mut current = input.clone();
        for _ in 2..ndim {
            let last = current.ndim() - 1;
            current = reduce_mean_last_axis(current, last);
        }

        // Restore spatial dims as size 1 via keepdims-equivalent reshape.
        let mut keepdims_shape = Vec::with_capacity(ndim);
        keepdims_shape.push(input.shape()[0]);
        keepdims_shape.push(input.shape()[1]);
        keepdims_shape.extend(std::iter::repeat_n(1usize, ndim - 2));
        reshape_preserving_dtype(current, keepdims_shape)
    }
}

fn reduce_mean_last_axis(t: Tensor, last_axis: usize) -> Tensor {
    match t {
        Tensor::Float32(arr) => {
            let reduced = arr.mean_axis(Axis(last_axis)).expect("mean_axis f32");
            Tensor::Float32(reduced)
        }
        Tensor::Float64(arr) => {
            let reduced = arr.mean_axis(Axis(last_axis)).expect("mean_axis f64");
            Tensor::Float64(reduced)
        }
        Tensor::Int64(arr) => {
            // ndarray's mean_axis isn't generic over Integer — compute
            // it as `sum / count`. ONNX defines GlobalAveragePool for
            // floating dtypes only in practice, but Int64 is included
            // for symmetry; division is integer (truncating).
            let axis_size = arr.shape()[last_axis] as i64;
            let summed = arr.sum_axis(Axis(last_axis));
            Tensor::Int64(summed.mapv(|v| v / axis_size))
        }
        Tensor::Int32(arr) => {
            let axis_size = arr.shape()[last_axis] as i32;
            let summed = arr.sum_axis(Axis(last_axis));
            Tensor::Int32(summed.mapv(|v| v / axis_size))
        }
        Tensor::Bool(_) => {
            panic!("GlobalAveragePool does not support Bool inputs");
        }
        Tensor::Int8(_) | Tensor::UInt8(_) => {
            // ONNX GlobalAveragePool is defined over floating dtypes
            // only in practice. INT8/UINT8 inputs would require a
            // dequantize step upstream — which is exactly what the
            // WS-4 quantization graph does (DequantizeLinear before
            // any reduction).
            panic!("GlobalAveragePool does not support {} inputs (dequantize first)", t.dtype());
        }
        Tensor::Float16(arr) => {
            // f32 accumulator — same precision contract M3.4 will use
            // for FP16 reductions (Softmax / ReduceMean) on GPU.
            let axis_size = arr.shape()[last_axis] as f32;
            let as_f32 = arr.mapv(|v| v.to_f32());
            let summed = as_f32.sum_axis(Axis(last_axis));
            Tensor::Float16(summed.mapv(|v| f16::from_f32(v / axis_size)))
        }
        Tensor::BFloat16(arr) => {
            // f32 accumulator — same precision contract WS-3.5 Y(2) uses
            // for BF16 reductions (Softmax / ReduceMean) on GPU.
            let axis_size = arr.shape()[last_axis] as f32;
            let as_f32 = arr.mapv(|v| v.to_f32());
            let summed = as_f32.sum_axis(Axis(last_axis));
            Tensor::BFloat16(summed.mapv(|v| half::bf16::from_f32(v / axis_size)))
        }
    }
}

fn reshape_preserving_dtype(t: Tensor, new_shape: Vec<usize>) -> Tensor {
    use ndarray::IxDyn;
    match t {
        Tensor::Float32(arr) => {
            Tensor::Float32(arr.into_shape_with_order(IxDyn(&new_shape)).expect("reshape f32"))
        }
        Tensor::Float64(arr) => {
            Tensor::Float64(arr.into_shape_with_order(IxDyn(&new_shape)).expect("reshape f64"))
        }
        Tensor::Int64(arr) => {
            Tensor::Int64(arr.into_shape_with_order(IxDyn(&new_shape)).expect("reshape i64"))
        }
        Tensor::Int32(arr) => {
            Tensor::Int32(arr.into_shape_with_order(IxDyn(&new_shape)).expect("reshape i32"))
        }
        Tensor::Bool(arr) => {
            Tensor::Bool(arr.into_shape_with_order(IxDyn(&new_shape)).expect("reshape bool"))
        }
        Tensor::Int8(arr) => {
            Tensor::Int8(arr.into_shape_with_order(IxDyn(&new_shape)).expect("reshape i8"))
        }
        Tensor::UInt8(arr) => {
            Tensor::UInt8(arr.into_shape_with_order(IxDyn(&new_shape)).expect("reshape u8"))
        }
        Tensor::Float16(arr) => {
            Tensor::Float16(arr.into_shape_with_order(IxDyn(&new_shape)).expect("reshape f16"))
        }
        Tensor::BFloat16(arr) => {
            Tensor::BFloat16(arr.into_shape_with_order(IxDyn(&new_shape)).expect("reshape bf16"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn global_average_pool_4d_nchw() {
        // [1, 2, 2, 2] — two channels with means 2.5 and 6.5.
        let input = Tensor::from_vec_f32(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![1, 2, 2, 2],
        );
        let out = GlobalAveragePool::forward(&[input], &NodeAttributes::new());
        assert_eq!(out.shape(), &[1, 2, 1, 1]);
        let vals = out.as_slice();
        assert!((vals[0] - 2.5).abs() < 1e-6, "channel 0 mean should be 2.5, got {}", vals[0]);
        assert!((vals[1] - 6.5).abs() < 1e-6, "channel 1 mean should be 6.5, got {}", vals[1]);
    }

    #[test]
    fn global_average_pool_5d_ncdhw() {
        // [1, 1, 2, 2, 2] = 8 values, mean = 4.5.
        let input = Tensor::from_vec_f32(
            (1..=8).map(|i| i as f32).collect(),
            vec![1, 1, 2, 2, 2],
        );
        let out = GlobalAveragePool::forward(&[input], &NodeAttributes::new());
        assert_eq!(out.shape(), &[1, 1, 1, 1, 1]);
        assert!((out.as_slice()[0] - 4.5).abs() < 1e-6);
    }

    #[test]
    fn global_average_pool_resnet_classifier_head_shape() {
        // ResNet-50: GAP input is [N, 2048, 7, 7]. Down-shape this for
        // testability: [2, 3, 4, 4] with each channel filled with a
        // constant equal to the channel index lets us verify means
        // independently.
        let mut data = Vec::with_capacity(2 * 3 * 4 * 4);
        for n in 0..2 {
            for c in 0..3 {
                for _ in 0..16 {
                    data.push((n * 3 + c) as f32);
                }
            }
        }
        let input = Tensor::from_vec_f32(data, vec![2, 3, 4, 4]);
        let out = GlobalAveragePool::forward(&[input], &NodeAttributes::new());
        assert_eq!(out.shape(), &[2, 3, 1, 1]);
        let vals = out.as_slice();
        // Each cell holds its (n*3+c) value; mean across the 16-element
        // spatial region equals that value.
        for n in 0..2 {
            for c in 0..3 {
                let expected = (n * 3 + c) as f32;
                let got = vals[n * 3 + c];
                assert!((got - expected).abs() < 1e-6, "[{},{}]: expected {}, got {}", n, c, expected, got);
            }
        }
    }
}
