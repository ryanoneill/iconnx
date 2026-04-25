//! ONNX Flatten operator: reshape an N-D tensor into a 2-D matrix.
//!
//! Spec: `Flatten(X, axis=1)` produces a 2-D tensor whose shape is
//! `[product(X.shape[..axis]), product(X.shape[axis..])]`. No data
//! movement — pure shape relabel. `axis` defaults to 1; negative values
//! count from the rear (`axis = -1` → `[product(0..ndim-1), shape[ndim-1]]`).
//! `axis = 0` collapses everything into a 1×N row vector (with the
//! prefix product of an empty range conventionally 1).
//!
//! Dtype-polymorphic — same memcpy-class semantics as Reshape, which
//! the GPU dispatch reuses directly.

use crate::attributes::NodeAttributes;
use crate::tensor::Tensor;
use ndarray::IxDyn;

/// ONNX Flatten.
pub struct Flatten;

impl Flatten {
    /// Resolve the axis attribute, normalizing negative values. Returns
    /// the split index in `[0, ndim]` (inclusive on both ends — `axis=0`
    /// is valid and produces a `[1, total]` row vector).
    pub fn resolve_axis(attrs: &NodeAttributes, ndim: usize) -> usize {
        let raw = attrs.get_int("axis").unwrap_or(1);
        if raw < 0 {
            ((ndim as i64 + raw).max(0) as usize).min(ndim)
        } else {
            (raw as usize).min(ndim)
        }
    }

    /// Compute the 2-D target shape from input shape + axis. Both halves
    /// fall back to 1 when their corresponding range is empty (matches
    /// ONNX Reference op for `axis = 0` and `axis = ndim`).
    pub fn target_shape(input_shape: &[usize], axis: usize) -> [usize; 2] {
        let outer: usize = input_shape[..axis].iter().product::<usize>().max(1);
        let inner: usize = input_shape[axis..].iter().product::<usize>().max(1);
        [outer, inner]
    }

    pub fn forward(inputs: &[Tensor], attrs: &NodeAttributes) -> Tensor {
        let input = &inputs[0];
        let ndim = input.ndim();
        let axis = Self::resolve_axis(attrs, ndim);
        let [outer, inner] = Self::target_shape(input.shape(), axis);
        let target = vec![outer, inner];

        match input {
            Tensor::Float32(arr) => Tensor::from_array(
                arr.clone().to_shape(IxDyn(&target)).expect("flatten").into_owned(),
            ),
            Tensor::Float64(arr) => Tensor::from_vec_f64(
                arr.iter().copied().collect(),
                target,
            ),
            Tensor::Int64(arr) => Tensor::from_array_i64(
                arr.clone().to_shape(IxDyn(&target)).expect("flatten").into_owned(),
            ),
            Tensor::Int32(arr) => Tensor::from_array_i32(
                arr.clone().to_shape(IxDyn(&target)).expect("flatten").into_owned(),
            ),
            Tensor::Bool(arr) => Tensor::from_vec_bool(
                arr.iter().copied().collect(),
                target,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn attrs_axis(axis: i64) -> NodeAttributes {
        let mut a = NodeAttributes::new();
        a.add_int("axis".into(), axis);
        a
    }

    #[test]
    fn flatten_axis_default_1_on_4d() {
        // [2, 3, 4, 5] with axis=1 → [2, 60].
        let input = Tensor::from_vec_f32(
            (0..120).map(|i| i as f32).collect(),
            vec![2, 3, 4, 5],
        );
        let out = Flatten::forward(&[input], &NodeAttributes::new());
        assert_eq!(out.shape(), &[2, 60]);
        // Element 0 of inner dim of row 1 = original [1,0,0,0] = 60.
        assert_eq!(out.as_slice()[60], 60.0);
    }

    #[test]
    fn flatten_axis_0_collapses_to_row_vector() {
        // axis=0 → [1, total]. Empty prefix product is 1 by convention.
        let input = Tensor::from_vec_f32(
            (0..24).map(|i| i as f32).collect(),
            vec![2, 3, 4],
        );
        let out = Flatten::forward(&[input], &attrs_axis(0));
        assert_eq!(out.shape(), &[1, 24]);
    }

    #[test]
    fn flatten_axis_negative_one_keeps_last_dim() {
        // axis=-1 on [2, 3, 4] → axis=2 → [6, 4].
        let input = Tensor::from_vec_f32(
            (0..24).map(|i| i as f32).collect(),
            vec![2, 3, 4],
        );
        let out = Flatten::forward(&[input], &attrs_axis(-1));
        assert_eq!(out.shape(), &[6, 4]);
    }

    #[test]
    fn flatten_axis_2_on_4d_resnet_classifier_pattern() {
        // [N, C, H, W] = [2, 5, 1, 1] flattened from axis=1 → [2, 5].
        // Mirrors the GAP→Flatten→FC head of ResNet-50.
        let input = Tensor::from_vec_f32(
            (0..10).map(|i| i as f32).collect(),
            vec![2, 5, 1, 1],
        );
        let out = Flatten::forward(&[input], &attrs_axis(1));
        assert_eq!(out.shape(), &[2, 5]);
        assert_eq!(out.as_slice()[5], 5.0);
    }

    #[test]
    fn flatten_int64_dispatches_correctly() {
        let input = Tensor::from_vec_i64(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        let out = Flatten::forward(&[input], &attrs_axis(1));
        assert_eq!(out.shape(), &[2, 3]);
        assert!(matches!(out, Tensor::Int64(_)));
    }
}
