//! ONNX Split operator: split a tensor along an axis into N outputs.
//!
//! Spec (opset 13+): `Split(input, split?)`.
//!   * Input 0: data tensor.
//!   * Input 1 (optional): `split` — Int64 1-D tensor of per-output sizes
//!     summing to `input.shape[axis]`.
//!   * Attribute `axis` (int, default 0; supports negative values
//!     counted from rear).
//!   * Attribute `num_outputs` (opset 18+): only used when the `split`
//!     input is absent — divides the axis into N approximately equal
//!     chunks (the last chunk receives the remainder if the dim doesn't
//!     divide evenly).
//!   * Output count is `split.len()` if provided, else `num_outputs`,
//!     else falls back to `node.outputs.len()` (the count the graph
//!     itself encodes).
//!
//! Multi-output: `forward` returns `Vec<Tensor>`, one entry per output
//! slice. Dtype-polymorphic — Split is a memcpy-class op with no math.

use crate::attributes::NodeAttributes;
use crate::tensor::Tensor;
use ndarray::{ArrayD, Axis};

/// ONNX Split.
pub struct Split;

impl Split {
    /// Compute the per-output split sizes from inputs / attributes.
    ///
    /// Resolution order matches the ONNX spec:
    ///   1. If `inputs[1]` is present and is a 1-D Int64 tensor, use it.
    ///   2. Otherwise, derive even sizes from `num_outputs` (opset 18+)
    ///      or, failing that, from `node.outputs.len()` carried via
    ///      `output_count_hint`.
    pub fn split_sizes(
        inputs: &[Tensor],
        attrs: &NodeAttributes,
        axis_size: usize,
        output_count_hint: usize,
    ) -> Vec<usize> {
        // Path 1: explicit split input.
        if let Some(split_tensor) = inputs.get(1) {
            let raw = split_tensor.as_slice_i64();
            if !raw.is_empty() {
                return raw.iter().map(|&s| s as usize).collect();
            }
        }

        // Path 2: num_outputs attribute (opset 18+) or graph-output count.
        let n = attrs
            .get_int("num_outputs")
            .map(|v| v as usize)
            .unwrap_or(output_count_hint)
            .max(1);

        // Even split with remainder on the last chunk (matches ONNX
        // Reference op's behaviour when num_outputs doesn't divide
        // axis_size evenly).
        let chunk = axis_size / n;
        let remainder = axis_size % n;
        let mut sizes = vec![chunk; n];
        if let Some(last) = sizes.last_mut() {
            *last += remainder;
        }
        sizes
    }

    /// Resolve the axis attribute, normalizing negative axes.
    pub fn resolve_axis(attrs: &NodeAttributes, ndim: usize) -> usize {
        let raw = attrs.get_int("axis").unwrap_or(0);
        if raw < 0 {
            (ndim as i64 + raw).max(0) as usize
        } else {
            (raw as usize).min(ndim.saturating_sub(1))
        }
    }

    /// Forward pass: split `inputs[0]` along `axis` into N tensors.
    ///
    /// Multi-output: returns `Vec<Tensor>`. Caller is responsible for
    /// distributing each entry to the corresponding ONNX output name.
    /// `output_count_hint` is consulted only when the spec resolution
    /// above can't pin N (no `split` input + no `num_outputs` attribute).
    pub fn forward(
        inputs: &[Tensor],
        attrs: &NodeAttributes,
        output_count_hint: usize,
    ) -> Vec<Tensor> {
        let input = &inputs[0];
        let ndim = input.ndim();
        let axis = Self::resolve_axis(attrs, ndim);
        let axis_size = input.shape()[axis];
        let sizes = Self::split_sizes(inputs, attrs, axis_size, output_count_hint);

        match input {
            Tensor::Float32(arr) => slice_into(arr, axis, &sizes, Tensor::Float32),
            Tensor::Float64(arr) => slice_into(arr, axis, &sizes, Tensor::Float64),
            Tensor::Int64(arr) => slice_into(arr, axis, &sizes, Tensor::Int64),
            Tensor::Int32(arr) => slice_into(arr, axis, &sizes, Tensor::Int32),
            Tensor::Bool(arr) => slice_into(arr, axis, &sizes, Tensor::Bool),
        }
    }
}

fn slice_into<T: Clone, F>(
    arr: &ArrayD<T>,
    axis: usize,
    sizes: &[usize],
    wrap: F,
) -> Vec<Tensor>
where
    F: Fn(ArrayD<T>) -> Tensor,
{
    let mut out = Vec::with_capacity(sizes.len());
    let mut start: usize = 0;
    for &size in sizes {
        let end = start + size;
        let slice = arr
            .slice_axis(Axis(axis), ndarray::Slice::from(start as isize..end as isize))
            .to_owned();
        out.push(wrap(slice));
        start = end;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn attrs_with_axis(axis: i64) -> NodeAttributes {
        let mut a = NodeAttributes::new();
        a.add_int("axis".into(), axis);
        a
    }

    #[test]
    fn split_evenly_into_3_via_num_outputs() {
        let mut attrs = NodeAttributes::new();
        attrs.add_int("axis".into(), 0);
        attrs.add_int("num_outputs".into(), 3);
        let input = Tensor::from_vec_f32((0..6).map(|i| i as f32).collect(), vec![6]);

        let outputs = Split::forward(&[input], &attrs, 3);
        assert_eq!(outputs.len(), 3);
        assert_eq!(outputs[0].shape(), &[2]);
        assert_eq!(outputs[0].as_slice(), vec![0.0, 1.0]);
        assert_eq!(outputs[1].as_slice(), vec![2.0, 3.0]);
        assert_eq!(outputs[2].as_slice(), vec![4.0, 5.0]);
    }

    #[test]
    fn split_with_explicit_equal_split_input() {
        let attrs = attrs_with_axis(0);
        let input = Tensor::from_vec_f32((0..6).map(|i| i as f32).collect(), vec![6]);
        let split = Tensor::from_vec_i64(vec![2, 2, 2], vec![3]);

        let outputs = Split::forward(&[input, split], &attrs, 3);
        assert_eq!(outputs.len(), 3);
        assert_eq!(outputs[0].as_slice(), vec![0.0, 1.0]);
        assert_eq!(outputs[1].as_slice(), vec![2.0, 3.0]);
        assert_eq!(outputs[2].as_slice(), vec![4.0, 5.0]);
    }

    #[test]
    fn split_with_unequal_split_input() {
        let attrs = attrs_with_axis(0);
        let input = Tensor::from_vec_f32((0..6).map(|i| i as f32).collect(), vec![6]);
        let split = Tensor::from_vec_i64(vec![1, 3, 2], vec![3]);

        let outputs = Split::forward(&[input, split], &attrs, 3);
        assert_eq!(outputs.len(), 3);
        assert_eq!(outputs[0].as_slice(), vec![0.0]);
        assert_eq!(outputs[1].as_slice(), vec![1.0, 2.0, 3.0]);
        assert_eq!(outputs[2].as_slice(), vec![4.0, 5.0]);
    }

    #[test]
    fn split_axis_negative_one_on_2d() {
        // axis = -1 on shape [2, 4]: split last dim into [2, 2].
        let attrs = attrs_with_axis(-1);
        let input = Tensor::from_vec_f32((0..8).map(|i| i as f32).collect(), vec![2, 4]);
        let split = Tensor::from_vec_i64(vec![2, 2], vec![2]);

        let outputs = Split::forward(&[input, split], &attrs, 2);
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].shape(), &[2, 2]);
        assert_eq!(outputs[1].shape(), &[2, 2]);
        // First half: cols 0..2 of each row.
        assert_eq!(outputs[0].as_slice(), vec![0.0, 1.0, 4.0, 5.0]);
        // Second half: cols 2..4 of each row.
        assert_eq!(outputs[1].as_slice(), vec![2.0, 3.0, 6.0, 7.0]);
    }

    #[test]
    fn split_axis_1_on_2d() {
        // axis = 1 on shape [2, 6]: split into [2, 2, 2] cols.
        let attrs = attrs_with_axis(1);
        let input = Tensor::from_vec_f32((0..12).map(|i| i as f32).collect(), vec![2, 6]);
        let split = Tensor::from_vec_i64(vec![2, 2, 2], vec![3]);

        let outputs = Split::forward(&[input, split], &attrs, 3);
        assert_eq!(outputs.len(), 3);
        assert_eq!(outputs[0].shape(), &[2, 2]);
        assert_eq!(outputs[0].as_slice(), vec![0.0, 1.0, 6.0, 7.0]);
        assert_eq!(outputs[1].as_slice(), vec![2.0, 3.0, 8.0, 9.0]);
        assert_eq!(outputs[2].as_slice(), vec![4.0, 5.0, 10.0, 11.0]);
    }

    #[test]
    fn split_int64_input_dispatches_correctly() {
        let attrs = attrs_with_axis(0);
        let input = Tensor::from_vec_i64(vec![10_i64, 20, 30, 40, 50, 60], vec![6]);
        let split = Tensor::from_vec_i64(vec![3, 3], vec![2]);

        let outputs = Split::forward(&[input, split], &attrs, 2);
        assert_eq!(outputs.len(), 2);
        match (&outputs[0], &outputs[1]) {
            (Tensor::Int64(a), Tensor::Int64(b)) => {
                assert_eq!(a.iter().copied().collect::<Vec<_>>(), vec![10, 20, 30]);
                assert_eq!(b.iter().copied().collect::<Vec<_>>(), vec![40, 50, 60]);
            }
            _ => panic!("expected Int64 outputs, got {:?} / {:?}", outputs[0], outputs[1]),
        }
    }

    #[test]
    fn split_uneven_via_num_outputs_remainder_on_last() {
        // 7 elements / 3 outputs: chunks [2, 2, 3] (remainder on last).
        let mut attrs = NodeAttributes::new();
        attrs.add_int("axis".into(), 0);
        attrs.add_int("num_outputs".into(), 3);
        let input = Tensor::from_vec_f32((0..7).map(|i| i as f32).collect(), vec![7]);

        let outputs = Split::forward(&[input], &attrs, 3);
        assert_eq!(outputs.len(), 3);
        assert_eq!(outputs[0].shape(), &[2]);
        assert_eq!(outputs[1].shape(), &[2]);
        assert_eq!(outputs[2].shape(), &[3]);
        assert_eq!(outputs[2].as_slice(), vec![4.0, 5.0, 6.0]);
    }
}
