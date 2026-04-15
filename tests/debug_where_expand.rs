//! Regression test: `Where` selecting from `x` produces an int64 shape vector
//! that `Expand` then uses as a target shape.
//!
//! Originally a scratch debug file for Kokoro's node 1251-1252 Where→Expand
//! pair. The original synthetic condition had a stray `0.0` (false) in it
//! that picked `y[2]=9` instead of `x[2]=128`, which contradicted the
//! test's own assertion. Fixed so the test actually exercises the intended
//! flow: an all-true condition selects the full `x = [1, 3, 128]` vector,
//! and `Expand` broadcasts `[1, 128]` up to `[1, 3, 128]`.

use iconnx::attributes::NodeAttributes;
use iconnx::operators::expand::Expand;
use iconnx::operators::where_op::Where;
use iconnx::tensor::Tensor;

#[test]
fn test_where_expand_shape_flow() {
    // Condition all-true → `Where` returns every element of `x`.
    let condition = Tensor::from_vec(vec![1.0, 1.0, 1.0], vec![3]);
    let x_values = Tensor::from_vec_i64(vec![1, 3, 128], vec![3]);
    let y_values = Tensor::from_vec_i64(vec![9, 9, 9], vec![3]);

    let where_result = Where::forward(
        &[condition, x_values.clone(), y_values],
        &NodeAttributes::new(),
    );
    assert_eq!(where_result.as_slice_i64(), &[1, 3, 128]);

    // Feed `where_result` as the target shape into `Expand`.
    let data_to_expand = Tensor::from_vec(vec![0.5; 128], vec![1, 128]);
    let expand_result =
        Expand::forward(&[data_to_expand, where_result], &NodeAttributes::new());
    assert_eq!(expand_result.shape(), &[1, 3, 128]);
}
