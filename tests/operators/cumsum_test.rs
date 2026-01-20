use iconnx::attributes::NodeAttributes;
/// Test CumSum operator for Iconnx
///
/// TDD: Tests written FIRST
/// CumSum: Cumulative sum along axis
use iconnx::tensor::Tensor;

/// Test 1: CumSum along axis 0 (1D)
#[test]
fn test_cumsum_1d() {
    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    let axis = Tensor::from_vec(vec![0.0], vec![1]);

    let result =
        iconnx::operators::cumsum::CumSum::forward(&[data, axis], &NodeAttributes::new());

    assert_eq!(result.shape(), &[4]);
    assert_eq!(result.as_slice(), vec![1.0, 3.0, 6.0, 10.0]);
}

/// Test 2: CumSum along axis 0 (2D)
#[test]
fn test_cumsum_2d_axis_0() {
    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
    let axis = Tensor::from_vec(vec![0.0], vec![1]);

    let result =
        iconnx::operators::cumsum::CumSum::forward(&[data, axis], &NodeAttributes::new());

    assert_eq!(result.shape(), &[3, 2]);
    assert_eq!(result.as_slice(), vec![1.0, 2.0, 4.0, 6.0, 9.0, 12.0]);
}

/// Test 3: CumSum along axis 1 (2D)
#[test]
fn test_cumsum_2d_axis_1() {
    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
    let axis = Tensor::from_vec(vec![1.0], vec![1]);

    let result =
        iconnx::operators::cumsum::CumSum::forward(&[data, axis], &NodeAttributes::new());

    assert_eq!(result.shape(), &[3, 2]);
    assert_eq!(result.as_slice(), vec![1.0, 3.0, 3.0, 7.0, 5.0, 11.0]);
}

/// Test 4: CumSum with single element
#[test]
fn test_cumsum_single() {
    let data = Tensor::from_vec(vec![42.0], vec![1]);
    let axis = Tensor::from_vec(vec![0.0], vec![1]);

    let result =
        iconnx::operators::cumsum::CumSum::forward(&[data, axis], &NodeAttributes::new());

    assert_eq!(result.as_slice(), vec![42.0]);
}

/// Test 5: CumSum node 1316 case - [13, 11, 2] -> [13, 24, 26]
#[test]
fn test_cumsum_node_1316() {
    let data = Tensor::from_vec_i64(vec![13, 11, 2], vec![3]);
    let axis = Tensor::from_vec_i64(vec![0], vec![1]);

    let result =
        iconnx::operators::cumsum::CumSum::forward(&[data, axis], &NodeAttributes::new());

    assert_eq!(result.shape(), &[3]);
    assert!(result.is_int64());
    assert_eq!(
        result.as_slice_i64(),
        vec![13, 24, 26],
        "CumSum should produce [13, 24, 26] from [13, 11, 2]"
    );
}
