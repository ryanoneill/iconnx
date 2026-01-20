use iconnx::attributes::NodeAttributes;
/// Test Gemm operator for Iconnx
///
/// TDD: Tests written FIRST
/// Gemm: Generalized matrix multiply alpha*A@B + beta*C
use iconnx::tensor::Tensor;

/// Test 1: Basic Gemm with alpha=1, beta=1, no transpose
#[test]
fn test_gemm_basic() {
    // A: 2x3, B: 3x2, C: 2x2
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
    let c = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);

    // alpha=1, beta=1 are defaults - no need to set
    let result =
        iconnx::operators::gemm::Gemm::forward(&[a, b, c], &NodeAttributes::new());

    // A@B = [[22, 28], [49, 64]]
    // A@B + C = [[23, 29], [50, 65]]
    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.as_slice(), vec![23.0, 29.0, 50.0, 65.0]);
}

/// Test 2: Gemm with alpha=2, beta=0 (no bias)
#[test]
fn test_gemm_alpha_only() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
    let c = Tensor::from_vec(vec![10.0, 10.0, 10.0, 10.0], vec![2, 2]);

    // Set alpha=2, beta=0 via attributes
    let mut attrs = NodeAttributes::new();
    attrs.add_float("alpha".to_string(), 2.0);
    attrs.add_float("beta".to_string(), 0.0);

    let result = iconnx::operators::gemm::Gemm::forward(&[a, b, c], &attrs);

    // A@B = [[1, 2], [3, 4]]
    // 2*A@B + 0*C = [[2, 4], [6, 8]]
    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.as_slice(), vec![2.0, 4.0, 6.0, 8.0]);
}

/// Test 3: Gemm with alpha=1, beta=2 (scaled bias)
#[test]
fn test_gemm_scaled_bias() {
    let a = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
    let b = Tensor::from_vec(vec![2.0, 0.0, 0.0, 2.0], vec![2, 2]);
    let c = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    // Set beta=2 via attributes
    let mut attrs = NodeAttributes::new();
    attrs.add_float("beta".to_string(), 2.0);

    let result = iconnx::operators::gemm::Gemm::forward(&[a, b, c], &attrs);

    // A@B = [[2, 0], [0, 2]]
    // A@B + 2*C = [[4, 4], [6, 10]]
    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.as_slice(), vec![4.0, 4.0, 6.0, 10.0]);
}

/// Test 4: Gemm without C (bias optional)
#[test]
fn test_gemm_no_bias() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

    // No bias (C), default alpha=1
    let result = iconnx::operators::gemm::Gemm::forward(&[a, b], &NodeAttributes::new());

    // A@B = [[19, 22], [43, 50]]
    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.as_slice(), vec![19.0, 22.0, 43.0, 50.0]);
}
