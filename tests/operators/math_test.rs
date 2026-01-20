use iconnx::attributes::NodeAttributes;
/// Test mathematical operators for Iconnx
///
/// TDD: Tests written FIRST
/// Math operators: Pow, Exp
use iconnx::tensor::Tensor;

/// Test Pow: element-wise x^y
#[test]
fn test_pow_basic() {
    let base = Tensor::from_vec(vec![2.0, 3.0, 4.0], vec![3]);
    let exp = Tensor::from_vec(vec![2.0], vec![1]); // Square everything

    let result =
        iconnx::operators::pow::Pow::forward(&[base, exp], &NodeAttributes::new());

    assert_eq!(result.shape(), &[3]);
    assert_eq!(result.as_slice(), vec![4.0, 9.0, 16.0]); // 2^2, 3^2, 4^2
}

/// Test Pow with different exponents
#[test]
fn test_pow_different_exponents() {
    let base = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
    let exp = Tensor::from_vec(vec![2.0, 3.0], vec![2]);

    let result =
        iconnx::operators::pow::Pow::forward(&[base, exp], &NodeAttributes::new());

    assert_eq!(result.as_slice(), vec![4.0, 27.0]); // 2^2=4, 3^3=27
}

/// Test Exp: element-wise e^x
#[test]
fn test_exp_basic() {
    let data = Tensor::from_vec(vec![0.0, 1.0, 2.0], vec![3]);

    let result = iconnx::operators::exp::Exp::forward(&[data], &NodeAttributes::new());

    assert_eq!(result.shape(), &[3]);

    let output = result.as_slice();
    assert!((output[0] - 1.0).abs() < 0.001); // e^0 = 1
    assert!((output[1] - 2.718).abs() < 0.01); // e^1 ≈ 2.718
    assert!((output[2] - 7.389).abs() < 0.01); // e^2 ≈ 7.389
}

/// Test Exp with negative values
#[test]
fn test_exp_negative() {
    let data = Tensor::from_vec(vec![-1.0], vec![1]);

    let result = iconnx::operators::exp::Exp::forward(&[data], &NodeAttributes::new());

    // e^-1 ≈ 0.368
    assert!((result.as_slice()[0] - 0.368).abs() < 0.01);
}
