use iconnx::attributes::NodeAttributes;
/// Test Conv operator for Iconnx
///
/// TDD: Tests written FIRST
/// Conv performs 2D convolution
/// ONNX spec: https://onnx.ai/onnx/operators/onnx__Conv.html
use iconnx::tensor::Tensor;

/// Test 1: Simple 2D convolution (no padding, stride=1)
#[test]
fn test_conv_2d_simple() {
    // Input: [1, 1, 3, 3] (batch=1, channels=1, height=3, width=3)
    // [[1, 2, 3],
    //  [4, 5, 6],
    //  [7, 8, 9]]
    let input = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        vec![1, 1, 3, 3],
    );

    // Kernel: [1, 1, 2, 2] (out_channels=1, in_channels=1, kh=2, kw=2)
    // [[1, 0],
    //  [0, 1]]  (identity-like kernel)
    let kernel = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], vec![1, 1, 2, 2]);

    let result =
        iconnx::operators::conv::Conv::forward(&[input, kernel], &NodeAttributes::new());

    // Output shape: [1, 1, 2, 2] (no padding, kernel 2x2)
    assert_eq!(result.shape(), &[1, 1, 2, 2]);

    // Values: top-left 2x2 convolved with kernel
    // [1*1 + 5*1, 2*1 + 6*1] = [6, 8]
    // [4*1 + 8*1, 5*1 + 9*1] = [12, 14]
    let output = result.as_slice();
    assert_eq!(output[0], 6.0); // (0,0)
    assert_eq!(output[1], 8.0); // (0,1)
    assert_eq!(output[2], 12.0); // (1,0)
    assert_eq!(output[3], 14.0); // (1,1)
}

/// Test 2: Conv with bias
#[test]
fn test_conv_with_bias() {
    let input = Tensor::from_vec(vec![1.0; 9], vec![1, 1, 3, 3]);
    let kernel = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 1, 2, 2]); // Sum kernel
    let bias = Tensor::from_vec(vec![10.0], vec![1]); // Add 10 to each output

    let result = iconnx::operators::conv::Conv::forward(
        &[input, kernel, bias],
        &NodeAttributes::new(),
    );

    assert_eq!(result.shape(), &[1, 1, 2, 2]);

    // Each output is sum of 4 ones (=4) plus bias (=10) = 14
    let output = result.as_slice();
    for &val in &output {
        assert_eq!(val, 14.0);
    }
}

/// Test 3: Conv with stride=2
#[test]
fn test_conv_stride_2() {
    // Input: [1, 1, 4, 4]
    let input = Tensor::from_vec(vec![1.0; 16], vec![1, 1, 4, 4]);
    let kernel = Tensor::from_vec(vec![1.0; 4], vec![1, 1, 2, 2]);

    // For now, test with stride=1 default
    // TODO: Add stride parameter support
    let result =
        iconnx::operators::conv::Conv::forward(&[input, kernel], &NodeAttributes::new());

    // With stride=1, kernel=2x2, input=4x4 → output=3x3
    assert_eq!(result.shape(), &[1, 1, 3, 3]);
}
