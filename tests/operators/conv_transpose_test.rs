use iconnx::attributes::NodeAttributes;
/// Test ConvTranspose operator for Iconnx
///
/// TDD: Tests written FIRST
/// ConvTranspose: Transposed convolution (deconvolution)
/// ONNX spec: https://onnx.ai/onnx/operators/onnx__ConvTranspose.html
use iconnx::tensor::Tensor;

/// Test 1: Simple 2D transposed convolution (upsample)
#[test]
fn test_conv_transpose_2d_simple() {
    // Input: [1, 1, 2, 2] (batch=1, channels=1, height=2, width=2)
    // [[1, 2],
    //  [3, 4]]
    let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);

    // Kernel: [1, 1, 2, 2] (in_channels=1, out_channels=1, kh=2, kw=2)
    // [[1, 0],
    //  [0, 1]]
    let kernel = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], vec![1, 1, 2, 2]);

    let result = iconnx::operators::conv_transpose::ConvTranspose::forward(
        &[input, kernel],
        &NodeAttributes::new(),
    );

    // Output shape: [1, 1, 3, 3] (stride=1, no padding)
    // With stride=1, kernel=2x2, input=2x2 → output = (2-1)*1 + 2 = 3
    assert_eq!(result.shape(), &[1, 1, 3, 3]);

    // Expected output (overlapping regions add):
    // Top-left: 1*1=1, Top-center: 1*0+2*1=2, Top-right: 2*0=0
    // Mid-left: 1*0+3*1=3, Center: 1*1+2*0+3*0+4*1=5, Mid-right: 2*1+4*0=2
    // Bottom-left: 3*0=0, Bottom-center: 3*1+4*0=3, Bottom-right: 4*1=4
    let expected = vec![1.0, 2.0, 0.0, 3.0, 5.0, 2.0, 0.0, 3.0, 4.0];
    assert_eq!(result.as_slice(), expected);
}

/// Test 2: ConvTranspose with bias
#[test]
fn test_conv_transpose_with_bias() {
    let input = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 1, 2, 2]);
    let kernel = Tensor::from_vec(vec![1.0; 4], vec![1, 1, 2, 2]); // All ones
    let bias = Tensor::from_vec(vec![5.0], vec![1]); // Add 5 to each output

    let result = iconnx::operators::conv_transpose::ConvTranspose::forward(
        &[input, kernel, bias],
        &NodeAttributes::new(),
    );

    assert_eq!(result.shape(), &[1, 1, 3, 3]);

    // Without bias: corners=1, edges=2, center=4
    // With bias (+5): corners=6, edges=7, center=9
    let output = result.as_slice();
    assert_eq!(output[0], 6.0); // top-left
    assert_eq!(output[1], 7.0); // top-center
    assert_eq!(output[4], 9.0); // center
}

/// Test 3: ConvTranspose identity (kernel=1x1)
#[test]
fn test_conv_transpose_identity() {
    let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
    let kernel = Tensor::from_vec(vec![1.0], vec![1, 1, 1, 1]); // 1x1 identity

    let result = iconnx::operators::conv_transpose::ConvTranspose::forward(
        &[input, kernel],
        &NodeAttributes::new(),
    );

    // With 1x1 kernel, output shape = input shape
    assert_eq!(result.shape(), &[1, 1, 2, 2]);
    assert_eq!(result.as_slice(), vec![1.0, 2.0, 3.0, 4.0]);
}
