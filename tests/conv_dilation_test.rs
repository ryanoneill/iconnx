use iconnx::attributes::NodeAttributes;
use iconnx::operators::conv::Conv;
/// Regression test for Conv operator with dilation attribute
///
/// Bug: Conv was ignoring dilations attribute
/// Fix: Read dilations from attributes and apply in convolution
use iconnx::tensor::Tensor;

#[test]
fn test_conv1d_with_dilation() {
    // This reproduces node 1739 from the Kokoro model
    // Input: [1, 256, 520], Kernel: [256, 256, 7]
    // Attributes: pads=[9, 9], dilations=[3], strides=[1]
    // Expected output: [1, 256, 520]

    // Smaller test case: [1, 3, 10], kernel [2, 3, 3], dilation=2
    // With dilation=2: effective_kernel = (3-1)*2 + 1 = 5
    // Output length = (10 + 2*1 - 5) / 1 + 1 = 8

    let input = Tensor::from_vec((0..30).map(|i| i as f32).collect(), vec![1, 3, 10]);
    let kernel = Tensor::from_vec(
        vec![1.0; 18], // 2 * 3 * 3
        vec![2, 3, 3],
    );

    let mut attrs = NodeAttributes::new();
    attrs.add_ints("pads".to_string(), vec![1, 1]);
    attrs.add_ints("dilations".to_string(), vec![2]);
    attrs.add_ints("strides".to_string(), vec![1]);

    let result = Conv::forward(&[input, kernel], &attrs);

    assert_eq!(result.shape(), &[1, 2, 8]);
}

#[test]
fn test_conv1d_shape_with_large_dilation() {
    // Reproduce exact scenario from node 1739
    // Input: [1, 256, 520], kernel: [256, 256, 7]
    // dilation=3, pads=[9,9], stride=1
    // Effective kernel = (7-1)*3 + 1 = 19
    // Output = (520 + 18 - 19) / 1 + 1 = 520

    let input = Tensor::from_vec(vec![1.0; 1 * 256 * 520], vec![1, 256, 520]);
    let kernel = Tensor::from_vec(vec![0.01; 256 * 256 * 7], vec![256, 256, 7]);

    let mut attrs = NodeAttributes::new();
    attrs.add_ints("pads".to_string(), vec![9, 9]);
    attrs.add_ints("dilations".to_string(), vec![3]);
    attrs.add_ints("strides".to_string(), vec![1]);

    let result = Conv::forward(&[input, kernel], &attrs);

    // Must produce output of length 520, not 532
    assert_eq!(result.shape(), &[1, 256, 520]);
}

#[test]
fn test_conv1d_no_dilation_default() {
    // Ensure backward compatibility: no dilation attribute defaults to 1

    let input = Tensor::from_vec((0..20).map(|i| i as f32).collect(), vec![1, 2, 10]);
    let kernel = Tensor::from_vec(
        vec![1.0; 6], // 2 * 2 * 3
        vec![1, 2, 3],
    );

    let attrs = NodeAttributes::new(); // No attributes

    let result = Conv::forward(&[input, kernel], &attrs);

    // Output = (10 + 0 - 3) / 1 + 1 = 8
    assert_eq!(result.shape(), &[1, 1, 8]);
}
