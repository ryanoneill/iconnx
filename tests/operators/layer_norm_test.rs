use iconnx::attributes::NodeAttributes;
/// Test LayerNormalization operator for Iconnx
///
/// TDD: Tests written FIRST
/// LayerNorm: Normalize along last axis
/// Formula: (x - mean) / sqrt(variance + epsilon) * scale + bias
/// ONNX spec: https://onnx.ai/onnx/operators/onnx__LayerNormalization.html
use iconnx::tensor::Tensor;

/// Test 1: Simple LayerNorm 1D
#[test]
fn test_layer_norm_1d() {
    // Input: [1, 2, 3, 4]
    // Mean: 2.5
    // Variance: 1.25
    // Std: sqrt(1.25) ≈ 1.118
    // Normalized: (x - 2.5) / 1.118
    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    let scale = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![4]);
    let bias = Tensor::from_vec(vec![0.0, 0.0, 0.0, 0.0], vec![4]);

    let result = iconnx::operators::layer_normalization::LayerNormalization::forward(
        &[data, scale, bias],
        &NodeAttributes::new(),
    );

    assert_eq!(result.shape(), &[4]);

    // Normalized values should have mean ≈ 0, std ≈ 1
    let output = result.as_slice();
    let mean: f32 = output.iter().sum::<f32>() / output.len() as f32;
    assert!(mean.abs() < 0.01, "Mean should be ~0, got {}", mean);
}

/// Test 2: LayerNorm 2D (along last axis)
#[test]
fn test_layer_norm_2d() {
    // [[1, 2, 3],
    //  [4, 5, 6]]
    // Normalize each row independently
    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let scale = Tensor::from_vec(vec![1.0, 1.0, 1.0], vec![3]);
    let bias = Tensor::from_vec(vec![0.0, 0.0, 0.0], vec![3]);

    let result = iconnx::operators::layer_normalization::LayerNormalization::forward(
        &[data, scale, bias],
        &NodeAttributes::new(),
    );

    assert_eq!(result.shape(), &[2, 3]);

    // Each row should be normalized independently
    let output = result.as_slice();

    // Row 0: [1,2,3] normalized
    let row0_mean: f32 = output[0..3].iter().sum::<f32>() / 3.0;
    assert!(row0_mean.abs() < 0.01);

    // Row 1: [4,5,6] normalized
    let row1_mean: f32 = output[3..6].iter().sum::<f32>() / 3.0;
    assert!(row1_mean.abs() < 0.01);
}

/// Test 3: LayerNorm with scale and bias
#[test]
fn test_layer_norm_scale_bias() {
    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
    let scale = Tensor::from_vec(vec![2.0, 2.0, 2.0], vec![3]); // Scale by 2
    let bias = Tensor::from_vec(vec![1.0, 1.0, 1.0], vec![3]); // Add 1

    let result = iconnx::operators::layer_normalization::LayerNormalization::forward(
        &[data, scale, bias],
        &NodeAttributes::new(),
    );

    assert_eq!(result.shape(), &[3]);

    // After normalization, scaled by 2 and biased by 1
    // Mean should be around 1 (bias), not 0
    let output = result.as_slice();
    let mean: f32 = output.iter().sum::<f32>() / output.len() as f32;
    assert!((mean - 1.0).abs() < 0.1, "Mean should be ~1, got {}", mean);
}
