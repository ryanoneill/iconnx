use iconnx::attributes::NodeAttributes;
/// Test Resize operator for Iconnx
///
/// TDD: Tests written FIRST
/// Resize: Tensor interpolation/upsampling
/// ONNX spec: https://onnx.ai/onnx/operators/onnx__Resize.html
use iconnx::tensor::Tensor;

/// Test 1: Resize 1D nearest neighbor (upsample 2x)
/// Default mode: coordinate_transformation_mode=asymmetric, nearest_mode=floor
#[test]
fn test_resize_1d_nearest_upsample() {
    // Input: [1, 2, 3] -> Output with nearest neighbor
    // Default: asymmetric transform + floor nearest mode
    // Mapping: output[i] -> input[floor(i/2.0)]
    // output[0] -> input[floor(0/2)=0] = 1.0
    // output[1] -> input[floor(1/2)=0] = 1.0
    // output[2] -> input[floor(2/2)=1] = 2.0
    // output[3] -> input[floor(3/2)=1] = 2.0
    // output[4] -> input[floor(4/2)=2] = 3.0
    // output[5] -> input[floor(5/2)=2] = 3.0
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);

    // roi (optional, empty for now)
    let roi = Tensor::from_vec(vec![], vec![0]);

    // scales: [1.0, 2.0] - batch dimension 1x, spatial dimension 2x
    let scales = Tensor::from_vec(vec![1.0, 2.0], vec![2]);

    let result = iconnx::operators::resize::Resize::forward(
        &[x, roi, scales],
        &NodeAttributes::new(),
    );

    assert_eq!(result.shape(), &[1, 6]);
    // With asymmetric + floor: [1, 1, 2, 2, 3, 3]
    assert_eq!(result.as_slice(), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
}

/// Test 2: Resize 2D nearest neighbor (downsample)
#[test]
fn test_resize_2d_nearest_downsample() {
    // 2x4 -> 2x2 (downsample by 0.5 in last dimension)
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 4]);

    let roi = Tensor::from_vec(vec![], vec![0]);

    // scales: [1.0, 0.5] - keep first dimension, halve second
    let scales = Tensor::from_vec(vec![1.0, 0.5], vec![2]);

    let result = iconnx::operators::resize::Resize::forward(
        &[x, roi, scales],
        &NodeAttributes::new(),
    );

    assert_eq!(result.shape(), &[2, 2]);
    // Nearest neighbor: picks [0, 2] from first row, [4, 6] from second row
    assert_eq!(result.as_slice(), vec![1.0, 3.0, 5.0, 7.0]);
}

/// Test 3: Resize 2D nearest neighbor (identity)
#[test]
fn test_resize_2d_identity() {
    // 2x3 -> 2x3 (no change)
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

    let roi = Tensor::from_vec(vec![], vec![0]);

    // scales: [1.0, 1.0] - no scaling
    let scales = Tensor::from_vec(vec![1.0, 1.0], vec![2]);

    let result = iconnx::operators::resize::Resize::forward(
        &[x, roi, scales],
        &NodeAttributes::new(),
    );

    assert_eq!(result.shape(), &[2, 3]);
    assert_eq!(result.as_slice(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

/// Test 4: Resize without roi input (regression test)
/// Default mode: coordinate_transformation_mode=asymmetric, nearest_mode=floor
#[test]
fn test_resize_without_roi() {
    // REGRESSION TEST: When roi is omitted (empty string in ONNX),
    // only 2 inputs are passed: [X, scales]
    // Was failing before with "requires at least 3 inputs"

    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    // scales: [1.0, 2.0] - keep first dimension, double second
    let scales = Tensor::from_vec(vec![1.0, 2.0], vec![2]);

    // Only 2 inputs - roi omitted
    let result =
        iconnx::operators::resize::Resize::forward(&[x, scales], &NodeAttributes::new());

    // Output shape: [2, 4]
    assert_eq!(result.shape(), &[2, 4]);

    // Nearest neighbor upsampling with asymmetric + floor:
    // Row 0: [1, 2] -> positions 0,1,2,3 -> coords 0,0.5,1,1.5 -> floor 0,0,1,1 -> [1,1,2,2]
    // Row 1: [3, 4] -> same -> [3,3,4,4]
    assert_eq!(
        result.as_slice(),
        vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]
    );
}

/// Test 5: Resize 3D tensor (like node 1386)
#[test]
fn test_resize_3d() {
    // Reproduce scenario from node 1386
    // Input: [1, 512, 26] -> scales to different size

    // Small representative case: [1, 2, 3]
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 2, 3]);

    let scales = Tensor::from_vec(vec![1.0, 1.0, 2.0], vec![3]);

    let result =
        iconnx::operators::resize::Resize::forward(&[x, scales], &NodeAttributes::new());

    // Output shape: [1, 2, 6]
    assert_eq!(result.shape(), &[1, 2, 6]);
}
