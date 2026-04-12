/// Test ConstantOfShape operator for Iconnx
///
/// TDD: Tests written FIRST
/// ConstantOfShape: Create tensor filled with constant value
use iconnx::attributes::NodeAttributes;
use iconnx::tensor::Tensor;

/// Test 1: Create 2x3 tensor filled with zeros (default)
#[test]
fn test_constant_of_shape_zeros() {
    let shape = Tensor::from_vec(vec![2.0, 3.0], vec![2]);

    // Default value is 0.0
    let result = iconnx::operators::constant_of_shape::ConstantOfShape::forward(
        &[shape],
        &NodeAttributes::new(),
    );

    assert_eq!(result.shape(), &[2, 3]);
    assert_eq!(result.as_slice(), vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
}

/// Test 2: Create 3x2 tensor filled with ones
#[test]
fn test_constant_of_shape_ones() {
    let shape = Tensor::from_vec(vec![3.0, 2.0], vec![2]);

    // Set value=1.0 via attributes
    let mut attrs = NodeAttributes::new();
    attrs.add_tensor("value".to_string(), Tensor::from_vec(vec![1.0], vec![]));

    let result =
        iconnx::operators::constant_of_shape::ConstantOfShape::forward(&[shape], &attrs);

    assert_eq!(result.shape(), &[3, 2]);
    assert_eq!(result.as_slice(), vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
}

/// Test 3: Create 1D tensor (vector) filled with custom value
#[test]
fn test_constant_of_shape_1d_custom() {
    let shape = Tensor::from_vec(vec![5.0], vec![1]);

    // Set value=PI via attributes (arbitrary test value, chosen as a
    // named constant so clippy's approx_constant lint stays happy)
    let mut attrs = NodeAttributes::new();
    attrs.add_tensor(
        "value".to_string(),
        Tensor::from_vec(vec![std::f32::consts::PI], vec![]),
    );

    let result =
        iconnx::operators::constant_of_shape::ConstantOfShape::forward(&[shape], &attrs);

    assert_eq!(result.shape(), &[5]);
    assert_eq!(result.as_slice(), vec![std::f32::consts::PI; 5]);
}

/// Test 4: Create 3D tensor filled with negative value
#[test]
fn test_constant_of_shape_3d() {
    let shape = Tensor::from_vec(vec![2.0, 2.0, 2.0], vec![3]);

    // Set value=-1.0 via attributes
    let mut attrs = NodeAttributes::new();
    attrs.add_tensor("value".to_string(), Tensor::from_vec(vec![-1.0], vec![]));

    let result =
        iconnx::operators::constant_of_shape::ConstantOfShape::forward(&[shape], &attrs);

    assert_eq!(result.shape(), &[2, 2, 2]);
    assert_eq!(
        result.as_slice(),
        vec![-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
    );
}
