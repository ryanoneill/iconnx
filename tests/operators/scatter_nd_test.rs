use iconnx::attributes::NodeAttributes;
/// Test ScatterND operator for Iconnx
///
/// TDD: Tests written FIRST
/// ScatterND: Scatter updates into tensor using indices
use iconnx::tensor::Tensor;

/// Test 1: Basic scatter into 1D tensor
#[test]
fn test_scatter_nd_1d_basic() {
    // Start with zeros
    let data = Tensor::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0], vec![5]);
    // Indices: scatter at positions [1] and [3]
    let indices = Tensor::from_vec(vec![1.0, 3.0], vec![2, 1]);
    // Updates: values to scatter
    let updates = Tensor::from_vec(vec![10.0, 20.0], vec![2]);

    let result = iconnx::operators::scatter_nd::ScatterND::forward(
        &[data, indices, updates],
        &NodeAttributes::new(),
    );

    assert_eq!(result.shape(), &[5]);
    assert_eq!(result.as_slice(), vec![0.0, 10.0, 0.0, 20.0, 0.0]);
}

/// Test 2: Scatter into 2D tensor (update individual elements)
#[test]
fn test_scatter_nd_2d_elements() {
    // 3x3 matrix of zeros
    let data = Tensor::from_vec(vec![0.0; 9], vec![3, 3]);
    // Indices: [[0,0], [1,1], [2,2]] - diagonal positions
    let indices = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0], vec![3, 2]);
    // Updates: values for diagonal
    let updates = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

    let result = iconnx::operators::scatter_nd::ScatterND::forward(
        &[data, indices, updates],
        &NodeAttributes::new(),
    );

    assert_eq!(result.shape(), &[3, 3]);
    assert_eq!(
        result.as_slice(),
        vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]
    );
}

/// Test 3: Scatter rows into 2D tensor
#[test]
fn test_scatter_nd_2d_rows() {
    // 3x2 matrix of zeros
    let data = Tensor::from_vec(vec![0.0; 6], vec![3, 2]);
    // Indices: [1] - update row 1
    let indices = Tensor::from_vec(vec![1.0], vec![1, 1]);
    // Updates: entire row [5.0, 6.0]
    let updates = Tensor::from_vec(vec![5.0, 6.0], vec![1, 2]);

    let result = iconnx::operators::scatter_nd::ScatterND::forward(
        &[data, indices, updates],
        &NodeAttributes::new(),
    );

    assert_eq!(result.shape(), &[3, 2]);
    assert_eq!(result.as_slice(), vec![0.0, 0.0, 5.0, 6.0, 0.0, 0.0]);
}

/// Test 4: Multiple updates to same location (later wins)
#[test]
fn test_scatter_nd_overwrites() {
    let data = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
    // Update position [1] twice
    let indices = Tensor::from_vec(vec![1.0, 1.0], vec![2, 1]);
    let updates = Tensor::from_vec(vec![10.0, 20.0], vec![2]);

    let result = iconnx::operators::scatter_nd::ScatterND::forward(
        &[data, indices, updates],
        &NodeAttributes::new(),
    );

    assert_eq!(result.shape(), &[3]);
    // Second update (20.0) should win
    assert_eq!(result.as_slice(), vec![1.0, 20.0, 3.0]);
}

/// Test 5: 3D indices (like Kokoro model node 2456)
/// indices.shape = [batch, extra_dim, index_depth]
#[test]
fn test_scatter_nd_3d_indices() {
    // Simulate Kokoro: data.shape = [1, 100], indices.shape = [3, 1, 2], updates.shape = [3, 1]
    let data = Tensor::from_vec(vec![0.0; 100], vec![1, 100]);

    // 3D indices: batch_dims = [3, 1], index_depth = 2
    // Each index is [row, col] into [1, 100]
    // indices values: [[0, 10], [0, 20], [0, 30]]
    let indices = Tensor::from_vec(
        vec![
            0.0, 10.0, // first update at [0, 10]
            0.0, 20.0, // second update at [0, 20]
            0.0, 30.0, // third update at [0, 30]
        ],
        vec![3, 1, 2],
    );

    // updates shape [3, 1] - one scalar per update
    let updates = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3, 1]);

    let result = iconnx::operators::scatter_nd::ScatterND::forward(
        &[data, indices, updates],
        &NodeAttributes::new(),
    );

    assert_eq!(result.shape(), &[1, 100]);
    let vals = result.as_slice();
    assert_eq!(vals[10], 1.0);
    assert_eq!(vals[20], 2.0);
    assert_eq!(vals[30], 3.0);
    assert_eq!(vals[0], 0.0); // unchanged
    assert_eq!(vals[50], 0.0); // unchanged
}

/// Test 6: Larger 3D indices matching exact Kokoro scenario
/// data.shape = [1, 15620], indices.shape = [19, 1, 2], updates.shape = [19, 1]
#[test]
fn test_scatter_nd_3d_indices_large() {
    let data = Tensor::from_vec(vec![0.0; 15620], vec![1, 15620]);

    // 19 updates, each with index depth 2
    // All first indices are 0 (valid for dim 0 of size 1)
    // Second indices are various positions in dim 1 (size 15620)
    let mut indices_data = Vec::with_capacity(19 * 2);
    for i in 0..19 {
        indices_data.push(0.0); // row index (always 0)
        indices_data.push((i * 100) as f32); // col index
    }
    let indices = Tensor::from_vec(indices_data, vec![19, 1, 2]);

    // 19 updates, each a scalar
    let updates_data: Vec<f32> = (1..=19).map(|i| i as f32).collect();
    let updates = Tensor::from_vec(updates_data, vec![19, 1]);

    let result = iconnx::operators::scatter_nd::ScatterND::forward(
        &[data, indices, updates],
        &NodeAttributes::new(),
    );

    assert_eq!(result.shape(), &[1, 15620]);
    let vals = result.as_slice();

    // Check that values were scattered correctly
    for i in 0..19 {
        assert_eq!(
            vals[i * 100],
            (i + 1) as f32,
            "Position {} should be {}",
            i * 100,
            i + 1
        );
    }
}
