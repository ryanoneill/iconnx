/// MatMul operator for Iconnx
///
/// Implements matrix multiplication (not element-wise!)
/// ONNX spec: https://onnx.ai/onnx/operators/onnx__MatMul.html
///
/// Behavior:
/// - 1D x 1D: dot product -> scalar
/// - 2D x 2D: standard matrix multiplication
/// - ND x ND: batched matrix multiplication (batch dimensions must broadcast)
use crate::tensor::Tensor;
use ndarray::{ArrayD, Axis};

/// MatMul operator - performs matrix multiplication
pub struct MatMul;

impl MatMul {
    /// Forward pass: matrix multiplication
    ///
    /// # Arguments
    /// * `inputs` - Exactly 2 input tensors [A, B]
    ///
    /// # Returns
    /// Tensor containing A @ B
    ///
    /// # Panics
    /// Panics if not exactly 2 inputs or shapes incompatible
    pub fn forward(
        inputs: &[Tensor],
        _attributes: &crate::attributes::NodeAttributes,
    ) -> Tensor {
        assert_eq!(inputs.len(), 2, "MatMul requires exactly 2 inputs");

        let a = inputs[0].to_array();
        let b = inputs[1].to_array();

        let a_ndim = a.ndim();
        let b_ndim = b.ndim();

        #[cfg(debug_assertions)]
        {}

        let result = match (a_ndim, b_ndim) {
            // 1D x 1D: dot product -> scalar
            (1, 1) => {
                let dot = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>();
                ArrayD::from_shape_vec(vec![], vec![dot]).expect("Failed to create scalar")
            }

            // 2D x 2D: standard matrix multiplication
            (2, 2) => {
                let a_2d = a.view().into_dimensionality::<ndarray::Ix2>().unwrap();
                let b_2d = b.view().into_dimensionality::<ndarray::Ix2>().unwrap();
                let result_2d = a_2d.dot(&b_2d);
                result_2d.into_dyn()
            }

            // 1D x 2D: treat 1D as row vector [1, N]
            (1, 2) => {
                let a_reshaped = a.view().insert_axis(Axis(0));
                let a_2d = a_reshaped.into_dimensionality::<ndarray::Ix2>().unwrap();
                let b_2d = b.view().into_dimensionality::<ndarray::Ix2>().unwrap();
                let result_2d = a_2d.dot(&b_2d);
                result_2d.into_dyn()
            }

            // 2D x 1D: treat 1D as column vector [N, 1]
            (2, 1) => {
                let a_2d = a.view().into_dimensionality::<ndarray::Ix2>().unwrap();
                let b_reshaped = b.view().insert_axis(Axis(1));
                let b_2d = b_reshaped.into_dimensionality::<ndarray::Ix2>().unwrap();
                let result_2d = a_2d.dot(&b_2d);
                result_2d.into_dyn()
            }

            // ND x ND: batched matrix multiplication with broadcasting
            _ if a_ndim >= 2 && b_ndim >= 2 => {
                // Get batch dimensions and matrix dimensions
                let a_shape = a.shape();
                let b_shape = b.shape();

                // Last 2 dims are matrix dims
                let a_batch: Vec<usize> = a_shape[..a_shape.len() - 2].to_vec();
                let b_batch: Vec<usize> = b_shape[..b_shape.len() - 2].to_vec();

                let a_rows = a_shape[a_shape.len() - 2];
                let a_cols = a_shape[a_shape.len() - 1];
                let b_rows = b_shape[b_shape.len() - 2];
                let b_cols = b_shape[b_shape.len() - 1];

                assert_eq!(
                    a_cols, b_rows,
                    "Inner dimensions must match: {} != {}",
                    a_cols, b_rows
                );

                // Handle broadcasting: if one has no batch dims, broadcast it
                let (out_batch, batch_size, b_is_broadcasted) =
                    if a_batch.is_empty() && !b_batch.is_empty() {
                        // A has no batch, B does - broadcast A
                        let size = b_batch.iter().product();
                        (b_batch.clone(), size, false)
                    } else if !a_batch.is_empty() && b_batch.is_empty() {
                        // B has no batch, A does - broadcast B
                        let size = a_batch.iter().product();
                        (a_batch.clone(), size, true)
                    } else if a_batch == b_batch {
                        // Same batch dims
                        let size = a_batch.iter().product();
                        (a_batch.clone(), size, false)
                    } else {
                        panic!(
                            "Batch dimensions must be compatible for broadcasting: {:?} vs {:?}",
                            a_batch, b_batch
                        );
                    };

                // Calculate output shape: batch_dims + [a_rows, b_cols]
                let mut out_shape = out_batch.clone();
                out_shape.push(a_rows);
                out_shape.push(b_cols);

                // Perform batched matmul
                let mut result_data = Vec::with_capacity(batch_size * a_rows * b_cols);

                // Get the 2D matrices (ignoring batch dims)
                let a_matrix_size = a_rows * a_cols;
                let b_matrix_size = b_rows * b_cols;

                for batch_idx in 0..batch_size {
                    // Extract 2D slice from A
                    // Use slice_axis to handle non-contiguous arrays
                    let a_batch_idx = if a_batch.is_empty() { 0 } else { batch_idx };

                    // For multidimensional arrays, we need to reshape and extract
                    // Convert to owned contiguous array first
                    let a_data: Vec<f32> = if let Some(slice) = a.as_slice() {
                        slice.to_vec()
                    } else {
                        // Non-contiguous - collect via iterator
                        a.iter().copied().collect()
                    };

                    let a_start = a_batch_idx * a_matrix_size;
                    let a_end = a_start + a_matrix_size;
                    let a_matrix = ndarray::Array2::from_shape_vec(
                        (a_rows, a_cols),
                        a_data[a_start..a_end].to_vec(),
                    )
                    .expect("Failed to reshape A slice");

                    // Extract 2D slice from B
                    let b_batch_idx = if b_is_broadcasted { 0 } else { batch_idx };

                    let b_data: Vec<f32> = if let Some(slice) = b.as_slice() {
                        slice.to_vec()
                    } else {
                        // Non-contiguous - collect via iterator
                        b.iter().copied().collect()
                    };

                    let b_start = b_batch_idx * b_matrix_size;
                    let b_end = b_start + b_matrix_size;
                    let b_matrix = ndarray::Array2::from_shape_vec(
                        (b_rows, b_cols),
                        b_data[b_start..b_end].to_vec(),
                    )
                    .expect("Failed to reshape B slice");

                    // Matrix multiply
                    let result_matrix = a_matrix.dot(&b_matrix);

                    // Append to result (result_matrix is contiguous)
                    if let Some(slice) = result_matrix.as_slice() {
                        result_data.extend_from_slice(slice);
                    } else {
                        result_data.extend(result_matrix.iter().copied());
                    }
                }

                #[cfg(debug_assertions)]
                {}

                ArrayD::from_shape_vec(out_shape, result_data).expect("Failed to create result")
            }

            _ => panic!("Unsupported MatMul dimensions: {}D x {}D", a_ndim, b_ndim),
        };

        Tensor::from_array(result)
    }
}
