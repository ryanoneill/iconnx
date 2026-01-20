/// Expand operator for Iconnx
///
/// Broadcast tensor to target shape
/// ONNX spec: https://onnx.ai/onnx/operators/onnx__Expand.html
use crate::tensor::Tensor;
use ndarray::ArrayD;

/// Expand operator - broadcast to target shape
pub struct Expand;

impl Expand {
    /// Forward pass: expand (broadcast) tensor
    ///
    /// # Arguments
    /// * `inputs` - [data, shape]
    ///   - data: input tensor
    ///   - shape: target shape (as f32 or int64 values)
    ///
    /// # Returns
    /// Broadcasted tensor
    ///
    /// # ONNX Broadcasting Semantics
    /// 1. Prepend 1s to shorter shape
    /// 2. For each dimension: must be equal OR one must be 1
    pub fn forward(
        inputs: &[Tensor],
        _attributes: &crate::attributes::NodeAttributes,
    ) -> Tensor {
        assert_eq!(
            inputs.len(),
            2,
            "Expand requires exactly 2 inputs [data, shape]"
        );

        // Extract target shape (Int64 or Float32)
        let target_shape_raw: Vec<i64> = if inputs[1].is_int64() {
            inputs[1].as_slice_i64().to_vec()
        } else {
            inputs[1].as_slice().iter().map(|&x| x as i64).collect()
        };

        let input_shape = inputs[0].shape();

        // ONNX Expand: When shape[i] = 1, output uses input dimension
        // Align input with target by prepending 1s to input
        let target_ndim = target_shape_raw.len();
        let input_ndim = input_shape.len();
        let prepend_count = target_ndim.saturating_sub(input_ndim);

        // Create aligned input shape by prepending 1s
        let mut aligned_input = vec![1; prepend_count];
        aligned_input.extend_from_slice(input_shape);

        // Build output shape: when shape[i]=1, use aligned_input[i]
        let mut output_shape: Vec<usize> = Vec::new();
        for (i, &val) in target_shape_raw.iter().enumerate() {
            if val == 1 {
                // Use input dimension (ONNX Expand semantics!)
                output_shape.push(aligned_input[i]);
            } else if val > 1 {
                output_shape.push(val as usize);
            } else {
                panic!(
                    "Expand: shape values must be >= 1, got {} at index {}",
                    val, i
                );
            }
        }

        // For broadcasting, use output_shape as the target
        let target_shape = output_shape.clone();

        #[cfg(debug_assertions)]
        {}

        // ONNX Expand uses numpy-style broadcasting with dimension prepending
        // If input has fewer dims than target, prepend 1s
        inputs[0].map_preserving_dtype(
            |data_f32| Self::broadcast_onnx(data_f32, &target_shape),
            |data_i64| Self::broadcast_onnx(data_i64, &target_shape),
        )
    }

    /// Broadcast array to target shape using ONNX semantics
    fn broadcast_onnx<T: Clone>(data: &ArrayD<T>, target_shape: &[usize]) -> ArrayD<T> {
        let input_shape = data.shape();
        let input_ndim = input_shape.len();
        let target_ndim = target_shape.len();

        // Step 1: Prepend 1s if input has fewer dimensions
        let mut expanded = data.clone();
        for _ in 0..(target_ndim.saturating_sub(input_ndim)) {
            expanded = expanded.insert_axis(ndarray::Axis(0));
        }

        let expanded_shape = expanded.shape();

        // Step 2: Broadcast each dimension
        // ndarray::broadcast should now work since we've prepended 1s
        expanded
            .broadcast(target_shape)
            .unwrap_or_else(|| {
                panic!(
                    "Expand: Cannot broadcast {:?} to {:?} even after prepending 1s. Expanded to {:?}",
                    input_shape, target_shape, expanded_shape
                )
            })
            .to_owned()
    }
}
