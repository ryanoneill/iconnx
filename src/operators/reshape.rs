/// Reshape operator for Iconnx
///
/// Changes tensor shape without modifying data
/// ONNX spec: https://onnx.ai/onnx/operators/onnx__Reshape.html
use crate::tensor::Tensor;
use ndarray::IxDyn;

/// Reshape operator - changes tensor dimensions
pub struct Reshape;

impl Reshape {
    /// Forward pass: reshape tensor to new shape
    ///
    /// # Arguments
    /// * `inputs` - [data_tensor, shape_tensor]
    ///   - data_tensor: The tensor to reshape
    ///   - shape_tensor: 1D tensor containing new shape (f32 values converted to usize)
    ///
    /// # Returns
    /// Reshaped tensor with same data, different shape
    ///
    /// # Panics
    /// Panics if inputs.len() != 2 or shapes are incompatible
    ///
    /// # ONNX Compatibility
    /// - Supports -1 dimension (infer from total size)
    /// - Shape is provided as a tensor (ONNX standard)
    pub fn forward(
        inputs: &[Tensor],
        _attributes: &crate::attributes::NodeAttributes,
    ) -> Tensor {
        assert_eq!(inputs.len(), 2, "Reshape requires exactly 2 inputs");

        let data = &inputs[0];
        let shape_tensor = &inputs[1];

        // Extract target shape from tensor (Int64 or Float32)
        let shape_values: Vec<i64> = if shape_tensor.is_int64() {
            shape_tensor.as_slice_i64()
        } else {
            shape_tensor.as_slice().iter().map(|&x| x as i64).collect()
        };

        let total_elements = data.len();
        let input_shape = data.shape();

        // Convert shape to usize, handling special ONNX values:
        // - 0 means: copy corresponding dimension from input shape
        // - -1 means: infer this dimension from total elements
        let mut target_shape: Vec<usize> = Vec::new();
        let mut infer_idx = None;

        for (i, &val) in shape_values.iter().enumerate() {
            if val == -1 {
                if infer_idx.is_some() {
                    panic!("Reshape: Only one dimension can be -1");
                }
                infer_idx = Some(i);
                target_shape.push(0); // Placeholder for inferred dimension
            } else if val == 0 {
                // Copy corresponding dimension from input shape
                if i >= input_shape.len() {
                    panic!(
                        "Reshape: Shape value 0 at index {} but input only has {} dimensions",
                        i,
                        input_shape.len()
                    );
                }
                target_shape.push(input_shape[i]);
            } else {
                assert!(
                    val > 0,
                    "Reshape: Shape values must be > 0, 0 (copy), or -1 (infer)"
                );
                target_shape.push(val as usize);
            }
        }

        // If we have a -1, infer that dimension
        if let Some(idx) = infer_idx {
            let known_product: usize = target_shape.iter().filter(|&&x| x > 0).product();
            if known_product == 0 {
                panic!("Reshape: Cannot infer dimension when product is 0");
            }
            target_shape[idx] = total_elements / known_product;
        }

        // Validate total elements match
        let target_elements: usize = if target_shape.is_empty() {
            1 // Scalar
        } else {
            target_shape.iter().product()
        };

        #[cfg(debug_assertions)]
        {}

        assert_eq!(
            total_elements, target_elements,
            "Reshape: Element count mismatch. Data has {} elements but target shape {:?} requires {}",
            total_elements, target_shape, target_elements
        );

        // Reshape using ndarray - handle different dtypes
        match data {
            Tensor::Float32(arr) => {
                let reshaped = arr
                    .clone()
                    .to_shape(IxDyn(&target_shape))
                    .expect("Failed to reshape tensor")
                    .into_owned();
                Tensor::from_array(reshaped)
            }
            Tensor::Int64(arr) => {
                let reshaped = arr
                    .clone()
                    .to_shape(IxDyn(&target_shape))
                    .expect("Failed to reshape tensor")
                    .into_owned();
                Tensor::from_array_i64(reshaped)
            }
            Tensor::Int32(arr) => {
                let reshaped = arr
                    .clone()
                    .to_shape(IxDyn(&target_shape))
                    .expect("Failed to reshape tensor")
                    .into_owned();
                Tensor::from_array_i32(reshaped)
            }
            _ => {
                // For other dtypes, convert to Float32
                let data_f32 = data.to_float32();
                Self::forward(&[data_f32, inputs[1].clone()], _attributes)
            }
        }
    }
}
