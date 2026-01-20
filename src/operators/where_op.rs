/// Where operator for Iconnx
///
/// Conditional selection: condition ? x : y
/// ONNX spec: https://onnx.ai/onnx/operators/onnx__Where.html
use crate::tensor::Tensor;
use ndarray::Zip;

/// Where operator - element-wise conditional selection
pub struct Where;

impl Where {
    /// Forward pass: element-wise conditional selection
    ///
    /// # Arguments
    /// * `inputs` - [condition, x, y]
    ///   - condition: boolean tensor (1.0 = true, 0.0 = false)
    ///   - x: values when condition is true
    ///   - y: values when condition is false
    ///
    /// # Returns
    /// Tensor with selected values
    pub fn forward(
        inputs: &[Tensor],
        _attributes: &crate::attributes::NodeAttributes,
    ) -> Tensor {
        assert_eq!(
            inputs.len(),
            3,
            "Where requires exactly 3 inputs [condition, x, y]"
        );

        // ONNX Where should preserve dtype of x and y (they must match)
        // Only condition needs to be boolean/numeric
        let condition = &inputs[0];
        let x = &inputs[1];
        let y = &inputs[2];

        #[cfg(debug_assertions)]
        {}

        // Convert condition to float for comparison
        let cond_f32 = condition.to_float32();

        // Determine broadcast shape (max of each dimension)
        let max_rank = *[cond_f32.ndim(), x.ndim(), y.ndim()].iter().max().unwrap();
        let mut broadcast_shape = vec![1; max_rank];

        for (i, dim) in cond_f32.shape().iter().rev().enumerate() {
            let idx = broadcast_shape.len() - 1 - i;
            broadcast_shape[idx] = broadcast_shape[idx].max(*dim);
        }
        for (i, dim) in x.shape().iter().rev().enumerate() {
            let idx = broadcast_shape.len() - 1 - i;
            broadcast_shape[idx] = broadcast_shape[idx].max(*dim);
        }
        for (i, dim) in y.shape().iter().rev().enumerate() {
            let idx = broadcast_shape.len() - 1 - i;
            broadcast_shape[idx] = broadcast_shape[idx].max(*dim);
        }

        // Broadcast condition (always as Float32 for comparison)
        let cond_bc = cond_f32
            .to_array()
            .broadcast(broadcast_shape.as_slice())
            .expect("Failed to broadcast condition")
            .to_owned();

        // Apply Where preserving dtype of x and y
        let result = Tensor::dispatch_multi(
            &[x.clone(), y.clone()],
            |arrays| {
                // Float32 path
                let x_bc = arrays[0]
                    .broadcast(broadcast_shape.as_slice())
                    .expect("Failed to broadcast x")
                    .to_owned();
                let y_bc = arrays[1]
                    .broadcast(broadcast_shape.as_slice())
                    .expect("Failed to broadcast y")
                    .to_owned();

                Zip::from(&cond_bc)
                    .and(&x_bc)
                    .and(&y_bc)
                    .map_collect(|&c, &x_val, &y_val| if c != 0.0 { x_val } else { y_val })
            },
            |arrays| {
                // Int64 path
                let x_bc = arrays[0]
                    .broadcast(broadcast_shape.as_slice())
                    .expect("Failed to broadcast x (int64)")
                    .to_owned();
                let y_bc = arrays[1]
                    .broadcast(broadcast_shape.as_slice())
                    .expect("Failed to broadcast y (int64)")
                    .to_owned();

                Zip::from(&cond_bc)
                    .and(&x_bc)
                    .and(&y_bc)
                    .map_collect(|&c, &x_val, &y_val| if c != 0.0 { x_val } else { y_val })
            },
        );

        // Debug logging removed - was placeholder code

        result
    }
}
