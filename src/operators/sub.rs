/// Sub operator for Iconnx
///
/// Implements element-wise subtraction with broadcasting support
/// ONNX spec: https://onnx.ai/onnx/operators/onnx__Sub.html
use crate::tensor::Tensor;

/// Sub operator - performs element-wise subtraction
pub struct Sub;

impl Sub {
    /// Forward pass: element-wise subtraction (A - B)
    ///
    /// # Arguments
    /// * `inputs` - Exactly 2 input tensors [A, B]
    ///
    /// # Returns
    /// Tensor containing A - B
    ///
    /// # Panics
    /// Panics if not exactly 2 inputs
    pub fn forward(
        inputs: &[Tensor],
        _attributes: &crate::attributes::NodeAttributes,
    ) -> Tensor {
        assert_eq!(inputs.len(), 2, "Sub requires exactly 2 inputs");

        #[cfg(debug_assertions)]
        {}

        Tensor::dispatch_multi(
            inputs,
            |arrays| {
                // Compute broadcast shape using numpy/ONNX rules
                let max_rank = arrays[0].ndim().max(arrays[1].ndim());

                // Align shapes from the right
                let a_shape = arrays[0].shape();
                let b_shape = arrays[1].shape();
                let mut broadcast_shape = Vec::with_capacity(max_rank);

                for i in 0..max_rank {
                    // Access from right: if max_rank=3 and a_shape.len()=2, then i=0 should skip, i=1,2 should use a_shape[0],a_shape[1]
                    let a_idx = if i + a_shape.len() >= max_rank {
                        i + a_shape.len() - max_rank
                    } else {
                        usize::MAX
                    };
                    let b_idx = if i + b_shape.len() >= max_rank {
                        i + b_shape.len() - max_rank
                    } else {
                        usize::MAX
                    };

                    let a_dim = if a_idx != usize::MAX {
                        a_shape[a_idx]
                    } else {
                        1
                    };
                    let b_dim = if b_idx != usize::MAX {
                        b_shape[b_idx]
                    } else {
                        1
                    };

                    let result_dim = if a_dim == b_dim {
                        a_dim
                    } else if a_dim == 1 {
                        b_dim
                    } else if b_dim == 1 {
                        a_dim
                    } else {
                        panic!(
                            "Sub: Incompatible shapes for broadcasting: {:?} and {:?}",
                            a_shape, b_shape
                        );
                    };

                    broadcast_shape.push(result_dim);
                }

                // Broadcast and subtract
                let a_bc = arrays[0]
                    .broadcast(broadcast_shape.as_slice())
                    .unwrap_or_else(|| {
                        panic!(
                            "Sub: Cannot broadcast {:?} to {:?}",
                            arrays[0].shape(),
                            broadcast_shape
                        )
                    })
                    .to_owned();
                let b_bc = arrays[1]
                    .broadcast(broadcast_shape.as_slice())
                    .unwrap_or_else(|| {
                        panic!(
                            "Sub: Cannot broadcast {:?} to {:?}",
                            arrays[1].shape(),
                            broadcast_shape
                        )
                    })
                    .to_owned();
                &a_bc - &b_bc
            },
            |arrays| {
                // Same for i64 - compute broadcast shape using numpy/ONNX rules
                let max_rank = arrays[0].ndim().max(arrays[1].ndim());

                // Align shapes from the right
                let a_shape = arrays[0].shape();
                let b_shape = arrays[1].shape();
                let mut broadcast_shape = Vec::with_capacity(max_rank);

                for i in 0..max_rank {
                    // Access from right: if max_rank=3 and a_shape.len()=2, then i=0 should skip, i=1,2 should use a_shape[0],a_shape[1]
                    let a_idx = if i + a_shape.len() >= max_rank {
                        i + a_shape.len() - max_rank
                    } else {
                        usize::MAX
                    };
                    let b_idx = if i + b_shape.len() >= max_rank {
                        i + b_shape.len() - max_rank
                    } else {
                        usize::MAX
                    };

                    let a_dim = if a_idx != usize::MAX {
                        a_shape[a_idx]
                    } else {
                        1
                    };
                    let b_dim = if b_idx != usize::MAX {
                        b_shape[b_idx]
                    } else {
                        1
                    };

                    let result_dim = if a_dim == b_dim {
                        a_dim
                    } else if a_dim == 1 {
                        b_dim
                    } else if b_dim == 1 {
                        a_dim
                    } else {
                        panic!(
                            "Sub: Incompatible shapes for broadcasting: {:?} and {:?}",
                            a_shape, b_shape
                        );
                    };

                    broadcast_shape.push(result_dim);
                }

                let a_bc = arrays[0]
                    .broadcast(broadcast_shape.as_slice())
                    .unwrap()
                    .to_owned();
                let b_bc = arrays[1]
                    .broadcast(broadcast_shape.as_slice())
                    .unwrap()
                    .to_owned();
                &a_bc - &b_bc
            },
        )
    }
}
