//! Reduction dispatch: ReduceMean, ReduceSum, LayerNormalization.
//!
//! Routes to `gpu_reduce_sum_axis`, `gpu_reduce_mean_axis`, and
//! `gpu_layer_norm` in `crate::cuda::reduction`. Today's GPU reductions
//! always operate on the last axis; the `keepdims` attribute is honored by
//! appending a trailing size-1 dimension after the kernel returns (mirrors
//! today's inference dispatch exactly).
//!
//! Dispatch target parity with today's `execute_gpu_operator_with_precomputed`
//! in `src/cuda/inference/mod.rs`: same kernel calls, same attribute
//! defaults (`keepdims=1`, `epsilon=1e-5`), same DEBUG_REDUCESUM trace
//! points. Kernel wrappers stay in their source modules; this file owns
//! only the dispatch match.

use std::collections::HashMap;

use crate::cuda::reduction::{gpu_layer_norm, gpu_reduce_mean_axis, gpu_reduce_sum_axis};
use crate::cuda::{CudaError, GpuTensor};
use crate::ir::PlannedOp;

use super::{resolve_inputs, Executor};

impl Executor {
    /// Dispatch a reduction op to its backing kernel.
    ///
    /// Handles ReduceSum, ReduceMean, and LayerNormalization. Softmax lives
    /// in `elementwise.rs` by plan convention even though it routes through
    /// `self.reduction_kernels`.
    pub(crate) fn dispatch_reduction(
        &self,
        op: &PlannedOp,
        values: &HashMap<String, GpuTensor>,
        weights: &HashMap<String, GpuTensor>,
    ) -> Result<GpuTensor, CudaError> {
        let inputs = resolve_inputs(&op.node.inputs, values, weights)?;
        let attributes = &op.node.attributes;
        let mut pool = self.memory_pool.borrow_mut();

        match op.node.op_type.as_str() {
            "ReduceSum" => {
                let keepdims = attributes.get_int("keepdims").unwrap_or(1) != 0;

                // Debug: trace ReduceSum input for duration predictor debugging.
                if std::env::var("DEBUG_REDUCESUM").is_ok() {
                    if let Ok(input_vals) = inputs[0].to_host_f32(&self.ctx) {
                        let display: Vec<_> = input_vals
                            .iter()
                            .take(10)
                            .map(|v| format!("{:.4}", v))
                            .collect();
                        eprintln!(
                            "DEBUG_REDUCESUM input: shape={:?}, first_10={:?}{}",
                            inputs[0].shape(),
                            display,
                            if input_vals.len() > 10 { "..." } else { "" }
                        );
                    }
                }

                let result = gpu_reduce_sum_axis(
                    &self.ctx,
                    &self.reduction_kernels,
                    &mut pool,
                    inputs[0],
                )?;

                if std::env::var("DEBUG_REDUCESUM").is_ok() {
                    if let Ok(output_vals) = result.to_host_f32(&self.ctx) {
                        eprintln!(
                            "DEBUG_REDUCESUM output: shape={:?}, values={:?}",
                            result.shape(),
                            output_vals
                        );
                    }
                }

                if keepdims {
                    let mut new_shape = result.shape().to_vec();
                    new_shape.push(1);
                    result.reshape(new_shape).ok_or_else(|| {
                        CudaError::Kernel("ReduceSum: failed to restore keepdims".into())
                    })
                } else {
                    Ok(result)
                }
            }

            "ReduceMean" => {
                let keepdims = attributes.get_int("keepdims").unwrap_or(1) != 0;
                let result = gpu_reduce_mean_axis(
                    &self.ctx,
                    &self.reduction_kernels,
                    &mut pool,
                    inputs[0],
                )?;
                if keepdims {
                    let mut new_shape = result.shape().to_vec();
                    new_shape.push(1);
                    result.reshape(new_shape).ok_or_else(|| {
                        CudaError::Kernel("ReduceMean: failed to restore keepdims".into())
                    })
                } else {
                    Ok(result)
                }
            }

            "LayerNormalization" => {
                let epsilon = attributes.get_float("epsilon").unwrap_or(1e-5);
                if inputs.len() < 3 {
                    return Err(CudaError::Kernel(
                        "LayerNormalization requires scale and bias inputs".into(),
                    ));
                }
                gpu_layer_norm(
                    &self.ctx,
                    &self.reduction_kernels,
                    &mut pool,
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    epsilon,
                )
            }

            other => Err(CudaError::Kernel(format!(
                "dispatch_reduction called with non-reduction op '{}'",
                other
            ))),
        }
    }
}
