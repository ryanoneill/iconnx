//! Miscellaneous dispatch: logical And, Or, Not.
//!
//! Ports the `"And"`, `"Or"`, `"Not"` arms from today's
//! `execute_gpu_operator_with_precomputed` in `src/cuda/inference/mod.rs`.
//! `And` and `Or` support ONNX-style broadcasting via the shared
//! `compute_broadcast_shape` + `maybe_expand` helpers; `Not` is unary
//! and passes the single input straight through.

use std::collections::HashMap;

use crate::cuda::inference::{compute_broadcast_shape, maybe_expand};
use crate::cuda::ops::{gpu_and, gpu_not, gpu_or};
use crate::cuda::{CudaError, GpuTensor};
use crate::ir::PlannedOp;

use super::{resolve_inputs, Executor};

impl Executor {
    /// Dispatch a single logical op (`And`, `Or`, `Not`) to its
    /// `gpu_*` kernel wrapper.
    ///
    /// The broadcasting path mirrors the binary-arith path used in
    /// `dispatch_elementwise`: same fast-path when shapes match, same
    /// `compute_broadcast_shape` fallback, same error message on
    /// non-broadcastable shapes.
    pub(crate) fn dispatch_misc(
        &self,
        op: &PlannedOp,
        values: &HashMap<String, GpuTensor>,
        weights: &HashMap<String, GpuTensor>,
    ) -> Result<GpuTensor, CudaError> {
        let inputs = resolve_inputs(&op.node.inputs, values, weights)?;
        let mut pool = self.memory_pool.borrow_mut();
        match op.node.op_type.as_str() {
            "And" => {
                let (a, b) = (inputs[0], inputs[1]);
                if a.shape() == b.shape() {
                    gpu_and(&self.ctx, &self.ops_kernels, a, b)
                } else if let Some(broadcast_shape) = compute_broadcast_shape(a.shape(), b.shape())
                {
                    let a_exp = maybe_expand(
                        &self.ctx,
                        &self.ops_kernels,
                        &mut pool,
                        a,
                        broadcast_shape.clone(),
                    )?;
                    let b_exp = maybe_expand(
                        &self.ctx,
                        &self.ops_kernels,
                        &mut pool,
                        b,
                        broadcast_shape,
                    )?;
                    gpu_and(&self.ctx, &self.ops_kernels, &a_exp, &b_exp)
                } else {
                    Err(CudaError::Kernel(format!(
                        "And: shapes {:?} and {:?} not broadcastable",
                        a.shape(),
                        b.shape()
                    )))
                }
            }
            "Or" => {
                let (a, b) = (inputs[0], inputs[1]);
                if a.shape() == b.shape() {
                    gpu_or(&self.ctx, &self.ops_kernels, a, b)
                } else if let Some(broadcast_shape) = compute_broadcast_shape(a.shape(), b.shape())
                {
                    let a_exp = maybe_expand(
                        &self.ctx,
                        &self.ops_kernels,
                        &mut pool,
                        a,
                        broadcast_shape.clone(),
                    )?;
                    let b_exp = maybe_expand(
                        &self.ctx,
                        &self.ops_kernels,
                        &mut pool,
                        b,
                        broadcast_shape,
                    )?;
                    gpu_or(&self.ctx, &self.ops_kernels, &a_exp, &b_exp)
                } else {
                    Err(CudaError::Kernel(format!(
                        "Or: shapes {:?} and {:?} not broadcastable",
                        a.shape(),
                        b.shape()
                    )))
                }
            }
            "Not" => gpu_not(&self.ctx, &self.ops_kernels, inputs[0]),
            other => Err(CudaError::Kernel(format!(
                "dispatch_misc: unsupported op_type '{}'",
                other
            ))),
        }
    }
}
