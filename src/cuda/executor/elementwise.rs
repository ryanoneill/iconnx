//! Elementwise op dispatch (arith, unary math, comparisons, activations).
//!
//! Covers: Add, Sub, Mul, Div, Pow, Exp, Sqrt, Sin, Cos, Tanh, Sigmoid,
//! LeakyRelu, Atan, Round, Floor, Clip, Softmax, Equal, Less, Greater,
//! GreaterOrEqual, LessOrEqual, NotEqual.
//!
//! Dispatch target parity with today's `execute_gpu_operator_with_precomputed`
//! in `src/cuda/inference/mod.rs`: same per-op `gpu_*` wrappers, same
//! broadcasting logic via `compute_broadcast_shape` + `maybe_expand`, same
//! attribute defaults. Kernel wrappers stay in their source modules;
//! this file owns only the dispatch match.
//!
//! Softmax lives here (rather than in `reduction.rs`) by plan convention —
//! the category split is organizational, not kernel-architectural: the
//! `gpu_softmax` call still routes through `self.reduction_kernels`.

use std::collections::HashMap;

use crate::cuda::inference::{compute_broadcast_shape, maybe_expand};
use crate::cuda::kernels::{
    gpu_add, gpu_clip, gpu_cos, gpu_div, gpu_erf, gpu_exp, gpu_floor, gpu_leaky_relu, gpu_mul,
    gpu_pow, gpu_round, gpu_sigmoid, gpu_sin, gpu_sqrt, gpu_sub, gpu_tanh,
};
use crate::cuda::ops::{
    gpu_atan, gpu_equal, gpu_greater, gpu_greater_or_equal, gpu_less, gpu_less_or_equal,
    gpu_not_equal,
};
use crate::cuda::reduction::gpu_softmax;
use crate::cuda::{CudaError, GpuTensor};
use crate::ir::PlannedOp;

use super::{resolve_inputs, Executor};

impl Executor {
    /// Dispatch a single elementwise op to its `gpu_*` kernel wrapper.
    ///
    /// Matches on `op.node.op_type`. Each arm mirrors the corresponding
    /// arm in today's `execute_gpu_operator_with_precomputed`: same kernel
    /// calls, same broadcasting fallback, same attribute reads. Unknown
    /// op_types return a Kernel error — the caller (`dispatch_op`)
    /// guarantees only the elementwise category reaches here.
    pub(crate) fn dispatch_elementwise(
        &self,
        op: &PlannedOp,
        values: &HashMap<String, GpuTensor>,
        weights: &HashMap<String, GpuTensor>,
    ) -> Result<GpuTensor, CudaError> {
        let inputs = resolve_inputs(&op.node.inputs, values, weights)?;
        let attrs = &op.node.attributes;
        let mut pool = self.memory_pool.borrow_mut();
        match op.node.op_type.as_str() {
            // --- Binary arith with broadcasting -------------------------
            "Add" => {
                let (a, b) = (inputs[0], inputs[1]);
                if a.shape() == b.shape() || a.len() == 1 || b.len() == 1 {
                    return gpu_add(&self.ctx, &self.elementwise_kernels, &mut pool, a, b);
                }
                if let Some(broadcast_shape) = compute_broadcast_shape(a.shape(), b.shape()) {
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
                    gpu_add(
                        &self.ctx,
                        &self.elementwise_kernels,
                        &mut pool,
                        &a_exp,
                        &b_exp,
                    )
                } else {
                    Err(CudaError::Kernel(format!(
                        "Add: shapes {:?} and {:?} are not broadcastable",
                        a.shape(),
                        b.shape()
                    )))
                }
            }
            "Sub" => {
                let (a, b) = (inputs[0], inputs[1]);
                if a.shape() == b.shape() {
                    gpu_sub(&self.ctx, &self.elementwise_kernels, &mut pool, a, b)
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
                    gpu_sub(
                        &self.ctx,
                        &self.elementwise_kernels,
                        &mut pool,
                        &a_exp,
                        &b_exp,
                    )
                } else {
                    Err(CudaError::Kernel(format!(
                        "Sub: shapes {:?} and {:?} are not broadcastable",
                        a.shape(),
                        b.shape()
                    )))
                }
            }
            "Mul" => {
                let (a, b) = (inputs[0], inputs[1]);
                if a.shape() == b.shape() {
                    gpu_mul(&self.ctx, &self.elementwise_kernels, &mut pool, a, b)
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
                    gpu_mul(
                        &self.ctx,
                        &self.elementwise_kernels,
                        &mut pool,
                        &a_exp,
                        &b_exp,
                    )
                } else {
                    Err(CudaError::Kernel(format!(
                        "Mul: shapes {:?} and {:?} are not broadcastable",
                        a.shape(),
                        b.shape()
                    )))
                }
            }
            "Div" => {
                let (a, b) = (inputs[0], inputs[1]);
                if a.shape() == b.shape() {
                    gpu_div(&self.ctx, &self.elementwise_kernels, &mut pool, a, b)
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
                    gpu_div(
                        &self.ctx,
                        &self.elementwise_kernels,
                        &mut pool,
                        &a_exp,
                        &b_exp,
                    )
                } else {
                    Err(CudaError::Kernel(format!(
                        "Div: shapes {:?} and {:?} are not broadcastable",
                        a.shape(),
                        b.shape()
                    )))
                }
            }
            "Pow" => {
                let (base, exp) = (inputs[0], inputs[1]);
                if base.shape() == exp.shape() {
                    gpu_pow(&self.ctx, &self.elementwise_kernels, &mut pool, base, exp)
                } else if let Some(broadcast_shape) =
                    compute_broadcast_shape(base.shape(), exp.shape())
                {
                    let base_exp = maybe_expand(
                        &self.ctx,
                        &self.ops_kernels,
                        &mut pool,
                        base,
                        broadcast_shape.clone(),
                    )?;
                    let exp_exp = maybe_expand(
                        &self.ctx,
                        &self.ops_kernels,
                        &mut pool,
                        exp,
                        broadcast_shape,
                    )?;
                    gpu_pow(
                        &self.ctx,
                        &self.elementwise_kernels,
                        &mut pool,
                        &base_exp,
                        &exp_exp,
                    )
                } else {
                    Err(CudaError::Kernel(format!(
                        "Pow: shapes {:?} and {:?} are not broadcastable",
                        base.shape(),
                        exp.shape()
                    )))
                }
            }

            // --- Unary math ---------------------------------------------
            "Exp" => gpu_exp(&self.ctx, &self.elementwise_kernels, &mut pool, inputs[0]),
            "Sqrt" => {
                // Optional debug probe for negative inputs (preserved from
                // today's dispatch; useful when tracing NaN pipelines).
                if std::env::var("DEBUG_SQRT").is_ok() {
                    let data = inputs[0].to_host_f32(&self.ctx)?;
                    let min_val = data.iter().cloned().fold(f32::INFINITY, f32::min);
                    let neg_count = data.iter().filter(|&&x| x < 0.0).count();
                    if neg_count > 0 || min_val < 0.0 {
                        eprintln!(
                            "DEBUG Sqrt: input has {} negative values, min={:.6e}, shape={:?}",
                            neg_count,
                            min_val,
                            inputs[0].shape()
                        );
                    }
                }
                gpu_sqrt(&self.ctx, &self.elementwise_kernels, &mut pool, inputs[0])
            }
            "Sin" => gpu_sin(&self.ctx, &self.elementwise_kernels, &mut pool, inputs[0]),
            "Cos" => gpu_cos(&self.ctx, &self.elementwise_kernels, &mut pool, inputs[0]),
            "Atan" => gpu_atan(&self.ctx, &self.ops_kernels, inputs[0]),
            "Round" => gpu_round(&self.ctx, &self.elementwise_kernels, &mut pool, inputs[0]),
            "Floor" => gpu_floor(&self.ctx, &self.elementwise_kernels, &mut pool, inputs[0]),

            // --- Activations --------------------------------------------
            "Tanh" => gpu_tanh(&self.ctx, &self.elementwise_kernels, &mut pool, inputs[0]),
            "Erf" => gpu_erf(&self.ctx, &self.elementwise_kernels, &mut pool, inputs[0]),
            "Sigmoid" => gpu_sigmoid(&self.ctx, &self.elementwise_kernels, &mut pool, inputs[0]),
            "LeakyRelu" => {
                let alpha = attrs.get_float("alpha").unwrap_or(0.01);
                gpu_leaky_relu(
                    &self.ctx,
                    &self.elementwise_kernels,
                    &mut pool,
                    inputs[0],
                    alpha,
                )
            }

            // --- Clip + Softmax -----------------------------------------
            "Clip" => {
                let min = attrs.get_float("min").unwrap_or(f32::MIN);
                let max = attrs.get_float("max").unwrap_or(f32::MAX);
                gpu_clip(
                    &self.ctx,
                    &self.elementwise_kernels,
                    &mut pool,
                    inputs[0],
                    min,
                    max,
                )
            }
            "Softmax" => {
                // GPU softmax operates on last axis. Port faithfully: today's
                // dispatch ignores the `axis` attribute here.
                gpu_softmax(&self.ctx, &self.reduction_kernels, &mut pool, inputs[0])
            }

            // --- Comparisons with broadcasting --------------------------
            "Equal" => {
                let (a, b) = (inputs[0], inputs[1]);
                if a.shape() == b.shape() {
                    gpu_equal(&self.ctx, &self.ops_kernels, a, b)
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
                    gpu_equal(&self.ctx, &self.ops_kernels, &a_exp, &b_exp)
                } else {
                    Err(CudaError::Kernel(format!(
                        "Equal: shapes {:?} and {:?} not broadcastable",
                        a.shape(),
                        b.shape()
                    )))
                }
            }
            "Less" => {
                let (a, b) = (inputs[0], inputs[1]);
                if a.shape() == b.shape() {
                    gpu_less(&self.ctx, &self.ops_kernels, a, b)
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
                    gpu_less(&self.ctx, &self.ops_kernels, &a_exp, &b_exp)
                } else {
                    Err(CudaError::Kernel(format!(
                        "Less: shapes {:?} and {:?} not broadcastable",
                        a.shape(),
                        b.shape()
                    )))
                }
            }
            "Greater" => {
                let (a, b) = (inputs[0], inputs[1]);
                if a.shape() == b.shape() {
                    gpu_greater(&self.ctx, &self.ops_kernels, a, b)
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
                    gpu_greater(&self.ctx, &self.ops_kernels, &a_exp, &b_exp)
                } else {
                    Err(CudaError::Kernel(format!(
                        "Greater: shapes {:?} and {:?} not broadcastable",
                        a.shape(),
                        b.shape()
                    )))
                }
            }
            "GreaterOrEqual" => {
                let (a, b) = (inputs[0], inputs[1]);
                if a.shape() == b.shape() {
                    gpu_greater_or_equal(&self.ctx, &self.ops_kernels, a, b)
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
                    gpu_greater_or_equal(&self.ctx, &self.ops_kernels, &a_exp, &b_exp)
                } else {
                    Err(CudaError::Kernel(format!(
                        "GreaterOrEqual: shapes {:?} and {:?} not broadcastable",
                        a.shape(),
                        b.shape()
                    )))
                }
            }
            "LessOrEqual" => {
                let (a, b) = (inputs[0], inputs[1]);
                if a.shape() == b.shape() {
                    gpu_less_or_equal(&self.ctx, &self.ops_kernels, a, b)
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
                    gpu_less_or_equal(&self.ctx, &self.ops_kernels, &a_exp, &b_exp)
                } else {
                    Err(CudaError::Kernel(format!(
                        "LessOrEqual: shapes {:?} and {:?} not broadcastable",
                        a.shape(),
                        b.shape()
                    )))
                }
            }
            "NotEqual" => {
                let (a, b) = (inputs[0], inputs[1]);
                if a.shape() == b.shape() {
                    gpu_not_equal(&self.ctx, &self.ops_kernels, a, b)
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
                    gpu_not_equal(&self.ctx, &self.ops_kernels, &a_exp, &b_exp)
                } else {
                    Err(CudaError::Kernel(format!(
                        "NotEqual: shapes {:?} and {:?} not broadcastable",
                        a.shape(),
                        b.shape()
                    )))
                }
            }

            other => Err(CudaError::Kernel(format!(
                "dispatch_elementwise called with non-elementwise op '{}'",
                other
            ))),
        }
    }
}
