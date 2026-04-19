//! Fused-pattern dispatch.
//!
//! Entered from `dispatch_op` when `op.kind == OpKind::FusedHead(pattern)`.
//! The fused pattern itself carries every tensor name needed to launch
//! the kernel — the lowering pass already resolved which non-head nodes
//! to skip (they're marked `OpKind::Skip` in-place on the `ExecutionPlan`).
//!
//! Each branch mirrors the corresponding arm of today's
//! `execute_fused_pattern` in `src/cuda/inference/mod.rs`: same
//! value/weight fallback lookup, same scalar-extraction behavior for
//! `eps` (DivRsqrt) and `p` (MulSinPowMulAdd), same `gpu_fused_*`
//! wrapper calls. Kernel wrappers stay where they live in
//! `cuda::kernels::fused`.

use std::collections::HashMap;

use crate::cuda::inference::fusion::FusedPattern;
use crate::cuda::kernels::fused::{
    gpu_fused_add_mul, gpu_fused_add_mul_add, gpu_fused_div_mul, gpu_fused_div_rsqrt,
    gpu_fused_gelu, gpu_fused_mul_add, gpu_fused_mul_sin_pow_mul_add, gpu_fused_sub_mul,
};
use crate::cuda::{CudaError, GpuTensor};
use crate::ir::{OpKind, PlannedOp};

use super::Executor;

impl Executor {
    /// Dispatch a `FusedHead` op to its fused kernel.
    ///
    /// The `OpKind::FusedHead(FusedPattern)` variant carries every input
    /// and output name needed; `op.node` itself is the head of the fused
    /// chain (its output name is the pattern's `output_name`) but the
    /// dispatch only consults `op.kind` for the fused pattern data.
    ///
    /// Inputs are resolved with the same "values first, then weights"
    /// fallback used by today's `execute_fused_pattern`. Each pattern
    /// branch returns a `Kernel` error with a descriptive message when a
    /// named input isn't found — the detection pass is supposed to
    /// guarantee all names resolve, so this only fires if a pattern was
    /// mis-lowered.
    pub(crate) fn dispatch_fused(
        &self,
        op: &PlannedOp,
        values: &HashMap<String, GpuTensor>,
        weights: &HashMap<String, GpuTensor>,
    ) -> Result<GpuTensor, CudaError> {
        let pattern = match &op.kind {
            OpKind::FusedHead(p) => p,
            _ => {
                return Err(CudaError::Kernel(format!(
                    "dispatch_fused called on non-FusedHead op '{}'",
                    op.node.name
                )))
            }
        };

        let mut pool = self.memory_pool.borrow_mut();
        match pattern {
            FusedPattern::DivRsqrt {
                numerator_input,
                variance_input,
                eps_input,
                ..
            } => {
                let numerator = values
                    .get(numerator_input)
                    .or_else(|| weights.get(numerator_input))
                    .ok_or_else(|| {
                        CudaError::Kernel(format!(
                            "Fused DivRsqrt: numerator '{}' not found",
                            numerator_input
                        ))
                    })?;
                let variance = values
                    .get(variance_input)
                    .or_else(|| weights.get(variance_input))
                    .ok_or_else(|| {
                        CudaError::Kernel(format!(
                            "Fused DivRsqrt: variance '{}' not found",
                            variance_input
                        ))
                    })?;
                let eps_tensor = values
                    .get(eps_input)
                    .or_else(|| weights.get(eps_input))
                    .ok_or_else(|| {
                        CudaError::Kernel(format!(
                            "Fused DivRsqrt: eps '{}' not found",
                            eps_input
                        ))
                    })?;

                // eps is typically a 1-element constant tensor.
                let eps = if eps_tensor.len() == 1 {
                    eps_tensor.to_host_f32(&self.ctx)?[0]
                } else {
                    1e-5
                };

                gpu_fused_div_rsqrt(
                    &self.ctx,
                    &self.fused_kernels,
                    &mut pool,
                    numerator,
                    variance,
                    eps,
                )
            }
            FusedPattern::AddMulAdd {
                x_input,
                a_input,
                b_input,
                c_input,
                ..
            } => {
                let x = values.get(x_input).or_else(|| weights.get(x_input)).ok_or_else(|| {
                    CudaError::Kernel(format!("Fused AddMulAdd: x '{}' not found", x_input))
                })?;
                let a = values.get(a_input).or_else(|| weights.get(a_input)).ok_or_else(|| {
                    CudaError::Kernel(format!("Fused AddMulAdd: a '{}' not found", a_input))
                })?;
                let b = values.get(b_input).or_else(|| weights.get(b_input)).ok_or_else(|| {
                    CudaError::Kernel(format!("Fused AddMulAdd: b '{}' not found", b_input))
                })?;
                let c = values.get(c_input).or_else(|| weights.get(c_input)).ok_or_else(|| {
                    CudaError::Kernel(format!("Fused AddMulAdd: c '{}' not found", c_input))
                })?;

                gpu_fused_add_mul_add(&self.ctx, &self.fused_kernels, &mut pool, x, a, b, c)
            }
            FusedPattern::Gelu { x_input, .. } => {
                let x = values.get(x_input).or_else(|| weights.get(x_input)).ok_or_else(|| {
                    CudaError::Kernel(format!("Fused Gelu: x '{}' not found", x_input))
                })?;

                gpu_fused_gelu(&self.ctx, &self.fused_kernels, &mut pool, x)
            }
            FusedPattern::MulAdd {
                a_input,
                b_input,
                c_input,
                ..
            } => {
                let a = values.get(a_input).or_else(|| weights.get(a_input)).ok_or_else(|| {
                    CudaError::Kernel(format!("Fused MulAdd: a '{}' not found", a_input))
                })?;
                let b = values.get(b_input).or_else(|| weights.get(b_input)).ok_or_else(|| {
                    CudaError::Kernel(format!("Fused MulAdd: b '{}' not found", b_input))
                })?;
                let c = values.get(c_input).or_else(|| weights.get(c_input)).ok_or_else(|| {
                    CudaError::Kernel(format!("Fused MulAdd: c '{}' not found", c_input))
                })?;

                gpu_fused_mul_add(&self.ctx, &self.fused_kernels, &mut pool, a, b, c)
            }
            FusedPattern::AddMul {
                a_input,
                b_input,
                c_input,
                ..
            } => {
                let a = values.get(a_input).or_else(|| weights.get(a_input)).ok_or_else(|| {
                    CudaError::Kernel(format!("Fused AddMul: a '{}' not found", a_input))
                })?;
                let b = values.get(b_input).or_else(|| weights.get(b_input)).ok_or_else(|| {
                    CudaError::Kernel(format!("Fused AddMul: b '{}' not found", b_input))
                })?;
                let c = values.get(c_input).or_else(|| weights.get(c_input)).ok_or_else(|| {
                    CudaError::Kernel(format!("Fused AddMul: c '{}' not found", c_input))
                })?;

                gpu_fused_add_mul(&self.ctx, &self.fused_kernels, &mut pool, a, b, c)
            }
            FusedPattern::SubMul {
                a_input,
                b_input,
                c_input,
                ..
            } => {
                let a = values.get(a_input).or_else(|| weights.get(a_input)).ok_or_else(|| {
                    CudaError::Kernel(format!("Fused SubMul: a '{}' not found", a_input))
                })?;
                let b = values.get(b_input).or_else(|| weights.get(b_input)).ok_or_else(|| {
                    CudaError::Kernel(format!("Fused SubMul: b '{}' not found", b_input))
                })?;
                let c = values.get(c_input).or_else(|| weights.get(c_input)).ok_or_else(|| {
                    CudaError::Kernel(format!("Fused SubMul: c '{}' not found", c_input))
                })?;

                gpu_fused_sub_mul(&self.ctx, &self.fused_kernels, &mut pool, a, b, c)
            }
            FusedPattern::DivMul {
                a_input,
                b_input,
                c_input,
                ..
            } => {
                let a = values.get(a_input).or_else(|| weights.get(a_input)).ok_or_else(|| {
                    CudaError::Kernel(format!("Fused DivMul: a '{}' not found", a_input))
                })?;
                let b = values.get(b_input).or_else(|| weights.get(b_input)).ok_or_else(|| {
                    CudaError::Kernel(format!("Fused DivMul: b '{}' not found", b_input))
                })?;
                let c = values.get(c_input).or_else(|| weights.get(c_input)).ok_or_else(|| {
                    CudaError::Kernel(format!("Fused DivMul: c '{}' not found", c_input))
                })?;

                gpu_fused_div_mul(&self.ctx, &self.fused_kernels, &mut pool, a, b, c)
            }
            FusedPattern::MulSinPowMulAdd {
                x_input,
                w0_input,
                w1_input,
                b_input,
                p_input,
                ..
            } => {
                let x = values.get(x_input).or_else(|| weights.get(x_input)).ok_or_else(|| {
                    CudaError::Kernel(format!(
                        "Fused MulSinPowMulAdd: x '{}' not found",
                        x_input
                    ))
                })?;
                let w0 = values.get(w0_input).or_else(|| weights.get(w0_input)).ok_or_else(
                    || {
                        CudaError::Kernel(format!(
                            "Fused MulSinPowMulAdd: w0 '{}' not found",
                            w0_input
                        ))
                    },
                )?;
                let w1 = values.get(w1_input).or_else(|| weights.get(w1_input)).ok_or_else(
                    || {
                        CudaError::Kernel(format!(
                            "Fused MulSinPowMulAdd: w1 '{}' not found",
                            w1_input
                        ))
                    },
                )?;
                let b = values.get(b_input).or_else(|| weights.get(b_input)).ok_or_else(|| {
                    CudaError::Kernel(format!(
                        "Fused MulSinPowMulAdd: b '{}' not found",
                        b_input
                    ))
                })?;
                let p_tensor = values
                    .get(p_input)
                    .or_else(|| weights.get(p_input))
                    .ok_or_else(|| {
                        CudaError::Kernel(format!(
                            "Fused MulSinPowMulAdd: p '{}' not found",
                            p_input
                        ))
                    })?;
                // p is a scalar exponent — fetch its single value.
                let p = p_tensor.to_host_f32(&self.ctx)?[0];

                gpu_fused_mul_sin_pow_mul_add(
                    &self.ctx,
                    &self.fused_kernels,
                    &mut pool,
                    x,
                    w0,
                    w1,
                    b,
                    p,
                )
            }
        }
    }
}
