//! MatMul / Gemm dispatch.
//!
//! Routes to `gpu_matmul`, `gpu_batched_matmul`, `gpu_matmul_2d_3d`, and
//! `gpu_gemm` in `crate::cuda::matmul`, all of which use `ctx.blas()`.
//! MatMul picks its backing kernel based on input ranks: 2x2 → matmul,
//! 3x3 → batched matmul, 3x2 → reshape-then-2D-matmul-then-reshape, 2x3
//! → matmul_2d_3d, 4x4 → reshape-then-3D-batched-matmul-then-reshape.
//!
//! Dispatch target parity with today's `execute_gpu_operator_with_precomputed`
//! in `src/cuda/inference/mod.rs`: same kernel calls, same attribute
//! defaults (`alpha=1.0`, `beta=0.0`, `transA=0`, `transB=0`), same rank
//! dispatch table. Kernel wrappers stay in their source modules; this file
//! owns only the dispatch match.

use std::collections::HashMap;

use crate::cuda::matmul::{gpu_batched_matmul, gpu_gemm, gpu_matmul, gpu_matmul_2d_3d};
use crate::cuda::{CudaError, GpuTensor};
use crate::ir::PlannedOp;

use super::{resolve_inputs, Executor};

impl Executor {
    /// Dispatch a MatMul or Gemm op to its backing cuBLAS wrapper.
    pub(crate) fn dispatch_matmul(
        &self,
        op: &PlannedOp,
        values: &HashMap<String, GpuTensor>,
        weights: &HashMap<String, GpuTensor>,
    ) -> Result<GpuTensor, CudaError> {
        let inputs = resolve_inputs(&op.node.inputs, values, weights)?;
        let attributes = &op.node.attributes;
        let mut pool = self.memory_pool.borrow_mut();

        match op.node.op_type.as_str() {
            "MatMul" => {
                let a_ndim = inputs[0].ndim();
                let b_ndim = inputs[1].ndim();
                match (a_ndim, b_ndim) {
                    (2, 2) => gpu_matmul(&self.ctx, &mut pool, inputs[0], inputs[1]),
                    (3, 3) => gpu_batched_matmul(&self.ctx, &mut pool, inputs[0], inputs[1]),
                    (3, 2) => {
                        // 3D x 2D: A[batch, M, K] @ B[K, N] = C[batch, M, N].
                        // Reshape A to 2D, matmul, reshape back.
                        let a_shape = inputs[0].shape();
                        let b_shape = inputs[1].shape();
                        let batch = a_shape[0];
                        let m = a_shape[1];
                        let k = a_shape[2];
                        let n = b_shape[1];

                        let a_reshaped =
                            inputs[0].clone().reshape(vec![batch * m, k]).ok_or_else(|| {
                                CudaError::Kernel("MatMul: failed to reshape A".into())
                            })?;

                        let result_2d =
                            gpu_matmul(&self.ctx, &mut pool, &a_reshaped, inputs[1])?;

                        result_2d.reshape(vec![batch, m, n]).ok_or_else(|| {
                            CudaError::Kernel("MatMul: failed to reshape result".into())
                        })
                    }
                    (2, 3) => {
                        // 2D x 3D: A[M, K] @ B[batch, K, N] = C[batch, M, N].
                        // Broadcasts A across the batch dimension.
                        gpu_matmul_2d_3d(&self.ctx, &mut pool, inputs[0], inputs[1])
                    }
                    (4, 4) => {
                        // 4D x 4D: [batch, heads, M, K] @ [batch, heads, K, N].
                        // Reshape to 3D, batched matmul, reshape back.
                        let a_shape = inputs[0].shape();
                        let b_shape = inputs[1].shape();
                        let batch = a_shape[0];
                        let heads = a_shape[1];
                        let m = a_shape[2];
                        let k = a_shape[3];
                        let n = b_shape[3];

                        let a_reshaped = inputs[0]
                            .clone()
                            .reshape(vec![batch * heads, m, k])
                            .ok_or_else(|| {
                                CudaError::Kernel("MatMul 4D: failed to reshape A".into())
                            })?;

                        let b_reshaped = inputs[1]
                            .clone()
                            .reshape(vec![batch * heads, k, n])
                            .ok_or_else(|| {
                                CudaError::Kernel("MatMul 4D: failed to reshape B".into())
                            })?;

                        let result_3d =
                            gpu_batched_matmul(&self.ctx, &mut pool, &a_reshaped, &b_reshaped)?;

                        result_3d.reshape(vec![batch, heads, m, n]).ok_or_else(|| {
                            CudaError::Kernel("MatMul 4D: failed to reshape result".into())
                        })
                    }
                    _ => Err(CudaError::Kernel(format!(
                        "MatMul: unsupported dimensions A={}D, B={}D",
                        a_ndim, b_ndim
                    ))),
                }
            }

            "Gemm" => {
                let alpha = attributes.get_float("alpha").unwrap_or(1.0);
                let beta = attributes.get_float("beta").unwrap_or(0.0);
                let trans_a = attributes.get_int("transA").unwrap_or(0) != 0;
                let trans_b = attributes.get_int("transB").unwrap_or(0) != 0;
                let c = if inputs.len() > 2 { Some(inputs[2]) } else { None };
                gpu_gemm(
                    &self.ctx,
                    &mut pool,
                    inputs[0],
                    inputs[1],
                    c,
                    alpha,
                    beta,
                    trans_a,
                    trans_b,
                )
            }

            other => Err(CudaError::Kernel(format!(
                "dispatch_matmul called with non-matmul op '{}'",
                other
            ))),
        }
    }
}
