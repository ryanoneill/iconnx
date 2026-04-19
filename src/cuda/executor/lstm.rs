//! LSTM dispatch.
//!
//! Takes the full [`ExecutionPlan`] because LSTM dispatch consumes both
//! `plan.lstm_weights` (the packed cuDNN weight cache populated eagerly
//! at lowering) and `plan.lstm_plan_cache` (the lazily populated
//! `RnnPlan` cache, keyed by `(W_ptr, R_ptr, seq_length)`).
//!
//! Dispatch target parity with today's `execute_gpu_operator_with_precomputed`
//! in `src/cuda/inference/mod.rs` (~line 1702): same cuDNN wrappers
//! (`prepare_lstm_plan`, `lstm_forward`), same plan-cache keying, same
//! optional-input slot handling. The only difference is that the packed
//! weights are looked up (not built) — lowering already packed them.
//!
//! All LSTMs in Kokoro use `batch_size=1`; plans are keyed on
//! `(W_ptr, R_ptr, seq_length)`. See `LstmPlanCache` docstring for the
//! invariant.

use std::collections::hash_map::Entry;
use std::collections::HashMap;

use crate::cuda::cudnn::{lstm_forward, prepare_lstm_plan};
use crate::cuda::{CudaError, GpuTensor};
use crate::ir::{ExecutionPlan, PlannedOp};

use super::{resolve_inputs, Executor};

impl Executor {
    /// Dispatch an LSTM op using the plan's packed weight cache and its
    /// lazy per-seq-length `RnnPlan` cache.
    pub(crate) fn dispatch_lstm(
        &self,
        op: &PlannedOp,
        values: &HashMap<String, GpuTensor>,
        plan: &ExecutionPlan,
    ) -> Result<GpuTensor, CudaError> {
        let weights = &plan.weights;
        let inputs = resolve_inputs(&op.node.inputs, values, weights)?;

        // ONNX LSTM input order: X, W, R, B?, sequence_lens?, initial_h?, initial_c?, P?
        // `resolve_inputs` drops empty names, so the positional indexing
        // below must go through the raw `op.node.inputs` for optionals.
        let x = inputs[0];
        let w = inputs[1];
        let r = inputs[2];

        // Optional inputs: fetch by ONNX slot number from the raw input
        // list, treating empty strings as "absent".
        let lookup_optional = |idx: usize| -> Option<&GpuTensor> {
            op.node.inputs.get(idx).and_then(|n| {
                if n.is_empty() {
                    None
                } else {
                    values.get(n).or_else(|| weights.get(n))
                }
            })
        };
        let h_init = lookup_optional(5);
        let c_init = lookup_optional(6);

        let w_shape = w.shape();
        let num_directions = w_shape[0];
        let hidden_size = w_shape[1] / 4;

        // Derive seq_length from the input tensor: X is either 3D
        // [seq_len, batch, input_size] or 2D [seq_len, input_size].
        let x_shape = x.shape();
        let seq_length = x_shape[0];

        // Lookup eagerly-packed weights by (w_ptr, r_ptr). Lowering's
        // `pack_lstm_weights_eagerly` populated `plan.lstm_weights` using
        // the same key derivation.
        let w_ptr = w.device_ptr(&self.ctx) as usize;
        let r_ptr = r.device_ptr(&self.ctx) as usize;
        let weight_key = (w_ptr, r_ptr);
        let packed = plan.lstm_weights.get(&weight_key).ok_or_else(|| {
            CudaError::Cudnn(format!(
                "LSTM '{}' weights not packed at lowering (key not found)",
                op.node.name
            ))
        })?;

        // Lazy RnnPlan cache keyed by (w_ptr, r_ptr, seq_length). The
        // borrow_mut must end before we take a shared borrow below —
        // scope the write explicitly.
        let plan_key = (w_ptr, r_ptr, seq_length);
        {
            let mut plan_cache = plan.lstm_plan_cache.borrow_mut();
            if let Entry::Vacant(e) = plan_cache.entry(plan_key) {
                // batch_size is always 1 for Kokoro's LSTMs.
                let rnn_plan = prepare_lstm_plan(&self.ctx, packed, 1, seq_length)?;
                e.insert(rnn_plan);
            }
        }

        // Shared borrow for the forward call.
        let plan_cache = plan.lstm_plan_cache.borrow();
        let rnn_plan = plan_cache.get(&plan_key).expect("just inserted above");

        lstm_forward(
            &self.ctx,
            packed,
            rnn_plan,
            x,
            h_init,
            c_init,
            hidden_size,
            num_directions,
        )
    }
}
