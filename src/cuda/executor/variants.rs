//! Instrumentation variants of [`Executor::run`].
//!
//! Each of these methods is a near-verbatim port of the corresponding
//! `run_with_*` method on today's `GpuGraphExecutor` (in
//! `src/cuda/inference/mod.rs` ~lines 3378-4847), adapted to drive the
//! new `ExecutionPlan`-based executor:
//!
//! - Iterate `plan.ops` instead of topo-sorting `self.nodes` each run.
//! - Read weights from `plan.weights` (kept off the `values` map).
//! - Dispatch through `self.dispatch_op(op, &values, plan)`, which handles
//!   `OpKind::FusedHead` + regular category dispatch in one call.
//! - `OpKind::Skip` is ignored (the corresponding fused head will fire on
//!   its own turn). Today's separate `nodes_to_skip` + `fused_patterns`
//!   machinery is collapsed into the plan's per-op `OpKind`.
//!
//! The four methods each wrap per-op dispatch with a different
//! instrumentation layer (validation, checkpoints, logging, profiling) but
//! all share the same memory-pool return-by-last-use logic ported in
//! [`super::Executor::return_dead_inputs`].

use std::collections::HashMap;

use crate::cuda::inference::{NodeExecutionLog, ProfileData};
use crate::cuda::{CudaError, GpuTensor};
use crate::ir::{ExecutionPlan, OpKind, PlannedOp};
use crate::tensor::Tensor;

use super::{collect_outputs, store_outputs, upload_tensor, Executor};

/// Derive the profiler/label string for a planned op.
///
/// Returns `"Fused_<VariantName>"` for `OpKind::FusedHead` and the raw
/// `node.op_type` for `OpKind::Regular`. `OpKind::Skip` never reaches
/// dispatch so callers don't need to handle it.
fn op_label(op: &PlannedOp) -> String {
    match &op.kind {
        OpKind::FusedHead(pattern) => {
            use crate::cuda::inference::fusion::FusedPattern::*;
            let suffix = match pattern {
                DivRsqrt { .. } => "DivRsqrt".to_string(),
                AddMulAdd { .. } => "AddMulAdd".to_string(),
                Gelu { .. } => "Gelu".to_string(),
                MulAdd { .. } => "MulAdd".to_string(),
                AddMul { .. } => "AddMul".to_string(),
                SubMul { .. } => "SubMul".to_string(),
                DivMul { .. } => "DivMul".to_string(),
                MulSinPowMulAdd { .. } => "MulSinPowMulAdd".to_string(),
                GeneralChain { chain_signature, .. } => {
                    format!("GeneralChain[{}]", chain_signature)
                }
            };
            format!("Fused_{}", suffix)
        }
        OpKind::Regular | OpKind::Skip => op.node.op_type.clone(),
    }
}

impl Executor {
    /// Run inference with tensor validation (NaN/Inf detection + stats).
    ///
    /// Port of today's `GpuGraphExecutor::run_with_validation` in
    /// `src/cuda/inference/mod.rs` (~line 3378). Stops on first error if
    /// `mode.stop_on_error` is true. Only available with the
    /// `debug-inference` feature.
    #[cfg(feature = "debug-inference")]
    pub fn run_with_validation(
        &self,
        plan: &ExecutionPlan,
        inputs: HashMap<String, Tensor>,
        output_names: &[&str],
        mode: crate::cuda::validation::ValidationMode,
    ) -> Result<
        (
            HashMap<String, Tensor>,
            crate::cuda::validation::ValidationReport,
        ),
        crate::cuda::validation::ValidationError,
    > {
        use crate::cuda::validation::{TensorValidator, ValidationError};

        let mut validator =
            TensorValidator::new(mode, &self.ctx).map_err(ValidationError::from)?;

        self.i64_cache.borrow_mut().clear();

        let mut values: HashMap<String, GpuTensor> = HashMap::new();
        let mut tensor_groups: HashMap<String, String> = HashMap::new();
        let mut group_remaining: HashMap<String, usize> = HashMap::new();

        for (name, tensor) in inputs {
            let gpu_tensor =
                upload_tensor(&self.ctx, &tensor).map_err(ValidationError::from)?;
            values.insert(name, gpu_tensor);
        }

        for (node_idx, op) in plan.ops.iter().enumerate() {
            if matches!(op.kind, OpKind::Skip) {
                continue;
            }

            if op.node.op_type == "Constant" {
                let value_tensor = op.node.attributes.resolve_constant_value().ok_or_else(|| {
                    ValidationError::Cuda(CudaError::Kernel(format!(
                        "Constant node '{}' missing all value-carrying attributes \
                         (value / value_int / value_ints / value_float / value_floats)",
                        op.node.name
                    )))
                })?;
                let gpu_tensor =
                    upload_tensor(&self.ctx, &value_tensor).map_err(ValidationError::from)?;
                if let Some(out_name) = op.node.outputs.first() {
                    values.insert(out_name.clone(), gpu_tensor);
                }
                continue;
            }

            let output = self
                .dispatch_op(op, &values, plan)
                .map_err(ValidationError::from)?;

            // Validate the primary output. For fused heads the "output
            // name" is the fused pattern's output_name (which is also
            // `op.node.outputs[0]` since the pattern head is the node
            // that produces the final output). For regular ops it's
            // simply `op.node.outputs[0]`.
            let validate_name = op
                .node
                .outputs
                .first()
                .cloned()
                .unwrap_or_default();
            let label = op_label(op);
            validator.validate(&validate_name, &label, &output, &self.ctx)?;

            store_outputs(
                op,
                vec![output],
                &mut values,
                &mut tensor_groups,
                &mut group_remaining,
                &plan.tensor_last_use,
                output_names,
            );

            self.return_dead_inputs(
                &op.node.inputs,
                node_idx,
                &plan.tensor_last_use,
                &mut values,
                &mut tensor_groups,
                &mut group_remaining,
            );
        }

        let outputs = collect_outputs(&values, output_names, &plan.weights, &self.ctx)
            .map_err(ValidationError::from)?;
        Ok((outputs, validator.report()))
    }

    /// Run inference with checkpoint capture and (optionally) ORT
    /// reference comparison.
    ///
    /// Port of today's `GpuGraphExecutor::run_with_checkpoints` in
    /// `src/cuda/inference/mod.rs` (~line 3689). Only available with the
    /// `debug-inference` feature.
    #[cfg(feature = "debug-inference")]
    pub fn run_with_checkpoints(
        &self,
        plan: &ExecutionPlan,
        inputs: HashMap<String, Tensor>,
        output_names: &[&str],
        config: crate::cuda::checkpoints::CheckpointConfig,
    ) -> Result<
        (
            HashMap<String, Tensor>,
            crate::cuda::checkpoints::CheckpointManager,
        ),
        CudaError,
    > {
        use crate::cuda::checkpoints::CheckpointManager;

        let mut checkpoint_mgr = CheckpointManager::new(config);
        if checkpoint_mgr.config.compare_to_ort {
            let _count = checkpoint_mgr.load_ort_references().map_err(|e| {
                CudaError::Kernel(format!("Failed to load ORT references: {}", e))
            })?;
        }

        self.i64_cache.borrow_mut().clear();

        let mut values: HashMap<String, GpuTensor> = HashMap::new();
        let mut tensor_groups: HashMap<String, String> = HashMap::new();
        let mut group_remaining: HashMap<String, usize> = HashMap::new();

        for (name, tensor) in inputs {
            let gpu_tensor = upload_tensor(&self.ctx, &tensor)?;
            values.insert(name, gpu_tensor);
        }

        for (node_idx, op) in plan.ops.iter().enumerate() {
            if matches!(op.kind, OpKind::Skip) {
                continue;
            }

            if op.node.op_type == "Constant" {
                let value_tensor = op.node.attributes.resolve_constant_value().ok_or_else(|| {
                    CudaError::Kernel(format!(
                        "Constant node '{}' missing all value-carrying attributes \
                         (value / value_int / value_ints / value_float / value_floats)",
                        op.node.name
                    ))
                })?;
                let gpu_tensor = upload_tensor(&self.ctx, &value_tensor)?;
                if let Some(out_name) = op.node.outputs.first() {
                    values.insert(out_name.clone(), gpu_tensor);
                }
                continue;
            }

            let output = self.dispatch_op(op, &values, plan)?;

            // Capture the checkpoint under the primary output name. For
            // fused heads this is the pattern's output_name; for regular
            // ops it's simply the node's first output.
            let capture_name = op
                .node
                .outputs
                .first()
                .cloned()
                .unwrap_or_default();
            let label = op_label(op);
            let _entry = checkpoint_mgr
                .capture(&capture_name, &label, &output, &self.ctx)
                .map_err(|e| CudaError::Kernel(format!("Checkpoint capture failed: {}", e)))?;

            store_outputs(
                op,
                vec![output],
                &mut values,
                &mut tensor_groups,
                &mut group_remaining,
                &plan.tensor_last_use,
                output_names,
            );

            self.return_dead_inputs(
                &op.node.inputs,
                node_idx,
                &plan.tensor_last_use,
                &mut values,
                &mut tensor_groups,
                &mut group_remaining,
            );
        }

        let outputs = collect_outputs(&values, output_names, &plan.weights, &self.ctx)?;
        Ok((outputs, checkpoint_mgr))
    }

    /// Execute with a per-node log of shapes, dtypes, and NaN presence.
    ///
    /// Port of today's `GpuGraphExecutor::run_with_logging` in
    /// `src/cuda/inference/mod.rs` (~line 4117). Used for comparing
    /// GPU-side shape traces against ORT traces to find divergences.
    ///
    /// Memory-pool returns are preserved (logging still runs full-length
    /// models); dtype checks and NaN detection mirror today's path.
    pub fn run_with_logging(
        &self,
        plan: &ExecutionPlan,
        inputs: HashMap<String, Tensor>,
        output_names: &[&str],
    ) -> Result<(HashMap<String, Tensor>, Vec<NodeExecutionLog>), CudaError> {
        use crate::cuda::tensor::DType;

        self.i64_cache.borrow_mut().clear();

        let mut execution_log: Vec<NodeExecutionLog> = Vec::new();
        let mut values: HashMap<String, GpuTensor> = HashMap::new();
        let mut tensor_groups: HashMap<String, String> = HashMap::new();
        let mut group_remaining: HashMap<String, usize> = HashMap::new();

        for (name, tensor) in inputs {
            let gpu_tensor = upload_tensor(&self.ctx, &tensor)?;
            execution_log.push(NodeExecutionLog {
                node_name: format!("INPUT:{}", name),
                op_type: "Input".to_string(),
                input_shapes: vec![],
                input_dtypes: vec![],
                output_shape: gpu_tensor.shape().to_vec(),
                output_dtype: gpu_tensor.dtype().name().to_string(),
                has_nan: false,
            });
            values.insert(name, gpu_tensor);
        }

        for (node_idx, op) in plan.ops.iter().enumerate() {
            if matches!(op.kind, OpKind::Skip) {
                continue;
            }

            if op.node.op_type == "Constant" {
                let value_tensor = op.node.attributes.resolve_constant_value().ok_or_else(|| {
                    CudaError::Kernel(format!(
                        "Constant node '{}' missing all value-carrying attributes \
                         (value / value_int / value_ints / value_float / value_floats)",
                        op.node.name
                    ))
                })?;
                let gpu_tensor = upload_tensor(&self.ctx, &value_tensor)?;
                if let Some(out_name) = op.node.outputs.first() {
                    values.insert(out_name.clone(), gpu_tensor);
                }
                continue;
            }

            // Record input shapes/dtypes BEFORE dispatch so the log reads
            // the tensors that dispatch will see.
            let mut input_shapes: Vec<Vec<usize>> = Vec::new();
            let mut input_dtypes: Vec<String> = Vec::new();
            for input_name in op.node.inputs.iter().filter(|n| !n.is_empty()) {
                if let Some(tensor) = values.get(input_name).or_else(|| plan.weights.get(input_name))
                {
                    input_shapes.push(tensor.shape().to_vec());
                    input_dtypes.push(tensor.dtype().name().to_string());
                }
            }

            let output = self.dispatch_op(op, &values, plan).map_err(|e| {
                CudaError::Kernel(format!(
                    "Node '{}' ({}): {} | Input shapes: {:?}",
                    op.node.name, op.node.op_type, e, input_shapes
                ))
            })?;

            // NaN check for Float32 outputs only. Today's path also
            // downloads the data for extensive per-op diagnostic
            // printing; that tracing is debug-only and intentionally
            // elided here to keep the ported executor focused on the
            // public logging contract.
            let has_nan = if output.dtype() == DType::Float32 {
                let data = output.to_host_f32(&self.ctx)?;
                data.iter().any(|x| x.is_nan())
            } else {
                false
            };

            execution_log.push(NodeExecutionLog {
                node_name: op.node.name.clone(),
                op_type: op_label(op),
                input_shapes,
                input_dtypes,
                output_shape: output.shape().to_vec(),
                output_dtype: output.dtype().name().to_string(),
                has_nan,
            });

            store_outputs(
                op,
                vec![output],
                &mut values,
                &mut tensor_groups,
                &mut group_remaining,
                &plan.tensor_last_use,
                output_names,
            );

            self.return_dead_inputs(
                &op.node.inputs,
                node_idx,
                &plan.tensor_last_use,
                &mut values,
                &mut tensor_groups,
                &mut group_remaining,
            );
        }

        let outputs = collect_outputs(&values, output_names, &plan.weights, &self.ctx)?;
        Ok((outputs, execution_log))
    }

    /// Execute with per-op CPU-dispatch timing and GPU-event timing.
    ///
    /// Port of today's `GpuGraphExecutor::run_with_profiling` in
    /// `src/cuda/inference/mod.rs` (~line 4532). CPU timing measures
    /// async enqueue cost via `Instant`; GPU timing measures kernel
    /// wall time via CUDA events (garboard's `GpuEventTimer`). Keys in
    /// both maps match: `"Add"`, `"Fused_DivRsqrt"`, etc.
    ///
    /// Note: today's profiling path in the old executor deliberately did
    /// not run memory-pool returns (the pool state is orthogonal to
    /// timing, and the old implementation skipped the reclamation loop).
    /// We preserve that behavior here so the profiler stays apples-to-
    /// apples with the old code — see the `_ = (&tensor_groups, ...)`
    /// suppression at the bottom of today's `run_with_profiling`.
    pub fn run_with_profiling(
        &self,
        plan: &ExecutionPlan,
        inputs: HashMap<String, Tensor>,
        output_names: &[&str],
    ) -> Result<(HashMap<String, Tensor>, ProfileData), CudaError> {
        use std::collections::BTreeMap;
        use std::time::Instant;

        use crate::cuda::inference::gpu_event_timer::GpuEventTimer;

        self.i64_cache.borrow_mut().clear();

        let mut profile = ProfileData::default();
        let total_start = Instant::now();

        let mut values: HashMap<String, GpuTensor> = HashMap::new();
        let mut tensor_groups: HashMap<String, String> = HashMap::new();
        let mut group_remaining: HashMap<String, usize> = HashMap::new();

        let upload_start = Instant::now();
        for (name, tensor) in inputs {
            let gpu_tensor = upload_tensor(&self.ctx, &tensor)?;
            values.insert(name, gpu_tensor);
        }
        profile.input_upload_us = upload_start.elapsed().as_micros() as u64;

        // Weights aren't copied into `values`; dispatch falls back to
        // `plan.weights`. Today's code zeroes this field for the same
        // reason.
        profile.weight_ref_us = 0;

        // Topo sort was already performed at lowering; the plan's op
        // sequence is frozen. Record 0 for parity with today's schema.
        profile.topo_sort_us = 0;

        let mut op_times: BTreeMap<String, (u64, usize)> = BTreeMap::new();
        let mut gpu_timer = GpuEventTimer::new(
            self.ctx.garboard_device(),
            self.ctx.garboard_stream(),
            plan.ops.len(),
        )?;

        let exec_start = Instant::now();
        for (node_idx, op) in plan.ops.iter().enumerate() {
            if matches!(op.kind, OpKind::Skip) {
                continue;
            }

            if op.node.op_type == "Constant" {
                let value_tensor = op.node.attributes.resolve_constant_value().ok_or_else(|| {
                    CudaError::Kernel(format!(
                        "Constant node '{}' missing all value-carrying attributes \
                         (value / value_int / value_ints / value_float / value_floats)",
                        op.node.name
                    ))
                })?;
                let gpu_tensor = upload_tensor(&self.ctx, &value_tensor)?;
                if let Some(out_name) = op.node.outputs.first() {
                    values.insert(out_name.clone(), gpu_tensor);
                }
                continue;
            }

            let label = op_label(op);
            let op_start = Instant::now();
            let output = self.dispatch_op(op, &values, plan)?;
            let op_us = op_start.elapsed().as_micros() as u64;

            let entry = op_times.entry(label.clone()).or_insert((0, 0));
            entry.0 += op_us;
            entry.1 += 1;
            gpu_timer.record(label)?;

            store_outputs(
                op,
                vec![output],
                &mut values,
                &mut tensor_groups,
                &mut group_remaining,
                &plan.tensor_last_use,
                output_names,
            );

            // Suppress unused warning for the pool-return bookkeeping
            // maps (profiling path intentionally skips returns — see the
            // doc comment on this method).
            let _ = node_idx;
        }
        let _ = (&tensor_groups, &group_remaining);
        profile.node_execution_us = exec_start.elapsed().as_micros() as u64;
        profile.op_times = op_times;
        profile.gpu_op_times = gpu_timer.finalize()?;

        let download_start = Instant::now();
        let outputs = collect_outputs(&values, output_names, &plan.weights, &self.ctx)?;
        profile.output_download_us = download_start.elapsed().as_micros() as u64;
        profile.total_us = total_start.elapsed().as_micros() as u64;

        Ok((outputs, profile))
    }
}
