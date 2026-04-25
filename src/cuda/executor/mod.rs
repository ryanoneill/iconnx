//! GPU runtime that dispatches an `ExecutionPlan` against hardware.
//!
//! `Executor` owns model-independent state: device, stream, cuBLAS, cuDNN,
//! kernel caches, memory pool, per-run Int64 cache. Running a plan is a
//! single pass over `plan.ops`, dispatching each op to its category handler
//! via the compact match in `run()`.
//!
//! See Phase 2 plan: `docs/superpowers/plans/2026-04-17-phase2-graph-ir.md`.

mod conv;
mod elementwise;
mod fused;
mod layout;
mod lstm;
mod matmul;
mod misc;
mod reduction;
mod stft;
mod variants;

use std::cell::RefCell;
use std::collections::HashMap;

use crate::cuda::conv::ConvKernelCache;
use crate::cuda::cufft::StftKernelCache;
use crate::cuda::kernels::fused::FusedKernelCache;
use crate::cuda::kernels::KernelCache;
use crate::cuda::lstm::LstmKernelCache;
use crate::cuda::memory_pool::GpuMemoryPool;
use crate::cuda::ops::OpsKernelCache;
use crate::cuda::reduction::ReductionKernelCache;
use crate::cuda::{CudaError, GpuTensor, IconnxCudaContext};
use crate::ir::{ExecutionPlan, OpKind, PlannedOp};
use crate::tensor::Tensor;

/// GPU runtime that dispatches a frozen [`ExecutionPlan`] against hardware.
///
/// Model-independent: all model-specific state (weights, ops, caches keyed
/// by the model's weight pointers) lives on the `ExecutionPlan`. The
/// `Executor` owns only the device context and kernel caches. One
/// `Executor` may run multiple `ExecutionPlan`s for different models.
pub struct Executor {
    pub ctx: IconnxCudaContext,
    pub elementwise_kernels: KernelCache,
    pub reduction_kernels: ReductionKernelCache,
    pub conv_kernels: ConvKernelCache,
    pub lstm_kernels: LstmKernelCache,
    pub ops_kernels: OpsKernelCache,
    pub stft_kernels: StftKernelCache,
    pub fused_kernels: FusedKernelCache,
    pub memory_pool: RefCell<GpuMemoryPool>,
    /// Per-run Int64 cache keyed by GPU device pointer. Cleared at the
    /// start of every `run()` because the memory pool reuses addresses
    /// across runs.
    pub i64_cache: RefCell<HashMap<u64, Vec<i64>>>,
    #[cfg(feature = "debug-inference")]
    pub validator: RefCell<Option<crate::cuda::validation::TensorValidator>>,
}

impl Executor {
    /// Construct a fresh executor: CUDA context + all kernel caches.
    pub fn new() -> Result<Self, CudaError> {
        let ctx = IconnxCudaContext::new()?;
        let elementwise_kernels = KernelCache::new(&ctx)?;
        let reduction_kernels = ReductionKernelCache::new(&ctx)?;
        let conv_kernels = ConvKernelCache::new(&ctx)?;
        let lstm_kernels = LstmKernelCache::new(&ctx)?;
        let ops_kernels = OpsKernelCache::new(&ctx)?;
        let stft_kernels = StftKernelCache::new(&ctx)?;
        let fused_kernels = FusedKernelCache::new(&ctx)?;
        Ok(Self {
            ctx,
            elementwise_kernels,
            reduction_kernels,
            conv_kernels,
            lstm_kernels,
            ops_kernels,
            stft_kernels,
            fused_kernels,
            memory_pool: RefCell::new(GpuMemoryPool::new()),
            i64_cache: RefCell::new(HashMap::new()),
            #[cfg(feature = "debug-inference")]
            validator: RefCell::new(None),
        })
    }

    /// Dispatch a whole plan's op sequence and return the requested outputs
    /// as CPU tensors.
    ///
    /// High-level structure:
    /// 1. Clear the per-run i64 cache (memory pool reuses pointers across
    ///    runs, so cached entries from prior runs are stale).
    /// 2. Upload CPU input tensors to the GPU. Weights are NOT copied into
    ///    `values`; dispatchers look up in `values` first, then
    ///    `plan.weights`.
    /// 3. Walk `plan.ops` in topo order. For each op: skip `OpKind::Skip`,
    ///    handle `Constant` inline (upload the attribute tensor), otherwise
    ///    route through `dispatch_op` and store the result.
    /// 4. After each op, return any tensor whose last use was this op back
    ///    to the memory pool. Multi-output nodes are tracked as a group so
    ///    the tensor is only returned when ALL group members are freed.
    /// 5. Download the requested outputs.
    ///
    /// Port of today's `GpuGraphExecutor::run` (in `src/cuda/inference/mod.rs`
    /// around lines 2866-3333). The `tensor_last_use` map is pre-computed by
    /// `ir::lowering` and lives on the plan.
    pub fn run(
        &self,
        plan: &ExecutionPlan,
        inputs: HashMap<String, Tensor>,
        output_names: &[&str],
    ) -> Result<HashMap<String, Tensor>, CudaError> {
        self.i64_cache.borrow_mut().clear();

        let mut values: HashMap<String, GpuTensor> = HashMap::new();

        // Track tensor groups for multi-output nodes.
        // When a node has multiple outputs, they share the same underlying
        // tensor (Arc clones). We group them together and only return to
        // the pool when ALL members are freed.
        // Maps each output name -> group ID (primary name).
        let mut tensor_groups: HashMap<String, String> = HashMap::new();
        // Maps group ID (primary) -> count of remaining references.
        let mut group_remaining: HashMap<String, usize> = HashMap::new();

        // Upload input tensors to GPU (preserving dtype).
        for (name, tensor) in inputs {
            let gpu_tensor = upload_tensor(&self.ctx, &tensor)?;
            values.insert(name, gpu_tensor);
        }

        // Weights are NOT copied into `values`; dispatchers fall back to
        // `plan.weights` on miss.

        for (node_idx, op) in plan.ops.iter().enumerate() {
            if matches!(op.kind, OpKind::Skip) {
                continue;
            }

            // Constant nodes don't dispatch to a kernel — we just upload
            // the attribute tensor into `values` under the node's output
            // name. Port of today's handling in
            // `cuda::inference::mod.rs::run`, ~line 2998.
            if op.node.op_type == "Constant" {
                let value_tensor = op.node.attributes.resolve_constant_value().ok_or_else(|| {
                    CudaError::Kernel(format!(
                        "Constant node '{}' missing all value-carrying attributes \
                         (value / value_int / value_ints / value_float / value_floats)",
                        op.node.name
                    ))
                })?;
                let gpu_tensor = upload_tensor(&self.ctx, &value_tensor)?;
                // Constants always have exactly one output in valid ONNX.
                if let Some(out_name) = op.node.outputs.first() {
                    values.insert(out_name.clone(), gpu_tensor);
                }
                continue;
            }

            // Wrap dispatch errors with the failing op's type and first
            // output name so the raw accessor message — e.g. `"data_f32
            // called on int64 tensor"` — becomes diagnosable: the message
            // reads `"Transpose (output=sqrted): data_f32 called on int64
            // tensor"`. Captures every error from every downstream op
            // wrapper (~143 `data_f32()` sites across 14 files) at a
            // single choke point, which scales where per-site
            // instrumentation doesn't. Only rewrites `CudaError::Kernel`
            // variants so other error kinds pass through unchanged.
            let dispatch_diag = |e: CudaError| match e {
                CudaError::Kernel(msg) => {
                    let out_name = op.node.outputs.first().cloned().unwrap_or_default();
                    CudaError::Kernel(format!(
                        "{} (output={}): {}",
                        op.node.op_type, out_name, msg
                    ))
                }
                other => other,
            };

            // Multi-output-aware dispatch. `dispatch_op_multi` handles the
            // Split → Vec<GpuTensor> routing and falls through to the
            // single-output path for everyone else. Same helper is used
            // by every variant in `variants.rs`.
            let outputs_vec = self
                .dispatch_op_multi(op, &values, plan)
                .map_err(dispatch_diag)?;
            store_outputs(
                op,
                outputs_vec,
                &mut values,
                &mut tensor_groups,
                &mut group_remaining,
                &plan.tensor_last_use,
                output_names,
            );

            // Memory management: return tensors to the pool when no longer
            // needed. Mirrors today's logic at
            // `src/cuda/inference/mod.rs::run` (~line 3279).
            self.return_dead_inputs(
                &op.node.inputs,
                node_idx,
                &plan.tensor_last_use,
                &mut values,
                &mut tensor_groups,
                &mut group_remaining,
            );
        }

        collect_outputs(&values, output_names, &plan.weights, &self.ctx)
    }

    /// Return any input tensor whose last-use index equals `node_idx` to
    /// the memory pool. Multi-output group tensors are only returned when
    /// all group members have been freed. Mirrors today's post-op loop in
    /// `src/cuda/inference/mod.rs::run` (~line 3279).
    pub(super) fn return_dead_inputs(
        &self,
        inputs: &[String],
        node_idx: usize,
        tensor_last_use: &HashMap<String, usize>,
        values: &mut HashMap<String, GpuTensor>,
        tensor_groups: &mut HashMap<String, String>,
        group_remaining: &mut HashMap<String, usize>,
    ) {
        for input_name in inputs {
            if input_name.is_empty() {
                continue;
            }
            let Some(&last_use_idx) = tensor_last_use.get(input_name) else {
                continue;
            };
            if last_use_idx != node_idx {
                continue;
            }
            let Some(tensor) = values.remove(input_name) else {
                continue;
            };
            if let Some(group) = tensor_groups.remove(input_name) {
                if let Some(count) = group_remaining.get_mut(&group) {
                    *count -= 1;
                    if *count == 0 {
                        group_remaining.remove(&group);
                        self.memory_pool.borrow_mut().return_tensor(tensor);
                    }
                    // Otherwise drop: Arc refcount decreases but still > 1.
                }
            } else {
                self.memory_pool.borrow_mut().return_tensor(tensor);
            }
        }
    }

    /// Multi-output-aware dispatch wrapper. Returns `Vec<GpuTensor>`
    /// with length 1 for single-output ops and length N for ops with
    /// genuinely distinct multi-output tensors (Split today; future
    /// TopK / NonMaxSuppression / beam-search ops can extend the
    /// match). Single source of truth so all execution paths (the
    /// main `Executor::run`, plus the 4 instrumented variants in
    /// `variants.rs`) route Split identically — earlier the Split
    /// arm was inlined only into the main loop, so profiled / logged
    /// / checkpointed / validated runs fell through to
    /// `dispatch_layout` and rejected with `"called with non-layout
    /// op 'Split'"`.
    pub(super) fn dispatch_op_multi(
        &self,
        op: &PlannedOp,
        values: &HashMap<String, GpuTensor>,
        plan: &ExecutionPlan,
    ) -> Result<Vec<GpuTensor>, CudaError> {
        match op.node.op_type.as_str() {
            "Split" => self.dispatch_split(op, values, plan),
            _ => Ok(vec![self.dispatch_op(op, values, plan)?]),
        }
    }

    /// Compact category dispatch. Matches `op.node.op_type` against each
    /// category's known op list, with `FusedHead` intercepted first and
    /// every unrecognized op_type falling through to layout.
    pub(super) fn dispatch_op(
        &self,
        op: &PlannedOp,
        values: &HashMap<String, GpuTensor>,
        plan: &ExecutionPlan,
    ) -> Result<GpuTensor, CudaError> {
        if matches!(op.kind, OpKind::FusedHead(_)) {
            return self.dispatch_fused(op, values, &plan.weights);
        }
        match op.node.op_type.as_str() {
            // Elementwise arith / unary math / comparisons / activations.
            "Add" | "Sub" | "Mul" | "Div" | "Pow" | "Exp" | "Erf" | "Sqrt" | "Sin" | "Cos"
            | "Tanh" | "Sigmoid" | "Relu" | "LeakyRelu" | "Atan" | "Round" | "Floor" | "Clip"
            | "Softmax" | "Equal" | "Less" | "Greater" | "GreaterOrEqual" | "LessOrEqual"
            | "NotEqual" => {
                self.dispatch_elementwise(op, values, &plan.weights)
            }

            "Conv" | "ConvTranspose" => self.dispatch_conv(op, values, &plan.weights),

            "ReduceMean" | "ReduceSum" | "LayerNormalization" | "GlobalAveragePool" => {
                self.dispatch_reduction(op, values, &plan.weights)
            }

            "Gemm" | "MatMul" => self.dispatch_matmul(op, values, &plan.weights),

            "LSTM" => self.dispatch_lstm(op, values, plan),

            "STFT" => self.dispatch_stft(op, values, &plan.weights),

            "And" | "Or" | "Not" => self.dispatch_misc(op, values, &plan.weights),

            // Everything else is layout/shape/metadata. Layout takes the
            // full plan because it reads `op.precomputed` and may consult
            // `plan.static_i64`.
            _ => self.dispatch_layout(op, values, plan),
        }
    }

}

/// Insert each tensor in `outputs_vec` into `values` under the matching
/// output name on the planned op's node.
///
/// Three shapes are supported, distinguished by `outputs_vec.len()` vs.
/// `op.node.outputs.len()`:
///
///   * **Single-output (common case):** one tensor in, one insert.
///   * **Aliased multi-output (LSTM-style):** one tensor in, multiple
///     output names; the same tensor is Arc-cloned under every name and
///     tracked as a group so the memory pool only returns it once ALL
///     group members are freed (see `return_dead_inputs`).
///   * **Distinct multi-output (Split-style):** N tensors in, N output
///     names; each tensor is inserted under its own name with no Arc
///     sharing and no group tracking — independent pool lifetimes via
///     `tensor_last_use`.
///
/// An output is considered "used" iff it appears in `tensor_last_use`
/// (consumed by a downstream op) OR is a final graph output. Unused
/// outputs aren't stored — otherwise their Arc clones would keep the
/// tensor pinned beyond its real last use.
///
/// Port of today's logic in `src/cuda/inference/mod.rs::run` (~line 3234),
/// extended in WS-2 M2.5 for genuine multi-output ops (Split).
pub(super) fn store_outputs(
    op: &PlannedOp,
    outputs_vec: Vec<GpuTensor>,
    values: &mut HashMap<String, GpuTensor>,
    tensor_groups: &mut HashMap<String, String>,
    group_remaining: &mut HashMap<String, usize>,
    tensor_last_use: &HashMap<String, usize>,
    output_names: &[&str],
) {
    // FusedHead ops are special: the head node's own outputs (e.g. the
    // first Mul's `t0`) are NOT where the fused kernel writes. The
    // fused pattern replaces the entire chain with a single kernel
    // whose result tops the chain — so it must be stored under the
    // pattern's `output_name` (the final node's output). Mirrors
    // today's behavior in `src/cuda/inference/mod.rs::run` (~line 3008).
    if let OpKind::FusedHead(pattern) = &op.kind {
        use crate::cuda::inference::fusion::FusedPattern;
        let fused_output_name = match pattern {
            FusedPattern::DivRsqrt { output_name, .. }
            | FusedPattern::AddMulAdd { output_name, .. }
            | FusedPattern::Gelu { output_name, .. }
            | FusedPattern::MulAdd { output_name, .. }
            | FusedPattern::AddMul { output_name, .. }
            | FusedPattern::SubMul { output_name, .. }
            | FusedPattern::DivMul { output_name, .. }
            | FusedPattern::MulSinPowMulAdd { output_name, .. }
            | FusedPattern::GeneralChain { output_name, .. } => output_name,
        };
        let output = outputs_vec
            .into_iter()
            .next()
            .expect("FusedHead dispatch must produce one tensor");
        values.insert(fused_output_name.clone(), output);
        return;
    }

    // Distinct multi-output (Split-style): each tensor is independent.
    // No Arc sharing, no group tracking — each output's pool lifetime is
    // governed by `tensor_last_use` like any other intermediate tensor.
    if outputs_vec.len() > 1 {
        let names = &op.node.outputs;
        for (name, tensor) in names.iter().zip(outputs_vec.into_iter()) {
            if !name.is_empty() {
                values.insert(name.clone(), tensor);
            }
        }
        return;
    }

    // Single-tensor return path (everything else, including LSTM-style
    // aliased multi-output where one buffer surfaces under multiple
    // output names with shared Arc + group tracking).
    let output = outputs_vec
        .into_iter()
        .next()
        .expect("dispatch must produce at least one tensor");

    let outputs = &op.node.outputs;
    if outputs.len() == 1 {
        values.insert(outputs[0].clone(), output);
        return;
    }

    // Multi-output node: track as a group so we only return to pool when
    // ALL members are freed.
    let non_empty_outputs: Vec<&String> =
        outputs.iter().filter(|name| !name.is_empty()).collect();
    if non_empty_outputs.is_empty() {
        return;
    }

    // Only store outputs that will actually be used (appear in
    // tensor_last_use or are graph outputs). Unused outputs would create
    // Arc clones that prevent pool returns.
    let output_name_set: std::collections::HashSet<&str> =
        output_names.iter().copied().collect();
    let used_outputs: Vec<&&String> = non_empty_outputs
        .iter()
        .filter(|name| {
            tensor_last_use.contains_key(**name)
                || output_name_set.contains(name.as_str())
        })
        .collect();

    let count = used_outputs.len();
    if count > 1 {
        let primary = used_outputs[0].to_string();
        group_remaining.insert(primary.clone(), count);
        for output_name in &used_outputs {
            tensor_groups.insert((***output_name).to_string(), primary.clone());
            values.insert((***output_name).to_string(), output.clone());
        }
    } else if count == 1 {
        values.insert((***used_outputs[0]).to_string(), output);
    }
    // count == 0: no used outputs, don't store the tensor at all.
}

/// Resolve a node's input names to borrowed `GpuTensor`s.
///
/// Lookup order is `values` first (intermediate tensors produced by prior
/// ops this run) then `weights` (model-constant tensors on the
/// `ExecutionPlan`). Empty input names (ONNX's "this optional input is
/// absent") are filtered out. Unknown names are a hard error — the
/// lowering pass is responsible for guaranteeing every non-empty input
/// name resolves.
///
/// Borrowed references rather than clones keep every dispatch allocation-
/// free on the hot path; the returned `Vec<&GpuTensor>` holds the same
/// `'a` as the maps it draws from.
pub(super) fn resolve_inputs<'a>(
    names: &[String],
    values: &'a HashMap<String, GpuTensor>,
    weights: &'a HashMap<String, GpuTensor>,
) -> Result<Vec<&'a GpuTensor>, CudaError> {
    names
        .iter()
        .filter(|n| !n.is_empty())
        .map(|n| {
            values
                .get(n)
                .or_else(|| weights.get(n))
                .ok_or_else(|| CudaError::Kernel(format!("Input '{}' not found", n)))
        })
        .collect()
}

/// Upload a CPU `Tensor` to the GPU, preserving its dtype where possible.
///
/// Bool tensors are cast to Float32 on upload (false → 0.0, true → 1.0)
/// to match iconnx's existing **f32-as-bool** convention — comparison
/// ops (Equal, Less, Greater, …) already produce Float32 outputs that
/// downstream Where consumes as condition. Encoding Bool initializers
/// as Float32 keeps every "bool-shaped" tensor on GPU under one
/// representation; the alternative (Int32-as-bool, or a true GPU Bool
/// dtype) would either fork the encoding or kick off the broader audit
/// L2.1 / L3.5 sweep, which belongs in its own scoped phase. GPT-2's
/// 1×1×1024×1024 causal-mask Constant is the M2.7.c first surface for
/// this path.
///
/// Float64 has no GPU representation today and reaches the catchall
/// `_ => as_slice()` path, which panics — match the audit L2.1 status
/// quo, fix when a roster model surfaces it.
pub(super) fn upload_tensor(
    ctx: &IconnxCudaContext,
    tensor: &Tensor,
) -> Result<GpuTensor, CudaError> {
    match tensor {
        Tensor::Float32(_) => {
            let data = tensor.as_slice();
            GpuTensor::from_host_f32(ctx, &data, tensor.shape().to_vec())
        }
        Tensor::Int64(_) => {
            let data = tensor.as_slice_i64();
            GpuTensor::from_host_i64(ctx, &data, tensor.shape().to_vec())
        }
        Tensor::Int32(_) => {
            let data = tensor.as_slice_i32();
            GpuTensor::from_host_i32(ctx, &data, tensor.shape().to_vec())
        }
        Tensor::Bool(arr) => {
            // Bool → Float32 with `false` mapping to 0.0 and `true` to
            // 1.0. Explicit conversion — `tensor.as_slice()` panics on
            // non-Float32 (tensor.rs:296), which the prior fallback hit.
            let data: Vec<f32> = arr.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
            GpuTensor::from_host_f32(ctx, &data, tensor.shape().to_vec())
        }
        _ => {
            // Float64: not yet representable on GPU; falls through to the
            // panicking accessor as before. Replace with explicit Float64
            // → Float32 cast (or a true Float64 GPU path) when a roster
            // model surfaces it.
            let data = tensor.as_slice();
            GpuTensor::from_host_f32(ctx, &data, tensor.shape().to_vec())
        }
    }
}

/// Download the requested outputs to CPU tensors. Outputs may come from
/// the per-run `values` map or from the plan's model weights (if a graph
/// output happens to coincide with a weight name — uncommon but legal).
pub(super) fn collect_outputs(
    values: &HashMap<String, GpuTensor>,
    output_names: &[&str],
    weights: &HashMap<String, GpuTensor>,
    ctx: &IconnxCudaContext,
) -> Result<HashMap<String, Tensor>, CudaError> {
    let mut outputs = HashMap::new();
    for name in output_names {
        let gpu_tensor = values
            .get(*name)
            .or_else(|| weights.get(*name))
            .ok_or_else(|| CudaError::Kernel(format!("Output '{}' not found", name)))?;
        let cpu_tensor = match gpu_tensor {
            GpuTensor::Float32 { shape, .. } => {
                let data = gpu_tensor.to_host_f32(ctx)?;
                Tensor::from_vec_f32(data, shape.clone())
            }
            GpuTensor::Int64 { shape, .. } => {
                let data = gpu_tensor.to_host_i64(ctx)?;
                Tensor::from_vec_i64(data, shape.clone())
            }
            GpuTensor::Int32 { shape, .. } => {
                let data = gpu_tensor.to_host_i32(ctx)?;
                Tensor::from_vec_i32(data, shape.clone())
            }
        };
        outputs.insert((*name).to_string(), cpu_tensor);
    }
    Ok(outputs)
}
