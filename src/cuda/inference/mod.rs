//! Strangler shim around the Phase 2 graph IR.
//!
//! `GpuGraphExecutor` preserves its public API (new / add_initializer /
//! add_node / run / run_with_*) but its internals are now a thin wrapper
//! around:
//!
//! - [`crate::ir::OptimizableGraphBuilder`] — accumulates initializers and
//!   nodes as the caller builds the graph.
//! - [`crate::cuda::executor::Executor`] — the GPU runtime that dispatches
//!   a compiled [`crate::ir::ExecutionPlan`].
//! - A lazily-compiled [`crate::ir::ExecutionPlan`] cache — lowered from
//!   the builder on first `run()` and reused across runs.
//!
//! The shim is kept so that existing consumers (Kokoro loader, tests, the
//! public library surface) continue to compile. New callers should build
//! against `ir::OptimizableGraphBuilder` + `Executor` directly; this
//! wrapper will be removed in a future phase once no public consumers
//! depend on it.

#![allow(clippy::needless_range_loop)] // Explicit indexing is clearer for tensor operations.

use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};

use crate::attributes::NodeAttributes;
use crate::cuda::memory_pool::GpuMemoryPool;
use crate::cuda::ops::{gpu_expand, OpsKernelCache};
use crate::ir;
use crate::tensor::Tensor;

use super::context::{CudaError, IconnxCudaContext};
use super::tensor::GpuTensor;

pub mod fusion;
pub mod gpu_event_timer;
pub mod testing;

/// Log entry for a single node execution (for debugging shape mismatches).
///
/// Produced by [`GpuGraphExecutor::run_with_logging`] (via
/// [`crate::cuda::executor::Executor::run_with_logging`]). Exists as its
/// own type so debug tooling can consume a per-op shape/dtype/NaN trace
/// independent of inference output.
#[derive(Debug, Clone)]
pub struct NodeExecutionLog {
    /// Node name (usually output tensor name).
    pub node_name: String,
    /// Operator type (e.g., "Add", "MatMul", "Slice").
    pub op_type: String,
    /// Input tensor shapes.
    pub input_shapes: Vec<Vec<usize>>,
    /// Input tensor dtypes.
    pub input_dtypes: Vec<String>,
    /// Output tensor shape.
    pub output_shape: Vec<usize>,
    /// Output tensor dtype.
    pub output_dtype: String,
    /// Whether the output contains NaN values.
    pub has_nan: bool,
}

impl NodeExecutionLog {
    /// Format as a single line for comparison.
    pub fn to_line(&self) -> String {
        let inputs: Vec<String> = self
            .input_shapes
            .iter()
            .zip(&self.input_dtypes)
            .map(|(s, d)| format!("{:?}({})", s, d))
            .collect();
        let nan_marker = if self.has_nan { " [NaN!]" } else { "" };
        format!(
            "{} [{}] -> {:?}({}){}",
            self.node_name,
            inputs.join(", "),
            self.output_shape,
            self.output_dtype,
            nan_marker
        )
    }
}

/// Profiling data for GPU inference.
///
/// Populated by [`GpuGraphExecutor::run_with_profiling`] (via
/// [`crate::cuda::executor::Executor::run_with_profiling`]).
#[derive(Debug, Default)]
pub struct ProfileData {
    /// Time to upload input tensors (microseconds).
    pub input_upload_us: u64,
    /// Time to insert weight references (microseconds). In the Phase 2
    /// executor this is always 0 (weights live on the plan and aren't
    /// copied on each run) — kept for backwards compatibility with
    /// existing consumers.
    pub weight_ref_us: u64,
    /// Time for topological sort (microseconds). Topo sort now happens
    /// once at lowering, so per-run value is 0 — kept for backwards
    /// compatibility with existing consumers.
    pub topo_sort_us: u64,
    /// Total node execution time (microseconds).
    pub node_execution_us: u64,
    /// Time to download outputs (microseconds).
    pub output_download_us: u64,
    /// Total time (microseconds).
    pub total_us: u64,
    /// Per-operator-type CPU dispatch timing: (total_us, count).
    pub op_times: BTreeMap<String, (u64, usize)>,
    /// Per-op GPU-side timing from CUDA events.
    ///
    /// Keys match `op_times` exactly: same op-type strings, same fused
    /// pattern labels. The two maps together give both halves: `op_times`
    /// measures CPU-side async enqueue cost, `gpu_op_times` measures
    /// kernel wall time measured via `cudaEventElapsedTime`.
    pub gpu_op_times: BTreeMap<String, (u64, usize)>,
}

impl ProfileData {
    /// Print a summary of the profiling data.
    pub fn print_summary(&self) {
        eprintln!("\n=== GPU Inference Profile ===");
        eprintln!("  Input upload:     {:>8} us", self.input_upload_us);
        eprintln!("  Weight refs:      {:>8} us", self.weight_ref_us);
        eprintln!("  Topo sort:        {:>8} us", self.topo_sort_us);
        eprintln!(
            "  Node execution:   {:>8} us ({:.2} ms)",
            self.node_execution_us,
            self.node_execution_us as f64 / 1000.0
        );
        eprintln!("  Output download:  {:>8} us", self.output_download_us);
        eprintln!(
            "  TOTAL:            {:>8} us ({:.2} ms)",
            self.total_us,
            self.total_us as f64 / 1000.0
        );

        eprintln!("\n=== Top Operators by Time ===");
        let mut sorted_ops: Vec<_> = self.op_times.iter().collect();
        sorted_ops.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));

        for (op, (time_us, count)) in sorted_ops.iter().take(15) {
            let avg = *time_us as f64 / *count as f64;
            eprintln!(
                "  {:20} {:>8} us ({:>4} calls, {:>6.1} us/call)",
                op, time_us, count, avg
            );
        }

        if !self.gpu_op_times.is_empty() {
            eprintln!("\n=== Top Operators by GPU-Side Time (CUDA events) ===");
            let mut sorted_gpu: Vec<_> = self.gpu_op_times.iter().collect();
            sorted_gpu.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));

            for (op, (time_us, count)) in sorted_gpu.iter().take(15) {
                let cpu_us = self.op_times.get(*op).map(|(t, _)| *t).unwrap_or(0);
                let overhead = cpu_us.saturating_sub(*time_us);
                eprintln!(
                    "  {:20} gpu={:>8} us cpu={:>8} us overhead={:>8} us ({:>4} calls)",
                    op, time_us, cpu_us, overhead, count
                );
            }
        }
    }
}

/// Tensor that can live on either CPU or GPU.
pub enum TensorLocation {
    Cpu(Tensor),
    Gpu(GpuTensor),
}

impl TensorLocation {
    /// Move tensor to GPU.
    pub fn to_gpu(&self, ctx: &IconnxCudaContext) -> Result<GpuTensor, CudaError> {
        match self {
            TensorLocation::Cpu(t) => {
                let data = t.as_slice();
                GpuTensor::from_host_f32(ctx, &data, t.shape().to_vec())
            }
            TensorLocation::Gpu(g) => {
                // Return a copy by copying to host then back to device.
                let data = g.to_host_f32(ctx)?;
                GpuTensor::from_host_f32(ctx, &data, g.shape().to_vec())
            }
        }
    }

    /// Move tensor to CPU.
    pub fn to_cpu(&self, ctx: &IconnxCudaContext) -> Result<Tensor, CudaError> {
        match self {
            TensorLocation::Cpu(t) => Ok(t.clone()),
            TensorLocation::Gpu(g) => {
                let data = g.to_host_f32(ctx)?;
                Ok(Tensor::from_vec_f32(data, g.shape().to_vec()))
            }
        }
    }
}

/// GPU-accelerated graph executor.
///
/// Thin shim over [`ir::OptimizableGraphBuilder`] +
/// [`crate::cuda::executor::Executor`] + a lazily-lowered
/// [`ir::ExecutionPlan`]. Mutations (`add_initializer`, `add_node`)
/// accumulate into the builder and invalidate the cached plan. The first
/// `run*` call after any mutation lowers the graph once; subsequent
/// calls reuse the cached plan.
pub struct GpuGraphExecutor {
    /// Accumulates the graph as the caller builds it. `RefCell` because
    /// `run*` methods take `&self` yet still need to swap the builder
    /// into `lower()` (which consumes it).
    builder: RefCell<ir::OptimizableGraphBuilder>,
    /// GPU runtime — owns device context, kernel caches, memory pool.
    executor: crate::cuda::executor::Executor,
    /// Lazily-built plan. `None` means "graph changed since last compile —
    /// re-lower on the next run". `Some` means "this plan is in sync with
    /// the current builder contents; skip lowering."
    compiled_plan: RefCell<Option<ir::ExecutionPlan>>,
}

impl GpuGraphExecutor {
    /// Create a new GPU graph executor.
    pub fn new() -> Result<Self, CudaError> {
        Self::new_with_profiling(false)
    }

    /// Create a new GPU graph executor with optional profiling output
    /// during initialization.
    pub fn new_with_profiling(profile: bool) -> Result<Self, CudaError> {
        use std::time::Instant;

        let total_start = Instant::now();
        let start = Instant::now();
        let executor = crate::cuda::executor::Executor::new()?;
        if profile {
            eprintln!("  Executor (kernel caches + ctx): {:?}", start.elapsed());
            eprintln!("  TOTAL initialization:          {:?}", total_start.elapsed());
        }

        Ok(Self {
            builder: RefCell::new(ir::OptimizableGraphBuilder::new()),
            executor,
            compiled_plan: RefCell::new(None),
        })
    }

    /// Add an initializer (weight) to the graph.
    ///
    /// Stored as a CPU tensor on the builder; uploaded to the GPU at
    /// lowering time. Supported dtypes: Float32, Int64, Int32. Other
    /// dtypes are skipped silently so downstream dispatch produces the
    /// "unsupported dtype" runtime error (matches historical behavior).
    pub fn add_initializer(&mut self, name: String, tensor: &Tensor) -> Result<(), CudaError> {
        self.builder
            .borrow_mut()
            .add_initializer(name, tensor.clone());
        // Graph changed — drop any cached plan so the next run re-lowers.
        *self.compiled_plan.borrow_mut() = None;
        Ok(())
    }

    /// Add a computation node.
    ///
    /// Slice / Reshape / Unsqueeze / Squeeze precomputation is handled by
    /// [`crate::ir::passes::precompute_params`] at lowering time; callers
    /// don't need to order their `add_node` calls in topological form.
    pub fn add_node(
        &mut self,
        name: &str,
        op_type: &str,
        inputs: Vec<&str>,
        outputs: Vec<&str>,
        attributes: NodeAttributes,
    ) {
        self.builder.borrow_mut().add_node(
            name.to_string(),
            op_type.to_string(),
            inputs.iter().map(|s| s.to_string()).collect(),
            outputs.iter().map(|s| s.to_string()).collect(),
            attributes,
        );
        *self.compiled_plan.borrow_mut() = None;
    }

    /// Get the CUDA context.
    pub fn context(&self) -> &IconnxCudaContext {
        &self.executor.ctx
    }

    /// Number of initializers lowered into the current plan, or 0 if the
    /// graph hasn't been compiled yet.
    ///
    /// Historically this returned the count of uploaded GPU weights. With
    /// the Phase 2 shim, weights are uploaded lazily at lowering — a
    /// caller who asks before the first `run()` gets the pre-lowering
    /// answer (0). After the first `run()`, the count reflects
    /// `ExecutionPlan::weights`.
    pub fn num_weights(&self) -> usize {
        self.compiled_plan
            .borrow()
            .as_ref()
            .map(|p| p.weights.len())
            .unwrap_or(0)
    }

    /// Approximate GPU memory used by weights, in bytes. `0` if the graph
    /// hasn't been compiled. Mirrors the historical 4-bytes-per-element
    /// approximation.
    pub fn weights_memory_bytes(&self) -> usize {
        self.compiled_plan
            .borrow()
            .as_ref()
            .map(|p| p.weights.values().map(|t| t.len() * 4).sum())
            .unwrap_or(0)
    }

    /// Memory pool statistics: `(hits, misses, hit_rate_percent)`.
    pub fn pool_stats(&self) -> (usize, usize, f64) {
        let pool = self.executor.memory_pool.borrow();
        let (hits, misses) = pool.stats();
        let hit_rate = pool.hit_rate();
        (hits, misses, hit_rate)
    }

    /// Reset memory pool statistics.
    pub fn reset_pool_stats(&self) {
        self.executor.memory_pool.borrow_mut().reset_stats();
    }

    /// Number of pooled slices currently held.
    pub fn pool_size(&self) -> usize {
        self.executor.memory_pool.borrow().pooled_count()
    }

    /// Number of failed tensor returns (Arc refcount > 1).
    pub fn pool_failed_returns(&self) -> usize {
        self.executor.memory_pool.borrow().failed_returns()
    }

    /// Allocation analysis: top N sizes by allocation count.
    /// Returns Vec of (size, alloc_count, return_count, reuse_rate).
    pub fn pool_allocation_analysis(&self, top_n: usize) -> Vec<(usize, usize, usize, f64)> {
        self.executor.memory_pool.borrow().allocation_analysis(top_n)
    }

    /// Sizes that have been allocated but never returned.
    pub fn pool_unreturned_sizes(&self) -> Vec<(usize, usize)> {
        self.executor.memory_pool.borrow().unreturned_sizes()
    }

    /// Count of unique allocation sizes.
    pub fn pool_unique_sizes(&self) -> usize {
        self.executor.memory_pool.borrow().unique_sizes()
    }

    /// Execute the full computation graph on GPU.
    ///
    /// # Arguments
    /// * `inputs` - Map of input names to CPU tensors (will be uploaded).
    /// * `output_names` - Names of outputs to compute.
    ///
    /// # Returns
    /// Map of output names to CPU tensors.
    pub fn run(
        &self,
        inputs: HashMap<String, Tensor>,
        output_names: Vec<&str>,
    ) -> Result<HashMap<String, Tensor>, CudaError> {
        self.ensure_compiled()?;
        let plan = self.compiled_plan.borrow();
        self.executor
            .run(plan.as_ref().unwrap(), inputs, &output_names)
    }

    /// Lower the accumulated graph into an `ExecutionPlan` if one isn't
    /// already cached. No-op on the hot path after the first run.
    ///
    /// Exposed as `pub` so the test-only facade in
    /// [`testing`] can force compilation and then inspect plan state
    /// without triggering a full `run()`. Production callers should just
    /// call `run*` directly; the plan is compiled implicitly.
    pub fn compile(&self) -> Result<(), CudaError> {
        self.ensure_compiled()
    }

    fn ensure_compiled(&self) -> Result<(), CudaError> {
        if self.compiled_plan.borrow().is_some() {
            return Ok(());
        }
        let graph = self.builder.borrow().clone().build();
        let plan = ir::lower(graph, &self.executor.ctx)?;
        *self.compiled_plan.borrow_mut() = Some(plan);
        Ok(())
    }

    /// Access the compiled plan for test-only inspection. Returns `None`
    /// if `compile()` / a `run*` method hasn't been called yet.
    ///
    /// Not intended for production use — callers should not depend on
    /// the plan's internal structure. Exposed so the testing facade can
    /// verify precompute / fusion-detection outputs.
    pub(crate) fn compiled_plan(&self) -> std::cell::Ref<'_, Option<ir::ExecutionPlan>> {
        self.compiled_plan.borrow()
    }

    /// Enable tensor validation.
    ///
    /// With the Phase 2 shim, validation is owned by the executor's
    /// built-in validator slot and a fresh [`crate::cuda::validation::TensorValidator`]
    /// is constructed here. The stored validator is consulted by the
    /// executor's validation-aware variants where available; the
    /// recommended path is [`Self::run_with_validation`] which takes the
    /// mode inline and doesn't rely on stored state.
    ///
    /// Only available with `debug-inference` feature flag.
    #[cfg(feature = "debug-inference")]
    pub fn enable_validation(
        &self,
        mode: super::validation::ValidationMode,
    ) -> Result<(), CudaError> {
        let validator = super::validation::TensorValidator::new(mode, &self.executor.ctx)?;
        *self.executor.validator.borrow_mut() = Some(validator);
        Ok(())
    }

    /// Disable tensor validation.
    ///
    /// Only available with `debug-inference` feature flag.
    #[cfg(feature = "debug-inference")]
    pub fn disable_validation(&self) {
        *self.executor.validator.borrow_mut() = None;
    }

    /// Get the validation report from the last run.
    ///
    /// Only available with `debug-inference` feature flag.
    #[cfg(feature = "debug-inference")]
    pub fn validation_report(&self) -> Option<super::validation::ValidationReport> {
        self.executor
            .validator
            .borrow()
            .as_ref()
            .map(|v| v.report())
    }

    /// Run inference with tensor validation (NaN/Inf detection + stats).
    ///
    /// Only available with `debug-inference` feature flag.
    #[cfg(feature = "debug-inference")]
    pub fn run_with_validation(
        &self,
        inputs: HashMap<String, Tensor>,
        output_names: Vec<&str>,
        mode: super::validation::ValidationMode,
    ) -> Result<
        (
            HashMap<String, Tensor>,
            super::validation::ValidationReport,
        ),
        super::validation::ValidationError,
    > {
        use super::validation::ValidationError;

        self.ensure_compiled().map_err(ValidationError::from)?;
        let plan = self.compiled_plan.borrow();
        self.executor
            .run_with_validation(plan.as_ref().unwrap(), inputs, &output_names, mode)
    }

    /// Run inference with checkpoint capture and (optionally) ORT
    /// reference comparison.
    ///
    /// Only available with `debug-inference` feature flag.
    #[cfg(feature = "debug-inference")]
    pub fn run_with_checkpoints(
        &self,
        inputs: HashMap<String, Tensor>,
        output_names: Vec<&str>,
        config: super::checkpoints::CheckpointConfig,
    ) -> Result<
        (
            HashMap<String, Tensor>,
            super::checkpoints::CheckpointManager,
        ),
        CudaError,
    > {
        self.ensure_compiled()?;
        let plan = self.compiled_plan.borrow();
        self.executor
            .run_with_checkpoints(plan.as_ref().unwrap(), inputs, &output_names, config)
    }

    /// Execute with a per-node log of shapes, dtypes, and NaN presence.
    pub fn run_with_logging(
        &self,
        inputs: HashMap<String, Tensor>,
        output_names: Vec<&str>,
    ) -> Result<(HashMap<String, Tensor>, Vec<NodeExecutionLog>), CudaError> {
        self.ensure_compiled()?;
        let plan = self.compiled_plan.borrow();
        self.executor
            .run_with_logging(plan.as_ref().unwrap(), inputs, &output_names)
    }

    /// Execute with per-op CPU-dispatch and GPU-event timing.
    pub fn run_with_profiling(
        &self,
        inputs: HashMap<String, Tensor>,
        output_names: Vec<&str>,
    ) -> Result<(HashMap<String, Tensor>, ProfileData), CudaError> {
        self.ensure_compiled()?;
        let plan = self.compiled_plan.borrow();
        self.executor
            .run_with_profiling(plan.as_ref().unwrap(), inputs, &output_names)
    }
}

/// Compute the broadcast shape for two tensor shapes using NumPy-style
/// broadcasting rules.
///
/// Returns None if the shapes are not broadcastable. Used by the executor's
/// per-category dispatch paths (`cuda::executor::*`).
pub(crate) fn compute_broadcast_shape(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let max_ndim = a.len().max(b.len());
    let mut result = Vec::with_capacity(max_ndim);

    let a_padded: Vec<usize> = std::iter::repeat_n(1, max_ndim - a.len())
        .chain(a.iter().copied())
        .collect();
    let b_padded: Vec<usize> = std::iter::repeat_n(1, max_ndim - b.len())
        .chain(b.iter().copied())
        .collect();

    for (a_dim, b_dim) in a_padded.iter().zip(b_padded.iter()) {
        if *a_dim == *b_dim {
            result.push(*a_dim);
        } else if *a_dim == 1 {
            result.push(*b_dim);
        } else if *b_dim == 1 {
            result.push(*a_dim);
        } else {
            return None;
        }
    }

    Some(result)
}

/// Expand tensor to target shape if needed, using Cow to avoid cloning
/// when shapes already match.
///
/// Memory-pool-friendly: doesn't increment the Arc refcount when no
/// expansion is needed, allowing the original tensor to be returned to
/// the pool later. Used by the executor's per-category dispatch paths.
pub(crate) fn maybe_expand<'a>(
    ctx: &IconnxCudaContext,
    kernels: &OpsKernelCache,
    pool: &mut GpuMemoryPool,
    tensor: &'a GpuTensor,
    target_shape: Vec<usize>,
) -> Result<Cow<'a, GpuTensor>, CudaError> {
    if tensor.shape() == target_shape.as_slice() {
        Ok(Cow::Borrowed(tensor))
    } else {
        Ok(Cow::Owned(gpu_expand(ctx, kernels, pool, tensor, target_shape)?))
    }
}
