//! GPU-accelerated graph execution for Iconnx
//!
//! Executes ONNX computation graphs using GPU operators where available,
//! with automatic fallback to CPU for unsupported operations.

#![allow(clippy::needless_range_loop)] // Explicit indexing is clearer for tensor operations

use super::context::{CudaError, IconnxCudaContext};
use super::conv::{add_bias, gpu_conv2d, gpu_conv_transpose_2d, Conv2dParams, ConvKernelCache};
use super::cudnn::{
    cudnn_conv_1d, cudnn_conv_transpose_1d, CudnnHandle, Workspace as CudnnWorkspace,
};
use super::cufft::StftKernelCache;
use super::kernels::fused::{
    gpu_fused_add_mul, gpu_fused_add_mul_add, gpu_fused_div_mul, gpu_fused_div_rsqrt,
    gpu_fused_gelu, gpu_fused_mul_add, gpu_fused_mul_sin_pow_mul_add, gpu_fused_sub_mul,
    FusedKernelCache,
};
use super::kernels::{
    gpu_abs, gpu_add, gpu_ceil, gpu_clip, gpu_cos, gpu_div, gpu_exp, gpu_floor, gpu_leaky_relu,
    gpu_log, gpu_mul, gpu_neg, gpu_pow, gpu_relu, gpu_round, gpu_sigmoid, gpu_sin, gpu_sqrt,
    gpu_sub, gpu_tanh, KernelCache,
};
use super::lstm::{gpu_lstm, LstmKernelCache};
use super::matmul::{gpu_batched_matmul, gpu_gemm, gpu_matmul, gpu_matmul_2d_3d};
use super::memory_pool::GpuMemoryPool;
use super::ops::{
    gpu_and, gpu_atan, gpu_cast, gpu_concat, gpu_constant_of_shape_direct, gpu_copy, gpu_cumsum,
    gpu_equal, gpu_expand, gpu_gather_from_gpu, gpu_greater, gpu_greater_or_equal, gpu_less,
    gpu_less_or_equal, gpu_nonzero, gpu_not, gpu_not_equal, gpu_or, gpu_pad, gpu_range,
    gpu_range_i64, gpu_resize, gpu_scatter_nd, gpu_shape, gpu_slice_nd, gpu_transpose_2d,
    gpu_transpose_nd, gpu_where, OpsKernelCache, ResizeCoordMode, ResizeMode, ResizeNearestMode,
};
use super::reduction::{
    gpu_layer_norm, gpu_reduce_mean_axis, gpu_reduce_sum_axis, gpu_softmax, ReductionKernelCache,
};
use super::tensor::DType;
use super::tensor::GpuTensor;
use crate::attributes::NodeAttributes;
use crate::operators::fftw_stft::fftw_stft;
use crate::tensor::Tensor;
use cudarc::driver::sys::CUdeviceptr;
use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};

pub mod fusion;
pub mod gpu_event_timer;
pub mod testing;

use self::fusion::{FusedPattern, FusedPatternInfo};
use self::gpu_event_timer::GpuEventTimer;

/// Log entry for a single node execution (for debugging shape mismatches)
#[derive(Debug, Clone)]
pub struct NodeExecutionLog {
    /// Node name (usually output tensor name)
    pub node_name: String,
    /// Operator type (e.g., "Add", "MatMul", "Slice")
    pub op_type: String,
    /// Input tensor shapes
    pub input_shapes: Vec<Vec<usize>>,
    /// Input tensor dtypes
    pub input_dtypes: Vec<String>,
    /// Output tensor shape
    pub output_shape: Vec<usize>,
    /// Output tensor dtype
    pub output_dtype: String,
    /// Whether output contains NaN values
    pub has_nan: bool,
}

impl NodeExecutionLog {
    /// Format as a single line for comparison
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

/// Profiling data for GPU inference
#[derive(Debug, Default)]
pub struct ProfileData {
    /// Time to upload input tensors (microseconds)
    pub input_upload_us: u64,
    /// Time to insert weight references (microseconds)
    pub weight_ref_us: u64,
    /// Time for topological sort (microseconds)
    pub topo_sort_us: u64,
    /// Total node execution time (microseconds)
    pub node_execution_us: u64,
    /// Time to download outputs (microseconds)
    pub output_download_us: u64,
    /// Total time (microseconds)
    pub total_us: u64,
    /// Per-operator-type timing: (total_us, count)
    pub op_times: BTreeMap<String, (u64, usize)>,
    /// Per-op GPU-side timing from CUDA events (excludes launch overhead
    /// and CPU-side dispatch bookkeeping).
    ///
    /// Keys match `op_times` exactly: same op-type strings, same fused
    /// pattern labels. For any key present in `op_times` the same key is
    /// present here, and the `count` values match. `total_us` here is
    /// the GPU-measured elapsed time; `total_us` in `op_times` is CPU
    /// wall-clock (includes launch overhead). The difference
    /// `op_times[k].0 - gpu_op_times[k].0` approximates per-op launch
    /// overhead. An entry may have `total_us == 0` when the GPU work
    /// is effectively instantaneous (metadata-only ops).
    pub gpu_op_times: BTreeMap<String, (u64, usize)>,
}

impl ProfileData {
    /// Print a summary of the profiling data
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

/// Tensor that can live on either CPU or GPU
pub enum TensorLocation {
    Cpu(Tensor),
    Gpu(GpuTensor),
}

impl TensorLocation {
    /// Move tensor to GPU
    pub fn to_gpu(&self, ctx: &IconnxCudaContext) -> Result<GpuTensor, CudaError> {
        match self {
            TensorLocation::Cpu(t) => {
                let data = t.as_slice();
                GpuTensor::from_host_f32(ctx, &data, t.shape().to_vec())
            }
            TensorLocation::Gpu(g) => {
                // Return a copy by copying to host then back to device
                let data = g.to_host_f32(ctx)?;
                GpuTensor::from_host_f32(ctx, &data, g.shape().to_vec())
            }
        }
    }

    /// Move tensor to CPU
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

/// GPU-accelerated graph executor
pub struct GpuGraphExecutor {
    /// CUDA context
    ctx: IconnxCudaContext,

    /// Compiled kernel caches
    elementwise_kernels: KernelCache,
    reduction_kernels: ReductionKernelCache,
    conv_kernels: ConvKernelCache,
    lstm_kernels: LstmKernelCache,
    ops_kernels: OpsKernelCache,
    stft_kernels: StftKernelCache,
    fused_kernels: FusedKernelCache,

    /// cuDNN handle for optimized convolutions
    cudnn_handle: CudnnHandle,
    /// cuDNN workspace for scratch memory (RefCell for interior mutability)
    cudnn_workspace: RefCell<CudnnWorkspace>,

    /// Dynamic cache for Int64 values (cleared each run due to memory pool pointer reuse)
    /// Keyed by GPU device pointer, stores computed Int64 values during a single run
    i64_cache: RefCell<HashMap<CUdeviceptr, Vec<i64>>>,

    /// Static cache for Int64 values keyed by tensor name (persists across runs)
    /// Used for initializers and Constant nodes - values are known at graph construction time
    /// This cache is NOT cleared between runs, providing ~10-15ms savings from avoided D2H transfers
    static_i64_cache: RefCell<HashMap<String, Vec<i64>>>,

    /// Model weights on GPU
    weights: HashMap<String, GpuTensor>,

    /// Execution nodes (same as CPU version)
    nodes: Vec<ExecutionNode>,

    /// Cached topological sort order (avoids recomputing ~2.6ms per inference)
    sorted_nodes_cache: RefCell<Option<Vec<ExecutionNode>>>,

    /// Detected fused patterns: maps head node name to pattern info
    /// Uses RefCell for interior mutability (detected on first run)
    fused_patterns: RefCell<HashMap<String, FusedPatternInfo>>,
    /// Set of node names to skip (they're part of a fused pattern)
    nodes_to_skip: RefCell<std::collections::HashSet<String>>,
    /// Flag to track if patterns have been detected
    patterns_detected: RefCell<bool>,

    /// GPU memory pool for tensor allocation reuse
    memory_pool: RefCell<GpuMemoryPool>,

    /// Tensor validator for NaN/Inf detection (compile-gated)
    #[cfg(feature = "debug-inference")]
    validator: RefCell<Option<super::validation::TensorValidator>>,
}

/// Pre-computed Slice parameters extracted at graph construction time.
/// When Slice inputs (starts, ends, axes, steps) come from initializers,
/// we can extract their values once and avoid D2H transfers entirely.
///
/// Widened to `pub(crate)` so `ExecutionNode` (also `pub(crate)`) does
/// not leak a less-visible field type.
#[derive(Clone, Debug, Default)]
pub(crate) struct PrecomputedSlice {
    /// Pre-computed starts values (if input is an initializer)
    pub(crate) starts: Option<Vec<i64>>,
    /// Pre-computed ends values (if input is an initializer)
    pub(crate) ends: Option<Vec<i64>>,
    /// Pre-computed axes values (if input is an initializer)
    pub(crate) axes: Option<Vec<i64>>,
    /// Pre-computed steps values (if input is an initializer)
    pub(crate) steps: Option<Vec<i64>>,
}

/// A node in the execution graph (fields used in future graph execution).
///
/// Visibility is `pub(crate)` so `cuda::inference::fusion::detection` and
/// the sibling `cuda::inference::testing` module can construct and read
/// these nodes. No production code outside `cuda::inference` touches
/// this type.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub(crate) struct ExecutionNode {
    pub(crate) name: String,
    pub(crate) op_type: String,
    pub(crate) inputs: Vec<String>,
    pub(crate) outputs: Vec<String>,
    pub(crate) attributes: NodeAttributes,
    /// Pre-computed Slice parameters (only for Slice nodes with constant inputs)
    pub(crate) precomputed_slice: Option<PrecomputedSlice>,
}

impl GpuGraphExecutor {
    /// Create new GPU graph executor
    pub fn new() -> Result<Self, CudaError> {
        Self::new_with_profiling(false)
    }

    /// Create new GPU graph executor with optional profiling output
    pub fn new_with_profiling(profile: bool) -> Result<Self, CudaError> {
        use std::time::Instant;

        let total_start = Instant::now();

        let start = Instant::now();
        let ctx = IconnxCudaContext::new()?;
        if profile {
            eprintln!("  CUDA context creation: {:?}", start.elapsed());
        }

        let start = Instant::now();
        let elementwise_kernels = KernelCache::new(&ctx)?;
        if profile {
            eprintln!("  Elementwise kernels (NVRTC): {:?}", start.elapsed());
        }

        let start = Instant::now();
        let reduction_kernels = ReductionKernelCache::new(&ctx)?;
        if profile {
            eprintln!("  Reduction kernels (NVRTC): {:?}", start.elapsed());
        }

        let start = Instant::now();
        let conv_kernels = ConvKernelCache::new(&ctx)?;
        if profile {
            eprintln!("  Conv kernels (NVRTC): {:?}", start.elapsed());
        }

        let start = Instant::now();
        let lstm_kernels = LstmKernelCache::new(&ctx)?;
        if profile {
            eprintln!("  LSTM kernels (NVRTC): {:?}", start.elapsed());
        }

        let start = Instant::now();
        let ops_kernels = OpsKernelCache::new(&ctx)?;
        if profile {
            eprintln!("  Ops kernels (NVRTC): {:?}", start.elapsed());
        }

        let start = Instant::now();
        let stft_kernels = StftKernelCache::new(&ctx)?;
        if profile {
            eprintln!("  STFT kernels (NVRTC): {:?}", start.elapsed());
        }

        let start = Instant::now();
        let fused_kernels = FusedKernelCache::new(&ctx)?;
        if profile {
            eprintln!("  Fused kernels (NVRTC): {:?}", start.elapsed());
        }

        let start = Instant::now();
        let cudnn_handle = CudnnHandle::new()?;
        cudnn_handle.set_stream(ctx.stream())?;
        let cudnn_workspace = RefCell::new(CudnnWorkspace::new());
        if profile {
            eprintln!("  cuDNN initialization: {:?}", start.elapsed());
        }

        if profile {
            eprintln!("  TOTAL initialization: {:?}", total_start.elapsed());
        }

        Ok(Self {
            ctx,
            elementwise_kernels,
            reduction_kernels,
            conv_kernels,
            lstm_kernels,
            ops_kernels,
            stft_kernels,
            fused_kernels,
            cudnn_handle,
            cudnn_workspace,
            i64_cache: RefCell::new(HashMap::new()),
            static_i64_cache: RefCell::new(HashMap::new()),
            weights: HashMap::new(),
            nodes: Vec::new(),
            sorted_nodes_cache: RefCell::new(None),
            fused_patterns: RefCell::new(HashMap::new()),
            nodes_to_skip: RefCell::new(std::collections::HashSet::new()),
            patterns_detected: RefCell::new(false),
            memory_pool: RefCell::new(GpuMemoryPool::new()),
            #[cfg(feature = "debug-inference")]
            validator: RefCell::new(None),
        })
    }

    /// Add an initializer (weight) - automatically uploads to GPU
    /// Supports Float32, Int64, and Int32 tensors
    /// Int64 tensors are cached by NAME in static_i64_cache for fast parameter retrieval
    /// (avoids D2H transfers across multiple inference runs)
    pub fn add_initializer(&mut self, name: String, tensor: &Tensor) -> Result<(), CudaError> {
        let gpu_tensor = match tensor {
            Tensor::Float32(_) => {
                let data = tensor.as_slice();
                GpuTensor::from_host_f32(&self.ctx, &data, tensor.shape().to_vec())?
            }
            Tensor::Int64(_) => {
                let data = tensor.as_slice_i64();
                let gpu = GpuTensor::from_host_i64(&self.ctx, &data, tensor.shape().to_vec())?;
                // Cache Int64 values by NAME for fast parameter retrieval (persists across runs)
                self.static_i64_cache
                    .borrow_mut()
                    .insert(name.clone(), data.to_vec());
                gpu
            }
            Tensor::Int32(_) => {
                let data = tensor.as_slice_i32();
                GpuTensor::from_host_i32(&self.ctx, &data, tensor.shape().to_vec())?
            }
            _ => {
                // Unsupported dtype - log and skip
                return Ok(());
            }
        };
        self.weights.insert(name, gpu_tensor);
        Ok(())
    }

    /// Add a computation node
    /// For Slice nodes, automatically pre-computes parameters from initializers
    pub fn add_node(
        &mut self,
        name: &str,
        op_type: &str,
        inputs: Vec<&str>,
        outputs: Vec<&str>,
        attributes: NodeAttributes,
    ) {
        // For Slice nodes, try to pre-compute parameters from initializers
        // This avoids D2H transfers entirely for static parameters
        let precomputed_slice = if op_type == "Slice" {
            self.try_precompute_slice_params(&inputs)
        } else {
            None
        };

        self.nodes.push(ExecutionNode {
            name: name.to_string(),
            op_type: op_type.to_string(),
            inputs: inputs.iter().map(|s| s.to_string()).collect(),
            outputs: outputs.iter().map(|s| s.to_string()).collect(),
            attributes,
            precomputed_slice,
        });
        // Invalidate topo sort cache when graph structure changes
        *self.sorted_nodes_cache.borrow_mut() = None;
        // Clear detected patterns (will be re-detected on next run)
        self.fused_patterns.borrow_mut().clear();
        self.nodes_to_skip.borrow_mut().clear();
        *self.patterns_detected.borrow_mut() = false;
    }

    /// Try to pre-compute Slice parameters from initializers
    /// Returns Some if any parameters can be pre-computed
    fn try_precompute_slice_params(&self, inputs: &[&str]) -> Option<PrecomputedSlice> {
        // ONNX Slice: inputs = [data, starts, ends, axes?, steps?]
        if inputs.len() < 3 {
            return None;
        }

        let static_cache = self.static_i64_cache.borrow();

        // Try to get starts from static cache (input[1])
        let starts = inputs.get(1).and_then(|&name| static_cache.get(name).cloned());

        // Try to get ends from static cache (input[2])
        let ends = inputs.get(2).and_then(|&name| static_cache.get(name).cloned());

        // Try to get axes from static cache (input[3], optional)
        let axes = inputs.get(3).and_then(|&name| {
            if name.is_empty() {
                None
            } else {
                static_cache.get(name).cloned()
            }
        });

        // Try to get steps from static cache (input[4], optional)
        let steps = inputs.get(4).and_then(|&name| {
            if name.is_empty() {
                None
            } else {
                static_cache.get(name).cloned()
            }
        });

        // Only return if at least one parameter was pre-computed
        if starts.is_some() || ends.is_some() || axes.is_some() || steps.is_some() {
            Some(PrecomputedSlice {
                starts,
                ends,
                axes,
                steps,
            })
        } else {
            None
        }
    }

    /// Detect fuseable operator patterns in the graph.
    /// Called automatically before first execution.
    fn detect_fused_patterns(&self) {
        if *self.patterns_detected.borrow() {
            return;
        }

        let (detected, skip_set) =
            fusion::detect_fused_patterns(&self.nodes, &self.weights, &self.ctx);

        *self.fused_patterns.borrow_mut() = detected;
        *self.nodes_to_skip.borrow_mut() = skip_set;
        *self.patterns_detected.borrow_mut() = true;
    }

    /// Execute a detected fused pattern
    fn execute_fused_pattern(
        &self,
        pattern_info: &FusedPatternInfo,
        values: &HashMap<String, GpuTensor>,
    ) -> Result<GpuTensor, CudaError> {
        let mut pool = self.memory_pool.borrow_mut();
        match &pattern_info.pattern {
            FusedPattern::DivRsqrt {
                numerator_input,
                variance_input,
                eps_input,
                ..
            } => {
                // Get inputs (checking both computed values and weights)
                let numerator = values
                    .get(numerator_input)
                    .or_else(|| self.weights.get(numerator_input))
                    .ok_or_else(|| {
                        CudaError::Kernel(format!(
                            "Fused DivRsqrt: numerator '{}' not found",
                            numerator_input
                        ))
                    })?;

                let variance = values
                    .get(variance_input)
                    .or_else(|| self.weights.get(variance_input))
                    .ok_or_else(|| {
                        CudaError::Kernel(format!(
                            "Fused DivRsqrt: variance '{}' not found",
                            variance_input
                        ))
                    })?;

                // eps is typically a constant scalar - try to get its value
                let eps_tensor = values
                    .get(eps_input)
                    .or_else(|| self.weights.get(eps_input))
                    .ok_or_else(|| {
                        CudaError::Kernel(format!(
                            "Fused DivRsqrt: eps '{}' not found",
                            eps_input
                        ))
                    })?;

                // Extract eps as scalar (it's typically a 1-element tensor)
                let eps = if eps_tensor.len() == 1 {
                    eps_tensor.to_host_f32(&self.ctx)?[0]
                } else {
                    // Fall back to a reasonable default if not scalar
                    1e-5
                };

                gpu_fused_div_rsqrt(&self.ctx, &self.fused_kernels, &mut pool, numerator, variance, eps)
            }
            FusedPattern::AddMulAdd {
                x_input,
                a_input,
                b_input,
                c_input,
                ..
            } => {
                let x = values
                    .get(x_input)
                    .or_else(|| self.weights.get(x_input))
                    .ok_or_else(|| {
                        CudaError::Kernel(format!("Fused AddMulAdd: x '{}' not found", x_input))
                    })?;
                let a = values
                    .get(a_input)
                    .or_else(|| self.weights.get(a_input))
                    .ok_or_else(|| {
                        CudaError::Kernel(format!("Fused AddMulAdd: a '{}' not found", a_input))
                    })?;
                let b = values
                    .get(b_input)
                    .or_else(|| self.weights.get(b_input))
                    .ok_or_else(|| {
                        CudaError::Kernel(format!("Fused AddMulAdd: b '{}' not found", b_input))
                    })?;
                let c = values
                    .get(c_input)
                    .or_else(|| self.weights.get(c_input))
                    .ok_or_else(|| {
                        CudaError::Kernel(format!("Fused AddMulAdd: c '{}' not found", c_input))
                    })?;

                gpu_fused_add_mul_add(&self.ctx, &self.fused_kernels, &mut pool, x, a, b, c)
            }
            FusedPattern::Gelu { x_input, .. } => {
                let x = values
                    .get(x_input)
                    .or_else(|| self.weights.get(x_input))
                    .ok_or_else(|| {
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
                let a = values
                    .get(a_input)
                    .or_else(|| self.weights.get(a_input))
                    .ok_or_else(|| {
                        CudaError::Kernel(format!("Fused MulAdd: a '{}' not found", a_input))
                    })?;
                let b = values
                    .get(b_input)
                    .or_else(|| self.weights.get(b_input))
                    .ok_or_else(|| {
                        CudaError::Kernel(format!("Fused MulAdd: b '{}' not found", b_input))
                    })?;
                let c = values
                    .get(c_input)
                    .or_else(|| self.weights.get(c_input))
                    .ok_or_else(|| {
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
                let a = values
                    .get(a_input)
                    .or_else(|| self.weights.get(a_input))
                    .ok_or_else(|| {
                        CudaError::Kernel(format!("Fused AddMul: a '{}' not found", a_input))
                    })?;
                let b = values
                    .get(b_input)
                    .or_else(|| self.weights.get(b_input))
                    .ok_or_else(|| {
                        CudaError::Kernel(format!("Fused AddMul: b '{}' not found", b_input))
                    })?;
                let c = values
                    .get(c_input)
                    .or_else(|| self.weights.get(c_input))
                    .ok_or_else(|| {
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
                let a = values
                    .get(a_input)
                    .or_else(|| self.weights.get(a_input))
                    .ok_or_else(|| {
                        CudaError::Kernel(format!("Fused SubMul: a '{}' not found", a_input))
                    })?;
                let b = values
                    .get(b_input)
                    .or_else(|| self.weights.get(b_input))
                    .ok_or_else(|| {
                        CudaError::Kernel(format!("Fused SubMul: b '{}' not found", b_input))
                    })?;
                let c = values
                    .get(c_input)
                    .or_else(|| self.weights.get(c_input))
                    .ok_or_else(|| {
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
                let a = values
                    .get(a_input)
                    .or_else(|| self.weights.get(a_input))
                    .ok_or_else(|| {
                        CudaError::Kernel(format!("Fused DivMul: a '{}' not found", a_input))
                    })?;
                let b = values
                    .get(b_input)
                    .or_else(|| self.weights.get(b_input))
                    .ok_or_else(|| {
                        CudaError::Kernel(format!("Fused DivMul: b '{}' not found", b_input))
                    })?;
                let c = values
                    .get(c_input)
                    .or_else(|| self.weights.get(c_input))
                    .ok_or_else(|| {
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
                let x = values
                    .get(x_input)
                    .or_else(|| self.weights.get(x_input))
                    .ok_or_else(|| {
                        CudaError::Kernel(format!(
                            "Fused MulSinPowMulAdd: x '{}' not found",
                            x_input
                        ))
                    })?;
                let w0 = values
                    .get(w0_input)
                    .or_else(|| self.weights.get(w0_input))
                    .ok_or_else(|| {
                        CudaError::Kernel(format!(
                            "Fused MulSinPowMulAdd: w0 '{}' not found",
                            w0_input
                        ))
                    })?;
                let w1 = values
                    .get(w1_input)
                    .or_else(|| self.weights.get(w1_input))
                    .ok_or_else(|| {
                        CudaError::Kernel(format!(
                            "Fused MulSinPowMulAdd: w1 '{}' not found",
                            w1_input
                        ))
                    })?;
                let b = values
                    .get(b_input)
                    .or_else(|| self.weights.get(b_input))
                    .ok_or_else(|| {
                        CudaError::Kernel(format!(
                            "Fused MulSinPowMulAdd: b '{}' not found",
                            b_input
                        ))
                    })?;
                let p_tensor = values
                    .get(p_input)
                    .or_else(|| self.weights.get(p_input))
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

    /// Get cached Int64 values for a tensor (avoids D2H transfer for constant parameters)
    /// Returns None if the tensor is not in the cache (computed value, not constant)
    fn get_cached_i64(&self, tensor: &GpuTensor) -> Option<Vec<i64>> {
        let ptr = tensor.device_ptr(&self.ctx);
        self.i64_cache.borrow().get(&ptr).cloned()
    }

    /// Cache Int64 values for a tensor by device pointer (used within a single run)
    fn cache_i64(&self, tensor: &GpuTensor, values: Vec<i64>) {
        let ptr = tensor.device_ptr(&self.ctx);
        self.i64_cache.borrow_mut().insert(ptr, values);
    }

    /// Get cached Int64 values by tensor name from static cache (persists across runs)
    /// Returns None if the tensor name is not in the static cache
    fn get_static_cached_i64(&self, name: &str) -> Option<Vec<i64>> {
        self.static_i64_cache.borrow().get(name).cloned()
    }

    /// Cache Int64 values by tensor name in static cache (persists across runs)
    /// Used for Constant nodes and values derived from initializers
    fn cache_static_i64(&self, name: &str, values: Vec<i64>) {
        self.static_i64_cache
            .borrow_mut()
            .insert(name.to_string(), values);
    }

    /// Get Int64 values by name, falling back to D2H transfer if not cached.
    /// This avoids device_ptr() calls which trigger GPU synchronization (~100μs overhead).
    ///
    /// IMPORTANT: This function does NOT cache fetched values in the static cache.
    /// Only truly static values (initializers, Constant nodes) are pre-cached.
    /// Dynamic values (e.g., Shape op outputs) depend on input shapes and must NOT
    /// be cached across inference runs, otherwise changing input sequence lengths
    /// will return stale cached values causing shape mismatches.
    ///
    /// Use this instead of get_cached_i64() for performance-critical paths like Reshape,
    /// Squeeze, Unsqueeze, Expand, Pad, Slice, etc.
    fn get_or_fetch_i64(&self, name: &str, tensor: &GpuTensor) -> Result<Vec<i64>, CudaError> {
        // Check static cache first (no GPU sync) - only hits for initializers/constants
        if let Some(cached) = self.get_static_cached_i64(name) {
            return Ok(cached);
        }

        // Fallback: D2H transfer
        // Note: We intentionally do NOT cache this result because the value may
        // depend on input shapes (e.g., Shape op outputs with varying sequence lengths)
        tensor.to_host_i64(&self.ctx)
    }

    /// Compute the last node index that uses each tensor
    /// This enables freeing tensors after their last use to save GPU memory
    fn compute_tensor_last_use(
        &self,
        sorted_nodes: &[ExecutionNode],
        output_names: &[&str],
    ) -> HashMap<String, usize> {
        let mut last_use: HashMap<String, usize> = HashMap::new();

        // Track outputs that must be kept until the end
        let _outputs_set: std::collections::HashSet<&str> = output_names.iter().copied().collect();

        // For each node, mark all its inputs as being used at this index
        for (idx, node) in sorted_nodes.iter().enumerate() {
            for input in &node.inputs {
                if !input.is_empty() && !self.weights.contains_key(input) {
                    // Only track computed tensors, not weights (weights are never freed)
                    last_use.insert(input.clone(), idx);
                }
            }
        }

        // Mark output tensors as used at the end (index = len, so they're never freed early)
        for output in output_names {
            if !self.weights.contains_key(*output) {
                last_use.insert(output.to_string(), sorted_nodes.len());
            }
        }

        last_use
    }

    /// Check if an operator has GPU implementation
    pub fn has_gpu_impl(op_type: &str) -> bool {
        matches!(
            op_type,
            // Elementwise binary
            "Add" | "Sub" | "Mul" | "Div" |
            // Activations
            "Sigmoid" | "Tanh" | "Relu" | "LeakyRelu" |
            // Math unary
            "Exp" | "Log" | "Sqrt" | "Pow" | "Sin" | "Cos" | "Abs" | "Neg" | "Atan" |
            // Utility
            "Clip" | "Floor" | "Ceil" | "Round" | "Identity" |
            // Matrix
            "MatMul" | "Gemm" |
            // Reductions
            "ReduceSum" | "ReduceMean" | "Softmax" | "LayerNormalization" |
            // Conv/LSTM
            "Conv" | "ConvTranspose" | "LSTM" |
            // Comparison
            "Equal" | "Less" | "Greater" |
            "GreaterOrEqual" | "LessOrEqual" | "NotEqual" |
            // Logical
            "And" | "Or" | "Not" |
            // Shape/layout (some are metadata-only)
            "Shape" | "Cast" | "ConstantOfShape" | "Transpose" | "Gather" | "Slice" | "Reshape" | "Squeeze" | "Unsqueeze" |
            // Memory operations
            "Concat" | "Expand" | "Pad" | "Resize" | "NonZero" | "ScatterND" |
            // Conditional
            "Where" |
            // Sequence
            "Range" | "CumSum" |
            // Signal processing
            "STFT"
        )
    }

    /// Execute an operator on GPU
    pub fn execute_gpu_operator(
        &self,
        op_type: &str,
        inputs: &[&GpuTensor],
        input_names: &[String],
        attributes: &NodeAttributes,
    ) -> Result<GpuTensor, CudaError> {
        self.execute_gpu_operator_with_precomputed(op_type, inputs, input_names, attributes, None)
    }

    /// Execute a GPU operator with optional pre-computed parameters
    /// For Slice operations, precomputed_slice provides already-extracted parameter values
    fn execute_gpu_operator_with_precomputed(
        &self,
        op_type: &str,
        inputs: &[&GpuTensor],
        input_names: &[String],
        attributes: &NodeAttributes,
        precomputed_slice: Option<&PrecomputedSlice>,
    ) -> Result<GpuTensor, CudaError> {
        let mut pool = self.memory_pool.borrow_mut();
        match op_type {
            // Binary elementwise ops - with broadcasting support
            "Add" => {
                let a = inputs[0];
                let b = inputs[1];

                // If shapes match exactly, fast path
                if a.shape() == b.shape() {
                    return gpu_add(&self.ctx, &self.elementwise_kernels, &mut pool, a, b);
                }

                // If one is scalar (len=1), use scalar broadcast (already implemented in gpu_add)
                if a.len() == 1 || b.len() == 1 {
                    return gpu_add(&self.ctx, &self.elementwise_kernels, &mut pool, a, b);
                }

                // Try broadcasting to match shapes
                if let Some(broadcast_shape) = compute_broadcast_shape(a.shape(), b.shape()) {
                    let a_expanded = maybe_expand(&self.ctx, &self.ops_kernels, &mut pool, a, broadcast_shape.clone())?;
                    let b_expanded = maybe_expand(&self.ctx, &self.ops_kernels, &mut pool, b, broadcast_shape)?;
                    gpu_add(
                        &self.ctx,
                        &self.elementwise_kernels,
                        &mut pool,
                        &a_expanded,
                        &b_expanded,
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
                    let a_expanded = maybe_expand(&self.ctx, &self.ops_kernels, &mut pool, a, broadcast_shape.clone())?;
                    let b_expanded = maybe_expand(&self.ctx, &self.ops_kernels, &mut pool, b, broadcast_shape)?;
                    gpu_sub(
                        &self.ctx,
                        &self.elementwise_kernels,
                        &mut pool,
                        &a_expanded,
                        &b_expanded,
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
                    let a_expanded = maybe_expand(&self.ctx, &self.ops_kernels, &mut pool, a, broadcast_shape.clone())?;
                    let b_expanded = maybe_expand(&self.ctx, &self.ops_kernels, &mut pool, b, broadcast_shape)?;
                    gpu_mul(
                        &self.ctx,
                        &self.elementwise_kernels,
                        &mut pool,
                        &a_expanded,
                        &b_expanded,
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
                    let a_expanded = maybe_expand(&self.ctx, &self.ops_kernels, &mut pool, a, broadcast_shape.clone())?;
                    let b_expanded = maybe_expand(&self.ctx, &self.ops_kernels, &mut pool, b, broadcast_shape)?;
                    gpu_div(
                        &self.ctx,
                        &self.elementwise_kernels,
                        &mut pool,
                        &a_expanded,
                        &b_expanded,
                    )
                } else {
                    Err(CudaError::Kernel(format!(
                        "Div: shapes {:?} and {:?} are not broadcastable",
                        a.shape(),
                        b.shape()
                    )))
                }
            }

            // Activation functions
            "Sigmoid" => gpu_sigmoid(&self.ctx, &self.elementwise_kernels, &mut pool, inputs[0]),
            "Tanh" => gpu_tanh(&self.ctx, &self.elementwise_kernels, &mut pool, inputs[0]),
            "Relu" => gpu_relu(&self.ctx, &self.elementwise_kernels, &mut pool, inputs[0]),
            "LeakyRelu" => {
                let alpha = attributes.get_float("alpha").unwrap_or(0.01);
                gpu_leaky_relu(&self.ctx, &self.elementwise_kernels, &mut pool, inputs[0], alpha)
            }

            // Math ops
            "Exp" => gpu_exp(&self.ctx, &self.elementwise_kernels, &mut pool, inputs[0]),
            "Log" => gpu_log(&self.ctx, &self.elementwise_kernels, &mut pool, inputs[0]),
            "Sqrt" => {
                // Debug: check for negative values that would cause NaN
                if std::env::var("DEBUG_SQRT").is_ok() {
                    let data = inputs[0].to_host_f32(&self.ctx)?;
                    let min_val = data.iter().cloned().fold(f32::INFINITY, f32::min);
                    let neg_count = data.iter().filter(|&&x| x < 0.0).count();
                    if neg_count > 0 || min_val < 0.0 {
                        eprintln!(
                            "DEBUG Sqrt: input has {} negative values, min={:.6e}, shape={:?}",
                            neg_count, min_val, inputs[0].shape()
                        );
                    }
                }
                gpu_sqrt(&self.ctx, &self.elementwise_kernels, &mut pool, inputs[0])
            }
            "Pow" => {
                // For Pow, the exponent might be a scalar or tensor
                let (base, exp) = (inputs[0], inputs[1]);
                if base.shape() == exp.shape() {
                    gpu_pow(&self.ctx, &self.elementwise_kernels, &mut pool, base, exp)
                } else if let Some(broadcast_shape) =
                    compute_broadcast_shape(base.shape(), exp.shape())
                {
                    let base_expanded = maybe_expand(&self.ctx, &self.ops_kernels, &mut pool, base, broadcast_shape.clone())?;
                    let exp_expanded = maybe_expand(&self.ctx, &self.ops_kernels, &mut pool, exp, broadcast_shape)?;
                    gpu_pow(
                        &self.ctx,
                        &self.elementwise_kernels,
                        &mut pool,
                        &base_expanded,
                        &exp_expanded,
                    )
                } else {
                    Err(CudaError::Kernel(format!(
                        "Pow: shapes {:?} and {:?} are not broadcastable",
                        base.shape(),
                        exp.shape()
                    )))
                }
            }
            "Sin" => gpu_sin(&self.ctx, &self.elementwise_kernels, &mut pool, inputs[0]),
            "Cos" => gpu_cos(&self.ctx, &self.elementwise_kernels, &mut pool, inputs[0]),
            "Abs" => gpu_abs(&self.ctx, &self.elementwise_kernels, &mut pool, inputs[0]),
            "Neg" => gpu_neg(&self.ctx, &self.elementwise_kernels, &mut pool, inputs[0]),

            // Utility ops
            "Clip" => {
                let min = attributes.get_float("min").unwrap_or(f32::MIN);
                let max = attributes.get_float("max").unwrap_or(f32::MAX);
                gpu_clip(&self.ctx, &self.elementwise_kernels, &mut pool, inputs[0], min, max)
            }
            "Floor" => gpu_floor(&self.ctx, &self.elementwise_kernels, &mut pool, inputs[0]),
            "Ceil" => gpu_ceil(&self.ctx, &self.elementwise_kernels, &mut pool, inputs[0]),
            "Round" => gpu_round(&self.ctx, &self.elementwise_kernels, &mut pool, inputs[0]),

            // Matrix ops - dispatch based on tensor dimensions
            "MatMul" => {
                let a_ndim = inputs[0].ndim();
                let b_ndim = inputs[1].ndim();
                match (a_ndim, b_ndim) {
                    (2, 2) => gpu_matmul(&self.ctx, &mut pool, inputs[0], inputs[1]),
                    (3, 3) => gpu_batched_matmul(&self.ctx, &mut pool, inputs[0], inputs[1]),
                    (3, 2) => {
                        // 3D x 2D: A[batch, M, K] @ B[K, N] = C[batch, M, N]
                        // Reshape A to 2D: [batch*M, K]
                        // MatMul: [batch*M, K] @ [K, N] = [batch*M, N]
                        // Reshape result to 3D: [batch, M, N]
                        let a_shape = inputs[0].shape();
                        let b_shape = inputs[1].shape();
                        let batch = a_shape[0];
                        let m = a_shape[1];
                        let k = a_shape[2];
                        let n = b_shape[1];

                        // Reshape A from [batch, M, K] to [batch*M, K]
                        let a_reshaped =
                            inputs[0].reshape(vec![batch * m, k]).ok_or_else(|| {
                                CudaError::Kernel("MatMul: failed to reshape A".into())
                            })?;

                        // 2D matmul: [batch*M, K] @ [K, N] = [batch*M, N]
                        let result_2d = gpu_matmul(&self.ctx, &mut pool, &a_reshaped, inputs[1])?;

                        // Reshape result to [batch, M, N]
                        result_2d.reshape(vec![batch, m, n]).ok_or_else(|| {
                            CudaError::Kernel("MatMul: failed to reshape result".into())
                        })
                    }
                    (2, 3) => {
                        // 2D x 3D: A[M, K] @ B[batch, K, N] = C[batch, M, N]
                        // This broadcasts A across the batch dimension
                        gpu_matmul_2d_3d(&self.ctx, &mut pool, inputs[0], inputs[1])
                    }
                    (4, 4) => {
                        // 4D x 4D: A[batch, heads, M, K] @ B[batch, heads, K, N] = C[batch, heads, M, N]
                        // Used for attention: [batch, heads, seq, d] @ [batch, heads, d, seq]
                        // Reshape to 3D, do batched matmul, reshape back
                        let a_shape = inputs[0].shape();
                        let b_shape = inputs[1].shape();
                        let batch = a_shape[0];
                        let heads = a_shape[1];
                        let m = a_shape[2];
                        let k = a_shape[3];
                        let n = b_shape[3];

                        // Reshape A from [batch, heads, M, K] to [batch*heads, M, K]
                        let a_reshaped =
                            inputs[0]
                                .reshape(vec![batch * heads, m, k])
                                .ok_or_else(|| {
                                    CudaError::Kernel("MatMul 4D: failed to reshape A".into())
                                })?;

                        // Reshape B from [batch, heads, K, N] to [batch*heads, K, N]
                        let b_reshaped =
                            inputs[1]
                                .reshape(vec![batch * heads, k, n])
                                .ok_or_else(|| {
                                    CudaError::Kernel("MatMul 4D: failed to reshape B".into())
                                })?;

                        // 3D batched matmul: [batch*heads, M, K] @ [batch*heads, K, N] = [batch*heads, M, N]
                        let result_3d = gpu_batched_matmul(&self.ctx, &mut pool, &a_reshaped, &b_reshaped)?;

                        // Reshape result to [batch, heads, M, N]
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
                let c = if inputs.len() > 2 {
                    Some(inputs[2])
                } else {
                    None
                };
                gpu_gemm(
                    &self.ctx, &mut pool, inputs[0], inputs[1], c, alpha, beta, trans_a, trans_b,
                )
            }

            // Reduction ops (note: current GPU impl reduces last axis only)
            "ReduceSum" => {
                let keepdims = attributes.get_int("keepdims").unwrap_or(1) != 0;

                // Debug: trace ReduceSum input/output for duration predictor debugging
                if std::env::var("DEBUG_REDUCESUM").is_ok() {
                    if let Ok(input_vals) = inputs[0].to_host_f32(&self.ctx) {
                        let display: Vec<_> = input_vals.iter().take(10).map(|v| format!("{:.4}", v)).collect();
                        eprintln!(
                            "DEBUG_REDUCESUM input: shape={:?}, first_10={:?}{}",
                            inputs[0].shape(),
                            display,
                            if input_vals.len() > 10 { "..." } else { "" }
                        );
                    }
                }

                let result = gpu_reduce_sum_axis(&self.ctx, &self.reduction_kernels, &mut pool, inputs[0])?;

                // Debug: trace ReduceSum output
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
                    // Add back the reduced dimension as size 1
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
                let result = gpu_reduce_mean_axis(&self.ctx, &self.reduction_kernels, &mut pool, inputs[0])?;
                if keepdims {
                    // Add back the reduced dimension as size 1
                    let mut new_shape = result.shape().to_vec();
                    new_shape.push(1);
                    result.reshape(new_shape).ok_or_else(|| {
                        CudaError::Kernel("ReduceMean: failed to restore keepdims".into())
                    })
                } else {
                    Ok(result)
                }
            }
            "Softmax" => {
                // GPU softmax operates on last axis
                gpu_softmax(&self.ctx, &self.reduction_kernels, &mut pool, inputs[0])
            }
            "LayerNormalization" => {
                let epsilon = attributes.get_float("epsilon").unwrap_or(1e-5);
                // LayerNormalization requires scale (gamma) and bias (beta)
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

            // Conv
            "Conv" => {
                let shape = inputs[0].shape();
                match shape.len() {
                    3 => {
                        // 1D conv: [batch, channels, length] - use cuDNN
                        let strides = attributes.get_ints("strides").unwrap_or(&[1]);
                        let pads = attributes.get_ints("pads").unwrap_or(&[0, 0]);
                        let dilations = attributes.get_ints("dilations").unwrap_or(&[1]);
                        let group = attributes.get_int("group").unwrap_or(1) as usize;

                        let stride = strides.first().copied().unwrap_or(1) as usize;
                        let padding = pads.first().copied().unwrap_or(0) as usize;
                        let dilation = dilations.first().copied().unwrap_or(1) as usize;

                        // Use cuDNN for convolution
                        let mut workspace = self.cudnn_workspace.borrow_mut();
                        let mut output = cudnn_conv_1d(
                            &self.cudnn_handle,
                            &self.ctx,
                            &mut workspace,
                            inputs[0],
                            inputs[1],
                            stride,
                            padding,
                            dilation,
                            group,
                        )?;

                        // Add bias if present (cuDNN doesn't include bias in conv)
                        if inputs.len() > 2 {
                            let bias = inputs[2];
                            let out_shape = output.shape();
                            let batch_size = out_shape[0];
                            let out_channels = out_shape[1];
                            let spatial_size = out_shape[2];
                            add_bias(
                                &self.ctx,
                                &self.conv_kernels,
                                &mut output,
                                bias,
                                batch_size,
                                out_channels,
                                spatial_size,
                            )?;
                        }

                        Ok(output)
                    }
                    4 => {
                        // 2D conv: [batch, channels, height, width]
                        let kernel_shape = attributes.get_ints("kernel_shape").unwrap_or(&[3, 3]);
                        let strides = attributes.get_ints("strides").unwrap_or(&[1, 1]);
                        let pads = attributes.get_ints("pads").unwrap_or(&[0, 0, 0, 0]);
                        let params = Conv2dParams {
                            kernel_h: kernel_shape.first().copied().unwrap_or(3) as usize,
                            kernel_w: kernel_shape.get(1).copied().unwrap_or(3) as usize,
                            stride_h: strides.first().copied().unwrap_or(1) as usize,
                            stride_w: strides.get(1).copied().unwrap_or(1) as usize,
                            pad_h: pads.first().copied().unwrap_or(0) as usize,
                            pad_w: pads.get(1).copied().unwrap_or(0) as usize,
                        };
                        let bias = if inputs.len() > 2 {
                            Some(inputs[2])
                        } else {
                            None
                        };
                        gpu_conv2d(
                            &self.ctx,
                            &self.conv_kernels,
                            &mut pool,
                            inputs[0],
                            inputs[1],
                            bias,
                            &params,
                        )
                    }
                    ndim => Err(CudaError::Kernel(format!(
                        "Conv expects 3D (1D conv) or 4D (2D conv) input, got {}D with shape {:?}",
                        ndim, shape
                    ))),
                }
            }

            // ConvTranspose (Deconvolution)
            "ConvTranspose" => {
                let shape = inputs[0].shape();
                let group = attributes.get_int("group").unwrap_or(1) as usize;
                if shape.len() == 3 {
                    // 1D ConvTranspose - use cuDNN
                    let strides = attributes.get_ints("strides").unwrap_or(&[1]);
                    let pads = attributes.get_ints("pads").unwrap_or(&[0, 0]);
                    let output_padding = attributes.get_ints("output_padding").unwrap_or(&[0]);
                    let dilations = attributes.get_ints("dilations").unwrap_or(&[1]);

                    let stride = strides.first().copied().unwrap_or(1) as usize;
                    let padding = pads.first().copied().unwrap_or(0) as usize;
                    let output_pad = output_padding.first().copied().unwrap_or(0) as usize;
                    let dilation = dilations.first().copied().unwrap_or(1) as usize;

                    // Use cuDNN for transposed convolution
                    let mut workspace = self.cudnn_workspace.borrow_mut();

                    // Debug: Print input stats for ConvTranspose
                    if std::env::var("DEBUG_CONV_TRANSPOSE").is_ok() {
                        let inp = inputs[0];
                        let kern = inputs[1];
                        eprintln!("DEBUG ConvTranspose input:");
                        eprintln!("  input shape: {:?}, kernel shape: {:?}", inp.shape(), kern.shape());
                        eprintln!("  stride={}, padding={}, output_pad={}, dilation={}, group={}",
                            stride, padding, output_pad, dilation, group);
                        if let Ok(data) = inp.to_host_f32(&self.ctx) {
                            let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
                            let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                            let sum: f32 = data.iter().sum();
                            eprintln!("  input range=[{:.6}, {:.6}], sum={:.6}", min, max, sum);
                        }
                        if let Ok(data) = kern.to_host_f32(&self.ctx) {
                            let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
                            let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                            let sum: f32 = data.iter().sum();
                            eprintln!("  kernel range=[{:.6}, {:.6}], sum={:.6}", min, max, sum);
                        }
                    }

                    let mut output = cudnn_conv_transpose_1d(
                        &self.cudnn_handle,
                        &self.ctx,
                        &mut workspace,
                        inputs[0],
                        inputs[1],
                        stride,
                        padding,
                        output_pad,
                        dilation,
                        group,
                    )?;

                    // Debug: Print output stats for ConvTranspose
                    if std::env::var("DEBUG_CONV_TRANSPOSE").is_ok() {
                        if let Ok(data) = output.to_host_f32(&self.ctx) {
                            let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
                            let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                            let sum: f32 = data.iter().sum();
                            eprintln!("DEBUG ConvTranspose output:");
                            eprintln!("  output shape: {:?}", output.shape());
                            eprintln!("  output range=[{:.6}, {:.6}], sum={:.6}", min, max, sum);
                        }
                    }

                    // Add bias if present (cuDNN doesn't include bias in conv)
                    if inputs.len() > 2 {
                        let bias = inputs[2];
                        let out_shape = output.shape();
                        let batch_size = out_shape[0];
                        let out_channels = out_shape[1];
                        let spatial_size = out_shape[2];
                        add_bias(
                            &self.ctx,
                            &self.conv_kernels,
                            &mut output,
                            bias,
                            batch_size,
                            out_channels,
                            spatial_size,
                        )?;
                    }

                    Ok(output)
                } else {
                    // 2D ConvTranspose
                    let kernel_shape = attributes.get_ints("kernel_shape").unwrap_or(&[3, 3]);
                    let strides = attributes.get_ints("strides").unwrap_or(&[1, 1]);
                    let pads = attributes.get_ints("pads").unwrap_or(&[0, 0, 0, 0]);
                    let output_padding = attributes.get_ints("output_padding").unwrap_or(&[0, 0]);
                    let params = Conv2dParams {
                        kernel_h: kernel_shape.first().copied().unwrap_or(3) as usize,
                        kernel_w: kernel_shape.get(1).copied().unwrap_or(3) as usize,
                        stride_h: strides.first().copied().unwrap_or(1) as usize,
                        stride_w: strides.get(1).copied().unwrap_or(1) as usize,
                        pad_h: pads.first().copied().unwrap_or(0) as usize,
                        pad_w: pads.get(1).copied().unwrap_or(0) as usize,
                    };
                    let bias = if inputs.len() > 2 {
                        Some(inputs[2])
                    } else {
                        None
                    };
                    let output_pad_h = output_padding.first().copied().unwrap_or(0) as usize;
                    let output_pad_w = output_padding.get(1).copied().unwrap_or(0) as usize;
                    gpu_conv_transpose_2d(
                        &self.ctx,
                        &self.conv_kernels,
                        &mut pool,
                        inputs[0],
                        inputs[1],
                        bias,
                        &params,
                        output_pad_h,
                        output_pad_w,
                    )
                }
            }

            // LSTM
            "LSTM" => {
                let w = inputs[1]; // Weight
                let r = inputs[2]; // Recurrence
                let b = if inputs.len() > 3 {
                    Some(inputs[3])
                } else {
                    None
                };
                let h_init = if inputs.len() > 5 {
                    Some(inputs[5])
                } else {
                    None
                };
                let c_init = if inputs.len() > 6 {
                    Some(inputs[6])
                } else {
                    None
                };
                gpu_lstm(
                    &self.ctx,
                    &self.lstm_kernels,
                    &self.ops_kernels,
                    &mut pool,
                    inputs[0],
                    w,
                    r,
                    b,
                    h_init,
                    c_init,
                )
            }

            // Comparison ops - with broadcasting support
            "Equal" => {
                let (a, b) = (inputs[0], inputs[1]);
                if a.shape() == b.shape() {
                    gpu_equal(&self.ctx, &self.ops_kernels, a, b)
                } else if let Some(broadcast_shape) = compute_broadcast_shape(a.shape(), b.shape())
                {
                    let a_exp = maybe_expand(&self.ctx, &self.ops_kernels, &mut pool, a, broadcast_shape.clone())?;
                    let b_exp = maybe_expand(&self.ctx, &self.ops_kernels, &mut pool, b, broadcast_shape)?;
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
                    let a_exp = maybe_expand(&self.ctx, &self.ops_kernels, &mut pool, a, broadcast_shape.clone())?;
                    let b_exp = maybe_expand(&self.ctx, &self.ops_kernels, &mut pool, b, broadcast_shape)?;
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
                    let a_exp = maybe_expand(&self.ctx, &self.ops_kernels, &mut pool, a, broadcast_shape.clone())?;
                    let b_exp = maybe_expand(&self.ctx, &self.ops_kernels, &mut pool, b, broadcast_shape)?;
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
                    let a_exp = maybe_expand(&self.ctx, &self.ops_kernels, &mut pool, a, broadcast_shape.clone())?;
                    let b_exp = maybe_expand(&self.ctx, &self.ops_kernels, &mut pool, b, broadcast_shape)?;
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
                    let a_exp = maybe_expand(&self.ctx, &self.ops_kernels, &mut pool, a, broadcast_shape.clone())?;
                    let b_exp = maybe_expand(&self.ctx, &self.ops_kernels, &mut pool, b, broadcast_shape)?;
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
                    let a_exp = maybe_expand(&self.ctx, &self.ops_kernels, &mut pool, a, broadcast_shape.clone())?;
                    let b_exp = maybe_expand(&self.ctx, &self.ops_kernels, &mut pool, b, broadcast_shape)?;
                    gpu_not_equal(&self.ctx, &self.ops_kernels, &a_exp, &b_exp)
                } else {
                    Err(CudaError::Kernel(format!(
                        "NotEqual: shapes {:?} and {:?} not broadcastable",
                        a.shape(),
                        b.shape()
                    )))
                }
            }

            // Logical ops - with broadcasting support
            "And" => {
                let (a, b) = (inputs[0], inputs[1]);
                if a.shape() == b.shape() {
                    gpu_and(&self.ctx, &self.ops_kernels, a, b)
                } else if let Some(broadcast_shape) = compute_broadcast_shape(a.shape(), b.shape())
                {
                    let a_exp = maybe_expand(&self.ctx, &self.ops_kernels, &mut pool, a, broadcast_shape.clone())?;
                    let b_exp = maybe_expand(&self.ctx, &self.ops_kernels, &mut pool, b, broadcast_shape)?;
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
                    let a_exp = maybe_expand(&self.ctx, &self.ops_kernels, &mut pool, a, broadcast_shape.clone())?;
                    let b_exp = maybe_expand(&self.ctx, &self.ops_kernels, &mut pool, b, broadcast_shape)?;
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

            // Math ops
            "Atan" => gpu_atan(&self.ctx, &self.ops_kernels, inputs[0]),

            // Conditional - with broadcasting support
            "Where" => {
                let (condition, x, y) = (inputs[0], inputs[1], inputs[2]);

                // Cast condition to Float32 if it's Int32/Int64 (e.g., from Bool cast)
                // Use Cow to avoid cloning when no cast is needed
                let condition: Cow<'_, GpuTensor> = if condition.dtype() != DType::Float32 {
                    Cow::Owned(gpu_cast(&self.ctx, &self.ops_kernels, condition, DType::Float32)?)
                } else {
                    Cow::Borrowed(condition)
                };

                // If all shapes match exactly, fast path
                if condition.shape() == x.shape() && x.shape() == y.shape() {
                    return gpu_where(&self.ctx, &self.ops_kernels, &mut pool, &condition, x, y);
                }

                // Compute broadcast shape for all three inputs
                let shape_cx =
                    compute_broadcast_shape(condition.shape(), x.shape()).ok_or_else(|| {
                        CudaError::Kernel(format!(
                            "Where: condition {:?} and x {:?} are not broadcastable",
                            condition.shape(),
                            x.shape()
                        ))
                    })?;
                let broadcast_shape =
                    compute_broadcast_shape(&shape_cx, y.shape()).ok_or_else(|| {
                        CudaError::Kernel(format!(
                            "Where: intermediate {:?} and y {:?} are not broadcastable",
                            shape_cx,
                            y.shape()
                        ))
                    })?;

                // Expand each input to broadcast shape
                let cond_expanded = maybe_expand(&self.ctx, &self.ops_kernels, &mut pool, &condition, broadcast_shape.clone())?;
                let x_expanded = maybe_expand(&self.ctx, &self.ops_kernels, &mut pool, x, broadcast_shape.clone())?;
                let y_expanded = maybe_expand(&self.ctx, &self.ops_kernels, &mut pool, y, broadcast_shape)?;

                gpu_where(
                    &self.ctx,
                    &self.ops_kernels,
                    &mut pool,
                    &cond_expanded,
                    &x_expanded,
                    &y_expanded,
                )
            }

            // Shape/layout ops
            "Shape" => {
                // Shape produces Int64 output - cache it for downstream ops (Reshape, etc.)
                let shape_values: Vec<i64> = inputs[0].shape().iter().map(|&d| d as i64).collect();
                let result = gpu_shape(&self.ctx, inputs[0])?;
                // Cache the shape values to avoid D2H in Reshape/Unsqueeze/etc.
                self.cache_i64(&result, shape_values);
                Ok(result)
            }
            "Cast" => {
                // ONNX Cast uses "to" attribute with TensorProto data type enum
                // TensorProto: FLOAT=1, INT32=6, INT64=7, BOOL=9
                let target_type = attributes
                    .get_int("to")
                    .ok_or_else(|| CudaError::Kernel("Cast: missing 'to' attribute".into()))?;
                let target_dtype = match target_type {
                    1 => DType::Float32, // FLOAT
                    6 => DType::Int32,   // INT32
                    7 => DType::Int64,   // INT64
                    9 => DType::Int32,   // BOOL -> treat as Int32 (0 or 1)
                    _ => {
                        return Err(CudaError::Kernel(format!(
                            "Cast: unsupported target type {}",
                            target_type
                        )))
                    }
                };
                gpu_cast(&self.ctx, &self.ops_kernels, inputs[0], target_dtype)
            }
            "ConstantOfShape" => {
                // ONNX ConstantOfShape creates a tensor of given shape filled with value
                // Value comes from "value" attribute (tensor) or defaults to 0
                let value = attributes
                    .get_tensor("value")
                    .map(|t| t.as_slice()[0])
                    .unwrap_or(0.0f32);

                // Get shape from cache or fetch via D2H (avoids device_ptr() overhead)
                let shape_name = input_names.first().map(|s| s.as_str()).unwrap_or("");
                let target_shape: Vec<usize> = self
                    .get_or_fetch_i64(shape_name, inputs[0])?
                    .iter()
                    .map(|&d| d as usize)
                    .collect();

                gpu_constant_of_shape_direct(&self.ctx, &self.ops_kernels, &mut pool, target_shape, value)
            }
            "Transpose" => {
                // N-dimensional transpose with perm attribute
                let shape = inputs[0].shape();
                let ndim = shape.len();

                // Get permutation from attributes, default to reverse order
                let mut perm: Vec<i64> = if let Some(perm_attr) = attributes.get_ints("perm") {
                    perm_attr.to_vec()
                } else {
                    // Default: reverse all dimensions
                    (0..ndim as i64).rev().collect()
                };

                // Handle perm length mismatch (can happen with dynamic shapes)
                // If perm is longer than ndim, truncate to only use the dimensions that exist
                // This handles cases where ONNX has static perm for varying tensor shapes
                if perm.len() > ndim {
                    // Filter perm to only include dimensions that exist in the input
                    // and remap them to valid indices
                    perm.retain(|&d| (d as usize) < ndim || (d < 0 && (ndim as i64 + d) >= 0));

                    // If still wrong length, generate default
                    if perm.len() != ndim {
                        perm = (0..ndim as i64).rev().collect();
                    }
                } else if perm.len() < ndim {
                    // perm is shorter - extend with identity mapping
                    for i in perm.len()..ndim {
                        perm.push(i as i64);
                    }
                }

                // Use optimized 2D path when applicable (Float32 only)
                // Use general N-D path for Int64/Int32 as 2D kernel only supports f32
                if shape.len() == 2 && perm == [1, 0] && inputs[0].dtype() == DType::Float32 {
                    gpu_transpose_2d(&self.ctx, &self.ops_kernels, &mut pool, inputs[0])
                } else {
                    gpu_transpose_nd(&self.ctx, &self.ops_kernels, &mut pool, inputs[0], &perm)
                }
            }
            "Gather" => {
                // Gather: inputs = [data, indices]
                // Use GPU-native gather to avoid D2H + H2D round trip
                let axis_raw = attributes.get_int("axis").unwrap_or(0);
                let ndim = inputs[0].shape().len() as i64;
                let axis = if axis_raw < 0 {
                    (ndim + axis_raw) as usize
                } else {
                    axis_raw as usize
                };
                let indices_shape = inputs[1].shape().to_vec();

                if axis == 0 {
                    // Simple case: gather along axis 0 - use GPU-native function
                    let result =
                        gpu_gather_from_gpu(&self.ctx, &self.ops_kernels, &mut pool, inputs[0], inputs[1])?;
                    Ok(result)
                } else {
                    // Gather along axis != 0: transpose axis to front, gather, transpose back
                    // Create permutation that moves axis to position 0
                    let input_shape = inputs[0].shape();
                    let mut perm: Vec<i64> = (0..ndim).collect();
                    perm.remove(axis);
                    perm.insert(0, axis as i64);

                    // Transpose: move axis to position 0
                    let transposed =
                        gpu_transpose_nd(&self.ctx, &self.ops_kernels, &mut pool, inputs[0], &perm)?;

                    // Gather along axis 0 - use GPU-native function
                    let gathered =
                        gpu_gather_from_gpu(&self.ctx, &self.ops_kernels, &mut pool, &transposed, inputs[1])?;

                    // Create inverse permutation to move axis back
                    // But indices_shape may replace the original axis dimension
                    // Result shape: input_shape[:axis] + indices_shape + input_shape[axis+1:]
                    // After gather: [indices...] + input_shape[:axis] + input_shape[axis+1:]
                    // We need to transpose indices dimensions back to position 'axis'
                    let gathered_shape = gathered.shape();
                    if indices_shape.is_empty() {
                        // Scalar indices - axis dimension is removed
                        // gathered_shape = transposed_shape without axis 0 (the original axis)
                        // Need to transpose to match: original_shape with axis removed
                        let mut inv_perm: Vec<i64> = Vec::with_capacity(gathered_shape.len());
                        for (new_pos, &old_pos) in perm.iter().skip(1).enumerate() {
                            // Find where old_pos should go
                            let target = if old_pos > axis as i64 {
                                old_pos - 1
                            } else {
                                old_pos
                            };
                            inv_perm.push(target);
                            let _ = new_pos; // suppress warning
                        }
                        // This is complex - for scalar indices, just return gathered directly
                        // The shape transformation handles axis removal correctly
                        Ok(gathered)
                    } else {
                        // Non-scalar indices - inverse permutation to restore axis order
                        // gathered shape: [indices_shape..., input_shape[:axis], input_shape[axis+1:]]
                        // We need to move indices_shape dims to position 'axis'
                        let indices_dims = indices_shape.len();
                        let remaining_dims = input_shape.len() - 1; // Minus the axis dimension

                        // Build inverse permutation
                        // Position 0..indices_dims came from indices, need to go to axis..axis+indices_dims
                        // Position indices_dims..indices_dims+axis came from dims 0..axis
                        // Position indices_dims+axis.. came from dims axis+1..
                        let total_dims = indices_dims + remaining_dims;
                        let mut inv_perm = vec![0i64; total_dims];
                        // Where does each output position come from?
                        // Output[0..axis] <- gathered[indices_dims..indices_dims+axis]
                        // Output[axis..axis+indices_dims] <- gathered[0..indices_dims]
                        // Output[axis+indices_dims..] <- gathered[indices_dims+axis..]
                        for i in 0..axis {
                            inv_perm[i] = (indices_dims + i) as i64;
                        }
                        for i in 0..indices_dims {
                            inv_perm[axis + i] = i as i64;
                        }
                        for i in axis..remaining_dims {
                            inv_perm[indices_dims + i] = (indices_dims + i) as i64;
                        }

                        gpu_transpose_nd(&self.ctx, &self.ops_kernels, &mut pool, &gathered, &inv_perm)
                    }
                }
            }
            "Slice" => {
                // ONNX Slice: inputs = [data, starts, ends, axes?, steps?]
                // Parameter priority:
                // 1. Pre-computed at graph construction (fastest - no lookup needed)
                // 2. Static cache by name (persists across runs)
                // 3. Dynamic cache by pointer (within single run)
                // 4. D2H transfer (slow path)
                if inputs.len() < 3 {
                    return Err(CudaError::Kernel(
                        "Slice requires at least 3 inputs: data, starts, ends".into(),
                    ));
                }

                // Get starts - prefer precomputed, then cache lookup (using get_or_fetch_i64 to avoid device_ptr overhead)
                let starts = if let Some(precomp) = precomputed_slice.and_then(|p| p.starts.as_ref()) {
                    precomp.clone()
                } else {
                    let starts_name = input_names.get(1).map(|s| s.as_str()).unwrap_or("");
                    self.get_or_fetch_i64(starts_name, inputs[1])?
                };

                // Get ends - prefer precomputed, then cache lookup
                let ends = if let Some(precomp) = precomputed_slice.and_then(|p| p.ends.as_ref()) {
                    precomp.clone()
                } else {
                    let ends_name = input_names.get(2).map(|s| s.as_str()).unwrap_or("");
                    self.get_or_fetch_i64(ends_name, inputs[2])?
                };

                // Debug: trace Slice operations
                if std::env::var("DEBUG_SLICE").is_ok() {
                    let axes_dbg = if inputs.len() > 3 && !inputs[3].is_empty() {
                        let name = input_names.get(3).map(|s| s.as_str()).unwrap_or("");
                        self.get_or_fetch_i64(name, inputs[3]).ok()
                    } else {
                        None
                    };
                    let steps_dbg = if inputs.len() > 4 && !inputs[4].is_empty() {
                        let name = input_names.get(4).map(|s| s.as_str()).unwrap_or("");
                        self.get_or_fetch_i64(name, inputs[4]).ok()
                    } else {
                        None
                    };
                    eprintln!(
                        "DEBUG Slice: input_shape={:?} starts={:?} ends={:?} axes={:?} steps={:?}",
                        inputs[0].shape(), starts, ends, axes_dbg, steps_dbg
                    );
                }

                // For ONNX Slice, starts and ends should have same length.
                // If they don't match, one of them might be a scalar that needs broadcasting.
                let starts_len = starts.len();
                let ends_len = ends.len();
                let (starts, ends) = if starts_len != ends_len {
                    if starts_len == 1 && ends_len > 1 {
                        // Broadcast starts to match ends length
                        (vec![starts[0]; ends_len], ends)
                    } else if ends_len == 1 && starts_len > 1 {
                        // Broadcast ends to match starts length
                        (starts, vec![ends[0]; starts_len])
                    } else {
                        // Take the minimum length (truncate the longer one)
                        // This handles cases where one tensor has extra dimensions
                        let len = starts_len.min(ends_len);
                        (starts[..len].to_vec(), ends[..len].to_vec())
                    }
                } else {
                    (starts, ends)
                };

                let num_slices = starts.len();

                // axes: prefer precomputed, then cache lookup
                let axes = if let Some(precomp) = precomputed_slice.and_then(|p| p.axes.as_ref()) {
                    if precomp.len() == num_slices {
                        Some(precomp.clone())
                    } else {
                        None
                    }
                } else if inputs.len() > 3 && !inputs[3].is_empty() {
                    let axes_name = input_names.get(3).map(|s| s.as_str()).unwrap_or("");
                    let axes_data = self.get_or_fetch_i64(axes_name, inputs[3])?;
                    if axes_data.len() == num_slices {
                        Some(axes_data)
                    } else {
                        // Length mismatch - use default (axes = 0..num_slices)
                        None
                    }
                } else {
                    None
                };

                // steps: prefer precomputed, then cache lookup
                let steps = if let Some(precomp) = precomputed_slice.and_then(|p| p.steps.as_ref()) {
                    if precomp.len() == num_slices {
                        Some(precomp.clone())
                    } else {
                        None
                    }
                } else if inputs.len() > 4 && !inputs[4].is_empty() {
                    let steps_name = input_names.get(4).map(|s| s.as_str()).unwrap_or("");
                    let steps_data = self.get_or_fetch_i64(steps_name, inputs[4])?;
                    if steps_data.len() == num_slices {
                        Some(steps_data)
                    } else {
                        // Length mismatch - use default (steps = all 1s)
                        None
                    }
                } else {
                    None
                };

                gpu_slice_nd(
                    &self.ctx,
                    &self.ops_kernels,
                    &mut pool,
                    inputs[0],
                    &starts,
                    &ends,
                    axes.as_deref(),
                    steps.as_deref(),
                )
            }
            "Reshape" => {
                // Reshape is metadata-only - we need to get shape from second input or attribute
                // Try to get shape from attribute (if provided) or second input tensor
                let input_shape = inputs[0].shape();
                let input_len = inputs[0].len();

                let shape_data: Vec<i64> = if let Some(shape_attr) = attributes.get_ints("shape") {
                    shape_attr.to_vec()
                } else if inputs.len() > 1 {
                    // Get shape from cache or fetch via D2H (avoids device_ptr() overhead)
                    let shape_name = input_names.get(1).map(|s| s.as_str()).unwrap_or("");
                    self.get_or_fetch_i64(shape_name, inputs[1])?
                } else {
                    return Err(CudaError::Kernel(
                        "Reshape requires shape attribute or second input".into(),
                    ));
                };

                // ONNX Reshape semantics:
                // - 0 means copy dimension from input
                // - -1 means infer this dimension
                let mut new_shape: Vec<usize> = Vec::with_capacity(shape_data.len());
                let mut neg_one_idx: Option<usize> = None;
                let mut known_product = 1usize;

                for (i, &dim) in shape_data.iter().enumerate() {
                    let resolved = if dim == 0 {
                        // Copy from input shape at same position
                        if i < input_shape.len() {
                            input_shape[i]
                        } else {
                            1 // Default if input shape is shorter
                        }
                    } else if dim == -1 {
                        neg_one_idx = Some(i);
                        1 // Placeholder, will be computed later
                    } else {
                        dim as usize
                    };
                    if dim != -1 {
                        known_product *= resolved;
                    }
                    new_shape.push(resolved);
                }

                // Compute the -1 dimension if present
                if let Some(idx) = neg_one_idx {
                    if known_product > 0 {
                        new_shape[idx] = input_len / known_product;
                    }
                }

                inputs[0].reshape(new_shape.clone()).ok_or_else(|| {
                    let new_len: usize = new_shape.iter().product();
                    CudaError::Kernel(format!(
                        "Reshape: element count mismatch. Input has {} elements (shape {:?}), target shape {:?} has {} elements. Raw shape_data: {:?}",
                        input_len, input_shape, new_shape, new_len, shape_data
                    ))
                })
            }
            "Squeeze" => {
                // Squeeze removes dimensions of size 1
                // Opset <= 12: axes from attribute (optional)
                // Opset 13+: axes from second input tensor (optional)
                let axes: Option<Vec<i64>> = if let Some(axes_attr) = attributes.get_ints("axes") {
                    Some(axes_attr.to_vec())
                } else if inputs.len() > 1 {
                    // Opset 13+: axes come from second input
                    // Get from cache or fetch via D2H (avoids device_ptr() overhead)
                    let axes_name = input_names.get(1).map(|s| s.as_str()).unwrap_or("");
                    Some(self.get_or_fetch_i64(axes_name, inputs[1])?)
                } else {
                    None // Squeeze all dimensions of size 1
                };

                let old_shape = inputs[0].shape();
                let new_shape: Vec<usize> = if let Some(axes) = axes {
                    // Normalize negative axes and squeeze specific axes
                    let ndim = old_shape.len() as i64;
                    let axes_set: std::collections::HashSet<usize> = axes
                        .iter()
                        .map(|&a| {
                            if a < 0 {
                                (ndim + a) as usize
                            } else {
                                a as usize
                            }
                        })
                        .collect();
                    old_shape
                        .iter()
                        .enumerate()
                        .filter(|(i, &dim)| !(dim == 1 && axes_set.contains(i)))
                        .map(|(_, &dim)| dim)
                        .collect()
                } else {
                    // Squeeze all dimensions of size 1
                    old_shape.iter().copied().filter(|&d| d != 1).collect()
                };
                inputs[0]
                    .reshape(new_shape)
                    .ok_or_else(|| CudaError::Kernel("Squeeze: element count mismatch".into()))
            }
            "Unsqueeze" => {
                // Unsqueeze inserts dimensions of size 1 at specified axes
                // Opset <= 12: axes from attribute
                // Opset 13+: axes from second input tensor
                let axes: Vec<i64> = if let Some(axes_attr) = attributes.get_ints("axes") {
                    axes_attr.to_vec()
                } else if inputs.len() > 1 {
                    // Opset 13+: axes come from second input
                    // Get from cache or fetch via D2H (avoids device_ptr() overhead)
                    let axes_name = input_names.get(1).map(|s| s.as_str()).unwrap_or("");
                    self.get_or_fetch_i64(axes_name, inputs[1])?
                } else {
                    return Err(CudaError::Kernel(
                        "Unsqueeze requires 'axes' attribute or second input tensor".into(),
                    ));
                };

                let old_shape = inputs[0].shape();
                let new_ndim = old_shape.len() + axes.len();
                let mut new_shape = vec![1usize; new_ndim];

                // Normalize negative axes
                let mut axes_sorted: Vec<usize> = axes
                    .iter()
                    .map(|&a| {
                        if a < 0 {
                            (new_ndim as i64 + a) as usize
                        } else {
                            a as usize
                        }
                    })
                    .collect();
                axes_sorted.sort();

                // Fill in non-1 dimensions
                let mut old_idx = 0;
                for i in 0..new_ndim {
                    if !axes_sorted.contains(&i) {
                        new_shape[i] = old_shape[old_idx];
                        old_idx += 1;
                    }
                }

                inputs[0]
                    .reshape(new_shape)
                    .ok_or_else(|| CudaError::Kernel("Unsqueeze: element count mismatch".into()))
            }
            "Identity" => gpu_copy(&self.ctx, &self.ops_kernels, &mut pool, inputs[0]),

            // Memory operations
            "Concat" => {
                let axis = attributes.get_int("axis").unwrap_or(0);
                gpu_concat(&self.ctx, &self.ops_kernels, &mut pool, inputs, axis)
            }
            "Expand" => {
                // Expand shape comes from second input (Int64 tensor) or attribute
                let out_shape: Vec<usize> = if let Some(shape_attr) = attributes.get_ints("shape") {
                    shape_attr.iter().map(|&x| x as usize).collect()
                } else if inputs.len() > 1 {
                    // Get shape from cache or fetch via D2H (avoids device_ptr() overhead)
                    let shape_name = input_names.get(1).map(|s| s.as_str()).unwrap_or("");
                    self.get_or_fetch_i64(shape_name, inputs[1])?
                        .iter()
                        .map(|&x| x as usize)
                        .collect()
                } else {
                    return Err(CudaError::Kernel(
                        "Expand requires shape attribute or second input".into(),
                    ));
                };
                gpu_expand(&self.ctx, &self.ops_kernels, &mut pool, inputs[0], out_shape)
            }
            "Pad" => {
                // pads format: [begin_0, begin_1, ..., end_0, end_1, ...]
                // In ONNX opset 11+, pads can be input 1 (not attribute)
                let pads: Vec<i64> = if let Some(attr_pads) = attributes.get_ints("pads") {
                    attr_pads.to_vec()
                } else if inputs.len() >= 2 {
                    // Get pads from cache or fetch via D2H (avoids device_ptr() overhead)
                    let pads_name = input_names.get(1).map(|s| s.as_str()).unwrap_or("");
                    self.get_or_fetch_i64(pads_name, inputs[1])?
                } else {
                    return Err(CudaError::Kernel(
                        "Pad requires 'pads' attribute or input".into(),
                    ));
                };
                let value = if inputs.len() > 2 {
                    // Constant value from third input
                    let constant_data = inputs[2].to_host_f32(&self.ctx)?;
                    constant_data.first().copied().unwrap_or(0.0)
                } else {
                    attributes.get_float("value").unwrap_or(0.0)
                };
                // Get padding mode: "constant" (default), "reflect", or "edge"
                let mode = attributes.get_string("mode").unwrap_or("constant");
                gpu_pad(
                    &self.ctx,
                    &self.ops_kernels,
                    &mut pool,
                    inputs[0],
                    &pads,
                    value,
                    mode,
                )
            }
            "Resize" => {
                // ONNX Resize: inputs = [X, roi, scales] or [X, roi, scales, sizes]
                // roi is often empty/unused. scales determines output shape.
                // scales_idx: 1 if 2 inputs (roi omitted), 2 if 3+ inputs
                let scales_idx = if inputs.len() == 2 { 1 } else { 2 };
                let scales = inputs[scales_idx].to_host_f32(&self.ctx)?;

                // Debug: trace Resize operations
                if std::env::var("DEBUG_RESIZE").is_ok() {
                    let inp_shape = inputs[0].shape();
                    let out_shape: Vec<usize> = inp_shape
                        .iter()
                        .zip(scales.iter())
                        .map(|(&dim, &scale)| (dim as f32 * scale).round() as usize)
                        .collect();
                    eprintln!(
                        "DEBUG Resize: input={:?} scales={:?} -> output={:?}",
                        inp_shape, scales, out_shape
                    );
                }

                // Read coordinate transformation mode (ONNX default: half_pixel)
                let coord_mode_str = attributes
                    .get_string("coordinate_transformation_mode")
                    .unwrap_or("half_pixel");
                let coord_mode = match coord_mode_str {
                    "asymmetric" => ResizeCoordMode::Asymmetric,
                    _ => ResizeCoordMode::HalfPixel, // half_pixel and others
                };

                // Read nearest mode (ONNX default: round_prefer_floor)
                let nearest_mode_str = attributes
                    .get_string("nearest_mode")
                    .unwrap_or("round_prefer_floor");
                let nearest_mode = match nearest_mode_str {
                    "floor" => ResizeNearestMode::Floor,
                    _ => ResizeNearestMode::RoundPreferFloor, // round_prefer_floor and others
                };

                // Read interpolation mode (ONNX default: nearest)
                let mode_str = attributes.get_string("mode").unwrap_or("nearest");
                let mode = match mode_str {
                    "linear" => ResizeMode::Linear,
                    _ => ResizeMode::Nearest, // nearest, cubic, etc. (only nearest/linear implemented)
                };

                gpu_resize(
                    &self.ctx,
                    &self.ops_kernels,
                    &mut pool,
                    inputs[0],
                    &scales,
                    coord_mode,
                    nearest_mode,
                    mode,
                )
            }
            "NonZero" => {
                // NonZero returns indices of non-zero elements as Int64
                gpu_nonzero(&self.ctx, inputs[0])
            }
            "ScatterND" => {
                // ScatterND: inputs = [data, indices, updates]
                // Scatter updates into data at given indices
                // Indices need to be extracted to CPU
                if inputs.len() < 3 {
                    return Err(CudaError::Kernel(
                        "ScatterND requires 3 inputs: data, indices, updates".into(),
                    ));
                }
                let indices_host = inputs[1].to_host_i64(&self.ctx)?;
                let indices_shape = inputs[1].shape().to_vec();
                gpu_scatter_nd(
                    &self.ctx,
                    &self.ops_kernels,
                    &mut pool,
                    inputs[0],
                    &indices_host,
                    inputs[2],
                    &indices_shape,
                )
            }

            // Sequence ops
            "Range" => {
                // ONNX Range: inputs = [start, limit, delta] (all scalars)
                // Output: sequence from start to limit (exclusive) with step delta
                if inputs.len() != 3 {
                    return Err(CudaError::Kernel(format!(
                        "Range requires 3 inputs, got {}",
                        inputs.len()
                    )));
                }
                let dtype = inputs[0].dtype();
                match dtype {
                    crate::cuda::tensor::DType::Float32 => {
                        // Float32 Range - D2H transfers required (no cache for f32 scalars)
                        let start_data = inputs[0].to_host_f32(&self.ctx)?;
                        let limit_data = inputs[1].to_host_f32(&self.ctx)?;
                        let delta_data = inputs[2].to_host_f32(&self.ctx)?;
                        let start = start_data[0];
                        let limit = limit_data[0];
                        let delta = delta_data[0];
                        let n = ((limit - start) / delta).ceil().max(0.0) as usize;

                        // Debug: trace Range parameters
                        if std::env::var("DEBUG_SHAPES").is_ok() {
                            eprintln!(
                                "DEBUG_RANGE_F32: start={}, limit={}, delta={}, n={}",
                                start, limit, delta, n
                            );
                        }

                        gpu_range(&self.ctx, &self.ops_kernels, start, delta, n)
                    }
                    crate::cuda::tensor::DType::Int64 => {
                        // Get scalars from cache or fetch via D2H (avoids device_ptr() overhead)
                        let start_name = input_names.first().map(|s| s.as_str()).unwrap_or("");
                        let start = self.get_or_fetch_i64(start_name, inputs[0])?[0];
                        let limit_name = input_names.get(1).map(|s| s.as_str()).unwrap_or("");
                        let limit = self.get_or_fetch_i64(limit_name, inputs[1])?[0];
                        let delta_name = input_names.get(2).map(|s| s.as_str()).unwrap_or("");
                        let delta = self.get_or_fetch_i64(delta_name, inputs[2])?[0];
                        // Debug: trace Range parameters
                        if std::env::var("DEBUG_SHAPES").is_ok() {
                            eprintln!(
                                "DEBUG_RANGE_I64: start={}, limit={}, delta={}",
                                start, limit, delta
                            );
                        }

                        // Compute number of elements in range [start, limit) with step delta
                        let n = if delta == 0 {
                            return Err(CudaError::Kernel("Range: delta cannot be 0".into()));
                        } else if delta > 0 {
                            if limit <= start {
                                0
                            } else {
                                // Ceiling division for positive delta
                                let diff = limit - start;
                                ((diff + delta - 1) / delta) as usize
                            }
                        } else {
                            // delta < 0
                            if limit >= start {
                                0
                            } else {
                                // Ceiling division for negative delta (start > limit)
                                let diff = start - limit;
                                let abs_delta = -delta;
                                ((diff + abs_delta - 1) / abs_delta) as usize
                            }
                        };
                        gpu_range_i64(&self.ctx, &self.ops_kernels, start, delta, n)
                    }
                    _ => Err(CudaError::Kernel(format!(
                        "Range: unsupported dtype {}",
                        dtype.name()
                    ))),
                }
            }
            "CumSum" => {
                let exclusive = attributes.get_int("exclusive").unwrap_or(0) != 0;
                let reverse = attributes.get_int("reverse").unwrap_or(0) != 0;

                // Get axis from inputs[1] (ONNX CumSum takes axis as second input)
                let axis = if inputs.len() > 1 {
                    let axis_tensor = inputs[1];
                    if axis_tensor.is_i64() {
                        axis_tensor.to_host_i64(&self.ctx)?[0]
                    } else if axis_tensor.is_i32() {
                        axis_tensor.to_host_i32(&self.ctx)?[0] as i64
                    } else {
                        return Err(CudaError::Kernel(
                            "CumSum: axis must be int64 or int32".to_string(),
                        ));
                    }
                } else {
                    // Default to last axis if not specified
                    -1
                };

                gpu_cumsum(&self.ctx, &self.ops_kernels, inputs[0], axis, exclusive, reverse)
            }

            // Signal processing
            "STFT" => {
                // ONNX STFT (opset 17):
                // Input 0: signal [batch, length] or [batch, length, 1]
                // Input 1: frame_step (scalar, int64) - hop length
                // Input 2: window (optional, float32)
                // Input 3: frame_length (optional, int64)
                // If frame_length not provided, use window length
                if inputs.len() < 2 {
                    return Err(CudaError::Kernel(
                        "STFT requires at least signal and frame_step inputs".to_string(),
                    ));
                }

                // Get hop length from input 1 (frame_step)
                let hop_length = {
                    let data = inputs[1].to_host_i64(&self.ctx)?;
                    data.first().copied().unwrap_or(512) as usize
                };

                // Window from input 2 (optional)
                let window = if inputs.len() > 2 {
                    Some(inputs[2])
                } else {
                    None
                };

                // Frame length from input 3 or infer from window
                let frame_length = if inputs.len() > 3 {
                    let data = inputs[3].to_host_i64(&self.ctx)?;
                    data.first().copied().unwrap_or(2048) as usize
                } else if let Some(w) = window {
                    w.shape().iter().product::<usize>()
                } else {
                    return Err(CudaError::Kernel(
                        "STFT needs frame_length input or window to infer frame size".to_string(),
                    ));
                };

                let onesided = attributes.get_int("onesided").unwrap_or(1) != 0;

                // Handle batched input: [batch, length] -> [length], then add batch back
                let signal_shape = inputs[0].shape();
                let (signal_1d, batch_size) = if signal_shape.len() == 2 {
                    // [batch, length] -> [length] for batch=1, else unsupported
                    if signal_shape[0] != 1 {
                        return Err(CudaError::Kernel(format!(
                            "STFT only supports batch=1, got shape {:?}",
                            signal_shape
                        )));
                    }
                    let signal_reshaped = inputs[0]
                        .reshape(vec![signal_shape[1]])
                        .ok_or_else(|| CudaError::Kernel("Failed to reshape signal".to_string()))?;
                    (signal_reshaped, 1)
                } else if signal_shape.len() == 3 && signal_shape[2] == 1 {
                    // [batch, length, 1] -> [length] for batch=1
                    if signal_shape[0] != 1 {
                        return Err(CudaError::Kernel(format!(
                            "STFT only supports batch=1, got shape {:?}",
                            signal_shape
                        )));
                    }
                    let signal_reshaped = inputs[0]
                        .reshape(vec![signal_shape[1]])
                        .ok_or_else(|| CudaError::Kernel("Failed to reshape signal".to_string()))?;
                    (signal_reshaped, 1)
                } else if signal_shape.len() == 1 {
                    (inputs[0].clone(), 1)
                } else {
                    return Err(CudaError::Kernel(format!(
                        "STFT signal must be 1D, 2D [batch, len] or 3D [batch, len, 1], got {:?}",
                        signal_shape
                    )));
                };

                // Use window if provided - use FFTW-based STFT for high-precision FFT
                if let Some(w) = window {
                    // Copy signal and window from GPU to CPU
                    let signal_cpu = signal_1d.to_host_f32(&self.ctx)?;
                    let window_cpu = w.to_host_f32(&self.ctx)?;

                    // Call FFTW-based STFT (pure Rust, high-precision FFT)
                    let (stft_data, stft_shape) =
                        fftw_stft(&signal_cpu, &window_cpu, frame_length, hop_length, onesided)
                            .map_err(|e| CudaError::Kernel(format!("FFTW STFT failed: {}", e)))?;

                    // Copy result back to GPU
                    // FFTW STFT returns [1, num_frames, fft_bins, 2]
                    // We need [batch, num_frames, fft_bins, 2]
                    let new_shape = if stft_shape.len() == 4 {
                        vec![batch_size, stft_shape[1], stft_shape[2], stft_shape[3]]
                    } else {
                        // [num_frames, fft_bins, 2] -> [batch, num_frames, fft_bins, 2]
                        vec![batch_size, stft_shape[0], stft_shape[1], stft_shape[2]]
                    };

                    GpuTensor::from_host_f32(&self.ctx, &stft_data, new_shape)
                } else {
                    Err(CudaError::Kernel(
                        "STFT requires window input for FFTW implementation".to_string(),
                    ))
                }
            }

            _ => Err(CudaError::Kernel(format!(
                "No GPU implementation for {}",
                op_type
            ))),
        }
    }

    /// Get the CUDA context
    pub fn context(&self) -> &IconnxCudaContext {
        &self.ctx
    }

    /// Get weights on GPU
    pub fn weights(&self) -> &HashMap<String, GpuTensor> {
        &self.weights
    }

    /// Get number of weights
    pub fn num_weights(&self) -> usize {
        self.weights.len()
    }

    /// Calculate total GPU memory used by weights
    pub fn weights_memory_bytes(&self) -> usize {
        self.weights.values().map(|t| t.len() * 4).sum()
    }

    /// Get memory pool statistics: (hits, misses, hit_rate_percent)
    pub fn pool_stats(&self) -> (usize, usize, f64) {
        let pool = self.memory_pool.borrow();
        let (hits, misses) = pool.stats();
        let hit_rate = pool.hit_rate();
        (hits, misses, hit_rate)
    }

    /// Reset memory pool statistics
    pub fn reset_pool_stats(&self) {
        self.memory_pool.borrow_mut().reset_stats();
    }

    /// Get number of pooled slices currently held
    pub fn pool_size(&self) -> usize {
        self.memory_pool.borrow().pooled_count()
    }

    /// Get number of failed tensor returns (Arc refcount > 1)
    pub fn pool_failed_returns(&self) -> usize {
        self.memory_pool.borrow().failed_returns()
    }

    /// Get allocation analysis: top N sizes by allocation count
    /// Returns Vec of (size, alloc_count, return_count, reuse_rate)
    pub fn pool_allocation_analysis(&self, top_n: usize) -> Vec<(usize, usize, usize, f64)> {
        self.memory_pool.borrow().allocation_analysis(top_n)
    }

    /// Get sizes that are allocated but never returned
    pub fn pool_unreturned_sizes(&self) -> Vec<(usize, usize)> {
        self.memory_pool.borrow().unreturned_sizes()
    }

    /// Get count of unique allocation sizes
    pub fn pool_unique_sizes(&self) -> usize {
        self.memory_pool.borrow().unique_sizes()
    }

    /// Execute the full computation graph on GPU
    ///
    /// # Arguments
    /// * `inputs` - Map of input names to CPU tensors (will be uploaded to GPU)
    /// * `output_names` - Names of outputs to compute
    ///
    /// # Returns
    /// Map of output names to CPU tensors (downloaded from GPU)
    pub fn run(
        &self,
        inputs: HashMap<String, Tensor>,
        output_names: Vec<&str>,
    ) -> Result<HashMap<String, Tensor>, CudaError> {
        // Clear the i64 cache before each run to avoid stale values from previous runs.
        // This is necessary because the memory pool reuses device pointers, and the cache
        // is keyed by device pointer. Without clearing, we'd get incorrect cached values
        // when a recycled pointer is looked up.
        self.i64_cache.borrow_mut().clear();

        // Storage for intermediate tensors on GPU
        let mut values: HashMap<String, GpuTensor> = HashMap::new();

        // Track tensor groups for multi-output nodes
        // When a node has multiple outputs, they share the same underlying tensor (Arc clones).
        // We group them together and only return to pool when ALL members are freed.
        // Maps each output name -> group ID (primary name)
        let mut tensor_groups: HashMap<String, String> = HashMap::new();
        // Maps group ID (primary) -> count of remaining references
        let mut group_remaining: HashMap<String, usize> = HashMap::new();

        // Upload input tensors to GPU (preserving dtype)
        for (name, tensor) in inputs {
            let gpu_tensor = match &tensor {
                Tensor::Float32(_) => {
                    let data = tensor.as_slice();
                    GpuTensor::from_host_f32(&self.ctx, &data, tensor.shape().to_vec())?
                }
                Tensor::Int64(_) => {
                    let data = tensor.as_slice_i64();
                    GpuTensor::from_host_i64(&self.ctx, &data, tensor.shape().to_vec())?
                }
                Tensor::Int32(_) => {
                    let data = tensor.as_slice_i32();
                    GpuTensor::from_host_i32(&self.ctx, &data, tensor.shape().to_vec())?
                }
                _ => {
                    // Bool and Float64 - convert to float32
                    let data = tensor.as_slice();
                    GpuTensor::from_host_f32(&self.ctx, &data, tensor.shape().to_vec())?
                }
            };
            values.insert(name, gpu_tensor);
        }

        // Note: Weights are NOT copied into values. Lookups check both maps.

        // Detect fuseable patterns (cached after first detection)
        self.detect_fused_patterns();

        // Topologically sort nodes
        let sorted_nodes = self.topological_sort()?;

        // Compute when each tensor is last used (for memory management)
        let tensor_last_use = self.compute_tensor_last_use(&sorted_nodes, &output_names);

        // Execute nodes in order
        for (node_idx, node) in sorted_nodes.iter().enumerate() {
            // Skip nodes that are part of a fused pattern (handled by pattern head)
            if self.nodes_to_skip.borrow().contains(&node.name) {
                continue;
            }

            // Check if this node is a pattern head - if so, execute fused kernel
            if let Some(pattern_info) = self.fused_patterns.borrow().get(&node.name).cloned() {
                let output = self.execute_fused_pattern(&pattern_info, &values)?;
                match &pattern_info.pattern {
                    FusedPattern::DivRsqrt { output_name, .. } => {
                        values.insert(output_name.clone(), output);
                    }
                    FusedPattern::AddMulAdd { output_name, .. } => {
                        values.insert(output_name.clone(), output);
                    }
                    FusedPattern::Gelu { output_name, .. } => {
                        values.insert(output_name.clone(), output);
                    }
                    FusedPattern::MulAdd { output_name, .. } => {
                        values.insert(output_name.clone(), output);
                    }
                    FusedPattern::AddMul { output_name, .. } => {
                        values.insert(output_name.clone(), output);
                    }
                    FusedPattern::SubMul { output_name, .. } => {
                        values.insert(output_name.clone(), output);
                    }
                    FusedPattern::DivMul { output_name, .. } => {
                        values.insert(output_name.clone(), output);
                    }
                    FusedPattern::MulSinPowMulAdd { output_name, .. } => {
                        values.insert(output_name.clone(), output);
                    }
                }
                continue;
            }

            // Handle Constant nodes specially
            if node.op_type == "Constant" {
                if let Some(value_tensor) = node.attributes.get_tensor("value") {
                    // Preserve the dtype of the constant tensor
                    let gpu_tensor = match value_tensor {
                        Tensor::Float32(_) => {
                            let data = value_tensor.as_slice();
                            GpuTensor::from_host_f32(
                                &self.ctx,
                                &data,
                                value_tensor.shape().to_vec(),
                            )?
                        }
                        Tensor::Int64(_) => {
                            let data = value_tensor.as_slice_i64();
                            let gpu = GpuTensor::from_host_i64(
                                &self.ctx,
                                &data,
                                value_tensor.shape().to_vec(),
                            )?;
                            // Cache Int64 constant by OUTPUT NAME for fast parameter retrieval
                            // This cache persists across runs (unlike device pointer cache)
                            self.cache_static_i64(&node.outputs[0], data.to_vec());
                            gpu
                        }
                        Tensor::Int32(_) => {
                            let data = value_tensor.as_slice_i32();
                            GpuTensor::from_host_i32(
                                &self.ctx,
                                &data,
                                value_tensor.shape().to_vec(),
                            )?
                        }
                        _ => {
                            // Bool and Float64 - convert to float32
                            let data = value_tensor.as_slice();
                            GpuTensor::from_host_f32(
                                &self.ctx,
                                &data,
                                value_tensor.shape().to_vec(),
                            )?
                        }
                    };
                    values.insert(node.outputs[0].clone(), gpu_tensor);
                    continue;
                } else {
                    return Err(CudaError::Kernel(format!(
                        "Constant node '{}' missing 'value' attribute",
                        node.name
                    )));
                }
            }

            // Check if we have GPU implementation
            if !Self::has_gpu_impl(&node.op_type) {
                return Err(CudaError::Kernel(format!(
                    "No GPU implementation for operator '{}' in node '{}'",
                    node.op_type, node.name
                )));
            }

            // Get input tensors (skip empty optional inputs)
            // Check values first (computed tensors), then weights (model parameters)
            let input_refs: Vec<&GpuTensor> = node
                .inputs
                .iter()
                .filter(|name| !name.is_empty())
                .map(|name| {
                    values
                        .get(name)
                        .or_else(|| self.weights.get(name))
                        .ok_or_else(|| {
                            CudaError::Kernel(format!(
                                "Missing input '{}' for node '{}' (op_type: {})",
                                name, node.name, node.op_type
                            ))
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;

            // Execute GPU operator (with precomputed slice params if available)
            let output = self
                .execute_gpu_operator_with_precomputed(
                    &node.op_type,
                    &input_refs,
                    &node.inputs,
                    &node.attributes,
                    node.precomputed_slice.as_ref(),
                )
                .map_err(|e| {
                    // Add context about which node failed
                    let input_dtypes: Vec<String> = input_refs
                        .iter()
                        .map(|t| t.dtype().name().to_string())
                        .collect();
                    CudaError::Kernel(format!(
                        "Node '{}' ({}): {} | Inputs: {} | Dtypes: {:?}",
                        node.name,
                        node.op_type,
                        e,
                        node.inputs.join(", "),
                        input_dtypes
                    ))
                })?;

            // Debug: trace ALL nodes when DEBUG_ALL_NODES is set
            if std::env::var("DEBUG_ALL_NODES").is_ok() {
                eprintln!("NODE: {} ({}) -> {:?}", node.name, node.op_type, output.shape());
            }

            // Debug: trace shapes for key nodes when DEBUG_SHAPES is set
            if std::env::var("DEBUG_SHAPES").is_ok() {
                // Match by op_type for ConvTranspose, Conv, and pattern matching for key nodes
                let is_key_node = node.op_type == "ConvTranspose"
                    || (node.op_type == "Conv" && node.name.contains("noise"))
                    || node.name.contains("Concat_3")
                    || node.name.contains("Add_2")
                    || node.name.contains("Add_8")
                    || node.name.contains("noise_res")
                    || node.name.contains("m_source")
                    || node.name.contains("generator/Pad")
                    || node.name.contains("generator/Reshape")
                    // Duration predictor path - trace sequence length calculation
                    || node.name.contains("predictor/Sigmoid")
                    || node.name.contains("predictor/ReduceSum")
                    // Duration predictor path (exact names from ort_shapes.txt)
                    || node.name == "/encoder/predictor/Sigmoid_output_0"
                    || node.name == "/encoder/predictor/ReduceSum_output_0"
                    || node.name == "/encoder/Div_output_0"
                    || node.name == "/encoder/Round_output_0"
                    || node.name == "/encoder/Clip_output_0"
                    || node.name == "/encoder/Cast_output_0"
                    || node.name == "/encoder/CumSum_output_0"
                    || node.name == "/encoder/Range_output_0"
                    // Trace ALL BERT/embedding nodes when DEBUG_BERT is set
                    || (std::env::var("DEBUG_BERT").is_ok() && (
                        node.name.contains("/encoder/bert") ||
                        node.name.contains("embedding") ||
                        node.name.contains("/encoder/text_encoder")
                    ))
                    || node.name.contains("encoder/Squeeze_1")
                    || node.name.contains("decoder/Unsqueeze_1")
                    || node.name.contains("decoder/Concat_output")
                    || node.name.contains("/encoder/Range")
                    || node.name.contains("encoder/Reshape_1")
                    // Additional nodes for sequence length debugging
                    || node.op_type == "Range"
                    || node.op_type == "Expand"
                    || node.name.contains("CumSum")
                    || node.name.contains("predictor/ReduceSum")
                    || (node.name.contains("/encoder/") && (
                        node.name.contains("Cast_") ||
                        node.name.contains("Div_") ||
                        node.name.contains("Round_") ||
                        node.name.contains("Clip_") ||
                        node.name.contains("ReduceSum")
                    ))
                    || node.name.contains("predictor/Sigmoid")
                    || node.name.contains("/encoder/predictor/")
                    || node.name == "/encoder/Div_output_0"
                    || node.name == "/encoder/Round_output_0"
                    || node.name == "/encoder/Clip_output_0"
                    || node.name == "/encoder/ReduceSum_output_0"
                    // Trace ALL encoder nodes for duration debugging
                    || (std::env::var("DEBUG_ENCODER").is_ok() && node.name.starts_with("/encoder/"));
                if is_key_node {
                    let input_shapes: Vec<_> = input_refs.iter().map(|t| format!("{:?}", t.shape())).collect();
                    eprintln!(
                        "DEBUG_SHAPES {}: {} | inputs: [{}] -> output: {:?}",
                        node.name,
                        node.op_type,
                        input_shapes.join(", "),
                        output.shape()
                    );
                }
            }

            // Debug: trace actual VALUES for duration predictor path nodes
            if std::env::var("DEBUG_DURATION_VALUES").is_ok() {
                // Match ORT node names from ort_shapes.txt:
                // /encoder/predictor/Sigmoid_output_0, /encoder/predictor/ReduceSum_output_0,
                // /encoder/Div_output_0, /encoder/Round_output_0, /encoder/Clip_output_0,
                // /encoder/Cast_output_0, /encoder/CumSum_output_0, /encoder/Range_output_0
                let is_duration_node = node.name == "/encoder/predictor/Sigmoid_output_0"
                    || node.name == "/encoder/predictor/ReduceSum_output_0"
                    || node.name == "/encoder/Div_output_0"
                    || node.name == "/encoder/Round_output_0"
                    || node.name == "/encoder/Clip_output_0"
                    || node.name == "/encoder/Cast_output_0"
                    || node.name == "/encoder/CumSum_output_0"
                    || node.name == "/encoder/Cast_1_output_0"
                    || node.name == "/encoder/Range_output_0";
                if is_duration_node {
                    match output.dtype() {
                        DType::Float32 => {
                            if let Ok(values) = output.to_host_f32(&self.ctx) {
                                let display_values: Vec<_> = values.iter().take(20).collect();
                                eprintln!(
                                    "DEBUG_DURATION_VALUES {}: shape={:?}, values={:?}{}",
                                    node.name,
                                    output.shape(),
                                    display_values,
                                    if values.len() > 20 { "..." } else { "" }
                                );
                            }
                        }
                        DType::Int64 => {
                            if let Ok(values) = output.to_host_i64(&self.ctx) {
                                let display_values: Vec<_> = values.iter().take(20).collect();
                                eprintln!(
                                    "DEBUG_DURATION_VALUES {}: shape={:?}, values={:?}{}",
                                    node.name,
                                    output.shape(),
                                    display_values,
                                    if values.len() > 20 { "..." } else { "" }
                                );
                            }
                        }
                        _ => {}
                    }
                }
            }

            // Cache Int64 outputs for downstream operations (Reshape, Expand, etc.)
            // that need to read shape values from computed tensors
            if output.dtype() == DType::Int64 {
                let values = output.to_host_i64(&self.ctx)?;
                self.cache_i64(&output, values);
            }

            // Store output(s)
            if node.outputs.len() == 1 {
                values.insert(node.outputs[0].clone(), output);
            } else {
                // Multi-output node - store same tensor for all outputs
                // Track them as a group so we only return to pool when ALL are freed
                let non_empty_outputs: Vec<_> = node.outputs.iter()
                    .filter(|name| !name.is_empty())
                    .collect();

                if !non_empty_outputs.is_empty() {
                    // Only store outputs that will actually be used:
                    // 1. Outputs that appear in tensor_last_use (used as inputs to other nodes)
                    // 2. Outputs that are graph outputs (needed for final result)
                    // Unused outputs would create Arc clones that prevent pool returns.
                    let output_name_set: std::collections::HashSet<&str> =
                        output_names.iter().copied().collect();
                    let used_outputs: Vec<_> = non_empty_outputs.iter()
                        .filter(|name| {
                            tensor_last_use.contains_key(**name) ||
                            output_name_set.contains(name.as_str())
                        })
                        .collect();

                    let count = used_outputs.len();

                    if count > 1 {
                        // Multiple used outputs - track as group so we only return
                        // to pool when ALL are freed
                        let primary = used_outputs[0].to_string();
                        group_remaining.insert(primary.clone(), count);

                        for output_name in &used_outputs {
                            tensor_groups.insert((***output_name).to_string(), primary.clone());
                            values.insert((***output_name).to_string(), output.clone());
                        }
                    } else if count == 1 {
                        // Single used output - no group tracking needed
                        values.insert((***used_outputs[0]).to_string(), output);
                    }
                    // count == 0: No used outputs, don't store the tensor at all
                }
            }

            // Memory management: Return tensors to pool when no longer needed
            // This enables memory reuse and avoids OOM for large models
            for input_name in &node.inputs {
                if !input_name.is_empty() {
                    if let Some(&last_use_idx) = tensor_last_use.get(input_name) {
                        if last_use_idx == node_idx {
                            // This node was the last consumer of this tensor
                            if let Some(tensor) = values.remove(input_name) {
                                // Check if this tensor is part of a multi-output group
                                if let Some(group) = tensor_groups.remove(input_name) {
                                    // Decrement group count
                                    if let Some(count) = group_remaining.get_mut(&group) {
                                        *count -= 1;
                                        if *count == 0 {
                                            // All group members freed - Arc refcount is now 1
                                            group_remaining.remove(&group);
                                            self.memory_pool.borrow_mut().return_tensor(tensor);
                                        }
                                        // Otherwise just drop (Arc refcount decreases but > 1)
                                    }
                                } else {
                                    // Not in a group, return directly
                                    self.memory_pool.borrow_mut().return_tensor(tensor);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Download requested outputs to CPU (preserving dtype)
        let mut outputs = HashMap::new();
        for name in output_names {
            let gpu_tensor = values
                .get(name)
                .or_else(|| self.weights.get(name))
                .ok_or_else(|| CudaError::Kernel(format!("Output '{}' not found", name)))?;
            let cpu_tensor = match gpu_tensor {
                GpuTensor::Float32 { shape, .. } => {
                    let data = gpu_tensor.to_host_f32(&self.ctx)?;
                    Tensor::from_vec_f32(data, shape.clone())
                }
                GpuTensor::Int64 { shape, .. } => {
                    let data = gpu_tensor.to_host_i64(&self.ctx)?;
                    Tensor::from_vec_i64(data, shape.clone())
                }
                GpuTensor::Int32 { shape, .. } => {
                    let data = gpu_tensor.to_host_i32(&self.ctx)?;
                    Tensor::from_vec_i32(data, shape.clone())
                }
            };
            outputs.insert(name.to_string(), cpu_tensor);
        }

        Ok(outputs)
    }

    /// Enable tensor validation for subsequent run() calls
    ///
    /// When enabled, each tensor output will be checked for NaN/Inf values
    /// and statistics will be logged. Returns error on first NaN/Inf.
    ///
    /// Only available with `debug-inference` feature flag.
    #[cfg(feature = "debug-inference")]
    pub fn enable_validation(
        &self,
        mode: super::validation::ValidationMode,
    ) -> Result<(), CudaError> {
        let validator = super::validation::TensorValidator::new(mode, &self.ctx)?;
        *self.validator.borrow_mut() = Some(validator);
        Ok(())
    }

    /// Disable tensor validation
    ///
    /// Only available with `debug-inference` feature flag.
    #[cfg(feature = "debug-inference")]
    pub fn disable_validation(&self) {
        *self.validator.borrow_mut() = None;
    }

    /// Get the validation report from the last run
    ///
    /// Only available with `debug-inference` feature flag.
    #[cfg(feature = "debug-inference")]
    pub fn validation_report(&self) -> Option<super::validation::ValidationReport> {
        self.validator
            .borrow()
            .as_ref()
            .map(|v| v.report())
    }

    /// Run inference with tensor validation
    ///
    /// Validates each tensor output for NaN/Inf values and returns a
    /// detailed report along with the outputs. Stops on first error if
    /// mode.stop_on_error is true.
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

        // Create a new validator for this run
        let mut validator = super::validation::TensorValidator::new(mode, &self.ctx)
            .map_err(ValidationError::from)?;

        // Clear caches
        self.i64_cache.borrow_mut().clear();

        // Storage for intermediate tensors
        let mut values: HashMap<String, GpuTensor> = HashMap::new();
        let mut tensor_groups: HashMap<String, String> = HashMap::new();
        let mut group_remaining: HashMap<String, usize> = HashMap::new();

        // Upload inputs
        for (name, tensor) in inputs {
            let gpu_tensor = match &tensor {
                Tensor::Float32(_) => {
                    let data = tensor.as_slice();
                    GpuTensor::from_host_f32(&self.ctx, &data, tensor.shape().to_vec())
                        .map_err(ValidationError::from)?
                }
                Tensor::Int64(_) => {
                    let data = tensor.as_slice_i64();
                    GpuTensor::from_host_i64(&self.ctx, &data, tensor.shape().to_vec())
                        .map_err(ValidationError::from)?
                }
                Tensor::Int32(_) => {
                    let data = tensor.as_slice_i32();
                    GpuTensor::from_host_i32(&self.ctx, &data, tensor.shape().to_vec())
                        .map_err(ValidationError::from)?
                }
                _ => {
                    let data = tensor.as_slice();
                    GpuTensor::from_host_f32(&self.ctx, &data, tensor.shape().to_vec())
                        .map_err(ValidationError::from)?
                }
            };
            values.insert(name, gpu_tensor);
        }

        // Detect patterns and sort nodes
        self.detect_fused_patterns();
        let sorted_nodes = self.topological_sort().map_err(ValidationError::from)?;
        let tensor_last_use = self.compute_tensor_last_use(&sorted_nodes, &output_names);

        // Execute nodes with validation
        for (node_idx, node) in sorted_nodes.iter().enumerate() {
            if self.nodes_to_skip.borrow().contains(&node.name) {
                continue;
            }

            // Handle fused patterns
            if let Some(pattern_info) = self.fused_patterns.borrow().get(&node.name).cloned() {
                let output = self
                    .execute_fused_pattern(&pattern_info, &values)
                    .map_err(ValidationError::from)?;

                let output_name = match &pattern_info.pattern {
                    FusedPattern::DivRsqrt { output_name, .. } => output_name.clone(),
                    FusedPattern::AddMulAdd { output_name, .. } => output_name.clone(),
                    FusedPattern::Gelu { output_name, .. } => output_name.clone(),
                    FusedPattern::MulAdd { output_name, .. } => output_name.clone(),
                    FusedPattern::AddMul { output_name, .. } => output_name.clone(),
                    FusedPattern::SubMul { output_name, .. } => output_name.clone(),
                    FusedPattern::DivMul { output_name, .. } => output_name.clone(),
                    FusedPattern::MulSinPowMulAdd { output_name, .. } => output_name.clone(),
                };

                // Validate fused output
                validator.validate(&output_name, "Fused", &output, &self.ctx)?;
                values.insert(output_name, output);
                continue;
            }

            // Handle Constant nodes
            if node.op_type == "Constant" {
                if let Some(value_tensor) = node.attributes.get_tensor("value") {
                    let gpu_tensor = match value_tensor {
                        Tensor::Float32(_) => {
                            let data = value_tensor.as_slice();
                            GpuTensor::from_host_f32(&self.ctx, &data, value_tensor.shape().to_vec())
                                .map_err(ValidationError::from)?
                        }
                        Tensor::Int64(_) => {
                            let data = value_tensor.as_slice_i64();
                            let gpu = GpuTensor::from_host_i64(
                                &self.ctx,
                                &data,
                                value_tensor.shape().to_vec(),
                            )
                            .map_err(ValidationError::from)?;
                            // Cache Int64 constant by OUTPUT NAME for fast parameter retrieval
                            self.cache_static_i64(&node.outputs[0], data.to_vec());
                            gpu
                        }
                        Tensor::Int32(_) => {
                            let data = value_tensor.as_slice_i32();
                            GpuTensor::from_host_i32(&self.ctx, &data, value_tensor.shape().to_vec())
                                .map_err(ValidationError::from)?
                        }
                        _ => {
                            let data = value_tensor.as_slice();
                            GpuTensor::from_host_f32(&self.ctx, &data, value_tensor.shape().to_vec())
                                .map_err(ValidationError::from)?
                        }
                    };
                    values.insert(node.outputs[0].clone(), gpu_tensor);
                    continue;
                }
            }

            // Check for GPU implementation
            if !Self::has_gpu_impl(&node.op_type) {
                return Err(ValidationError::Cuda(CudaError::Kernel(format!(
                    "No GPU implementation for operator '{}' in node '{}'",
                    node.op_type, node.name
                ))));
            }

            // Get input tensors
            let input_refs: Vec<&GpuTensor> = node
                .inputs
                .iter()
                .filter(|name| !name.is_empty())
                .map(|name| {
                    values
                        .get(name)
                        .or_else(|| self.weights.get(name))
                        .ok_or_else(|| {
                            ValidationError::Cuda(CudaError::Kernel(format!(
                                "Missing input '{}' for node '{}'",
                                name, node.name
                            )))
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;

            // Execute operator (with precomputed slice params if available)
            let output = self
                .execute_gpu_operator_with_precomputed(
                    &node.op_type,
                    &input_refs,
                    &node.inputs,
                    &node.attributes,
                    node.precomputed_slice.as_ref(),
                )
                .map_err(|e| {
                    ValidationError::Cuda(CudaError::Kernel(format!(
                        "Node '{}' ({}): {}",
                        node.name, node.op_type, e
                    )))
                })?;

            // Validate output
            validator.validate(&node.outputs[0], &node.op_type, &output, &self.ctx)?;

            // Cache Int64 outputs
            if output.dtype() == DType::Int64 {
                let vals = output.to_host_i64(&self.ctx).map_err(ValidationError::from)?;
                self.cache_i64(&output, vals);
            }

            // Store output
            if node.outputs.len() == 1 {
                values.insert(node.outputs[0].clone(), output);
            } else {
                let non_empty_outputs: Vec<_> = node.outputs.iter()
                    .filter(|name| !name.is_empty())
                    .collect();

                if !non_empty_outputs.is_empty() {
                    let output_name_set: std::collections::HashSet<&str> =
                        output_names.iter().copied().collect();
                    let used_outputs: Vec<_> = non_empty_outputs.iter()
                        .filter(|name| {
                            tensor_last_use.contains_key(**name) ||
                            output_name_set.contains(name.as_str())
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
                }
            }

            // Memory management (same as run())
            for input_name in &node.inputs {
                if !input_name.is_empty() {
                    if let Some(&last_use_idx) = tensor_last_use.get(input_name) {
                        if last_use_idx == node_idx {
                            if let Some(tensor) = values.remove(input_name) {
                                if let Some(group) = tensor_groups.remove(input_name) {
                                    if let Some(count) = group_remaining.get_mut(&group) {
                                        *count -= 1;
                                        if *count == 0 {
                                            group_remaining.remove(&group);
                                            self.memory_pool.borrow_mut().return_tensor(tensor);
                                        }
                                    }
                                } else {
                                    self.memory_pool.borrow_mut().return_tensor(tensor);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Download outputs
        let mut outputs = HashMap::new();
        for name in output_names {
            let gpu_tensor = values
                .get(name)
                .or_else(|| self.weights.get(name))
                .ok_or_else(|| {
                    ValidationError::Cuda(CudaError::Kernel(format!("Output '{}' not found", name)))
                })?;
            let cpu_tensor = match gpu_tensor {
                GpuTensor::Float32 { shape, .. } => {
                    let data = gpu_tensor.to_host_f32(&self.ctx).map_err(ValidationError::from)?;
                    Tensor::from_vec_f32(data, shape.clone())
                }
                GpuTensor::Int64 { shape, .. } => {
                    let data = gpu_tensor.to_host_i64(&self.ctx).map_err(ValidationError::from)?;
                    Tensor::from_vec_i64(data, shape.clone())
                }
                GpuTensor::Int32 { shape, .. } => {
                    let data = gpu_tensor.to_host_i32(&self.ctx).map_err(ValidationError::from)?;
                    Tensor::from_vec_i32(data, shape.clone())
                }
            };
            outputs.insert(name.to_string(), cpu_tensor);
        }

        Ok((outputs, validator.report()))
    }

    /// Run inference with checkpoint capture and ORT comparison
    ///
    /// Captures intermediate tensors and compares them against ORT reference
    /// values loaded from the fixture directory. Useful for finding where
    /// GPU execution diverges from ORT.
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
        // Create checkpoint manager and load ORT references
        let mut checkpoint_mgr = super::checkpoints::CheckpointManager::new(config);
        if checkpoint_mgr.config.compare_to_ort {
            let _count = checkpoint_mgr.load_ort_references().map_err(|e| {
                CudaError::Kernel(format!("Failed to load ORT references: {}", e))
            })?;
        }

        // Clear caches
        self.i64_cache.borrow_mut().clear();

        // Storage for intermediate tensors
        let mut values: HashMap<String, GpuTensor> = HashMap::new();
        let mut tensor_groups: HashMap<String, String> = HashMap::new();
        let mut group_remaining: HashMap<String, usize> = HashMap::new();

        // Upload inputs
        for (name, tensor) in inputs {
            let gpu_tensor = match &tensor {
                Tensor::Float32(_) => {
                    let data = tensor.as_slice();
                    GpuTensor::from_host_f32(&self.ctx, &data, tensor.shape().to_vec())?
                }
                Tensor::Int64(_) => {
                    let data = tensor.as_slice_i64();
                    GpuTensor::from_host_i64(&self.ctx, &data, tensor.shape().to_vec())?
                }
                Tensor::Int32(_) => {
                    let data = tensor.as_slice_i32();
                    GpuTensor::from_host_i32(&self.ctx, &data, tensor.shape().to_vec())?
                }
                _ => {
                    let data = tensor.as_slice();
                    GpuTensor::from_host_f32(&self.ctx, &data, tensor.shape().to_vec())?
                }
            };
            values.insert(name, gpu_tensor);
        }

        // Detect patterns and sort nodes
        self.detect_fused_patterns();
        let sorted_nodes = self.topological_sort()?;
        let tensor_last_use = self.compute_tensor_last_use(&sorted_nodes, &output_names);

        // Execute nodes with checkpoint capture
        for (node_idx, node) in sorted_nodes.iter().enumerate() {
            if self.nodes_to_skip.borrow().contains(&node.name) {
                continue;
            }

            // Handle fused patterns
            if let Some(pattern_info) = self.fused_patterns.borrow().get(&node.name).cloned() {
                let output = self.execute_fused_pattern(&pattern_info, &values)?;

                let output_name = match &pattern_info.pattern {
                    FusedPattern::DivRsqrt { output_name, .. } => output_name.clone(),
                    FusedPattern::AddMulAdd { output_name, .. } => output_name.clone(),
                    FusedPattern::Gelu { output_name, .. } => output_name.clone(),
                    FusedPattern::MulAdd { output_name, .. } => output_name.clone(),
                    FusedPattern::AddMul { output_name, .. } => output_name.clone(),
                    FusedPattern::SubMul { output_name, .. } => output_name.clone(),
                    FusedPattern::DivMul { output_name, .. } => output_name.clone(),
                    FusedPattern::MulSinPowMulAdd { output_name, .. } => output_name.clone(),
                };

                // Capture checkpoint
                let _entry = checkpoint_mgr
                    .capture(&output_name, "Fused", &output, &self.ctx)
                    .map_err(|e| CudaError::Kernel(format!("Checkpoint capture failed: {}", e)))?;

                values.insert(output_name, output);
                continue;
            }

            // Handle Constant nodes
            if node.op_type == "Constant" {
                if let Some(value_tensor) = node.attributes.get_tensor("value") {
                    let gpu_tensor = match value_tensor {
                        Tensor::Float32(_) => {
                            let data = value_tensor.as_slice();
                            GpuTensor::from_host_f32(&self.ctx, &data, value_tensor.shape().to_vec())?
                        }
                        Tensor::Int64(_) => {
                            let data = value_tensor.as_slice_i64();
                            let gpu = GpuTensor::from_host_i64(
                                &self.ctx,
                                &data,
                                value_tensor.shape().to_vec(),
                            )?;
                            // Cache Int64 constant by OUTPUT NAME for fast parameter retrieval
                            self.cache_static_i64(&node.outputs[0], data.to_vec());
                            gpu
                        }
                        Tensor::Int32(_) => {
                            let data = value_tensor.as_slice_i32();
                            GpuTensor::from_host_i32(&self.ctx, &data, value_tensor.shape().to_vec())?
                        }
                        _ => {
                            let data = value_tensor.as_slice();
                            GpuTensor::from_host_f32(&self.ctx, &data, value_tensor.shape().to_vec())?
                        }
                    };
                    values.insert(node.outputs[0].clone(), gpu_tensor);
                    continue;
                }
            }

            // Check for GPU implementation
            if !Self::has_gpu_impl(&node.op_type) {
                return Err(CudaError::Kernel(format!(
                    "No GPU implementation for operator '{}' in node '{}'",
                    node.op_type, node.name
                )));
            }

            // Get input tensors
            let input_refs: Vec<&GpuTensor> = node
                .inputs
                .iter()
                .filter(|name| !name.is_empty())
                .map(|name| {
                    values
                        .get(name)
                        .or_else(|| self.weights.get(name))
                        .ok_or_else(|| {
                            CudaError::Kernel(format!(
                                "Missing input '{}' for node '{}'",
                                name, node.name
                            ))
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;

            // Execute operator (with precomputed slice params if available)
            let output = self
                .execute_gpu_operator_with_precomputed(
                    &node.op_type,
                    &input_refs,
                    &node.inputs,
                    &node.attributes,
                    node.precomputed_slice.as_ref(),
                )
                .map_err(|e| {
                    CudaError::Kernel(format!(
                        "Node '{}' ({}): {}",
                        node.name, node.op_type, e
                    ))
                })?;

            // Debug: trace ALL nodes when DEBUG_ALL_NODES is set
            if std::env::var("DEBUG_ALL_NODES").is_ok() {
                // For Slice operations in bert path, also show input names
                if node.op_type == "Slice" && node.name.contains("bert") {
                    let input_shapes: Vec<_> = input_refs.iter().map(|t| format!("{:?}", t.shape())).collect();
                    eprintln!(
                        "NODE: {} ({}) inputs={:?} shapes=[{}] -> {:?}",
                        node.name, node.op_type, node.inputs, input_shapes.join(", "), output.shape()
                    );
                } else {
                    eprintln!("NODE: {} ({}) -> {:?}", node.name, node.op_type, output.shape());
                }

                // Detailed trace for key predictor nodes
                if node.name == "/encoder/predictor/Sigmoid_output_0" {
                    if let Ok(vals) = output.to_host_f32(&self.ctx) {
                        let sum_per_phoneme: Vec<f32> = (0..5).map(|p| {
                            vals[p * 50..(p + 1) * 50].iter().sum::<f32>()
                        }).collect();
                        eprintln!("  => Sigmoid sum per phoneme: {:?}", sum_per_phoneme);
                        eprintln!("  => Sigmoid min: {:.6}, max: {:.6}, mean: {:.6}",
                            vals.iter().cloned().fold(f32::INFINITY, f32::min),
                            vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
                            vals.iter().sum::<f32>() / vals.len() as f32);
                    }
                }

                // Trace predictor and text_encoder LSTM outputs
                if (node.name.contains("predictor") || node.name.contains("text_encoder/lstm")) && node.name.contains("LSTM_output") {
                    if let Ok(vals) = output.to_host_f32(&self.ctx) {
                        let min = vals.iter().cloned().fold(f32::INFINITY, f32::min);
                        let max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let mean = vals.iter().sum::<f32>() / vals.len() as f32;
                        eprintln!("  => {} min: {:.6}, max: {:.6}, mean: {:.6}", node.name, min, max, mean);
                    }
                }

                // Trace encoder/decoder Slice outputs and predictor Gemm
                if node.name == "/encoder/Slice_output_0"
                    || node.name == "/decoder/Slice_output_0"
                    || node.name == "/encoder/predictor/text_encoder/lstms.1/fc/Gemm_output_0"
                    || node.name.contains("ConstantOfShape") {
                    if let Ok(vals) = output.to_host_f32(&self.ctx) {
                        let min = vals.iter().cloned().fold(f32::INFINITY, f32::min);
                        let max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let mean = vals.iter().sum::<f32>() / vals.len() as f32;
                        eprintln!("  => {} min: {:.6}, max: {:.6}, mean: {:.6}", node.name, min, max, mean);
                    }
                }

                // Trace LSTM inputs for lstms.0
                if node.name == "/encoder/predictor/text_encoder/lstms.0/Transpose_output_0" {
                    if let Ok(vals) = output.to_host_f32(&self.ctx) {
                        let min = vals.iter().cloned().fold(f32::INFINITY, f32::min);
                        let max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let mean = vals.iter().sum::<f32>() / vals.len() as f32;
                        eprintln!("  => LSTM.0 input: min={:.6}, max={:.6}, mean={:.6}", min, max, mean);
                    }
                }

                if node.name.contains("word_embeddings/Gather") || node.name.contains("text_encoder/embedding/Gather") {
                    if let Ok(vals) = output.to_host_f32(&self.ctx) {
                        let min = vals.iter().cloned().fold(f32::INFINITY, f32::min);
                        let max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let mean = vals.iter().sum::<f32>() / vals.len() as f32;
                        eprintln!("  => {} min: {:.6}, max: {:.6}, mean: {:.6}", node.name, min, max, mean);
                    }
                }
                // Trace Unsqueeze_1 value (this is the slice 'ends' parameter)
                if node.name == "/encoder/bert/Unsqueeze_1_output_0" {
                    if let Ok(vals) = output.to_host_i64(&self.ctx) {
                        eprintln!("  => bert/Unsqueeze_1 value = {:?}", vals);
                    }
                }
                // Trace bert/Gather_1 which produces the sequence length
                if node.name == "/encoder/bert/Gather_1_output_0" {
                    if let Ok(vals) = output.to_host_i64(&self.ctx) {
                        eprintln!("  => bert/Gather_1 value = {:?}", vals);
                    }
                }
                // Trace embedding Gather inputs
                if node.name.contains("word_embeddings/Gather") || node.name.contains("text_encoder/embedding/Gather") {
                    eprintln!("  => {} inputs: {:?}", node.name, node.inputs);
                    for (i, input_name) in node.inputs.iter().enumerate() {
                        if let Some(tensor) = values.get(input_name) {
                            if tensor.dtype() == DType::Int64 {
                                if let Ok(vals) = tensor.to_host_i64(&self.ctx) {
                                    eprintln!("    input[{}] '{}' = {:?}", i, input_name, vals);
                                }
                            } else {
                                eprintln!("    input[{}] '{}' shape = {:?}", i, input_name, tensor.shape());
                            }
                        }
                    }
                }
            }

            // Debug: trace shapes for key nodes when DEBUG_SHAPES is set
            if std::env::var("DEBUG_SHAPES").is_ok() {
                let is_key_node = node.op_type == "ConvTranspose"
                    || (node.op_type == "Conv" && node.name.contains("noise"))
                    || node.name.contains("Concat_3")
                    || node.name.contains("Add_2")
                    || node.name.contains("Add_8")
                    || node.name.contains("noise_res")
                    || node.name.contains("m_source")
                    || node.name.contains("generator/Pad")
                    || node.name.contains("generator/Reshape")
                    // BERT/embedding/text_encoder when DEBUG_BERT is set
                    || (std::env::var("DEBUG_BERT").is_ok() && (
                        node.name.contains("/encoder/bert") ||
                        node.name.contains("embedding") ||
                        node.name.contains("/encoder/text_encoder")
                    ));
                if is_key_node {
                    let input_shapes: Vec<_> = input_refs.iter().map(|t| format!("{:?}", t.shape())).collect();
                    eprintln!(
                        "DEBUG_SHAPES {}: {} | inputs: [{}] -> output: {:?}",
                        node.name,
                        node.op_type,
                        input_shapes.join(", "),
                        output.shape()
                    );
                }
            }

            // Capture checkpoint
            let _entry = checkpoint_mgr
                .capture(&node.outputs[0], &node.op_type, &output, &self.ctx)
                .map_err(|e| CudaError::Kernel(format!("Checkpoint capture failed: {}", e)))?;

            // Cache Int64 outputs
            if output.dtype() == DType::Int64 {
                let vals = output.to_host_i64(&self.ctx)?;
                self.cache_i64(&output, vals);
            }

            // Store output
            if node.outputs.len() == 1 {
                values.insert(node.outputs[0].clone(), output);
            } else {
                let non_empty_outputs: Vec<_> = node.outputs.iter()
                    .filter(|name| !name.is_empty())
                    .collect();

                if !non_empty_outputs.is_empty() {
                    let output_name_set: std::collections::HashSet<&str> =
                        output_names.iter().copied().collect();
                    let used_outputs: Vec<_> = non_empty_outputs.iter()
                        .filter(|name| {
                            tensor_last_use.contains_key(**name) ||
                            output_name_set.contains(name.as_str())
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
                }
            }

            // Memory management
            for input_name in &node.inputs {
                if !input_name.is_empty() {
                    if let Some(&last_use_idx) = tensor_last_use.get(input_name) {
                        if last_use_idx == node_idx {
                            if let Some(tensor) = values.remove(input_name) {
                                if let Some(group) = tensor_groups.remove(input_name) {
                                    if let Some(count) = group_remaining.get_mut(&group) {
                                        *count -= 1;
                                        if *count == 0 {
                                            group_remaining.remove(&group);
                                            self.memory_pool.borrow_mut().return_tensor(tensor);
                                        }
                                    }
                                } else {
                                    self.memory_pool.borrow_mut().return_tensor(tensor);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Download outputs
        let mut outputs = HashMap::new();
        for name in output_names {
            let gpu_tensor = values
                .get(name)
                .or_else(|| self.weights.get(name))
                .ok_or_else(|| CudaError::Kernel(format!("Output '{}' not found", name)))?;
            let cpu_tensor = match gpu_tensor {
                GpuTensor::Float32 { shape, .. } => {
                    let data = gpu_tensor.to_host_f32(&self.ctx)?;
                    Tensor::from_vec_f32(data, shape.clone())
                }
                GpuTensor::Int64 { shape, .. } => {
                    let data = gpu_tensor.to_host_i64(&self.ctx)?;
                    Tensor::from_vec_i64(data, shape.clone())
                }
                GpuTensor::Int32 { shape, .. } => {
                    let data = gpu_tensor.to_host_i32(&self.ctx)?;
                    Tensor::from_vec_i32(data, shape.clone())
                }
            };
            outputs.insert(name.to_string(), cpu_tensor);
        }

        Ok((outputs, checkpoint_mgr))
    }

    /// Execute with logging of tensor shapes at each node
    ///
    /// Used for debugging shape mismatches by comparing against ORT execution.
    /// Returns both the outputs and a log of each node's execution.
    pub fn run_with_logging(
        &self,
        inputs: HashMap<String, Tensor>,
        output_names: Vec<&str>,
    ) -> Result<(HashMap<String, Tensor>, Vec<NodeExecutionLog>), CudaError> {
        // Clear the i64 cache before each run (see run() for explanation)
        self.i64_cache.borrow_mut().clear();

        let mut execution_log: Vec<NodeExecutionLog> = Vec::new();
        let mut values: HashMap<String, GpuTensor> = HashMap::new();

        // Track tensor groups for multi-output nodes
        let mut tensor_groups: HashMap<String, String> = HashMap::new();
        let mut group_remaining: HashMap<String, usize> = HashMap::new();

        // Upload inputs
        for (name, tensor) in inputs {
            let gpu_tensor = match &tensor {
                Tensor::Float32(_) => {
                    let data = tensor.as_slice();
                    GpuTensor::from_host_f32(&self.ctx, &data, tensor.shape().to_vec())?
                }
                Tensor::Int64(_) => {
                    let data = tensor.as_slice_i64();
                    GpuTensor::from_host_i64(&self.ctx, &data, tensor.shape().to_vec())?
                }
                Tensor::Int32(_) => {
                    let data = tensor.as_slice_i32();
                    GpuTensor::from_host_i32(&self.ctx, &data, tensor.shape().to_vec())?
                }
                _ => {
                    let data = tensor.as_slice();
                    GpuTensor::from_host_f32(&self.ctx, &data, tensor.shape().to_vec())?
                }
            };
            // Log input tensor
            execution_log.push(NodeExecutionLog {
                node_name: format!("INPUT:{}", name),
                op_type: "Input".to_string(),
                input_shapes: vec![],
                input_dtypes: vec![],
                output_shape: gpu_tensor.shape().to_vec(),
                output_dtype: gpu_tensor.dtype().name().to_string(),
                has_nan: false, // Input tensors don't have NaN
            });
            values.insert(name, gpu_tensor);
        }

        // Get sorted nodes
        let sorted_nodes = self.topological_sort()?;
        let tensor_last_use = self.compute_tensor_last_use(&sorted_nodes, &output_names);

        // Execute nodes
        for (node_idx, node) in sorted_nodes.iter().enumerate() {
            // Skip Constant nodes for logging (they're handled separately)
            if node.op_type == "Constant" {
                if let Some(value_tensor) = node.attributes.get_tensor("value") {
                    let gpu_tensor = match value_tensor {
                        Tensor::Float32(_) => {
                            let data = value_tensor.as_slice();
                            GpuTensor::from_host_f32(
                                &self.ctx,
                                &data,
                                value_tensor.shape().to_vec(),
                            )?
                        }
                        Tensor::Int64(_) => {
                            let data = value_tensor.as_slice_i64();
                            let gpu = GpuTensor::from_host_i64(
                                &self.ctx,
                                &data,
                                value_tensor.shape().to_vec(),
                            )?;
                            // Cache Int64 constant by OUTPUT NAME for fast parameter retrieval
                            self.cache_static_i64(&node.outputs[0], data.to_vec());
                            gpu
                        }
                        Tensor::Int32(_) => {
                            let data = value_tensor.as_slice_i32();
                            GpuTensor::from_host_i32(
                                &self.ctx,
                                &data,
                                value_tensor.shape().to_vec(),
                            )?
                        }
                        _ => {
                            let data = value_tensor.as_slice();
                            GpuTensor::from_host_f32(
                                &self.ctx,
                                &data,
                                value_tensor.shape().to_vec(),
                            )?
                        }
                    };
                    if !node.outputs.is_empty() {
                        values.insert(node.outputs[0].clone(), gpu_tensor);
                    }
                }
                continue;
            }

            // Get input tensors and record shapes
            let mut input_shapes: Vec<Vec<usize>> = Vec::new();
            let mut input_dtypes: Vec<String> = Vec::new();
            let input_refs: Vec<&GpuTensor> = node
                .inputs
                .iter()
                .filter(|name| !name.is_empty())
                .map(|name| {
                    values
                        .get(name)
                        .or_else(|| self.weights.get(name))
                        .ok_or_else(|| {
                            CudaError::Kernel(format!(
                                "Missing input '{}' for node '{}'",
                                name, node.name
                            ))
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;

            for tensor in &input_refs {
                input_shapes.push(tensor.shape().to_vec());
                input_dtypes.push(tensor.dtype().name().to_string());
            }

            // Execute operator (with precomputed slice params if available)
            let output = self
                .execute_gpu_operator_with_precomputed(
                    &node.op_type,
                    &input_refs,
                    &node.inputs,
                    &node.attributes,
                    node.precomputed_slice.as_ref(),
                )
                .map_err(|e| {
                    CudaError::Kernel(format!(
                        "Node '{}' ({}): {} | Input shapes: {:?}",
                        node.name, node.op_type, e, input_shapes
                    ))
                })?;

            // Cache Int64 outputs for downstream operations (Reshape, Expand, etc.)
            if output.dtype() == DType::Int64 {
                let cached_values = output.to_host_i64(&self.ctx)?;
                self.cache_i64(&output, cached_values);
            }

            // Check for NaN values (only for float tensors, to find where NaN is introduced)
            let (has_nan, has_all_zeros, min_val, max_val) = if output.dtype() == DType::Float32 {
                let data = output.to_host_f32(&self.ctx)?;
                let nan = data.iter().any(|x| x.is_nan());
                let zeros = data.iter().all(|&x| x.abs() < 1e-30);
                let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                (nan, zeros, min, max)
            } else {
                (false, false, 0.0, 0.0)
            };

            // Print diagnostic for operators near where NaN might occur
            if node.op_type == "STFT"
                || has_nan
                || has_all_zeros
                || node.name.contains("/generator/Gather")
                || node.name.contains("/generator/Pow")
                || node.name.contains("/generator/Div")
                || node.name.contains("/generator/Slice")
                || node.name.contains("/generator/Exp")
                || node.name.contains("/generator/Sin")
                || node.name.contains("/generator/conv_post")
                || node.name.contains("/istft/")
            {
                eprintln!(
                    "DEBUG {} ({}): has_nan={}, all_zeros={}, range=[{:.6e}, {:.6e}], shape={:?}",
                    node.name,
                    node.op_type,
                    has_nan,
                    has_all_zeros,
                    min_val,
                    max_val,
                    output.shape()
                );
                // For Div ops, also print input info
                if node.op_type == "Div" && node.inputs.len() >= 2 {
                    eprintln!("  Div inputs: '{}' / '{}'", node.inputs[0], node.inputs[1]);
                }
                // For Slice/Gather ops in iSTFT, print more detail
                if (node.op_type == "Slice" || node.op_type == "Gather")
                    && node.name.contains("/istft/")
                {
                    eprintln!("  {} inputs: {:?}", node.op_type, node.inputs);
                    // Get shapes of inputs
                    for (i, input_name) in node.inputs.iter().enumerate() {
                        if let Some(tensor) = values.get(input_name) {
                            // For Float32, also show range
                            if tensor.dtype() == DType::Float32 {
                                if let Ok(data) = tensor.to_host_f32(&self.ctx) {
                                    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
                                    let max =
                                        data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                                    eprintln!("    input[{}] '{}' shape={:?} dtype={} range=[{:.6}, {:.6}]",
                                        i, input_name, tensor.shape(), tensor.dtype().name(), min, max);
                                }
                            } else {
                                eprintln!(
                                    "    input[{}] '{}' shape={:?} dtype={}",
                                    i,
                                    input_name,
                                    tensor.shape(),
                                    tensor.dtype().name()
                                );
                            }
                        }
                    }
                }
                // For Cos/Sin ops, print input details to trace phase source
                if (node.op_type == "Cos" || node.op_type == "Sin") && !node.inputs.is_empty() {
                    eprintln!("  {} input: '{}'", node.op_type, node.inputs[0]);
                    if let Some(tensor) = values.get(&node.inputs[0]) {
                        if tensor.dtype() == DType::Float32 {
                            if let Ok(data) = tensor.to_host_f32(&self.ctx) {
                                let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
                                let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                                eprintln!(
                                    "    input '{}' shape={:?} range=[{:.6}, {:.6}]",
                                    node.inputs[0],
                                    tensor.shape(),
                                    min,
                                    max
                                );
                            }
                        }
                    }
                }
                // For ConvTranspose in iSTFT, print detailed inputs
                if node.op_type == "ConvTranspose" && node.name.contains("/istft/") {
                    eprintln!("  ConvTranspose inputs: {:?}", node.inputs);
                    for (i, input_name) in node.inputs.iter().enumerate() {
                        if let Some(tensor) = values.get(input_name) {
                            if tensor.dtype() == DType::Float32 {
                                if let Ok(data) = tensor.to_host_f32(&self.ctx) {
                                    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
                                    let max =
                                        data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                                    let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
                                    let sum: f32 = data.iter().sum();
                                    eprintln!("    input[{}] '{}' shape={:?} range=[{:.6}, {:.6}] mean={:.6} sum={:.6}",
                                        i, input_name, tensor.shape(), min, max, mean, sum);
                                }
                            }
                        } else if let Some(weight) = self.weights.get(input_name) {
                            if weight.dtype() == DType::Float32 {
                                if let Ok(data) = weight.to_host_f32(&self.ctx) {
                                    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
                                    let max =
                                        data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                                    let sum: f32 = data.iter().sum();
                                    eprintln!("    input[{}] '{}' (weight) shape={:?} range=[{:.6}, {:.6}] sum={:.6}",
                                        i, input_name, weight.shape(), min, max, sum);
                                }
                            }
                        }
                    }
                }
                // For Mul ops in iSTFT (especially Mul_4), print inputs
                if node.op_type == "Mul" && node.name.contains("Mul_4") {
                    eprintln!("  Mul_4 inputs: {:?}", node.inputs);
                    for (i, input_name) in node.inputs.iter().enumerate() {
                        if let Some(tensor) = values.get(input_name) {
                            if tensor.dtype() == DType::Float32 {
                                if let Ok(data) = tensor.to_host_f32(&self.ctx) {
                                    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
                                    let max =
                                        data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                                    eprintln!(
                                        "    input[{}] '{}' shape={:?} range=[{:.6}, {:.6}]",
                                        i,
                                        input_name,
                                        tensor.shape(),
                                        min,
                                        max
                                    );
                                    // If it's a scalar or small tensor, print all values
                                    if data.len() <= 10 {
                                        eprintln!("      values: {:?}", data);
                                    }
                                }
                            }
                        } else if let Some(weight) = self.weights.get(input_name) {
                            // Check if it's a weight (constant)
                            if weight.dtype() == DType::Float32 {
                                if let Ok(data) = weight.to_host_f32(&self.ctx) {
                                    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
                                    let max =
                                        data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                                    eprintln!("    input[{}] '{}' (weight) shape={:?} range=[{:.6}, {:.6}]",
                                        i, input_name, weight.shape(), min, max);
                                    if data.len() <= 10 {
                                        eprintln!("      values: {:?}", data);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Log this node
            execution_log.push(NodeExecutionLog {
                node_name: node.name.clone(),
                op_type: node.op_type.clone(),
                input_shapes,
                input_dtypes,
                output_shape: output.shape().to_vec(),
                output_dtype: output.dtype().name().to_string(),
                has_nan, // Track NaN presence
            });

            // Store output
            if node.outputs.len() == 1 {
                values.insert(node.outputs[0].clone(), output);
            } else {
                // Multi-output node - track as a group for proper pool return
                let non_empty_outputs: Vec<_> = node.outputs.iter()
                    .filter(|name| !name.is_empty())
                    .collect();

                if !non_empty_outputs.is_empty() {
                    // Only store outputs that will actually be used
                    let output_name_set: std::collections::HashSet<&str> =
                        output_names.iter().copied().collect();
                    let used_outputs: Vec<_> = non_empty_outputs.iter()
                        .filter(|name| {
                            tensor_last_use.contains_key(**name) ||
                            output_name_set.contains(name.as_str())
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
                }
            }

            // Memory management: Return tensors to pool when no longer needed
            for input_name in &node.inputs {
                if !input_name.is_empty() {
                    if let Some(&last_use_idx) = tensor_last_use.get(input_name) {
                        if last_use_idx == node_idx {
                            if let Some(tensor) = values.remove(input_name) {
                                // Check if part of a multi-output group
                                if let Some(group) = tensor_groups.remove(input_name) {
                                    if let Some(count) = group_remaining.get_mut(&group) {
                                        *count -= 1;
                                        if *count == 0 {
                                            group_remaining.remove(&group);
                                            self.memory_pool.borrow_mut().return_tensor(tensor);
                                        }
                                    }
                                } else {
                                    self.memory_pool.borrow_mut().return_tensor(tensor);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Download outputs
        let mut outputs = HashMap::new();
        for name in output_names {
            let gpu_tensor = values
                .get(name)
                .or_else(|| self.weights.get(name))
                .ok_or_else(|| CudaError::Kernel(format!("Output '{}' not found", name)))?;
            let cpu_tensor = match gpu_tensor {
                GpuTensor::Float32 { shape, .. } => {
                    let data = gpu_tensor.to_host_f32(&self.ctx)?;
                    Tensor::from_vec_f32(data, shape.clone())
                }
                GpuTensor::Int64 { shape, .. } => {
                    let data = gpu_tensor.to_host_i64(&self.ctx)?;
                    Tensor::from_vec_i64(data, shape.clone())
                }
                GpuTensor::Int32 { shape, .. } => {
                    let data = gpu_tensor.to_host_i32(&self.ctx)?;
                    Tensor::from_vec_i32(data, shape.clone())
                }
            };
            outputs.insert(name.to_string(), cpu_tensor);
        }

        Ok((outputs, execution_log))
    }

    /// Execute the full computation graph on GPU with detailed profiling
    ///
    /// Returns timing breakdown in addition to outputs
    pub fn run_with_profiling(
        &self,
        inputs: HashMap<String, Tensor>,
        output_names: Vec<&str>,
    ) -> Result<(HashMap<String, Tensor>, ProfileData), CudaError> {
        use std::collections::BTreeMap;
        use std::time::Instant;

        // Clear the i64 cache before each run (see run() for explanation)
        self.i64_cache.borrow_mut().clear();

        let mut profile = ProfileData::default();
        let total_start = Instant::now();

        // Storage for intermediate tensors on GPU
        let mut values: HashMap<String, GpuTensor> = HashMap::new();

        // Track tensor groups for multi-output nodes
        let mut tensor_groups: HashMap<String, String> = HashMap::new();
        let mut group_remaining: HashMap<String, usize> = HashMap::new();

        // Upload input tensors to GPU
        let upload_start = Instant::now();
        for (name, tensor) in inputs {
            let gpu_tensor = match &tensor {
                Tensor::Float32(_) => {
                    let data = tensor.as_slice();
                    GpuTensor::from_host_f32(&self.ctx, &data, tensor.shape().to_vec())?
                }
                Tensor::Int64(_) => {
                    let data = tensor.as_slice_i64();
                    GpuTensor::from_host_i64(&self.ctx, &data, tensor.shape().to_vec())?
                }
                Tensor::Int32(_) => {
                    let data = tensor.as_slice_i32();
                    GpuTensor::from_host_i32(&self.ctx, &data, tensor.shape().to_vec())?
                }
                _ => {
                    let data = tensor.as_slice();
                    GpuTensor::from_host_f32(&self.ctx, &data, tensor.shape().to_vec())?
                }
            };
            values.insert(name, gpu_tensor);
        }
        profile.input_upload_us = upload_start.elapsed().as_micros() as u64;

        // Note: Weights are NOT copied into values. Lookups check both maps.
        // This eliminates ~4.5ms of HashMap insertion overhead.
        profile.weight_ref_us = 0;

        // Detect fuseable patterns (cached after first detection)
        self.detect_fused_patterns();

        // Topological sort
        let sort_start = Instant::now();
        let sorted_nodes = self.topological_sort()?;
        profile.topo_sort_us = sort_start.elapsed().as_micros() as u64;

        // Execute nodes with per-op-type timing
        let mut op_times: BTreeMap<String, (u64, usize)> = BTreeMap::new();
        // GPU-side timing via CUDA events. Allocates events fresh per run.
        let mut gpu_timer = GpuEventTimer::new(
            self.ctx.context().clone(),
            self.ctx.stream(),
            sorted_nodes.len(),
        )?;
        let exec_start = Instant::now();

        for node in sorted_nodes {
            // Skip nodes that are part of a fused pattern (handled by pattern head)
            if self.nodes_to_skip.borrow().contains(&node.name) {
                continue;
            }

            // Check if this node is a pattern head - if so, execute fused kernel
            if let Some(pattern_info) = self.fused_patterns.borrow().get(&node.name).cloned() {
                let op_start = Instant::now();
                let output = self.execute_fused_pattern(&pattern_info, &values)?;
                let op_us = op_start.elapsed().as_micros() as u64;

                // Track as "Fused_<pattern_type>" in profiler
                let pattern_name = match &pattern_info.pattern {
                    FusedPattern::DivRsqrt { output_name, .. } => {
                        values.insert(output_name.clone(), output);
                        "Fused_DivRsqrt"
                    }
                    FusedPattern::AddMulAdd { output_name, .. } => {
                        values.insert(output_name.clone(), output);
                        "Fused_AddMulAdd"
                    }
                    FusedPattern::Gelu { output_name, .. } => {
                        values.insert(output_name.clone(), output);
                        "Fused_Gelu"
                    }
                    FusedPattern::MulAdd { output_name, .. } => {
                        values.insert(output_name.clone(), output);
                        "Fused_MulAdd"
                    }
                    FusedPattern::AddMul { output_name, .. } => {
                        values.insert(output_name.clone(), output);
                        "Fused_AddMul"
                    }
                    FusedPattern::SubMul { output_name, .. } => {
                        values.insert(output_name.clone(), output);
                        "Fused_SubMul"
                    }
                    FusedPattern::DivMul { output_name, .. } => {
                        values.insert(output_name.clone(), output);
                        "Fused_DivMul"
                    }
                    FusedPattern::MulSinPowMulAdd { output_name, .. } => {
                        values.insert(output_name.clone(), output);
                        "Fused_MulSinPowMulAdd"
                    }
                };

                let entry = op_times.entry(pattern_name.to_string()).or_insert((0, 0));
                entry.0 += op_us;
                entry.1 += 1;
                gpu_timer.record(pattern_name.to_string())?;

                continue;
            }
            if node.op_type == "Constant" {
                if let Some(value_tensor) = node.attributes.get_tensor("value") {
                    let gpu_tensor = match value_tensor {
                        Tensor::Float32(_) => {
                            let data = value_tensor.as_slice();
                            GpuTensor::from_host_f32(
                                &self.ctx,
                                &data,
                                value_tensor.shape().to_vec(),
                            )?
                        }
                        Tensor::Int64(_) => {
                            let data = value_tensor.as_slice_i64();
                            let gpu = GpuTensor::from_host_i64(
                                &self.ctx,
                                &data,
                                value_tensor.shape().to_vec(),
                            )?;
                            // Cache Int64 constant by OUTPUT NAME for fast parameter retrieval
                            // This cache persists across runs (unlike device pointer cache)
                            self.cache_static_i64(&node.outputs[0], data.to_vec());
                            gpu
                        }
                        Tensor::Int32(_) => {
                            let data = value_tensor.as_slice_i32();
                            GpuTensor::from_host_i32(
                                &self.ctx,
                                &data,
                                value_tensor.shape().to_vec(),
                            )?
                        }
                        _ => {
                            let data = value_tensor.as_slice();
                            GpuTensor::from_host_f32(
                                &self.ctx,
                                &data,
                                value_tensor.shape().to_vec(),
                            )?
                        }
                    };
                    values.insert(node.outputs[0].clone(), gpu_tensor);
                    continue;
                }
            }

            if !Self::has_gpu_impl(&node.op_type) {
                return Err(CudaError::Kernel(format!(
                    "No GPU implementation for operator '{}'",
                    node.op_type
                )));
            }

            // Check values first (computed tensors), then weights (model parameters)
            let input_refs: Vec<&GpuTensor> = node
                .inputs
                .iter()
                .filter(|name| !name.is_empty())
                .map(|name| {
                    values
                        .get(name)
                        .or_else(|| self.weights.get(name))
                        .ok_or_else(|| {
                            CudaError::Kernel(format!(
                                "Missing input '{}' for node '{}'",
                                name, node.name
                            ))
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;

            // Time the operation (async - measures enqueue time, not execution time)
            let op_start = Instant::now();
            let output = self.execute_gpu_operator_with_precomputed(
                &node.op_type,
                &input_refs,
                &node.inputs,
                &node.attributes,
                node.precomputed_slice.as_ref(),
            )?;
            let op_us = op_start.elapsed().as_micros() as u64;

            // Track per-op-type timing
            let entry = op_times.entry(node.op_type.clone()).or_insert((0, 0));
            entry.0 += op_us;
            entry.1 += 1;
            gpu_timer.record(node.op_type.clone())?;

            if node.outputs.len() == 1 {
                values.insert(node.outputs[0].clone(), output);
            } else {
                // Multi-output node - track as group for consistent API
                let non_empty_outputs: Vec<_> = node.outputs.iter()
                    .filter(|name| !name.is_empty())
                    .collect();

                if !non_empty_outputs.is_empty() {
                    let primary = non_empty_outputs[0].to_string();
                    let count = non_empty_outputs.len();
                    group_remaining.insert(primary.clone(), count);

                    for output_name in &non_empty_outputs {
                        tensor_groups.insert((*output_name).to_string(), primary.clone());
                        values.insert((*output_name).to_string(), output.clone());
                    }
                }
            }
        }
        // Suppress unused warnings (no memory management in profiling mode)
        let _ = (&tensor_groups, &group_remaining);
        profile.node_execution_us = exec_start.elapsed().as_micros() as u64;
        profile.op_times = op_times;
        profile.gpu_op_times = gpu_timer.finalize()?;

        // Download outputs
        let download_start = Instant::now();
        let mut outputs = HashMap::new();
        for name in output_names {
            let gpu_tensor = values
                .get(name)
                .or_else(|| self.weights.get(name))
                .ok_or_else(|| CudaError::Kernel(format!("Output '{}' not found", name)))?;
            let cpu_tensor = match gpu_tensor {
                GpuTensor::Float32 { shape, .. } => {
                    let data = gpu_tensor.to_host_f32(&self.ctx)?;
                    Tensor::from_vec_f32(data, shape.clone())
                }
                GpuTensor::Int64 { shape, .. } => {
                    let data = gpu_tensor.to_host_i64(&self.ctx)?;
                    Tensor::from_vec_i64(data, shape.clone())
                }
                GpuTensor::Int32 { shape, .. } => {
                    let data = gpu_tensor.to_host_i32(&self.ctx)?;
                    Tensor::from_vec_i32(data, shape.clone())
                }
            };
            outputs.insert(name.to_string(), cpu_tensor);
        }
        profile.output_download_us = download_start.elapsed().as_micros() as u64;
        profile.total_us = total_start.elapsed().as_micros() as u64;

        Ok((outputs, profile))
    }

    /// Topologically sort nodes for execution order (with caching)
    fn topological_sort(&self) -> Result<Vec<ExecutionNode>, CudaError> {
        // Check cache first
        if let Some(ref cached) = *self.sorted_nodes_cache.borrow() {
            return Ok(cached.clone());
        }

        // Compute topo sort
        let sorted = self.compute_topological_sort()?;

        // Store in cache
        *self.sorted_nodes_cache.borrow_mut() = Some(sorted.clone());

        Ok(sorted)
    }

    /// Actually compute the topological sort (internal helper)
    fn compute_topological_sort(&self) -> Result<Vec<ExecutionNode>, CudaError> {
        use std::collections::HashSet;

        let mut sorted = Vec::new();
        let mut visited = HashSet::new();
        let mut temp_mark = HashSet::new();

        // Build name -> node map
        let node_map: HashMap<String, &ExecutionNode> =
            self.nodes.iter().map(|n| (n.name.clone(), n)).collect();

        // Build output -> producer node map
        let output_producers: HashMap<String, &str> = self
            .nodes
            .iter()
            .flat_map(|n| n.outputs.iter().map(|o| (o.clone(), n.name.as_str())))
            .collect();

        fn visit<'a>(
            node: &'a ExecutionNode,
            node_map: &HashMap<String, &'a ExecutionNode>,
            output_producers: &HashMap<String, &str>,
            visited: &mut HashSet<String>,
            temp_mark: &mut HashSet<String>,
            sorted: &mut Vec<ExecutionNode>,
            weights: &HashSet<String>,
        ) -> Result<(), CudaError> {
            if visited.contains(&node.name) {
                return Ok(());
            }
            if temp_mark.contains(&node.name) {
                return Err(CudaError::Kernel(format!(
                    "Cycle detected at node '{}'",
                    node.name
                )));
            }

            temp_mark.insert(node.name.clone());

            // Visit dependencies (nodes that produce our inputs)
            for input in &node.inputs {
                if input.is_empty() || weights.contains(input) {
                    continue; // Skip optional/weight inputs
                }
                if let Some(producer_name) = output_producers.get(input) {
                    if let Some(producer_node) = node_map.get(*producer_name) {
                        visit(
                            producer_node,
                            node_map,
                            output_producers,
                            visited,
                            temp_mark,
                            sorted,
                            weights,
                        )?;
                    }
                }
            }

            temp_mark.remove(&node.name);
            visited.insert(node.name.clone());
            sorted.push(node.clone());

            Ok(())
        }

        // Get weight names as a set for quick lookup
        let weight_names: HashSet<String> = self.weights.keys().cloned().collect();

        // Visit all nodes
        for node in &self.nodes {
            visit(
                node,
                &node_map,
                &output_producers,
                &mut visited,
                &mut temp_mark,
                &mut sorted,
                &weight_names,
            )?;
        }

        Ok(sorted)
    }
}

/// Compute the broadcast shape for two tensor shapes using NumPy-style broadcasting rules.
///
/// Returns None if the shapes are not broadcastable.
fn compute_broadcast_shape(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    // Pad shorter shape with 1s on the left
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
            // Dimensions don't match and neither is 1
            return None;
        }
    }

    Some(result)
}

/// Expand tensor to target shape if needed, using Cow to avoid cloning when shapes match.
///
/// This is more memory-pool-friendly than cloning because it doesn't increment
/// the Arc refcount when no expansion is needed, allowing the original tensor
/// to be returned to the pool later.
fn maybe_expand<'a>(
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
