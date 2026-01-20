//! GPU Tensor Validation Layer
//!
//! Provides efficient validation of GPU tensors during inference:
//! - NaN/Inf detection
//! - Shape verification
//! - Statistical analysis (min/max/mean)
//!
//! All validation is performed on-device to minimize data transfers.
//! This module is compile-gated behind the `debug-inference` feature.

use super::context::{CudaError, IconnxCudaContext};
use super::tensor::GpuTensor;
use cudarc::driver::{CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};
use std::sync::Arc;

/// Validation mode flags
#[derive(Clone, Copy, Debug)]
pub struct ValidationMode {
    /// Check for NaN values
    pub check_nan: bool,
    /// Check for Inf values
    pub check_inf: bool,
    /// Stop execution on first error
    pub stop_on_error: bool,
    /// Log statistics (min/max/mean) for each tensor
    pub log_stats: bool,
}

impl Default for ValidationMode {
    fn default() -> Self {
        Self {
            check_nan: true,
            check_inf: true,
            stop_on_error: true,
            log_stats: false,
        }
    }
}

impl ValidationMode {
    /// Create a mode that checks everything
    pub fn full() -> Self {
        Self {
            check_nan: true,
            check_inf: true,
            stop_on_error: true,
            log_stats: true,
        }
    }

    /// Create a mode for just NaN/Inf checking (fast)
    pub fn nan_inf_only() -> Self {
        Self {
            check_nan: true,
            check_inf: true,
            stop_on_error: true,
            log_stats: false,
        }
    }
}

/// Validation result for a single tensor
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Tensor name (usually node output name)
    pub tensor_name: String,
    /// Operator type that produced this tensor
    pub op_type: String,
    /// Tensor shape
    pub shape: Vec<usize>,
    /// Number of elements
    pub numel: usize,
    /// Whether tensor contains NaN values
    pub has_nan: bool,
    /// Whether tensor contains Inf values
    pub has_inf: bool,
    /// Count of NaN values
    pub nan_count: usize,
    /// Count of Inf values
    pub inf_count: usize,
    /// Minimum value (excluding NaN/Inf)
    pub min: f32,
    /// Maximum value (excluding NaN/Inf)
    pub max: f32,
    /// Mean value (excluding NaN/Inf)
    pub mean: f32,
}

impl ValidationResult {
    /// Check if this result indicates an issue
    pub fn has_issues(&self) -> bool {
        self.has_nan || self.has_inf
    }

    /// Format as a single line for logging
    pub fn to_line(&self) -> String {
        let issues = match (self.has_nan, self.has_inf) {
            (true, true) => format!(" [NaN:{}! Inf:{}!]", self.nan_count, self.inf_count),
            (true, false) => format!(" [NaN:{}!]", self.nan_count),
            (false, true) => format!(" [Inf:{}!]", self.inf_count),
            (false, false) => String::new(),
        };
        format!(
            "{} ({}) {:?} range=[{:.6e}, {:.6e}]{}",
            self.tensor_name, self.op_type, self.shape, self.min, self.max, issues
        )
    }
}

/// Validation error
#[derive(Debug, Clone)]
pub enum ValidationError {
    /// Tensor contains NaN values
    NanDetected {
        node_name: String,
        op_type: String,
        count: usize,
    },
    /// Tensor contains Inf values
    InfDetected {
        node_name: String,
        op_type: String,
        count: usize,
    },
    /// Shape mismatch
    ShapeMismatch {
        node_name: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    /// CUDA error during validation
    Cuda(CudaError),
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationError::NanDetected {
                node_name,
                op_type,
                count,
            } => {
                write!(
                    f,
                    "NaN detected in {} ({}): {} NaN values",
                    node_name, op_type, count
                )
            }
            ValidationError::InfDetected {
                node_name,
                op_type,
                count,
            } => {
                write!(
                    f,
                    "Inf detected in {} ({}): {} Inf values",
                    node_name, op_type, count
                )
            }
            ValidationError::ShapeMismatch {
                node_name,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "Shape mismatch in {}: expected {:?}, got {:?}",
                    node_name, expected, actual
                )
            }
            ValidationError::Cuda(e) => write!(f, "CUDA error during validation: {}", e),
        }
    }
}

impl std::error::Error for ValidationError {}

impl From<CudaError> for ValidationError {
    fn from(e: CudaError) -> Self {
        ValidationError::Cuda(e)
    }
}

/// Validation report summarizing all issues found
#[derive(Debug, Default)]
pub struct ValidationReport {
    /// All validation results
    pub results: Vec<ValidationResult>,
    /// First node with NaN values
    pub first_nan_node: Option<String>,
    /// First node with Inf values
    pub first_inf_node: Option<String>,
    /// Total number of NaN values found
    pub total_nan_count: usize,
    /// Total number of Inf values found
    pub total_inf_count: usize,
}

impl ValidationReport {
    /// Check if any issues were found
    pub fn has_issues(&self) -> bool {
        self.first_nan_node.is_some() || self.first_inf_node.is_some()
    }

    /// Get a summary of issues
    pub fn summary(&self) -> String {
        if !self.has_issues() {
            return "No issues found".to_string();
        }

        let mut parts = Vec::new();
        if let Some(ref node) = self.first_nan_node {
            parts.push(format!(
                "First NaN at: {} ({} total)",
                node, self.total_nan_count
            ));
        }
        if let Some(ref node) = self.first_inf_node {
            parts.push(format!(
                "First Inf at: {} ({} total)",
                node, self.total_inf_count
            ));
        }
        parts.join("; ")
    }
}

/// CUDA kernel source for tensor validation
const VALIDATE_KERNEL_SRC: &str = r#"
extern "C" __global__ void validate_f32(
    const float* data,
    unsigned int len,
    unsigned int* nan_count,
    unsigned int* inf_count,
    float* min_val,
    float* max_val,
    double* sum
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Shared memory for block-level reduction
    __shared__ unsigned int s_nan[256];
    __shared__ unsigned int s_inf[256];
    __shared__ float s_min[256];
    __shared__ float s_max[256];
    __shared__ double s_sum[256];

    unsigned int tid = threadIdx.x;

    // Initialize shared memory
    s_nan[tid] = 0;
    s_inf[tid] = 0;
    s_min[tid] = 3.402823e+38f;  // FLT_MAX
    s_max[tid] = -3.402823e+38f; // -FLT_MAX
    s_sum[tid] = 0.0;

    // Each thread processes multiple elements
    for (unsigned int i = idx; i < len; i += blockDim.x * gridDim.x) {
        float val = data[i];

        if (isnan(val)) {
            s_nan[tid]++;
        } else if (isinf(val)) {
            s_inf[tid]++;
        } else {
            if (val < s_min[tid]) s_min[tid] = val;
            if (val > s_max[tid]) s_max[tid] = val;
            s_sum[tid] += (double)val;
        }
    }

    __syncthreads();

    // Block-level reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_nan[tid] += s_nan[tid + s];
            s_inf[tid] += s_inf[tid + s];
            if (s_min[tid + s] < s_min[tid]) s_min[tid] = s_min[tid + s];
            if (s_max[tid + s] > s_max[tid]) s_max[tid] = s_max[tid + s];
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 writes block result
    if (tid == 0) {
        atomicAdd(nan_count, s_nan[0]);
        atomicAdd(inf_count, s_inf[0]);

        // Atomic min/max for floats (using atomicCAS)
        // For simplicity, we'll just use global atomics which are slower
        // but correct. A production version would use atomicMin/Max with int casting.

        // Use block-local results and atomic for final reduction
        float old_min, old_max;
        do {
            old_min = *min_val;
            if (s_min[0] >= old_min) break;
        } while (atomicCAS((unsigned int*)min_val, __float_as_uint(old_min), __float_as_uint(s_min[0])) != __float_as_uint(old_min));

        do {
            old_max = *max_val;
            if (s_max[0] <= old_max) break;
        } while (atomicCAS((unsigned int*)max_val, __float_as_uint(old_max), __float_as_uint(s_max[0])) != __float_as_uint(old_max));

        // Atomic add for sum (double precision)
        atomicAdd((unsigned long long*)sum, __double_as_longlong(s_sum[0]));
    }
}
"#;

/// Cache for validation kernel
pub struct ValidationKernelCache {
    validate_f32: CudaFunction,
}

impl ValidationKernelCache {
    /// Compile and cache validation kernels
    pub fn new(ctx: &IconnxCudaContext) -> Result<Self, CudaError> {
        let ptx = cudarc::nvrtc::compile_ptx(VALIDATE_KERNEL_SRC)
            .map_err(|e| CudaError::Kernel(format!("Validation kernel compile failed: {}", e)))?;

        let module = ctx
            .context()
            .load_module(ptx)
            .map_err(|e| CudaError::Kernel(format!("Module load failed: {}", e)))?;

        let validate_f32 = module
            .load_function("validate_f32")
            .map_err(|e| CudaError::Kernel(format!("Function load failed: {}", e)))?;

        Ok(Self { validate_f32 })
    }

    /// Validate an f32 tensor on GPU
    /// Returns (nan_count, inf_count, min, max, mean)
    pub fn validate_f32(
        &self,
        data: &CudaSlice<f32>,
        len: usize,
        ctx: &IconnxCudaContext,
    ) -> Result<(usize, usize, f32, f32, f32), CudaError> {
        if len == 0 {
            return Ok((0, 0, 0.0, 0.0, 0.0));
        }

        // Allocate output buffers
        let mut nan_count = ctx.alloc_zeros::<u32>(1)?;
        let mut inf_count = ctx.alloc_zeros::<u32>(1)?;
        let mut min_val = ctx.htod(&[f32::MAX])?;
        let mut max_val = ctx.htod(&[f32::MIN])?;
        let mut sum = ctx.alloc_zeros::<f64>(1)?;

        // Calculate grid dimensions
        let block_size = 256usize;
        let grid_size = ((len + block_size - 1) / block_size).min(1024);

        // Launch kernel using stream.launch_builder pattern
        let cfg = LaunchConfig {
            block_dim: (block_size as u32, 1, 1),
            grid_dim: (grid_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let len_u32 = len as u32;

        unsafe {
            ctx.stream()
                .launch_builder(&self.validate_f32)
                .arg(data)
                .arg(&len_u32)
                .arg(&mut nan_count)
                .arg(&mut inf_count)
                .arg(&mut min_val)
                .arg(&mut max_val)
                .arg(&mut sum)
                .launch(cfg)
                .map_err(|e| CudaError::Kernel(format!("Validation kernel launch failed: {}", e)))?;
        }

        ctx.sync()?;

        // Read results
        let nan_result = ctx.dtoh::<u32>(&nan_count)?;
        let inf_result = ctx.dtoh::<u32>(&inf_count)?;
        let min_result = ctx.dtoh::<f32>(&min_val)?;
        let max_result = ctx.dtoh::<f32>(&max_val)?;
        let sum_result = ctx.dtoh::<f64>(&sum)?;

        let valid_count = len - nan_result[0] as usize - inf_result[0] as usize;
        let mean = if valid_count > 0 {
            (sum_result[0] / valid_count as f64) as f32
        } else {
            0.0
        };

        Ok((
            nan_result[0] as usize,
            inf_result[0] as usize,
            min_result[0],
            max_result[0],
            mean,
        ))
    }
}

/// Tensor validator that accumulates issues during execution
pub struct TensorValidator {
    mode: ValidationMode,
    kernel_cache: Arc<ValidationKernelCache>,
    results: Vec<ValidationResult>,
    first_nan_node: Option<String>,
    first_inf_node: Option<String>,
    total_nan_count: usize,
    total_inf_count: usize,
}

impl TensorValidator {
    /// Create a new validator with the given mode
    pub fn new(mode: ValidationMode, ctx: &IconnxCudaContext) -> Result<Self, CudaError> {
        let kernel_cache = Arc::new(ValidationKernelCache::new(ctx)?);
        Ok(Self {
            mode,
            kernel_cache,
            results: Vec::new(),
            first_nan_node: None,
            first_inf_node: None,
            total_nan_count: 0,
            total_inf_count: 0,
        })
    }

    /// Create a validator with a pre-existing kernel cache (for sharing)
    pub fn with_cache(mode: ValidationMode, kernel_cache: Arc<ValidationKernelCache>) -> Self {
        Self {
            mode,
            kernel_cache,
            results: Vec::new(),
            first_nan_node: None,
            first_inf_node: None,
            total_nan_count: 0,
            total_inf_count: 0,
        }
    }

    /// Validate a tensor
    pub fn validate(
        &mut self,
        name: &str,
        op_type: &str,
        tensor: &GpuTensor,
        ctx: &IconnxCudaContext,
    ) -> Result<(), ValidationError> {
        // Only validate Float32 tensors (Int32/Int64 can't have NaN/Inf)
        if !tensor.is_f32() {
            return Ok(());
        }

        let numel = tensor.len();

        // Get underlying slice and validate
        let slice = tensor.data_f32()?;
        let (nan_count, inf_count, min, max, mean) =
            self.kernel_cache.validate_f32(slice, numel, ctx)?;

        let has_nan = nan_count > 0;
        let has_inf = inf_count > 0;

        let result = ValidationResult {
            tensor_name: name.to_string(),
            op_type: op_type.to_string(),
            shape: tensor.shape().to_vec(),
            numel,
            has_nan,
            has_inf,
            nan_count,
            inf_count,
            min,
            max,
            mean,
        };

        // Track first occurrences
        if has_nan && self.first_nan_node.is_none() {
            self.first_nan_node = Some(name.to_string());
        }
        if has_inf && self.first_inf_node.is_none() {
            self.first_inf_node = Some(name.to_string());
        }

        self.total_nan_count += nan_count;
        self.total_inf_count += inf_count;

        // Log if enabled
        if self.mode.log_stats || result.has_issues() {
            eprintln!("VALIDATE: {}", result.to_line());
        }

        self.results.push(result);

        // Stop on error if configured
        if self.mode.stop_on_error {
            if self.mode.check_nan && has_nan {
                return Err(ValidationError::NanDetected {
                    node_name: name.to_string(),
                    op_type: op_type.to_string(),
                    count: nan_count,
                });
            }
            if self.mode.check_inf && has_inf {
                return Err(ValidationError::InfDetected {
                    node_name: name.to_string(),
                    op_type: op_type.to_string(),
                    count: inf_count,
                });
            }
        }

        Ok(())
    }

    /// Get the validation report
    pub fn report(&self) -> ValidationReport {
        ValidationReport {
            results: self.results.clone(),
            first_nan_node: self.first_nan_node.clone(),
            first_inf_node: self.first_inf_node.clone(),
            total_nan_count: self.total_nan_count,
            total_inf_count: self.total_inf_count,
        }
    }

    /// Check if any issues were found
    pub fn has_issues(&self) -> bool {
        self.first_nan_node.is_some() || self.first_inf_node.is_some()
    }

    /// Reset the validator for a new inference run
    pub fn reset(&mut self) {
        self.results.clear();
        self.first_nan_node = None;
        self.first_inf_node = None;
        self.total_nan_count = 0;
        self.total_inf_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires CUDA GPU
    fn test_validation_kernel_basic() {
        let ctx = IconnxCudaContext::new().expect("Failed to create context");
        let cache = ValidationKernelCache::new(&ctx).expect("Failed to create kernel cache");

        // Test with normal data
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let gpu_data = ctx.htod(&data).expect("Failed to upload");
        let (nan, inf, min, max, mean) = cache
            .validate_f32(&gpu_data, data.len(), &ctx)
            .expect("Validation failed");

        assert_eq!(nan, 0);
        assert_eq!(inf, 0);
        assert!((min - 1.0).abs() < 1e-6);
        assert!((max - 5.0).abs() < 1e-6);
        assert!((mean - 3.0).abs() < 1e-6);
    }

    #[test]
    #[ignore] // Requires CUDA GPU
    fn test_validation_kernel_with_nan() {
        let ctx = IconnxCudaContext::new().expect("Failed to create context");
        let cache = ValidationKernelCache::new(&ctx).expect("Failed to create kernel cache");

        // Test with NaN values
        let data = vec![1.0f32, f32::NAN, 3.0, f32::NAN, 5.0];
        let gpu_data = ctx.htod(&data).expect("Failed to upload");
        let (nan, inf, _min, _max, _mean) = cache
            .validate_f32(&gpu_data, data.len(), &ctx)
            .expect("Validation failed");

        assert_eq!(nan, 2);
        assert_eq!(inf, 0);
    }

    #[test]
    #[ignore] // Requires CUDA GPU
    fn test_validation_kernel_with_inf() {
        let ctx = IconnxCudaContext::new().expect("Failed to create context");
        let cache = ValidationKernelCache::new(&ctx).expect("Failed to create kernel cache");

        // Test with Inf values
        let data = vec![1.0f32, f32::INFINITY, 3.0, f32::NEG_INFINITY, 5.0];
        let gpu_data = ctx.htod(&data).expect("Failed to upload");
        let (nan, inf, _min, _max, _mean) = cache
            .validate_f32(&gpu_data, data.len(), &ctx)
            .expect("Validation failed");

        assert_eq!(nan, 0);
        assert_eq!(inf, 2);
    }

    #[test]
    #[ignore] // Requires CUDA GPU
    fn test_tensor_validator() {
        let ctx = IconnxCudaContext::new().expect("Failed to create context");
        let mode = ValidationMode::nan_inf_only();
        let mut validator = TensorValidator::new(mode, &ctx).expect("Failed to create validator");

        // Create a tensor with NaN
        let data = vec![1.0f32, f32::NAN, 3.0];
        let tensor = GpuTensor::from_host_f32(&ctx, &data, vec![3]).expect("Failed to create");

        let result = validator.validate("test_node", "TestOp", &tensor, &ctx);
        assert!(result.is_err());

        match result {
            Err(ValidationError::NanDetected { count, .. }) => assert_eq!(count, 1),
            _ => panic!("Expected NanDetected error"),
        }
    }
}
