use crate::cuda::bridge::GbKernelArg;
use super::context::{CudaError, IconnxCudaContext};
use super::memory_pool::GpuMemoryPool;
use super::tensor::GpuTensor;
/// GPU Elementwise Kernels using NVRTC
///
/// Provides GPU-accelerated elementwise operations compiled at runtime via NVRTC.
/// Kernels are compiled once and cached for reuse.
use cudarc::driver::{CudaFunction, CudaModule, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use std::collections::HashMap;
use std::sync::Arc;

/// Manages compiled CUDA kernels with caching
pub struct KernelCache {
    // Module must be kept alive to keep functions valid
    #[allow(dead_code)]
    module: Arc<CudaModule>,
    functions: HashMap<&'static str, CudaFunction>,
}

impl KernelCache {
    /// Compile all elementwise kernels and create the cache
    pub fn new(ctx: &IconnxCudaContext) -> Result<Self, CudaError> {
        let ptx = compile_ptx(ELEMENTWISE_KERNELS)
            .map_err(|e| CudaError::Kernel(format!("NVRTC compilation failed: {:?}", e)))?;

        let module = ctx
            .context()
            .load_module(ptx)
            .map_err(|e| CudaError::Kernel(format!("Module load failed: {}", e)))?;

        let mut functions = HashMap::new();

        // Load all kernel functions
        for name in KERNEL_NAMES {
            let func = module.load_function(name).map_err(|e| {
                CudaError::Kernel(format!("Failed to load kernel '{}': {}", name, e))
            })?;
            functions.insert(*name, func);
        }

        Ok(Self { module, functions })
    }

    /// Get a compiled kernel function by name
    pub fn get(&self, name: &'static str) -> Option<&CudaFunction> {
        self.functions.get(name)
    }
}

/// List of all kernel function names
const KERNEL_NAMES: &[&str] = &[
    // Binary ops
    "add_kernel",
    "add_i64_kernel",
    "sub_kernel",
    "sub_i64_kernel",
    "mul_kernel",
    "mul_i64_kernel",
    "div_kernel",
    // Activations
    "sigmoid_kernel",
    "tanh_kernel",
    "relu_kernel",
    "leaky_relu_kernel",
    // Math ops
    "exp_kernel",
    "log_kernel",
    "sqrt_kernel",
    "pow_kernel",
    "sin_kernel",
    "cos_kernel",
    "abs_kernel",
    "neg_kernel",
    // Utility ops
    "clip_kernel",
    "floor_kernel",
    "ceil_kernel",
    "round_kernel",
    // Scalar ops
    "add_scalar_kernel",
    "add_i64_scalar_kernel",
    "mul_scalar_kernel",
    // Scalar-from-pointer ops (avoids D2H transfer)
    "add_scalar_ptr_kernel",
    "add_i64_scalar_ptr_kernel",
];

/// All elementwise CUDA kernel source code
const ELEMENTWISE_KERNELS: &str = r#"
// Binary elementwise operations: out[i] = a[i] op b[i]

extern "C" __global__ void add_kernel(float* out, const float* a, const float* b, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}

extern "C" __global__ void sub_kernel(float* out, const float* a, const float* b, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] - b[i];
    }
}

extern "C" __global__ void mul_kernel(float* out, const float* a, const float* b, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] * b[i];
    }
}

extern "C" __global__ void div_kernel(float* out, const float* a, const float* b, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Add epsilon protection to prevent 0/0 = NaN
        // Use signed epsilon to preserve sign of denominator
        const float eps = 1e-8f;
        float denom = b[i];
        if (fabsf(denom) < eps) {
            denom = (denom >= 0.0f) ? eps : -eps;
        }
        out[i] = a[i] / denom;
    }
}

// Binary elementwise operations for Int64
extern "C" __global__ void add_i64_kernel(long long* out, const long long* a, const long long* b, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}

extern "C" __global__ void sub_i64_kernel(long long* out, const long long* a, const long long* b, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] - b[i];
    }
}

extern "C" __global__ void mul_i64_kernel(long long* out, const long long* a, const long long* b, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] * b[i];
    }
}

// Activation functions

extern "C" __global__ void sigmoid_kernel(float* out, const float* inp, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = 1.0f / (1.0f + expf(-inp[i]));
    }
}

extern "C" __global__ void tanh_kernel(float* out, const float* inp, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = tanhf(inp[i]);
    }
}

extern "C" __global__ void relu_kernel(float* out, const float* inp, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = fmaxf(0.0f, inp[i]);
    }
}

extern "C" __global__ void leaky_relu_kernel(float* out, const float* inp, float alpha, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = inp[i];
        out[i] = x >= 0.0f ? x : alpha * x;
    }
}

// Math operations

extern "C" __global__ void exp_kernel(float* out, const float* inp, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = expf(inp[i]);
    }
}

extern "C" __global__ void log_kernel(float* out, const float* inp, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = logf(inp[i]);
    }
}

extern "C" __global__ void sqrt_kernel(float* out, const float* inp, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = sqrtf(inp[i]);
    }
}

extern "C" __global__ void pow_kernel(float* out, const float* base, const float* exp, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = powf(base[i], exp[i]);
    }
}

extern "C" __global__ void sin_kernel(float* out, const float* inp, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = sinf(inp[i]);
    }
}

extern "C" __global__ void cos_kernel(float* out, const float* inp, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = cosf(inp[i]);
    }
}

extern "C" __global__ void abs_kernel(float* out, const float* inp, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = fabsf(inp[i]);
    }
}

extern "C" __global__ void neg_kernel(float* out, const float* inp, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = -inp[i];
    }
}

// Utility operations

extern "C" __global__ void clip_kernel(float* out, const float* inp, float min_val, float max_val, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = fminf(fmaxf(inp[i], min_val), max_val);
    }
}

extern "C" __global__ void floor_kernel(float* out, const float* inp, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = floorf(inp[i]);
    }
}

extern "C" __global__ void ceil_kernel(float* out, const float* inp, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = ceilf(inp[i]);
    }
}

extern "C" __global__ void round_kernel(float* out, const float* inp, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = roundf(inp[i]);
    }
}

// Scalar operations: out[i] = a[i] op scalar

extern "C" __global__ void add_scalar_kernel(float* out, const float* a, float scalar, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + scalar;
    }
}

extern "C" __global__ void add_i64_scalar_kernel(long long* out, const long long* a, long long scalar, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + scalar;
    }
}

extern "C" __global__ void mul_scalar_kernel(float* out, const float* a, float scalar, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] * scalar;
    }
}

// Scalar-from-pointer operations: read scalar from GPU memory (avoids D2H transfer)

extern "C" __global__ void add_scalar_ptr_kernel(float* out, const float* a, const float* scalar_ptr, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + scalar_ptr[0];
    }
}

extern "C" __global__ void add_i64_scalar_ptr_kernel(long long* out, const long long* a, const long long* scalar_ptr, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + scalar_ptr[0];
    }
}
"#;

/// Launch configuration for elementwise operations
fn elementwise_config(n: usize) -> LaunchConfig {
    LaunchConfig::for_num_elems(n as u32)
}

// =============================================================================
// Binary Operations
// =============================================================================

/// GPU Add: out = a + b (elementwise with broadcasting)
/// Supports Float32 and Int64 dtypes
pub fn gpu_add(
    ctx: &IconnxCudaContext,
    kernels: &KernelCache,
    pool: &mut GpuMemoryPool,
    a: &GpuTensor,
    b: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    // Handle scalar broadcasting: if one tensor is scalar (len=1), broadcast it
    if b.len() == 1 && a.len() > 1 {
        // Broadcast scalar b to match a's shape
        return gpu_add_broadcast_scalar(ctx, kernels, pool, a, b);
    }
    if a.len() == 1 && b.len() > 1 {
        // Broadcast scalar a to match b's shape
        return gpu_add_broadcast_scalar(ctx, kernels, pool, b, a);
    }

    // Regular elementwise add - shapes must match
    if a.shape() != b.shape() {
        return Err(CudaError::Kernel(format!(
            "Add: shapes must match or be broadcastable, got {:?} and {:?}",
            a.shape(),
            b.shape()
        )));
    }

    let n = a.len();

    // Dispatch based on dtype
    match (a, b) {
        (GpuTensor::Int64 { .. }, GpuTensor::Int64 { .. }) => {
            let mut out = pool.get_tensor_i64(ctx, a.shape().to_vec())?;
            let func = kernels
                .get("add_i64_kernel")
                .ok_or_else(|| CudaError::Kernel("add_i64_kernel not found".into()))?;
            unsafe {
                ctx.stream()
                    .launch_builder(func)
                    .arg(&GbKernelArg::new_mut(out.data_i64_mut()?))
                    .arg(&GbKernelArg::new(a.data_i64()?))
                    .arg(&GbKernelArg::new(b.data_i64()?))
                    .arg(&n)
                    .launch(elementwise_config(n))
                    .map_err(|e| CudaError::Kernel(e.to_string()))?;
            }
            Ok(out)
        }
        _ => {
            // Default Float32 path
            let mut out = pool.get_tensor_f32(ctx, a.shape().to_vec())?;
            let func = kernels
                .get("add_kernel")
                .ok_or_else(|| CudaError::Kernel("add_kernel not found".into()))?;
            unsafe {
                ctx.stream()
                    .launch_builder(func)
                    .arg(&GbKernelArg::new_mut(out.data_f32_mut()?))
                    .arg(&GbKernelArg::new(a.data_f32()?))
                    .arg(&GbKernelArg::new(b.data_f32()?))
                    .arg(&n)
                    .launch(elementwise_config(n))
                    .map_err(|e| CudaError::Kernel(e.to_string()))?;
            }
            Ok(out)
        }
    }
}

/// GPU Add with scalar tensor: out = tensor + scalar_tensor (broadcast scalar to all elements)
/// Internal function used by gpu_add when one operand is scalar
/// Uses pointer-based kernels to avoid D2H memory transfer
fn gpu_add_broadcast_scalar(
    ctx: &IconnxCudaContext,
    kernels: &KernelCache,
    pool: &mut GpuMemoryPool,
    tensor: &GpuTensor,
    scalar: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    let n = tensor.len();

    // Dispatch based on dtype - use pointer-based kernels to avoid D2H transfer
    match (tensor, scalar) {
        (GpuTensor::Int64 { .. }, GpuTensor::Int64 { .. }) => {
            let mut out = pool.get_tensor_i64(ctx, tensor.shape().to_vec())?;

            let func = kernels
                .get("add_i64_scalar_ptr_kernel")
                .ok_or_else(|| CudaError::Kernel("add_i64_scalar_ptr_kernel not found".into()))?;

            unsafe {
                ctx.stream()
                    .launch_builder(func)
                    .arg(&GbKernelArg::new_mut(out.data_i64_mut()?))
                    .arg(&GbKernelArg::new(tensor.data_i64()?))
                    .arg(&GbKernelArg::new(scalar.data_i64()?)) // Pass GPU pointer directly
                    .arg(&n)
                    .launch(elementwise_config(n))
                    .map_err(|e| CudaError::Kernel(e.to_string()))?;
            }

            Ok(out)
        }
        _ => {
            // Default Float32 path
            let mut out = pool.get_tensor_f32(ctx, tensor.shape().to_vec())?;

            let func = kernels
                .get("add_scalar_ptr_kernel")
                .ok_or_else(|| CudaError::Kernel("add_scalar_ptr_kernel not found".into()))?;

            unsafe {
                ctx.stream()
                    .launch_builder(func)
                    .arg(&GbKernelArg::new_mut(out.data_f32_mut()?))
                    .arg(&GbKernelArg::new(tensor.data_f32()?))
                    .arg(&GbKernelArg::new(scalar.data_f32()?)) // Pass GPU pointer directly
                    .arg(&n)
                    .launch(elementwise_config(n))
                    .map_err(|e| CudaError::Kernel(e.to_string()))?;
            }

            Ok(out)
        }
    }
}

/// GPU Sub: out = a - b (elementwise)
pub fn gpu_sub(
    ctx: &IconnxCudaContext,
    kernels: &KernelCache,
    pool: &mut GpuMemoryPool,
    a: &GpuTensor,
    b: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    assert_eq!(a.shape(), b.shape(), "Sub: shapes must match");
    let n = a.len();

    // Dispatch based on dtype
    match (a, b) {
        (GpuTensor::Int64 { .. }, GpuTensor::Int64 { .. }) => {
            let mut out = pool.get_tensor_i64(ctx, a.shape().to_vec())?;
            let func = kernels
                .get("sub_i64_kernel")
                .ok_or_else(|| CudaError::Kernel("sub_i64_kernel not found".into()))?;
            unsafe {
                ctx.stream()
                    .launch_builder(func)
                    .arg(&GbKernelArg::new_mut(out.data_i64_mut()?))
                    .arg(&GbKernelArg::new(a.data_i64()?))
                    .arg(&GbKernelArg::new(b.data_i64()?))
                    .arg(&n)
                    .launch(elementwise_config(n))
                    .map_err(|e| CudaError::Kernel(e.to_string()))?;
            }
            Ok(out)
        }
        _ => {
            // Default Float32 path
            let mut out = pool.get_tensor_f32(ctx, a.shape().to_vec())?;
            let func = kernels
                .get("sub_kernel")
                .ok_or_else(|| CudaError::Kernel("sub_kernel not found".into()))?;
            unsafe {
                ctx.stream()
                    .launch_builder(func)
                    .arg(&GbKernelArg::new_mut(out.data_f32_mut()?))
                    .arg(&GbKernelArg::new(a.data_f32()?))
                    .arg(&GbKernelArg::new(b.data_f32()?))
                    .arg(&n)
                    .launch(elementwise_config(n))
                    .map_err(|e| CudaError::Kernel(e.to_string()))?;
            }
            Ok(out)
        }
    }
}

/// GPU Mul: out = a * b (elementwise)
pub fn gpu_mul(
    ctx: &IconnxCudaContext,
    kernels: &KernelCache,
    pool: &mut GpuMemoryPool,
    a: &GpuTensor,
    b: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    assert_eq!(a.shape(), b.shape(), "Mul: shapes must match");
    let n = a.len();

    // Dispatch based on dtype
    match (a, b) {
        (GpuTensor::Int64 { .. }, GpuTensor::Int64 { .. }) => {
            let mut out = pool.get_tensor_i64(ctx, a.shape().to_vec())?;
            let func = kernels
                .get("mul_i64_kernel")
                .ok_or_else(|| CudaError::Kernel("mul_i64_kernel not found".into()))?;
            unsafe {
                ctx.stream()
                    .launch_builder(func)
                    .arg(&GbKernelArg::new_mut(out.data_i64_mut()?))
                    .arg(&GbKernelArg::new(a.data_i64()?))
                    .arg(&GbKernelArg::new(b.data_i64()?))
                    .arg(&n)
                    .launch(elementwise_config(n))
                    .map_err(|e| CudaError::Kernel(e.to_string()))?;
            }
            Ok(out)
        }
        _ => {
            // Default Float32 path
            let mut out = pool.get_tensor_f32(ctx, a.shape().to_vec())?;
            let func = kernels
                .get("mul_kernel")
                .ok_or_else(|| CudaError::Kernel("mul_kernel not found".into()))?;
            unsafe {
                ctx.stream()
                    .launch_builder(func)
                    .arg(&GbKernelArg::new_mut(out.data_f32_mut()?))
                    .arg(&GbKernelArg::new(a.data_f32()?))
                    .arg(&GbKernelArg::new(b.data_f32()?))
                    .arg(&n)
                    .launch(elementwise_config(n))
                    .map_err(|e| CudaError::Kernel(e.to_string()))?;
            }
            Ok(out)
        }
    }
}

/// GPU Div: out = a / b (elementwise)
pub fn gpu_div(
    ctx: &IconnxCudaContext,
    kernels: &KernelCache,
    pool: &mut GpuMemoryPool,
    a: &GpuTensor,
    b: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    assert_eq!(a.shape(), b.shape(), "Div: shapes must match");
    let n = a.len();
    let mut out = pool.get_tensor_f32(ctx, a.shape().to_vec())?;

    let func = kernels
        .get("div_kernel")
        .ok_or_else(|| CudaError::Kernel("div_kernel not found".into()))?;

    unsafe {
        ctx.stream()
            .launch_builder(func)
            .arg(&GbKernelArg::new_mut(out.data_f32_mut()?))
            .arg(&GbKernelArg::new(a.data_f32()?))
            .arg(&GbKernelArg::new(b.data_f32()?))
            .arg(&n)
            .launch(elementwise_config(n))
            .map_err(|e| CudaError::Kernel(e.to_string()))?;
    }

    Ok(out)
}

// =============================================================================
// Activation Functions
// =============================================================================

/// GPU Sigmoid: out = 1 / (1 + exp(-x))
pub fn gpu_sigmoid(
    ctx: &IconnxCudaContext,
    kernels: &KernelCache,
    pool: &mut GpuMemoryPool,
    input: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    let n = input.len();
    let mut out = pool.get_tensor_f32(ctx, input.shape().to_vec())?;

    let func = kernels
        .get("sigmoid_kernel")
        .ok_or_else(|| CudaError::Kernel("sigmoid_kernel not found".into()))?;

    unsafe {
        ctx.stream()
            .launch_builder(func)
            .arg(&GbKernelArg::new_mut(out.data_f32_mut()?))
            .arg(&GbKernelArg::new(input.data_f32()?))
            .arg(&n)
            .launch(elementwise_config(n))
            .map_err(|e| CudaError::Kernel(e.to_string()))?;
    }

    Ok(out)
}

/// GPU Tanh: out = tanh(x)
pub fn gpu_tanh(
    ctx: &IconnxCudaContext,
    kernels: &KernelCache,
    pool: &mut GpuMemoryPool,
    input: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    let n = input.len();
    let mut out = pool.get_tensor_f32(ctx, input.shape().to_vec())?;

    let func = kernels
        .get("tanh_kernel")
        .ok_or_else(|| CudaError::Kernel("tanh_kernel not found".into()))?;

    unsafe {
        ctx.stream()
            .launch_builder(func)
            .arg(&GbKernelArg::new_mut(out.data_f32_mut()?))
            .arg(&GbKernelArg::new(input.data_f32()?))
            .arg(&n)
            .launch(elementwise_config(n))
            .map_err(|e| CudaError::Kernel(e.to_string()))?;
    }

    Ok(out)
}

/// GPU ReLU: out = max(0, x)
pub fn gpu_relu(
    ctx: &IconnxCudaContext,
    kernels: &KernelCache,
    pool: &mut GpuMemoryPool,
    input: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    let n = input.len();
    let mut out = pool.get_tensor_f32(ctx, input.shape().to_vec())?;

    let func = kernels
        .get("relu_kernel")
        .ok_or_else(|| CudaError::Kernel("relu_kernel not found".into()))?;

    unsafe {
        ctx.stream()
            .launch_builder(func)
            .arg(&GbKernelArg::new_mut(out.data_f32_mut()?))
            .arg(&GbKernelArg::new(input.data_f32()?))
            .arg(&n)
            .launch(elementwise_config(n))
            .map_err(|e| CudaError::Kernel(e.to_string()))?;
    }

    Ok(out)
}

/// GPU LeakyReLU: out = x >= 0 ? x : alpha * x
pub fn gpu_leaky_relu(
    ctx: &IconnxCudaContext,
    kernels: &KernelCache,
    pool: &mut GpuMemoryPool,
    input: &GpuTensor,
    alpha: f32,
) -> Result<GpuTensor, CudaError> {
    let n = input.len();
    let mut out = pool.get_tensor_f32(ctx, input.shape().to_vec())?;

    let func = kernels
        .get("leaky_relu_kernel")
        .ok_or_else(|| CudaError::Kernel("leaky_relu_kernel not found".into()))?;

    unsafe {
        ctx.stream()
            .launch_builder(func)
            .arg(&GbKernelArg::new_mut(out.data_f32_mut()?))
            .arg(&GbKernelArg::new(input.data_f32()?))
            .arg(&alpha)
            .arg(&n)
            .launch(elementwise_config(n))
            .map_err(|e| CudaError::Kernel(e.to_string()))?;
    }

    Ok(out)
}

// =============================================================================
// Math Operations
// =============================================================================

/// GPU Exp: out = exp(x)
pub fn gpu_exp(
    ctx: &IconnxCudaContext,
    kernels: &KernelCache,
    pool: &mut GpuMemoryPool,
    input: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    let n = input.len();
    let mut out = pool.get_tensor_f32(ctx, input.shape().to_vec())?;

    let func = kernels
        .get("exp_kernel")
        .ok_or_else(|| CudaError::Kernel("exp_kernel not found".into()))?;

    unsafe {
        ctx.stream()
            .launch_builder(func)
            .arg(&GbKernelArg::new_mut(out.data_f32_mut()?))
            .arg(&GbKernelArg::new(input.data_f32()?))
            .arg(&n)
            .launch(elementwise_config(n))
            .map_err(|e| CudaError::Kernel(e.to_string()))?;
    }

    Ok(out)
}

/// GPU Log: out = log(x)
pub fn gpu_log(
    ctx: &IconnxCudaContext,
    kernels: &KernelCache,
    pool: &mut GpuMemoryPool,
    input: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    let n = input.len();
    let mut out = pool.get_tensor_f32(ctx, input.shape().to_vec())?;

    let func = kernels
        .get("log_kernel")
        .ok_or_else(|| CudaError::Kernel("log_kernel not found".into()))?;

    unsafe {
        ctx.stream()
            .launch_builder(func)
            .arg(&GbKernelArg::new_mut(out.data_f32_mut()?))
            .arg(&GbKernelArg::new(input.data_f32()?))
            .arg(&n)
            .launch(elementwise_config(n))
            .map_err(|e| CudaError::Kernel(e.to_string()))?;
    }

    Ok(out)
}

/// GPU Sqrt: out = sqrt(x)
pub fn gpu_sqrt(
    ctx: &IconnxCudaContext,
    kernels: &KernelCache,
    pool: &mut GpuMemoryPool,
    input: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    let n = input.len();
    let mut out = pool.get_tensor_f32(ctx, input.shape().to_vec())?;

    let func = kernels
        .get("sqrt_kernel")
        .ok_or_else(|| CudaError::Kernel("sqrt_kernel not found".into()))?;

    unsafe {
        ctx.stream()
            .launch_builder(func)
            .arg(&GbKernelArg::new_mut(out.data_f32_mut()?))
            .arg(&GbKernelArg::new(input.data_f32()?))
            .arg(&n)
            .launch(elementwise_config(n))
            .map_err(|e| CudaError::Kernel(e.to_string()))?;
    }

    Ok(out)
}

/// GPU Pow: out = base ^ exp (elementwise)
pub fn gpu_pow(
    ctx: &IconnxCudaContext,
    kernels: &KernelCache,
    pool: &mut GpuMemoryPool,
    base: &GpuTensor,
    exp: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    assert_eq!(base.shape(), exp.shape(), "Pow: shapes must match");
    let n = base.len();
    let mut out = pool.get_tensor_f32(ctx, base.shape().to_vec())?;

    let func = kernels
        .get("pow_kernel")
        .ok_or_else(|| CudaError::Kernel("pow_kernel not found".into()))?;

    unsafe {
        ctx.stream()
            .launch_builder(func)
            .arg(&GbKernelArg::new_mut(out.data_f32_mut()?))
            .arg(&GbKernelArg::new(base.data_f32()?))
            .arg(&GbKernelArg::new(exp.data_f32()?))
            .arg(&n)
            .launch(elementwise_config(n))
            .map_err(|e| CudaError::Kernel(e.to_string()))?;
    }

    Ok(out)
}

/// GPU Sin: out = sin(x)
pub fn gpu_sin(
    ctx: &IconnxCudaContext,
    kernels: &KernelCache,
    pool: &mut GpuMemoryPool,
    input: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    let n = input.len();
    let mut out = pool.get_tensor_f32(ctx, input.shape().to_vec())?;

    let func = kernels
        .get("sin_kernel")
        .ok_or_else(|| CudaError::Kernel("sin_kernel not found".into()))?;

    unsafe {
        ctx.stream()
            .launch_builder(func)
            .arg(&GbKernelArg::new_mut(out.data_f32_mut()?))
            .arg(&GbKernelArg::new(input.data_f32()?))
            .arg(&n)
            .launch(elementwise_config(n))
            .map_err(|e| CudaError::Kernel(e.to_string()))?;
    }

    Ok(out)
}

/// GPU Cos: out = cos(x)
pub fn gpu_cos(
    ctx: &IconnxCudaContext,
    kernels: &KernelCache,
    pool: &mut GpuMemoryPool,
    input: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    let n = input.len();
    let mut out = pool.get_tensor_f32(ctx, input.shape().to_vec())?;

    let func = kernels
        .get("cos_kernel")
        .ok_or_else(|| CudaError::Kernel("cos_kernel not found".into()))?;

    unsafe {
        ctx.stream()
            .launch_builder(func)
            .arg(&GbKernelArg::new_mut(out.data_f32_mut()?))
            .arg(&GbKernelArg::new(input.data_f32()?))
            .arg(&n)
            .launch(elementwise_config(n))
            .map_err(|e| CudaError::Kernel(e.to_string()))?;
    }

    Ok(out)
}

/// GPU Abs: out = |x|
pub fn gpu_abs(
    ctx: &IconnxCudaContext,
    kernels: &KernelCache,
    pool: &mut GpuMemoryPool,
    input: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    let n = input.len();
    let mut out = pool.get_tensor_f32(ctx, input.shape().to_vec())?;

    let func = kernels
        .get("abs_kernel")
        .ok_or_else(|| CudaError::Kernel("abs_kernel not found".into()))?;

    unsafe {
        ctx.stream()
            .launch_builder(func)
            .arg(&GbKernelArg::new_mut(out.data_f32_mut()?))
            .arg(&GbKernelArg::new(input.data_f32()?))
            .arg(&n)
            .launch(elementwise_config(n))
            .map_err(|e| CudaError::Kernel(e.to_string()))?;
    }

    Ok(out)
}

/// GPU Neg: out = -x
pub fn gpu_neg(
    ctx: &IconnxCudaContext,
    kernels: &KernelCache,
    pool: &mut GpuMemoryPool,
    input: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    let n = input.len();
    let mut out = pool.get_tensor_f32(ctx, input.shape().to_vec())?;

    let func = kernels
        .get("neg_kernel")
        .ok_or_else(|| CudaError::Kernel("neg_kernel not found".into()))?;

    unsafe {
        ctx.stream()
            .launch_builder(func)
            .arg(&GbKernelArg::new_mut(out.data_f32_mut()?))
            .arg(&GbKernelArg::new(input.data_f32()?))
            .arg(&n)
            .launch(elementwise_config(n))
            .map_err(|e| CudaError::Kernel(e.to_string()))?;
    }

    Ok(out)
}

// =============================================================================
// Utility Operations
// =============================================================================

/// GPU Clip: out = clamp(x, min, max)
pub fn gpu_clip(
    ctx: &IconnxCudaContext,
    kernels: &KernelCache,
    pool: &mut GpuMemoryPool,
    input: &GpuTensor,
    min_val: f32,
    max_val: f32,
) -> Result<GpuTensor, CudaError> {
    let n = input.len();
    let mut out = pool.get_tensor_f32(ctx, input.shape().to_vec())?;

    let func = kernels
        .get("clip_kernel")
        .ok_or_else(|| CudaError::Kernel("clip_kernel not found".into()))?;

    unsafe {
        ctx.stream()
            .launch_builder(func)
            .arg(&GbKernelArg::new_mut(out.data_f32_mut()?))
            .arg(&GbKernelArg::new(input.data_f32()?))
            .arg(&min_val)
            .arg(&max_val)
            .arg(&n)
            .launch(elementwise_config(n))
            .map_err(|e| CudaError::Kernel(e.to_string()))?;
    }

    Ok(out)
}

/// GPU Floor: out = floor(x)
pub fn gpu_floor(
    ctx: &IconnxCudaContext,
    kernels: &KernelCache,
    pool: &mut GpuMemoryPool,
    input: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    let n = input.len();
    let mut out = pool.get_tensor_f32(ctx, input.shape().to_vec())?;

    let func = kernels
        .get("floor_kernel")
        .ok_or_else(|| CudaError::Kernel("floor_kernel not found".into()))?;

    unsafe {
        ctx.stream()
            .launch_builder(func)
            .arg(&GbKernelArg::new_mut(out.data_f32_mut()?))
            .arg(&GbKernelArg::new(input.data_f32()?))
            .arg(&n)
            .launch(elementwise_config(n))
            .map_err(|e| CudaError::Kernel(e.to_string()))?;
    }

    Ok(out)
}

/// GPU Ceil: out = ceil(x)
pub fn gpu_ceil(
    ctx: &IconnxCudaContext,
    kernels: &KernelCache,
    pool: &mut GpuMemoryPool,
    input: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    let n = input.len();
    let mut out = pool.get_tensor_f32(ctx, input.shape().to_vec())?;

    let func = kernels
        .get("ceil_kernel")
        .ok_or_else(|| CudaError::Kernel("ceil_kernel not found".into()))?;

    unsafe {
        ctx.stream()
            .launch_builder(func)
            .arg(&GbKernelArg::new_mut(out.data_f32_mut()?))
            .arg(&GbKernelArg::new(input.data_f32()?))
            .arg(&n)
            .launch(elementwise_config(n))
            .map_err(|e| CudaError::Kernel(e.to_string()))?;
    }

    Ok(out)
}

/// GPU Round: out = round(x)
pub fn gpu_round(
    ctx: &IconnxCudaContext,
    kernels: &KernelCache,
    pool: &mut GpuMemoryPool,
    input: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    let n = input.len();
    let mut out = pool.get_tensor_f32(ctx, input.shape().to_vec())?;

    let func = kernels
        .get("round_kernel")
        .ok_or_else(|| CudaError::Kernel("round_kernel not found".into()))?;

    unsafe {
        ctx.stream()
            .launch_builder(func)
            .arg(&GbKernelArg::new_mut(out.data_f32_mut()?))
            .arg(&GbKernelArg::new(input.data_f32()?))
            .arg(&n)
            .launch(elementwise_config(n))
            .map_err(|e| CudaError::Kernel(e.to_string()))?;
    }

    Ok(out)
}

// =============================================================================
// Scalar Operations
// =============================================================================

/// GPU Add Scalar: out = a + scalar
pub fn gpu_add_scalar(
    ctx: &IconnxCudaContext,
    kernels: &KernelCache,
    pool: &mut GpuMemoryPool,
    a: &GpuTensor,
    scalar: f32,
) -> Result<GpuTensor, CudaError> {
    let n = a.len();
    let mut out = pool.get_tensor_f32(ctx, a.shape().to_vec())?;

    let func = kernels
        .get("add_scalar_kernel")
        .ok_or_else(|| CudaError::Kernel("add_scalar_kernel not found".into()))?;

    unsafe {
        ctx.stream()
            .launch_builder(func)
            .arg(&GbKernelArg::new_mut(out.data_f32_mut()?))
            .arg(&GbKernelArg::new(a.data_f32()?))
            .arg(&scalar)
            .arg(&n)
            .launch(elementwise_config(n))
            .map_err(|e| CudaError::Kernel(e.to_string()))?;
    }

    Ok(out)
}

/// GPU Mul Scalar: out = a * scalar
pub fn gpu_mul_scalar(
    ctx: &IconnxCudaContext,
    kernels: &KernelCache,
    pool: &mut GpuMemoryPool,
    a: &GpuTensor,
    scalar: f32,
) -> Result<GpuTensor, CudaError> {
    let n = a.len();
    let mut out = pool.get_tensor_f32(ctx, a.shape().to_vec())?;

    let func = kernels
        .get("mul_scalar_kernel")
        .ok_or_else(|| CudaError::Kernel("mul_scalar_kernel not found".into()))?;

    unsafe {
        ctx.stream()
            .launch_builder(func)
            .arg(&GbKernelArg::new_mut(out.data_f32_mut()?))
            .arg(&GbKernelArg::new(a.data_f32()?))
            .arg(&scalar)
            .arg(&n)
            .launch(elementwise_config(n))
            .map_err(|e| CudaError::Kernel(e.to_string()))?;
    }

    Ok(out)
}

pub mod fused;

#[cfg(test)]
mod tests;
