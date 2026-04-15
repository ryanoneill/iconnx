//! GPU Elementwise Kernels via garboard NVRTC + TypedKernel.
//!
//! This is the first kernel cache migrated from cudarc launches to
//! garboard's typed launch pattern (Phase 1 plan step 4). The kernel
//! source strings are unchanged — NVRTC compiles them either way. What
//! changes is the Rust side: `Module::typed_kernel::<Args>` gives us
//! compile-time parameter-type checking at the cost of the `unsafe`
//! declaration of the kernel's C signature.
//!
//! Kernels are resolved per-launch via `Module::typed_kernel`, which is
//! a thin wrapper over `cuModuleGetFunction`. That lookup cost (~200 ns)
//! is small compared to kernel execution; caching the typed kernel as a
//! struct field is not viable because `TypedKernel<'d, Args>` bakes
//! concrete reference lifetimes into `Args`, and `&mut T` is invariant
//! in `T` so a stored `'static`-ref kernel cannot be launched with
//! local-ref args. Per-launch resolution sidesteps the lifetime puzzle.

use garboard::{DeviceSlice, LaunchConfig, Module, Program};

use super::context::{CudaError, IconnxCudaContext};
use super::memory_pool::GpuMemoryPool;
use super::tensor::GpuTensor;

/// Compiled elementwise kernels loaded as a garboard `Module`.
pub struct KernelCache {
    module: Module<'static>,
}

impl KernelCache {
    /// Compile all elementwise kernels and create the cache.
    pub fn new(ctx: &IconnxCudaContext) -> Result<Self, CudaError> {
        let program =
            Program::compile_for_device(ELEMENTWISE_KERNELS, ctx.garboard_device(), &[])
                .map_err(|e| {
                    CudaError::Kernel(format!("NVRTC compilation failed: {}", e))
                })?;

        let module = ctx
            .garboard_device()
            .load_module(&program)
            .map_err(|e| CudaError::Kernel(format!("Module load failed: {}", e)))?;

        // Eagerly validate that every kernel name resolves — catches
        // typos or missing `extern "C"` at cache-construction time rather
        // than on first launch.
        for name in KERNEL_NAMES {
            module.function(name).map_err(|e| {
                CudaError::Kernel(format!("Failed to load kernel '{}': {}", name, e))
            })?;
        }

        Ok(Self { module })
    }

    /// Access to the underlying module for launch sites.
    pub(crate) fn module(&self) -> &Module<'static> {
        &self.module
    }
}

/// List of all kernel function names — validated at cache construction.
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

/// All elementwise CUDA kernel source code.
///
/// Largely unchanged from the cudarc implementation. The one tweak:
/// `-INFINITY` (from `math.h`) is replaced with the IEEE-754 bit
/// pattern via `__int_as_float(0xff800000)` because NVRTC under garboard
/// does not auto-include `math_constants.h` when a target architecture
/// is specified. `reduction.rs` has always used this workaround; it's
/// now applied consistently here too.
const ELEMENTWISE_KERNELS: &str = r#"
// Negative infinity as a float — bit pattern 0xff800000.
// NVRTC cannot include <math_constants.h> when a target architecture is
// specified, so the raw bit cast is the portable workaround.
#define ICONNX_NEG_INF_F __int_as_float(0xff800000)

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

extern "C" __global__ void sigmoid_kernel(float* out, const float* x, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = 1.0f / (1.0f + expf(-x[i]));
    }
}

extern "C" __global__ void tanh_kernel(float* out, const float* x, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = tanhf(x[i]);
    }
}

extern "C" __global__ void relu_kernel(float* out, const float* x, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = fmaxf(0.0f, x[i]);
    }
}

extern "C" __global__ void leaky_relu_kernel(float* out, const float* x, float alpha, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        out[i] = v >= 0.0f ? v : alpha * v;
    }
}

// Math operations

extern "C" __global__ void exp_kernel(float* out, const float* x, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = expf(x[i]);
    }
}

extern "C" __global__ void log_kernel(float* out, const float* x, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Guard against log(0) and log(negative)
        out[i] = x[i] > 0.0f ? logf(x[i]) : ICONNX_NEG_INF_F;
    }
}

extern "C" __global__ void sqrt_kernel(float* out, const float* x, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Guard against sqrt(negative)
        out[i] = x[i] >= 0.0f ? sqrtf(x[i]) : 0.0f;
    }
}

extern "C" __global__ void pow_kernel(float* out, const float* base, const float* exp_vals, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = powf(base[i], exp_vals[i]);
    }
}

extern "C" __global__ void sin_kernel(float* out, const float* x, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = sinf(x[i]);
    }
}

extern "C" __global__ void cos_kernel(float* out, const float* x, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = cosf(x[i]);
    }
}

extern "C" __global__ void abs_kernel(float* out, const float* x, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = fabsf(x[i]);
    }
}

extern "C" __global__ void neg_kernel(float* out, const float* x, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = -x[i];
    }
}

// Utility operations

extern "C" __global__ void clip_kernel(float* out, const float* x, float min_val, float max_val, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = fminf(fmaxf(x[i], min_val), max_val);
    }
}

extern "C" __global__ void floor_kernel(float* out, const float* x, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = floorf(x[i]);
    }
}

extern "C" __global__ void ceil_kernel(float* out, const float* x, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = ceilf(x[i]);
    }
}

extern "C" __global__ void round_kernel(float* out, const float* x, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = roundf(x[i]);
    }
}

// Scalar operations

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

// Scalar-from-pointer ops: scalar value is read from GPU memory (first element of pointer).
// This avoids the D2H transfer that would otherwise be needed to pass the scalar by value.

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

// ============================================================================
// Per-signature launch helpers
// ============================================================================
//
// Each helper factors out the boilerplate for a given kernel signature:
// resolve the typed kernel, build a LaunchConfig, launch on the garboard
// stream. The `unsafe { typed_kernel::<SIG>(name) }` declares the kernel's
// C signature; a mismatch would show up as a CUDA runtime error (wrong
// parameter layout) rather than a Rust type error.

/// Binary elementwise f32: `(out, a, b, n)`. Covers add/sub/mul/div/pow
/// and the scalar-from-pointer kernels.
fn launch_binary_f32(
    ctx: &IconnxCudaContext,
    kernels: &KernelCache,
    name: &'static str,
    out: &mut DeviceSlice<'static, f32>,
    a: &DeviceSlice<'static, f32>,
    b: &DeviceSlice<'static, f32>,
    n: usize,
) -> Result<(), CudaError> {
    // SAFETY: The kernel's C signature is
    // `(float* out, const float* a, const float* b, size_t n)`.
    let kernel = unsafe {
        kernels.module().typed_kernel::<(
            &mut DeviceSlice<'_, f32>,
            &DeviceSlice<'_, f32>,
            &DeviceSlice<'_, f32>,
            usize,
        )>(name)
    }
    .map_err(|e| CudaError::Kernel(format!("{} lookup failed: {}", name, e)))?;

    kernel
        .launch(
            ctx.garboard_stream(),
            &LaunchConfig::for_num_elems(n as u32),
            (out, a, b, n),
        )
        .map_err(|e| CudaError::Kernel(format!("{} launch failed: {}", name, e)))
}

/// Binary elementwise i64: `(out, a, b, n)`. Covers add_i64 / sub_i64 /
/// mul_i64 and the i64 scalar-from-pointer kernel.
fn launch_binary_i64(
    ctx: &IconnxCudaContext,
    kernels: &KernelCache,
    name: &'static str,
    out: &mut DeviceSlice<'static, i64>,
    a: &DeviceSlice<'static, i64>,
    b: &DeviceSlice<'static, i64>,
    n: usize,
) -> Result<(), CudaError> {
    // SAFETY: kernel signature is `(long long*, const long long*, const long long*, size_t)`.
    let kernel = unsafe {
        kernels.module().typed_kernel::<(
            &mut DeviceSlice<'_, i64>,
            &DeviceSlice<'_, i64>,
            &DeviceSlice<'_, i64>,
            usize,
        )>(name)
    }
    .map_err(|e| CudaError::Kernel(format!("{} lookup failed: {}", name, e)))?;

    kernel
        .launch(
            ctx.garboard_stream(),
            &LaunchConfig::for_num_elems(n as u32),
            (out, a, b, n),
        )
        .map_err(|e| CudaError::Kernel(format!("{} launch failed: {}", name, e)))
}

/// Unary elementwise f32: `(out, x, n)`. Covers all the simple unary
/// math and activation kernels.
fn launch_unary_f32(
    ctx: &IconnxCudaContext,
    kernels: &KernelCache,
    name: &'static str,
    out: &mut DeviceSlice<'static, f32>,
    x: &DeviceSlice<'static, f32>,
    n: usize,
) -> Result<(), CudaError> {
    // SAFETY: kernel signature is `(float* out, const float* x, size_t n)`.
    let kernel = unsafe {
        kernels.module().typed_kernel::<(
            &mut DeviceSlice<'_, f32>,
            &DeviceSlice<'_, f32>,
            usize,
        )>(name)
    }
    .map_err(|e| CudaError::Kernel(format!("{} lookup failed: {}", name, e)))?;

    kernel
        .launch(
            ctx.garboard_stream(),
            &LaunchConfig::for_num_elems(n as u32),
            (out, x, n),
        )
        .map_err(|e| CudaError::Kernel(format!("{} launch failed: {}", name, e)))
}

/// Unary f32 with one scalar f32 parameter: `(out, x, scalar, n)`.
/// Covers leaky_relu (alpha), add_scalar, mul_scalar.
fn launch_unary_f32_with_scalar(
    ctx: &IconnxCudaContext,
    kernels: &KernelCache,
    name: &'static str,
    out: &mut DeviceSlice<'static, f32>,
    x: &DeviceSlice<'static, f32>,
    scalar: f32,
    n: usize,
) -> Result<(), CudaError> {
    // SAFETY: kernel signature is `(float* out, const float* x, float scalar, size_t n)`.
    let kernel = unsafe {
        kernels.module().typed_kernel::<(
            &mut DeviceSlice<'_, f32>,
            &DeviceSlice<'_, f32>,
            f32,
            usize,
        )>(name)
    }
    .map_err(|e| CudaError::Kernel(format!("{} lookup failed: {}", name, e)))?;

    kernel
        .launch(
            ctx.garboard_stream(),
            &LaunchConfig::for_num_elems(n as u32),
            (out, x, scalar, n),
        )
        .map_err(|e| CudaError::Kernel(format!("{} launch failed: {}", name, e)))
}

/// Unary i64 with one scalar i64 parameter: `(out, a, scalar, n)`.
/// Covers add_i64_scalar.
#[allow(dead_code)] // add_i64_scalar is declared but not wired to a gpu_* wrapper yet.
fn launch_unary_i64_with_scalar(
    ctx: &IconnxCudaContext,
    kernels: &KernelCache,
    name: &'static str,
    out: &mut DeviceSlice<'static, i64>,
    a: &DeviceSlice<'static, i64>,
    scalar: i64,
    n: usize,
) -> Result<(), CudaError> {
    // SAFETY: kernel signature is `(long long* out, const long long* a, long long scalar, size_t n)`.
    let kernel = unsafe {
        kernels.module().typed_kernel::<(
            &mut DeviceSlice<'_, i64>,
            &DeviceSlice<'_, i64>,
            i64,
            usize,
        )>(name)
    }
    .map_err(|e| CudaError::Kernel(format!("{} lookup failed: {}", name, e)))?;

    kernel
        .launch(
            ctx.garboard_stream(),
            &LaunchConfig::for_num_elems(n as u32),
            (out, a, scalar, n),
        )
        .map_err(|e| CudaError::Kernel(format!("{} launch failed: {}", name, e)))
}

/// Clip: `(out, x, min, max, n)`.
fn launch_clip(
    ctx: &IconnxCudaContext,
    kernels: &KernelCache,
    out: &mut DeviceSlice<'static, f32>,
    x: &DeviceSlice<'static, f32>,
    min_val: f32,
    max_val: f32,
    n: usize,
) -> Result<(), CudaError> {
    // SAFETY: clip_kernel signature is
    // `(float* out, const float* x, float min_val, float max_val, size_t n)`.
    let kernel = unsafe {
        kernels.module().typed_kernel::<(
            &mut DeviceSlice<'_, f32>,
            &DeviceSlice<'_, f32>,
            f32,
            f32,
            usize,
        )>("clip_kernel")
    }
    .map_err(|e| CudaError::Kernel(format!("clip_kernel lookup failed: {}", e)))?;

    kernel
        .launch(
            ctx.garboard_stream(),
            &LaunchConfig::for_num_elems(n as u32),
            (out, x, min_val, max_val, n),
        )
        .map_err(|e| CudaError::Kernel(format!("clip_kernel launch failed: {}", e)))
}

// =============================================================================
// Binary Operations
// =============================================================================

/// GPU Add: out = a + b (elementwise with broadcasting).
/// Supports Float32 and Int64 dtypes.
pub fn gpu_add(
    ctx: &IconnxCudaContext,
    kernels: &KernelCache,
    pool: &mut GpuMemoryPool,
    a: &GpuTensor,
    b: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    // Handle scalar broadcasting: if one tensor is scalar (len=1), broadcast it.
    if b.len() == 1 && a.len() > 1 {
        return gpu_add_broadcast_scalar(ctx, kernels, pool, a, b);
    }
    if a.len() == 1 && b.len() > 1 {
        return gpu_add_broadcast_scalar(ctx, kernels, pool, b, a);
    }

    // Regular elementwise add - shapes must match.
    if a.shape() != b.shape() {
        return Err(CudaError::Kernel(format!(
            "Add: shapes must match or be broadcastable, got {:?} and {:?}",
            a.shape(),
            b.shape()
        )));
    }

    let n = a.len();

    // Dispatch based on dtype.
    match (a, b) {
        (GpuTensor::Int64 { .. }, GpuTensor::Int64 { .. }) => {
            let mut out = pool.get_tensor_i64(ctx, a.shape().to_vec())?;
            launch_binary_i64(
                ctx,
                kernels,
                "add_i64_kernel",
                out.data_i64_mut()?,
                a.data_i64()?,
                b.data_i64()?,
                n,
            )?;
            Ok(out)
        }
        _ => {
            let mut out = pool.get_tensor_f32(ctx, a.shape().to_vec())?;
            launch_binary_f32(
                ctx,
                kernels,
                "add_kernel",
                out.data_f32_mut()?,
                a.data_f32()?,
                b.data_f32()?,
                n,
            )?;
            Ok(out)
        }
    }
}

/// GPU Add with scalar tensor: out = tensor + scalar_tensor (broadcast
/// scalar to all elements). Internal — used by `gpu_add` when one
/// operand is scalar. Uses pointer-based kernels to avoid D2H transfer.
fn gpu_add_broadcast_scalar(
    ctx: &IconnxCudaContext,
    kernels: &KernelCache,
    pool: &mut GpuMemoryPool,
    tensor: &GpuTensor,
    scalar: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    let n = tensor.len();

    match (tensor, scalar) {
        (GpuTensor::Int64 { .. }, GpuTensor::Int64 { .. }) => {
            let mut out = pool.get_tensor_i64(ctx, tensor.shape().to_vec())?;
            launch_binary_i64(
                ctx,
                kernels,
                "add_i64_scalar_ptr_kernel",
                out.data_i64_mut()?,
                tensor.data_i64()?,
                scalar.data_i64()?,
                n,
            )?;
            Ok(out)
        }
        _ => {
            let mut out = pool.get_tensor_f32(ctx, tensor.shape().to_vec())?;
            launch_binary_f32(
                ctx,
                kernels,
                "add_scalar_ptr_kernel",
                out.data_f32_mut()?,
                tensor.data_f32()?,
                scalar.data_f32()?,
                n,
            )?;
            Ok(out)
        }
    }
}

/// GPU Sub: out = a - b (elementwise).
pub fn gpu_sub(
    ctx: &IconnxCudaContext,
    kernels: &KernelCache,
    pool: &mut GpuMemoryPool,
    a: &GpuTensor,
    b: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    assert_eq!(a.shape(), b.shape(), "Sub: shapes must match");
    let n = a.len();

    match (a, b) {
        (GpuTensor::Int64 { .. }, GpuTensor::Int64 { .. }) => {
            let mut out = pool.get_tensor_i64(ctx, a.shape().to_vec())?;
            launch_binary_i64(
                ctx,
                kernels,
                "sub_i64_kernel",
                out.data_i64_mut()?,
                a.data_i64()?,
                b.data_i64()?,
                n,
            )?;
            Ok(out)
        }
        _ => {
            let mut out = pool.get_tensor_f32(ctx, a.shape().to_vec())?;
            launch_binary_f32(
                ctx,
                kernels,
                "sub_kernel",
                out.data_f32_mut()?,
                a.data_f32()?,
                b.data_f32()?,
                n,
            )?;
            Ok(out)
        }
    }
}

/// GPU Mul: out = a * b (elementwise).
pub fn gpu_mul(
    ctx: &IconnxCudaContext,
    kernels: &KernelCache,
    pool: &mut GpuMemoryPool,
    a: &GpuTensor,
    b: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    assert_eq!(a.shape(), b.shape(), "Mul: shapes must match");
    let n = a.len();

    match (a, b) {
        (GpuTensor::Int64 { .. }, GpuTensor::Int64 { .. }) => {
            let mut out = pool.get_tensor_i64(ctx, a.shape().to_vec())?;
            launch_binary_i64(
                ctx,
                kernels,
                "mul_i64_kernel",
                out.data_i64_mut()?,
                a.data_i64()?,
                b.data_i64()?,
                n,
            )?;
            Ok(out)
        }
        _ => {
            let mut out = pool.get_tensor_f32(ctx, a.shape().to_vec())?;
            launch_binary_f32(
                ctx,
                kernels,
                "mul_kernel",
                out.data_f32_mut()?,
                a.data_f32()?,
                b.data_f32()?,
                n,
            )?;
            Ok(out)
        }
    }
}

/// GPU Div: out = a / b (elementwise).
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
    launch_binary_f32(
        ctx,
        kernels,
        "div_kernel",
        out.data_f32_mut()?,
        a.data_f32()?,
        b.data_f32()?,
        n,
    )?;
    Ok(out)
}

// =============================================================================
// Activation Functions
// =============================================================================

/// GPU Sigmoid: out = 1 / (1 + exp(-x)).
pub fn gpu_sigmoid(
    ctx: &IconnxCudaContext,
    kernels: &KernelCache,
    pool: &mut GpuMemoryPool,
    input: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    let n = input.len();
    let mut out = pool.get_tensor_f32(ctx, input.shape().to_vec())?;
    launch_unary_f32(
        ctx,
        kernels,
        "sigmoid_kernel",
        out.data_f32_mut()?,
        input.data_f32()?,
        n,
    )?;
    Ok(out)
}

/// GPU Tanh: out = tanh(x).
pub fn gpu_tanh(
    ctx: &IconnxCudaContext,
    kernels: &KernelCache,
    pool: &mut GpuMemoryPool,
    input: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    let n = input.len();
    let mut out = pool.get_tensor_f32(ctx, input.shape().to_vec())?;
    launch_unary_f32(
        ctx,
        kernels,
        "tanh_kernel",
        out.data_f32_mut()?,
        input.data_f32()?,
        n,
    )?;
    Ok(out)
}

/// GPU ReLU: out = max(0, x).
pub fn gpu_relu(
    ctx: &IconnxCudaContext,
    kernels: &KernelCache,
    pool: &mut GpuMemoryPool,
    input: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    let n = input.len();
    let mut out = pool.get_tensor_f32(ctx, input.shape().to_vec())?;
    launch_unary_f32(
        ctx,
        kernels,
        "relu_kernel",
        out.data_f32_mut()?,
        input.data_f32()?,
        n,
    )?;
    Ok(out)
}

/// GPU LeakyReLU: out = x >= 0 ? x : alpha * x.
pub fn gpu_leaky_relu(
    ctx: &IconnxCudaContext,
    kernels: &KernelCache,
    pool: &mut GpuMemoryPool,
    input: &GpuTensor,
    alpha: f32,
) -> Result<GpuTensor, CudaError> {
    let n = input.len();
    let mut out = pool.get_tensor_f32(ctx, input.shape().to_vec())?;
    launch_unary_f32_with_scalar(
        ctx,
        kernels,
        "leaky_relu_kernel",
        out.data_f32_mut()?,
        input.data_f32()?,
        alpha,
        n,
    )?;
    Ok(out)
}

// =============================================================================
// Math Operations
// =============================================================================

/// GPU Exp: out = exp(x).
pub fn gpu_exp(
    ctx: &IconnxCudaContext,
    kernels: &KernelCache,
    pool: &mut GpuMemoryPool,
    input: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    let n = input.len();
    let mut out = pool.get_tensor_f32(ctx, input.shape().to_vec())?;
    launch_unary_f32(
        ctx,
        kernels,
        "exp_kernel",
        out.data_f32_mut()?,
        input.data_f32()?,
        n,
    )?;
    Ok(out)
}

/// GPU Log: out = log(x).
pub fn gpu_log(
    ctx: &IconnxCudaContext,
    kernels: &KernelCache,
    pool: &mut GpuMemoryPool,
    input: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    let n = input.len();
    let mut out = pool.get_tensor_f32(ctx, input.shape().to_vec())?;
    launch_unary_f32(
        ctx,
        kernels,
        "log_kernel",
        out.data_f32_mut()?,
        input.data_f32()?,
        n,
    )?;
    Ok(out)
}

/// GPU Sqrt: out = sqrt(x).
pub fn gpu_sqrt(
    ctx: &IconnxCudaContext,
    kernels: &KernelCache,
    pool: &mut GpuMemoryPool,
    input: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    let n = input.len();
    let mut out = pool.get_tensor_f32(ctx, input.shape().to_vec())?;
    launch_unary_f32(
        ctx,
        kernels,
        "sqrt_kernel",
        out.data_f32_mut()?,
        input.data_f32()?,
        n,
    )?;
    Ok(out)
}

/// GPU Pow: out = base ^ exp (elementwise).
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
    launch_binary_f32(
        ctx,
        kernels,
        "pow_kernel",
        out.data_f32_mut()?,
        base.data_f32()?,
        exp.data_f32()?,
        n,
    )?;
    Ok(out)
}

/// GPU Sin: out = sin(x).
pub fn gpu_sin(
    ctx: &IconnxCudaContext,
    kernels: &KernelCache,
    pool: &mut GpuMemoryPool,
    input: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    let n = input.len();
    let mut out = pool.get_tensor_f32(ctx, input.shape().to_vec())?;
    launch_unary_f32(
        ctx,
        kernels,
        "sin_kernel",
        out.data_f32_mut()?,
        input.data_f32()?,
        n,
    )?;
    Ok(out)
}

/// GPU Cos: out = cos(x).
pub fn gpu_cos(
    ctx: &IconnxCudaContext,
    kernels: &KernelCache,
    pool: &mut GpuMemoryPool,
    input: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    let n = input.len();
    let mut out = pool.get_tensor_f32(ctx, input.shape().to_vec())?;
    launch_unary_f32(
        ctx,
        kernels,
        "cos_kernel",
        out.data_f32_mut()?,
        input.data_f32()?,
        n,
    )?;
    Ok(out)
}

/// GPU Abs: out = |x|.
pub fn gpu_abs(
    ctx: &IconnxCudaContext,
    kernels: &KernelCache,
    pool: &mut GpuMemoryPool,
    input: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    let n = input.len();
    let mut out = pool.get_tensor_f32(ctx, input.shape().to_vec())?;
    launch_unary_f32(
        ctx,
        kernels,
        "abs_kernel",
        out.data_f32_mut()?,
        input.data_f32()?,
        n,
    )?;
    Ok(out)
}

/// GPU Neg: out = -x.
pub fn gpu_neg(
    ctx: &IconnxCudaContext,
    kernels: &KernelCache,
    pool: &mut GpuMemoryPool,
    input: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    let n = input.len();
    let mut out = pool.get_tensor_f32(ctx, input.shape().to_vec())?;
    launch_unary_f32(
        ctx,
        kernels,
        "neg_kernel",
        out.data_f32_mut()?,
        input.data_f32()?,
        n,
    )?;
    Ok(out)
}

// =============================================================================
// Utility Operations
// =============================================================================

/// GPU Clip: out = clamp(x, min, max).
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
    launch_clip(
        ctx,
        kernels,
        out.data_f32_mut()?,
        input.data_f32()?,
        min_val,
        max_val,
        n,
    )?;
    Ok(out)
}

/// GPU Floor: out = floor(x).
pub fn gpu_floor(
    ctx: &IconnxCudaContext,
    kernels: &KernelCache,
    pool: &mut GpuMemoryPool,
    input: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    let n = input.len();
    let mut out = pool.get_tensor_f32(ctx, input.shape().to_vec())?;
    launch_unary_f32(
        ctx,
        kernels,
        "floor_kernel",
        out.data_f32_mut()?,
        input.data_f32()?,
        n,
    )?;
    Ok(out)
}

/// GPU Ceil: out = ceil(x).
pub fn gpu_ceil(
    ctx: &IconnxCudaContext,
    kernels: &KernelCache,
    pool: &mut GpuMemoryPool,
    input: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    let n = input.len();
    let mut out = pool.get_tensor_f32(ctx, input.shape().to_vec())?;
    launch_unary_f32(
        ctx,
        kernels,
        "ceil_kernel",
        out.data_f32_mut()?,
        input.data_f32()?,
        n,
    )?;
    Ok(out)
}

/// GPU Round: out = round(x).
pub fn gpu_round(
    ctx: &IconnxCudaContext,
    kernels: &KernelCache,
    pool: &mut GpuMemoryPool,
    input: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    let n = input.len();
    let mut out = pool.get_tensor_f32(ctx, input.shape().to_vec())?;
    launch_unary_f32(
        ctx,
        kernels,
        "round_kernel",
        out.data_f32_mut()?,
        input.data_f32()?,
        n,
    )?;
    Ok(out)
}

// =============================================================================
// Scalar Operations
// =============================================================================

/// GPU Add Scalar: out = a + scalar.
pub fn gpu_add_scalar(
    ctx: &IconnxCudaContext,
    kernels: &KernelCache,
    pool: &mut GpuMemoryPool,
    a: &GpuTensor,
    scalar: f32,
) -> Result<GpuTensor, CudaError> {
    let n = a.len();
    let mut out = pool.get_tensor_f32(ctx, a.shape().to_vec())?;
    launch_unary_f32_with_scalar(
        ctx,
        kernels,
        "add_scalar_kernel",
        out.data_f32_mut()?,
        a.data_f32()?,
        scalar,
        n,
    )?;
    Ok(out)
}

/// GPU Mul Scalar: out = a * scalar.
pub fn gpu_mul_scalar(
    ctx: &IconnxCudaContext,
    kernels: &KernelCache,
    pool: &mut GpuMemoryPool,
    a: &GpuTensor,
    scalar: f32,
) -> Result<GpuTensor, CudaError> {
    let n = a.len();
    let mut out = pool.get_tensor_f32(ctx, a.shape().to_vec())?;
    launch_unary_f32_with_scalar(
        ctx,
        kernels,
        "mul_scalar_kernel",
        out.data_f32_mut()?,
        a.data_f32()?,
        scalar,
        n,
    )?;
    Ok(out)
}

pub mod fused;

#[cfg(test)]
mod tests;
