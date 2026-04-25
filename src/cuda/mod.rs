/// CUDA GPU acceleration module for Iconnx.
///
/// Provides GPU-accelerated inference on top of [`garboard`], which handles
/// device discovery, memory allocation, stream management, NVRTC kernel
/// compilation, cuBLAS GEMMs, and cuDNN conv/RNN operations.
///
/// Key components:
/// - `IconnxCudaContext` — Device management and memory allocation
/// - `GpuTensor` — GPU tensor storage (wraps garboard `DeviceSlice`)
/// - Operators: GPU-accelerated operator implementations
mod context;
mod conv;
pub mod cudnn;
pub mod cufft;
pub mod executor;
pub mod inference;
mod kernels;
mod lstm;
mod matmul;
mod memory_pool;
mod ops;
mod reduction;
mod tensor;

// Debug inference modules (compile-gated)
#[cfg(feature = "debug-inference")]
pub mod checkpoints;
#[cfg(feature = "debug-inference")]
pub mod differential;
#[cfg(feature = "debug-inference")]
pub mod validation;

// Re-exports for test access - these are used by tests/iconnx/* even though
// rustc doesn't see the usage within the library crate itself
#[allow(unused_imports)]
pub use context::{CudaError, IconnxCudaContext};
#[allow(unused_imports)]
pub use conv::{
    add_bias, gpu_col2im, gpu_col2im_1d, gpu_conv1d, gpu_conv2d, gpu_conv_transpose_1d,
    gpu_conv_transpose_2d, gpu_im2col, gpu_im2col_1d, Conv1dParams, Conv2dParams, ConvKernelCache,
};
#[allow(unused_imports)]
pub use inference::{GpuGraphExecutor, NodeExecutionLog, ProfileData, TensorLocation};
#[allow(unused_imports)]
pub use kernels::{
    gpu_abs,
    // Binary ops
    gpu_add,
    // Scalar ops
    gpu_add_scalar,
    gpu_ceil,
    // Utility ops
    gpu_clip,
    gpu_cos,
    gpu_div,
    gpu_erf,
    // Math ops
    gpu_exp,
    gpu_floor,
    gpu_leaky_relu,
    gpu_log,
    gpu_mul,
    gpu_mul_scalar,
    gpu_neg,
    gpu_pow,
    gpu_relu,
    gpu_round,
    // Activations
    gpu_sigmoid,
    gpu_sin,
    gpu_sqrt,
    gpu_sub,
    gpu_tanh,
    KernelCache,
};
#[allow(unused_imports)]
pub use kernels::fused::{
    gpu_fused_add_mul, gpu_fused_add_mul_add, gpu_fused_div_mul, gpu_fused_div_rsqrt,
    gpu_fused_gelu, gpu_fused_mul_add, gpu_fused_mul_sin_pow_mul_add, gpu_fused_sub_mul,
    FusedKernelCache,
};
#[allow(unused_imports)]
pub use lstm::{gpu_lstm, LstmKernelCache};
#[allow(unused_imports)]
pub use matmul::{gpu_batched_matmul, gpu_gemm, gpu_matmul, gpu_matmul_2d_3d};
#[allow(unused_imports)]
pub use ops::{
    // Logical ops
    gpu_and,
    // Math ops
    gpu_atan,
    gpu_copy,
    gpu_cumsum,
    // Quantization ops (WS-4)
    gpu_dequantize_linear,
    // Comparison ops
    gpu_equal,
    gpu_gather,
    gpu_greater,
    gpu_greater_or_equal,
    gpu_less,
    gpu_less_or_equal,
    gpu_not,
    gpu_not_equal,
    gpu_or,
    // Sequence ops
    gpu_range,
    gpu_slice_contiguous,
    // Shape/layout ops
    gpu_transpose_2d,
    // Conditional ops
    gpu_where,
    OpsKernelCache,
};
#[allow(unused_imports)]
pub use reduction::{
    gpu_layer_norm, gpu_reduce_mean_axis, gpu_reduce_sum_all, gpu_reduce_sum_axis, gpu_softmax,
    ReductionKernelCache,
};
#[allow(unused_imports)]
pub use memory_pool::GpuMemoryPool;
#[allow(unused_imports)]
pub use tensor::{GpuTensor, TypedSlice};

/// Check if CUDA is available on this system.
///
/// Probes garboard's CUDA backend by attempting to create a `Device` for
/// ordinal 0. Returns `false` if no CUDA driver / device is present.
pub fn cuda_available() -> bool {
    garboard::Device::new_cuda(0).is_ok()
}

/// Get the number of available CUDA devices.
///
/// Garboard does not expose `cuDeviceGetCount` on the safe surface, so we
/// probe ordinals 0..16 until `Device::new_cuda` fails. The upper bound
/// is a sanity limit; no realistic system has more than 16 GPUs visible
/// to a single process.
pub fn cuda_device_count() -> usize {
    let mut count = 0i32;
    while garboard::Device::new_cuda(count).is_ok() {
        count += 1;
        if count > 16 {
            break;
        }
    }
    count as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_availability() {
        // This test will pass even if CUDA is not available
        let available = cuda_available();
        println!("CUDA available: {}", available);

        if available {
            let count = cuda_device_count();
            println!("CUDA devices: {}", count);
            assert!(count > 0);
        }
    }
}
