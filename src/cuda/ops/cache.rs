//! Kernel cache for ops module (garboard `Module`).
//!
//! All ops kernels are loaded into a single garboard `Module` and launched
//! via the typed-kernel API on garboard's user stream.

use super::kernels::{KERNEL_NAMES, OPS_KERNELS};
use crate::cuda::context::{CudaError, IconnxCudaContext};
use garboard::{Module, Program};

/// Compiled utility-operation kernels.
pub struct OpsKernelCache {
    garboard_module: Module<'static>,
}

impl OpsKernelCache {
    /// Compile all utility kernels.
    pub fn new(ctx: &IconnxCudaContext) -> Result<Self, CudaError> {
        let garboard_program =
            Program::compile_for_device(OPS_KERNELS, ctx.garboard_device(), &[]).map_err(|e| {
                CudaError::Kernel(format!("NVRTC compilation failed: {}", e))
            })?;
        let garboard_module = ctx
            .garboard_device()
            .load_module(&garboard_program)
            .map_err(|e| CudaError::Kernel(format!("Module load failed: {}", e)))?;
        for name in KERNEL_NAMES {
            garboard_module.function(name).map_err(|e| {
                CudaError::Kernel(format!("Failed to load kernel '{}': {}", name, e))
            })?;
        }

        Ok(Self { garboard_module })
    }

    /// Access to the garboard module for kernel lookups.
    pub(crate) fn module(&self) -> &Module<'static> {
        &self.garboard_module
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_kernel_compilation() {
        let ctx = IconnxCudaContext::new().expect("Failed to create CUDA context");
        let cache = OpsKernelCache::new(&ctx).expect("Failed to compile kernels");

        // Spot-check a representative slice of kernel names — the full
        // KERNEL_NAMES list is already validated by `OpsKernelCache::new`.
        for name in [
            "transpose_2d_kernel",
            "transpose_general_kernel",
            "transpose_general_i32_kernel",
            "transpose_general_i64_kernel",
            "where_kernel",
            "equal_kernel",
            "less_kernel",
            "greater_kernel",
            "gather_kernel",
            "slice_kernel",
            "slice_nd_kernel",
            "slice_nd_i32_kernel",
            "slice_nd_i64_kernel",
            "greater_or_equal_kernel",
            "less_or_equal_kernel",
            "not_equal_kernel",
            "and_kernel",
            "or_kernel",
            "not_kernel",
            "atan_kernel",
            "range_kernel",
            "cumsum_kernel",
            "copy_kernel",
            "expand_kernel",
            "pad_kernel",
            "pad_edge_kernel",
            "pad_reflect_kernel",
            "resize_nearest_kernel",
            "resize_linear_3d_kernel",
            "scatter_nd_kernel",
        ] {
            assert!(
                cache.module().function(name).is_ok(),
                "kernel {name} should be present in the garboard ops module"
            );
        }

        println!("All ops kernels compiled successfully!");
    }
}
