//! Kernel cache for convolution operations.
//!
//! Mixed-backend: most kernels (1D im2col, 1D col2im, add_bias) run
//! through garboard's `Module` + `TypedKernel`. The 2D im2col/col2im
//! kernels have 14 parameters each, which exceeds garboard's current
//! `KernelArgs` tuple impl cap of 12. Those two kernels remain on
//! cudarc until garboard's tuple impls are extended. Both backends load
//! the full kernel source — the unused entries are harmless and keep
//! the source a single authoritative copy.

use super::kernels::{CONV_KERNELS, KERNEL_NAMES};
use crate::cuda::context::{CudaError, IconnxCudaContext};
use cudarc::driver::{CudaFunction, CudaModule};
use cudarc::nvrtc::compile_ptx;
use garboard::{Module, Program};
use std::sync::Arc;

/// Names of kernels that currently launch via cudarc due to the 12-arg
/// `KernelArgs` cap in garboard. See `cache.rs` module docs.
const CUDARC_BRIDGE_KERNELS: &[&str] = &["im2col_kernel", "col2im_kernel"];

/// Compiled convolution kernels.
pub struct ConvKernelCache {
    /// Garboard module — holds all kernels; used by all launches except
    /// the 14-arg 2D im2col/col2im pair.
    garboard_module: Module<'static>,
    /// Cudarc module for the two 14-arg kernels that exceed garboard's
    /// 12-tuple limit. Kept alive for the duration of the cache.
    #[allow(dead_code)]
    cudarc_module: Arc<CudaModule>,
    /// Pre-resolved cudarc functions for the bridge kernels, keyed by
    /// name (only `im2col_kernel` and `col2im_kernel`).
    cudarc_functions: std::collections::HashMap<&'static str, CudaFunction>,
}

impl ConvKernelCache {
    /// Compile all convolution kernels and create the cache.
    pub fn new(ctx: &IconnxCudaContext) -> Result<Self, CudaError> {
        // Garboard side: compile once, validate every name resolves.
        let garboard_program =
            Program::compile_for_device(CONV_KERNELS, ctx.garboard_device(), &[])
                .map_err(|e| CudaError::Kernel(format!("NVRTC compilation failed: {}", e)))?;
        let garboard_module = ctx
            .garboard_device()
            .load_module(&garboard_program)
            .map_err(|e| CudaError::Kernel(format!("Module load failed: {}", e)))?;
        for name in KERNEL_NAMES {
            garboard_module.function(name).map_err(|e| {
                CudaError::Kernel(format!("Failed to load kernel '{}': {}", name, e))
            })?;
        }

        // Cudarc side: compile the same source, pre-resolve only the
        // kernels that still need a cudarc launch path.
        let ptx = compile_ptx(CONV_KERNELS)
            .map_err(|e| CudaError::Kernel(format!("cudarc NVRTC compile failed: {:?}", e)))?;
        let cudarc_module = ctx
            .context()
            .load_module(ptx)
            .map_err(|e| CudaError::Kernel(format!("cudarc module load failed: {}", e)))?;
        let mut cudarc_functions = std::collections::HashMap::new();
        for name in CUDARC_BRIDGE_KERNELS {
            let func = cudarc_module.load_function(name).map_err(|e| {
                CudaError::Kernel(format!("cudarc load {} failed: {}", name, e))
            })?;
            cudarc_functions.insert(*name, func);
        }

        Ok(Self {
            garboard_module,
            cudarc_module,
            cudarc_functions,
        })
    }

    /// Access to the garboard module for the migrated kernels.
    pub(crate) fn module(&self) -> &Module<'static> {
        &self.garboard_module
    }

    /// Cudarc function for the 14-arg bridge kernels. Returns `None`
    /// for names outside [`CUDARC_BRIDGE_KERNELS`].
    pub(crate) fn cudarc_function(&self, name: &'static str) -> Option<&CudaFunction> {
        self.cudarc_functions.get(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_kernel_compilation() {
        let ctx = IconnxCudaContext::new().expect("Failed to create CUDA context");
        let cache = ConvKernelCache::new(&ctx).expect("Failed to compile kernels");

        // Garboard module should have every kernel.
        for name in ["im2col_kernel", "col2im_kernel", "im2col_1d_kernel", "col2im_1d_kernel", "add_bias_kernel"] {
            assert!(
                cache.module().function(name).is_ok(),
                "kernel {name} should be present in the garboard module"
            );
        }

        // Cudarc module holds pre-resolved functions for the 14-arg kernels.
        assert!(cache.cudarc_function("im2col_kernel").is_some());
        assert!(cache.cudarc_function("col2im_kernel").is_some());
        assert!(cache.cudarc_function("im2col_1d_kernel").is_none());

        println!("All convolution kernels compiled successfully!");
    }
}
