//! Kernel cache for ops module (garboard `Module`).
//!
//! Mixed-backend: most ops kernels run through garboard's TypedKernel,
//! but a handful of layout/ops kernels exceed garboard's 12-element
//! `KernelArgs` tuple cap and stay on cudarc via the `GbKernelArg`
//! bridge until the cap is lifted upstream. Both backends load the
//! same source; unused entries are harmless.

use super::kernels::{KERNEL_NAMES, OPS_KERNELS};
use crate::cuda::context::{CudaError, IconnxCudaContext};
use cudarc::driver::{CudaFunction, CudaModule};
use cudarc::nvrtc::compile_ptx;
use garboard::{Module, Program};
use std::sync::Arc;

/// Names of ops kernels that currently launch via cudarc. A mix of:
/// - kernels with more than 12 parameters (over garboard's KernelArgs
///   tuple cap — will migrate when the cap is lifted upstream);
/// - kernels in layout/mod.rs that remain on cudarc for this PR to
///   keep its scope bounded. Those will migrate in a follow-up PR
///   alongside a helper-function refactor of layout.
const CUDARC_BRIDGE_KERNELS: &[&str] = &[
    // Layout/mod.rs — deferred to follow-up migration PR.
    "concat_kernel",
    "concat_i32_kernel",
    "concat_i64_kernel",
    "copy_kernel",
    "expand_kernel",
    "expand_i32_kernel",
    "expand_i64_kernel",
    "fill_kernel",
    "gather_kernel",
    "gather_i64_kernel",
    "pad_kernel",
    "pad_edge_kernel",
    "pad_reflect_kernel",
    // pad_3d_scalar_kernel — migrated to garboard (packed params).
    // pad_4d_scalar_kernel — migrated to garboard (packed params).
    // pad_reflect_3d_kernel — migrated to garboard (packed params).
    "resize_nearest_kernel",
    // resize_3d_scalar_kernel — migrated to garboard (packed params).
    // resize_4d_scalar_kernel — migrated to garboard (packed params).
    "resize_linear_3d_kernel",
    // resize_linear_4d_kernel — migrated to garboard (packed params).
    "scatter_nd_kernel",
    "slice_kernel",
    "slice_nd_kernel",
    "slice_nd_i32_kernel",
    "slice_nd_i64_kernel",
    "slice_3d_scalar_kernel",
    "slice_3d_i64_scalar_kernel",
    // slice_4d_scalar_kernel — migrated to garboard (packed params).
    // slice_4d_i64_scalar_kernel — migrated to garboard (packed params).
    "transpose_2d_kernel",
    "transpose_general_kernel",
    "transpose_general_i64_kernel",
    "where_kernel",
    "where_i32_kernel",
    "where_i64_kernel",
];

/// Compiled utility-operation kernels.
pub struct OpsKernelCache {
    garboard_module: Module<'static>,
    #[allow(dead_code)]
    cudarc_module: Arc<CudaModule>,
    cudarc_functions: std::collections::HashMap<&'static str, CudaFunction>,
}

impl OpsKernelCache {
    /// Compile all utility kernels.
    pub fn new(ctx: &IconnxCudaContext) -> Result<Self, CudaError> {
        // Garboard side.
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

        // Cudarc side: pre-resolve only the bridge kernels.
        let ptx = compile_ptx(OPS_KERNELS)
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

    /// Access to the garboard module for migrated kernels.
    pub(crate) fn module(&self) -> &Module<'static> {
        &self.garboard_module
    }

    /// Cudarc function for the over-12-arg bridge kernels. Returns
    /// `None` for names outside [`CUDARC_BRIDGE_KERNELS`].
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
        let cache = OpsKernelCache::new(&ctx).expect("Failed to compile kernels");

        // Garboard module should have every kernel.
        for name in [
            "transpose_2d_kernel",
            "where_kernel",
            "equal_kernel",
            "less_kernel",
            "greater_kernel",
            "gather_kernel",
            "slice_kernel",
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
            "scatter_nd_kernel",
        ] {
            assert!(
                cache.module().function(name).is_ok(),
                "kernel {name} should be present in the garboard ops module"
            );
        }

        // Bridge kernels have cudarc functions pre-resolved.
        assert!(cache.cudarc_function("slice_nd_kernel").is_some());
        assert!(cache.cudarc_function("transpose_general_kernel").is_some());
        // Comparison kernels are migrated to garboard; no cudarc entry.
        assert!(cache.cudarc_function("equal_kernel").is_none());

        println!("All ops kernels compiled successfully!");
    }
}
