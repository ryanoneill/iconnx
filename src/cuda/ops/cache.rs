//! Kernel cache for ops module

use super::kernels::{KERNEL_NAMES, OPS_KERNELS};
use crate::cuda::context::{CudaError, IconnxCudaContext};
use cudarc::driver::{CudaFunction, CudaModule};
use cudarc::nvrtc::compile_ptx;
use std::collections::HashMap;
use std::sync::Arc;

/// Manages compiled utility operation kernels
pub struct OpsKernelCache {
    #[allow(dead_code)]
    module: Arc<CudaModule>,
    functions: HashMap<&'static str, CudaFunction>,
}

impl OpsKernelCache {
    /// Compile all utility kernels
    pub fn new(ctx: &IconnxCudaContext) -> Result<Self, CudaError> {
        let ptx = compile_ptx(OPS_KERNELS)
            .map_err(|e| CudaError::Kernel(format!("NVRTC compilation failed: {:?}", e)))?;

        let module = ctx
            .context()
            .load_module(ptx)
            .map_err(|e| CudaError::Kernel(format!("Module load failed: {}", e)))?;

        let mut functions = HashMap::new();

        for name in KERNEL_NAMES {
            let func = module.load_function(name).map_err(|e| {
                CudaError::Kernel(format!("Failed to load kernel '{}': {}", name, e))
            })?;
            functions.insert(*name, func);
        }

        Ok(Self { module, functions })
    }

    /// Get a kernel function by name
    pub fn get(&self, name: &'static str) -> Option<&CudaFunction> {
        self.functions.get(name)
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

        // Original kernels
        assert!(cache.get("transpose_2d_kernel").is_some());
        assert!(cache.get("where_kernel").is_some());
        assert!(cache.get("equal_kernel").is_some());
        assert!(cache.get("less_kernel").is_some());
        assert!(cache.get("greater_kernel").is_some());
        assert!(cache.get("gather_kernel").is_some());
        assert!(cache.get("slice_kernel").is_some());

        // New comparison/logical kernels
        assert!(cache.get("greater_or_equal_kernel").is_some());
        assert!(cache.get("less_or_equal_kernel").is_some());
        assert!(cache.get("not_equal_kernel").is_some());
        assert!(cache.get("and_kernel").is_some());
        assert!(cache.get("or_kernel").is_some());
        assert!(cache.get("not_kernel").is_some());

        // New utility kernels
        assert!(cache.get("atan_kernel").is_some());
        assert!(cache.get("range_kernel").is_some());
        assert!(cache.get("cumsum_kernel").is_some());
        assert!(cache.get("copy_kernel").is_some());
        assert!(cache.get("expand_kernel").is_some());
        assert!(cache.get("pad_kernel").is_some());
        assert!(cache.get("scatter_nd_kernel").is_some());

        println!("All ops kernels compiled successfully!");
    }
}
