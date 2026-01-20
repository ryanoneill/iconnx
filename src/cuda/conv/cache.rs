//! Kernel cache for convolution operations

use super::kernels::{CONV_KERNELS, KERNEL_NAMES};
use crate::cuda::context::{CudaError, IconnxCudaContext};
use cudarc::driver::{CudaFunction, CudaModule};
use cudarc::nvrtc::compile_ptx;
use std::collections::HashMap;
use std::sync::Arc;

/// Manages compiled convolution kernels with caching
pub struct ConvKernelCache {
    #[allow(dead_code)]
    module: Arc<CudaModule>,
    functions: HashMap<&'static str, CudaFunction>,
}

impl ConvKernelCache {
    /// Compile all convolution kernels and create the cache
    pub fn new(ctx: &IconnxCudaContext) -> Result<Self, CudaError> {
        let ptx = compile_ptx(CONV_KERNELS)
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
        let cache = ConvKernelCache::new(&ctx).expect("Failed to compile kernels");

        assert!(cache.get("im2col_kernel").is_some());
        assert!(cache.get("col2im_kernel").is_some());
        assert!(cache.get("im2col_1d_kernel").is_some());
        assert!(cache.get("col2im_1d_kernel").is_some());
        assert!(cache.get("add_bias_kernel").is_some());

        println!("All convolution kernels compiled successfully!");
    }
}
