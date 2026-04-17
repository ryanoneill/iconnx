//! Kernel cache for convolution operations (garboard `Module`).

use super::kernels::{CONV_KERNELS, KERNEL_NAMES};
use crate::cuda::context::{CudaError, IconnxCudaContext};
use garboard::{Module, Program};

/// Compiled convolution kernels (im2col, col2im, add_bias variants)
/// loaded as a garboard `Module`. cuDNN-backed conv/conv-transpose go
/// through garboard's `DnnContext` (see `cuda::cudnn`); this cache only
/// covers the custom im2col-style kernels.
pub struct ConvKernelCache {
    module: Module<'static>,
}

impl ConvKernelCache {
    /// Compile all convolution kernels and create the cache.
    pub fn new(ctx: &IconnxCudaContext) -> Result<Self, CudaError> {
        let program = Program::compile_for_device(CONV_KERNELS, ctx.garboard_device(), &[])
            .map_err(|e| CudaError::Kernel(format!("NVRTC compilation failed: {}", e)))?;

        let module = ctx
            .garboard_device()
            .load_module(&program)
            .map_err(|e| CudaError::Kernel(format!("Module load failed: {}", e)))?;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_kernel_compilation() {
        let ctx = IconnxCudaContext::new().expect("Failed to create CUDA context");
        let cache = ConvKernelCache::new(&ctx).expect("Failed to compile kernels");

        for name in [
            "im2col_kernel",
            "col2im_kernel",
            "im2col_1d_kernel",
            "col2im_1d_kernel",
            "add_bias_kernel",
        ] {
            assert!(
                cache.module().function(name).is_ok(),
                "kernel {name} should be present in conv module"
            );
        }

        println!("All convolution kernels compiled successfully!");
    }
}
