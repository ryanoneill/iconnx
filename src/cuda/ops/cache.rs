//! Kernel cache for ops module (garboard `Module`).
//!
//! All ops kernels are loaded into a single garboard `Module` and launched
//! via the typed-kernel API on garboard's user stream. Source is composed
//! from per-category constants — `kernels::OPS_KERNELS` for layout/utility
//! ops plus `quantize_kernels::QUANTIZE_KERNELS` for the WS-4 quantization
//! family — concatenated at startup before NVRTC compilation. This keeps
//! per-file source within the project's 1000-line guideline as the kernel
//! catalogue grows.

use super::kernels::{KERNEL_NAMES, OPS_KERNELS};
use super::quantize_kernels::{QUANTIZE_KERNEL_NAMES, QUANTIZE_KERNELS};
use crate::cuda::context::{CudaError, IconnxCudaContext};
use garboard::{Module, Program};

/// Compiled utility-operation kernels.
pub struct OpsKernelCache {
    garboard_module: Module<'static>,
}

impl OpsKernelCache {
    /// Compile all utility kernels. Source from every per-category
    /// `*_KERNELS` constant is concatenated then handed to NVRTC as a
    /// single translation unit.
    pub fn new(ctx: &IconnxCudaContext) -> Result<Self, CudaError> {
        let combined_src = format!("{}\n{}", OPS_KERNELS, QUANTIZE_KERNELS);
        let garboard_program =
            Program::compile_for_device(&combined_src, ctx.garboard_device(), &[]).map_err(
                |e| CudaError::Kernel(format!("NVRTC compilation failed: {}", e)),
            )?;
        let garboard_module = ctx
            .garboard_device()
            .load_module(&garboard_program)
            .map_err(|e| CudaError::Kernel(format!("Module load failed: {}", e)))?;
        for name in KERNEL_NAMES.iter().chain(QUANTIZE_KERNEL_NAMES.iter()) {
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

    /// Microbenchmark: per-call cost of small host-to-device uploads via
    /// `IconnxCudaContext::htod_usize` (which goes through garboard's
    /// `Stream::copy_host_to_device`, the asynchronous
    /// `cuMemcpyHtoDAsync`). Diagnostic kept around for regression
    /// triage — every layout op does a shape/stride upload, so any
    /// regression here compounds across the ~2400 ops in a Kokoro
    /// inference.
    #[test]
    #[ignore = "microbenchmark — requires CUDA GPU"]
    fn bench_htod_usize_cost() {
        use std::time::Instant;
        let ctx = IconnxCudaContext::new().expect("ctx");

        const ITERATIONS: u32 = 1_000;
        let small = vec![1usize, 2, 3, 4, 5, 6, 7, 8];

        // Warm.
        let _ = ctx.htod_usize(&small).unwrap();

        let start = Instant::now();
        for _ in 0..ITERATIONS {
            let _ = ctx.htod_usize(&small).unwrap();
        }
        let elapsed = start.elapsed();
        let per_call_ns = elapsed.as_nanos() as f64 / ITERATIONS as f64;
        println!(
            "htod_usize(8) x {} = {:?} (avg {:.0} ns/call = {:.2} us/call)",
            ITERATIONS,
            elapsed,
            per_call_ns,
            per_call_ns / 1000.0,
        );

        // Also a slightly larger one to see size dependence.
        let medium: Vec<usize> = (0..32).collect();
        let _ = ctx.htod_usize(&medium).unwrap();
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            let _ = ctx.htod_usize(&medium).unwrap();
        }
        let elapsed = start.elapsed();
        let per_call_ns = elapsed.as_nanos() as f64 / ITERATIONS as f64;
        println!(
            "htod_usize(32) x {} = {:?} (avg {:.0} ns/call = {:.2} us/call)",
            ITERATIONS,
            elapsed,
            per_call_ns,
            per_call_ns / 1000.0,
        );

        // alloc_zeros + drop, since pool misses go through this path.
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            let _ = ctx.alloc_zeros::<f32>(1024).unwrap();
        }
        let elapsed = start.elapsed();
        let per_call_ns = elapsed.as_nanos() as f64 / ITERATIONS as f64;
        println!(
            "alloc_zeros(1024 f32)+drop x {} = {:?} (avg {:.0} ns/call = {:.2} us/call)",
            ITERATIONS,
            elapsed,
            per_call_ns,
            per_call_ns / 1000.0,
        );
    }

    /// Microbenchmark: per-call cost of `Module::typed_kernel<...>("name")`.
    ///
    /// Diagnostic for the post-PR-5b benchmark plateau (~200ms vs. 109ms
    /// baseline). The per-launch typed_kernel lookup is currently the
    /// suspected source of regression — every iconnx kernel call goes
    /// through `cuModuleGetFunction` on the driver, with no caching in
    /// garboard's `Module`. Run via:
    ///   cargo test --release --features cuda -- --include-ignored \
    ///     bench_typed_kernel_lookup_cost --nocapture
    #[test]
    #[ignore = "microbenchmark — requires CUDA GPU"]
    fn bench_typed_kernel_lookup_cost() {
        use std::time::Instant;
        let ctx = IconnxCudaContext::new().expect("ctx");
        let cache = OpsKernelCache::new(&ctx).expect("cache");

        type TransposeArgs<'a> = (
            &'a mut garboard::DeviceSlice<'a, f32>,
            &'a garboard::DeviceSlice<'a, f32>,
            usize,
            usize,
        );

        const ITERATIONS: u32 = 10_000;

        // Warm: do one call so any first-time-load cost is not in the loop.
        let _ = unsafe {
            cache
                .module()
                .typed_kernel::<TransposeArgs<'_>>("transpose_2d_kernel")
        };

        let start = Instant::now();
        for _ in 0..ITERATIONS {
            let _ = unsafe {
                cache
                    .module()
                    .typed_kernel::<TransposeArgs<'_>>("transpose_2d_kernel")
            };
        }
        let elapsed = start.elapsed();
        let per_call_ns = elapsed.as_nanos() as f64 / ITERATIONS as f64;
        println!(
            "typed_kernel x {} = {:?} (avg {:.0} ns/call = {:.2} us/call)",
            ITERATIONS,
            elapsed,
            per_call_ns,
            per_call_ns / 1000.0,
        );

        let start = Instant::now();
        for _ in 0..ITERATIONS {
            let _ = cache.module().function("transpose_2d_kernel");
        }
        let elapsed = start.elapsed();
        let per_call_ns = elapsed.as_nanos() as f64 / ITERATIONS as f64;
        println!(
            "module.function x {} = {:?} (avg {:.0} ns/call = {:.2} us/call)",
            ITERATIONS,
            elapsed,
            per_call_ns,
            per_call_ns / 1000.0,
        );
    }

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
