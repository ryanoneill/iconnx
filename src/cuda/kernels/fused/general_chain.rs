//! Runtime dispatcher for `FusedPattern::GeneralChain` — NVRTC-compile
//! each unique `chain_signature` on first use, cache the compiled
//! module, dispatch each call on the cached handle.
//!
//! The dispatcher owns the compile + launch path for single-input
//! general elementwise chains. The kernel source is produced by
//! [`crate::ir::passes::elementwise_fusion_codegen::generate`] and
//! compiled once per unique signature; identical signatures reuse the
//! same compile via the `FusedKernelCache::general_chain` cache.
//!
//! Why cache the module and re-resolve the typed kernel per launch:
//! `TypedKernel<'d, Args>` bakes concrete reference lifetimes into
//! `Args`, so a stored `'static`-ref typed kernel cannot be launched
//! with local-lifetime arguments (invariance of `&mut T`). The ~200 ns
//! resolution cost per launch is negligible compared to kernel execution.

use garboard::{DeviceSlice, Program};

use super::{elementwise_config, FusedKernelCache, GeneralChainKernel};
use crate::attributes::NodeAttributes;
use crate::cuda::context::{CudaError, IconnxCudaContext};
use crate::cuda::memory_pool::GpuMemoryPool;
use crate::cuda::tensor::GpuTensor;
use crate::ir::passes::elementwise_fusion_codegen;

/// Dispatch a `GeneralChain` fusion: compile the kernel for
/// `chain_signature` if not already cached, then launch it over the
/// single input tensor.
///
/// Returns a freshly-allocated output tensor with the same shape as
/// `x`. The caller (executor) is responsible for registering the output
/// under the pattern's `output_name` in the run-time values map.
pub fn gpu_fused_general_chain(
    ctx: &IconnxCudaContext,
    cache: &FusedKernelCache,
    pool: &mut GpuMemoryPool,
    chain_signature: &str,
    ops: &[(String, NodeAttributes)],
    x: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    ensure_compiled(ctx, cache, chain_signature, ops)?;

    let n = x.len();
    let mut out = pool.get_tensor_f32(ctx, x.shape().to_vec())?;

    // Re-resolve the typed kernel on each launch. `TypedKernel<'d,
    // Args>` invariance over `&mut T` rules out storing a ready-made
    // handle with local-lifetime args; the cost is a `cuModuleGet
    // Function` lookup (~200ns) which is negligible compared to the
    // kernel body. See the module-level doc comment.
    let chain_cache = cache.general_chain_cache().borrow();
    let kernel_entry = chain_cache.get(chain_signature).ok_or_else(|| {
        CudaError::Kernel(format!(
            "GeneralChain '{}': cache missing after compile",
            chain_signature
        ))
    })?;

    // SAFETY: the kernel source emitted by
    // `elementwise_fusion_codegen::generate` has signature
    // `(float* y, const float* x, int n)`. The C signature is fixed by
    // the codegen module and verified by its unit tests.
    let typed = unsafe {
        kernel_entry.module.typed_kernel::<(
            &mut DeviceSlice<'_, f32>,
            &DeviceSlice<'_, f32>,
            i32,
        )>(&kernel_entry.entry_name)
    }
    .map_err(|e| {
        CudaError::Kernel(format!(
            "GeneralChain '{}' typed_kernel lookup failed: {}",
            chain_signature, e
        ))
    })?;

    typed
        .launch(
            ctx.garboard_stream(),
            &elementwise_config(n),
            (out.data_f32_mut()?, x.data_f32()?, n as i32),
        )
        .map_err(|e| {
            CudaError::Kernel(format!(
                "GeneralChain '{}' launch failed: {}",
                chain_signature, e
            ))
        })?;

    Ok(out)
}

/// Ensure the kernel for `chain_signature` is compiled and cached. No-op
/// if already present. Separates the borrow-mut needed during compile
/// from the borrow needed during launch.
fn ensure_compiled(
    ctx: &IconnxCudaContext,
    cache: &FusedKernelCache,
    chain_signature: &str,
    ops: &[(String, NodeAttributes)],
) -> Result<(), CudaError> {
    if cache.general_chain_cache().borrow().contains_key(chain_signature) {
        return Ok(());
    }

    let elementwise_fusion_codegen::GeneratedKernel { source, entry_name } =
        elementwise_fusion_codegen::generate(chain_signature, ops);

    let program = Program::compile_for_device(&source, ctx.garboard_device(), &[])
        .map_err(|e| {
            CudaError::Kernel(format!(
                "GeneralChain NVRTC compile failed for '{}': {}",
                chain_signature, e
            ))
        })?;

    let module = ctx
        .garboard_device()
        .load_module(&program)
        .map_err(|e| {
            CudaError::Kernel(format!(
                "GeneralChain module load failed for '{}': {}",
                chain_signature, e
            ))
        })?;

    // Eagerly verify the entry symbol resolves — catches codegen /
    // sanitizer bugs at compile time rather than first launch.
    module.function(&entry_name).map_err(|e| {
        CudaError::Kernel(format!(
            "GeneralChain '{}' entry '{}' missing from compiled module: {}",
            chain_signature, entry_name, e
        ))
    })?;

    cache.general_chain_cache().borrow_mut().insert(
        chain_signature.to_string(),
        GeneralChainKernel {
            module,
            entry_name,
        },
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attributes::NodeAttributes;

    fn setup() -> (IconnxCudaContext, FusedKernelCache, GpuMemoryPool) {
        let ctx = IconnxCudaContext::new().expect("ctx");
        let cache = FusedKernelCache::new(&ctx).expect("fused cache");
        let pool = GpuMemoryPool::new();
        (ctx, cache, pool)
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn compile_sqrt_exp_chain_and_launch_matches_hand_computed() {
        let (ctx, cache, mut pool) = setup();

        // Input: 8 values in [0.1, 0.8], exp(sqrt(x)) computed on host.
        let host: Vec<f32> = (1..=8).map(|i| i as f32 * 0.1).collect();
        let expected: Vec<f32> = host.iter().map(|v| v.sqrt().exp()).collect();
        let x = GpuTensor::from_host_f32(&ctx, &host, vec![8]).expect("upload");

        let ops = vec![
            ("Sqrt".to_string(), NodeAttributes::new()),
            ("Exp".to_string(), NodeAttributes::new()),
        ];
        let out = gpu_fused_general_chain(
            &ctx,
            &cache,
            &mut pool,
            "Sqrt|Exp",
            &ops,
            &x,
        )
        .expect("launch");
        let got = out.to_host_f32(&ctx).expect("download");

        assert_eq!(got.len(), expected.len());
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!(
                (g - e).abs() < 1e-5,
                "Sqrt|Exp mismatch: got {} expected {}",
                g,
                e
            );
        }
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn compile_is_cached_across_launches() {
        let (ctx, cache, mut pool) = setup();

        let host: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let x = GpuTensor::from_host_f32(&ctx, &host, vec![4]).expect("upload");

        let ops = vec![("Tanh".to_string(), NodeAttributes::new()), ("Sqrt".to_string(), NodeAttributes::new())];

        // First launch compiles; second launch must reuse the cached
        // module (no second NVRTC compile). We can't directly observe
        // the compile count, but we can verify the cache size is 1
        // after two launches of the same signature.
        let _ = gpu_fused_general_chain(&ctx, &cache, &mut pool, "Tanh|Sqrt", &ops, &x).unwrap();
        let _ = gpu_fused_general_chain(&ctx, &cache, &mut pool, "Tanh|Sqrt", &ops, &x).unwrap();

        assert_eq!(
            cache.general_chain_cache().borrow().len(),
            1,
            "identical signatures must coalesce in the cache"
        );
    }
}
