//! WS-3.5 Y(2) sub-2c — BFloat16 ReduceMean with f32 accumulator.

#![cfg(feature = "cuda")]

use half::bf16;
use iconnx::cuda::{
    gpu_reduce_mean_axis, GpuMemoryPool, GpuTensor, IconnxCudaContext, ReductionKernelCache,
};

#[test]
#[ignore = "requires CUDA GPU"]
fn reduce_mean_bfloat16_axis_last() {
    let ctx = IconnxCudaContext::new().expect("ctx");
    if ctx.compute_capability().0 < 8 {
        return;
    }
    let kernels = ReductionKernelCache::new(&ctx).expect("kernels");
    let mut pool = GpuMemoryPool::new();

    // Input [2, 3]: [[1, 2, 3], [4, 5, 6]]; axis=last; expected = [2.0, 5.0].
    let host: Vec<bf16> = (1..=6).map(|i| bf16::from_f32(i as f32)).collect();
    let g = GpuTensor::from_host_bf16(&ctx, &host, vec![2, 3]).expect("upload");
    let result = gpu_reduce_mean_axis(&ctx, &kernels, &mut pool, &g).expect("reduce_mean");
    let back = result.to_host_bf16(&ctx).expect("download");
    let expected = [2.0_f32, 5.0];
    for (got, want) in back.iter().zip(expected.iter()) {
        let diff = (got.to_f32() - *want).abs();
        assert!(diff < 1e-2, "got {got:?} want {want} (BF16 1e-2 tol)");
    }
}
