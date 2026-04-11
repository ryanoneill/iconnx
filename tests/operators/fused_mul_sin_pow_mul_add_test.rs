//! Tests for the fused Mul->Sin->Pow->Mul->Add kernel.
//!
//! Mathematical identity: y = sin(x * w0)^p * w1 + b
//!
//! The fused kernel must match the unfused chain of `gpu_mul`, `gpu_sin`,
//! `gpu_pow`, `gpu_mul`, `gpu_add` to within 1e-5 per element.

#![cfg(feature = "cuda")]

use iconnx::cuda::{
    gpu_add, gpu_fused_mul_sin_pow_mul_add, gpu_mul, gpu_pow, gpu_sin, FusedKernelCache,
    GpuMemoryPool, GpuTensor, IconnxCudaContext, KernelCache,
};

/// Build a GPU f32 tensor from a host Vec.
fn upload(ctx: &IconnxCudaContext, data: Vec<f32>, shape: Vec<usize>) -> GpuTensor {
    GpuTensor::from_host_f32(ctx, &data, shape).expect("upload failed")
}

/// Run the unfused reference chain: y = sin(x * w0)^p * w1 + b
#[allow(clippy::too_many_arguments)]
fn unfused_reference(
    ctx: &IconnxCudaContext,
    elem_cache: &KernelCache,
    pool: &mut GpuMemoryPool,
    x: &GpuTensor,
    w0: &GpuTensor,
    w1: &GpuTensor,
    b: &GpuTensor,
    p: f32,
) -> GpuTensor {
    let t0 = gpu_mul(ctx, elem_cache, pool, x, w0).expect("gpu_mul t0");
    let t1 = gpu_sin(ctx, elem_cache, pool, &t0).expect("gpu_sin t1");
    // gpu_pow requires matching shapes, so build a full-shape exponent tensor
    // filled with `p` and matching the shape of t1.
    let p_data = vec![p; t1.len()];
    let p_tensor =
        GpuTensor::from_host_f32(ctx, &p_data, t1.shape().to_vec()).expect("p tensor");
    let t2 = gpu_pow(ctx, elem_cache, pool, &t1, &p_tensor).expect("gpu_pow t2");
    let t3 = gpu_mul(ctx, elem_cache, pool, &t2, w1).expect("gpu_mul t3");
    gpu_add(ctx, elem_cache, pool, &t3, b).expect("gpu_add y")
}

/// Compare two float tensors elementwise within a tolerance.
fn assert_close(ctx: &IconnxCudaContext, a: &GpuTensor, b: &GpuTensor, tol: f32) {
    let av = a.to_host_f32(ctx).expect("to_host_f32 a");
    let bv = b.to_host_f32(ctx).expect("to_host_f32 b");
    assert_eq!(av.len(), bv.len(), "length mismatch");
    for (i, (lhs, rhs)) in av.iter().zip(bv.iter()).enumerate() {
        let diff = (lhs - rhs).abs();
        assert!(
            diff <= tol,
            "mismatch at index {i}: fused={lhs}, unfused={rhs}, diff={diff}"
        );
    }
}

#[test]
fn naive_vs_fused_matches_same_shape() {
    let ctx = IconnxCudaContext::new().expect("CUDA context");
    let elem_cache = KernelCache::new(&ctx).expect("KernelCache");
    let fused_cache = FusedKernelCache::new(&ctx).expect("FusedKernelCache");
    let mut pool = GpuMemoryPool::new();

    // Deterministic synthetic inputs: shape [4, 8] so the test runs fast.
    let n = 4 * 8;
    let x_data: Vec<f32> = (0..n).map(|i| 0.01 * i as f32).collect();
    let w0_data: Vec<f32> = (0..n).map(|i| 0.5 + 0.03 * i as f32).collect();
    let w1_data: Vec<f32> = (0..n).map(|i| 1.0 - 0.02 * i as f32).collect();
    let b_data: Vec<f32> = (0..n).map(|i| 0.1 * i as f32).collect();

    let x = upload(&ctx, x_data, vec![4, 8]);
    let w0 = upload(&ctx, w0_data, vec![4, 8]);
    let w1 = upload(&ctx, w1_data, vec![4, 8]);
    let b = upload(&ctx, b_data, vec![4, 8]);
    let p: f32 = 2.0;

    let reference =
        unfused_reference(&ctx, &elem_cache, &mut pool, &x, &w0, &w1, &b, p);
    let fused = gpu_fused_mul_sin_pow_mul_add(
        &ctx, &fused_cache, &mut pool, &x, &w0, &w1, &b, p,
    )
    .expect("gpu_fused_mul_sin_pow_mul_add");

    assert_close(&ctx, &fused, &reference, 1e-5);
}

#[test]
fn broadcasting_w0_w1_b_per_channel() {
    let ctx = IconnxCudaContext::new().expect("CUDA context");
    let elem_cache = KernelCache::new(&ctx).expect("KernelCache");
    let fused_cache = FusedKernelCache::new(&ctx).expect("FusedKernelCache");
    let mut pool = GpuMemoryPool::new();

    // x is [N, C], w0/w1/b are [1, C] broadcast along the leading dim.
    let n_rows = 3usize;
    let channels = 8usize;
    let total = n_rows * channels;

    let x_data: Vec<f32> = (0..total).map(|i| 0.01 * i as f32 + 0.2).collect();
    let w0_row: Vec<f32> = (0..channels).map(|i| 0.5 + 0.05 * i as f32).collect();
    let w1_row: Vec<f32> = (0..channels).map(|i| 1.0 - 0.03 * i as f32).collect();
    let b_row: Vec<f32> = (0..channels).map(|i| 0.1 * i as f32).collect();

    let x = upload(&ctx, x_data.clone(), vec![n_rows, channels]);
    let w0 = upload(&ctx, w0_row.clone(), vec![1, channels]);
    let w1 = upload(&ctx, w1_row.clone(), vec![1, channels]);
    let b = upload(&ctx, b_row.clone(), vec![1, channels]);
    let p: f32 = 2.0;

    // Build the ground truth by tiling the broadcast rows to the full shape
    // and running the unfused reference chain.
    let mut w0_tiled = Vec::with_capacity(total);
    let mut w1_tiled = Vec::with_capacity(total);
    let mut b_tiled = Vec::with_capacity(total);
    for _ in 0..n_rows {
        w0_tiled.extend_from_slice(&w0_row);
        w1_tiled.extend_from_slice(&w1_row);
        b_tiled.extend_from_slice(&b_row);
    }
    let w0_full = upload(&ctx, w0_tiled, vec![n_rows, channels]);
    let w1_full = upload(&ctx, w1_tiled, vec![n_rows, channels]);
    let b_full = upload(&ctx, b_tiled, vec![n_rows, channels]);

    let reference = unfused_reference(
        &ctx, &elem_cache, &mut pool, &x, &w0_full, &w1_full, &b_full, p,
    );
    let fused = gpu_fused_mul_sin_pow_mul_add(
        &ctx, &fused_cache, &mut pool, &x, &w0, &w1, &b, p,
    )
    .expect("gpu_fused_mul_sin_pow_mul_add broadcast");

    assert_close(&ctx, &fused, &reference, 1e-5);
}
