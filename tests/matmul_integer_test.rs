//! WS-4 M4.6 GPU contract tests: `gpu_matmul_integer`.
//!
//! Each test seeds known INT8 / UINT8 inputs through the GPU kernel and
//! compares to the CPU forward (`MatMulInteger::forward`) at exact i32
//! equality. Integer GEMM is bit-deterministic at this op layer — there
//! is no floating-point intermediate, so any GPU/CPU divergence is a
//! kernel correctness bug, not a rounding artifact.
//!
//! The K=3072 test pins the DistilBERT-FFN inner-dim accumulator-width
//! invariant on the GPU side; its CPU twin is the gating verification
//! that ran first per the WS-4 leadline-2026-04-25 directive.

#![cfg(feature = "cuda")]

use iconnx::cuda::{
    gpu_matmul_integer, GpuMemoryPool, GpuTensor, IconnxCudaContext, OpsKernelCache,
};
use iconnx::operators::matmul_integer::MatMulInteger;
use iconnx::{NodeAttributes, Tensor};

/// Helper: pull an Int32 GPU tensor back to host.
fn dtoh_i32(ctx: &IconnxCudaContext, t: &GpuTensor) -> Vec<i32> {
    t.to_host_i32(ctx).expect("to_host_i32")
}

/// Helper: i32 slice from a CPU `Tensor::Int32` for direct comparison.
fn cpu_int32(t: &Tensor) -> Vec<i32> {
    match t {
        Tensor::Int32(arr) => arr.iter().copied().collect(),
        _ => panic!("expected Int32 tensor"),
    }
}

#[test]
#[ignore = "requires CUDA GPU"]
fn gpu_matmul_integer_uint8_int8_distilbert_combo_no_zps() {
    // Mirrors the CPU `matmul_integer_uint8_int8_distilbert_combo_no_zero_points`
    // test on the GPU. 2x3 UINT8 × 3x2 INT8 → 2x2 INT32.
    let ctx = IconnxCudaContext::new().expect("ctx");
    let cache = OpsKernelCache::new(&ctx).expect("cache");
    let mut pool = GpuMemoryPool::new();

    let a_host: Vec<u8> = vec![1, 2, 3, 4, 5, 6];
    let b_host: Vec<i8> = vec![1, 2, 3, 4, 5, 6];
    let a_gpu = GpuTensor::from_host_u8(&ctx, &a_host, vec![2, 3]).expect("a");
    let b_gpu = GpuTensor::from_host_i8(&ctx, &b_host, vec![3, 2]).expect("b");

    let out_gpu = gpu_matmul_integer(&ctx, &cache, &mut pool, &a_gpu, &b_gpu, None, None)
        .expect("matmul_integer");
    let out = dtoh_i32(&ctx, &out_gpu);
    assert_eq!(out_gpu.shape(), &[2, 2]);

    // CPU oracle.
    let cpu = MatMulInteger::forward(
        &[
            Tensor::from_vec_u8(a_host, vec![2, 3]),
            Tensor::from_vec_i8(b_host, vec![3, 2]),
        ],
        &NodeAttributes::new(),
    );
    assert_eq!(out, cpu_int32(&cpu));
    // Spot-check the known expected values from the plan.
    assert_eq!(out, vec![22_i32, 28, 49, 64]);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn gpu_matmul_integer_distilbert_shape_uint8_int8_k_768() {
    // M=4, K=768, N=4 — DistilBERT-INT8 hidden-size matmul. a=ones, b=ones
    // → every cell == K (sanity that the GEMM walks the full inner dim).
    let ctx = IconnxCudaContext::new().expect("ctx");
    let cache = OpsKernelCache::new(&ctx).expect("cache");
    let mut pool = GpuMemoryPool::new();

    let a_host: Vec<u8> = vec![1_u8; 4 * 768];
    let b_host: Vec<i8> = vec![1_i8; 768 * 4];
    let a_gpu = GpuTensor::from_host_u8(&ctx, &a_host, vec![4, 768]).expect("a");
    let b_gpu = GpuTensor::from_host_i8(&ctx, &b_host, vec![768, 4]).expect("b");

    let out_gpu = gpu_matmul_integer(&ctx, &cache, &mut pool, &a_gpu, &b_gpu, None, None)
        .expect("matmul_integer");
    let out = dtoh_i32(&ctx, &out_gpu);
    assert_eq!(out_gpu.shape(), &[4, 4]);
    assert!(out.iter().all(|&v| v == 768), "every cell must equal K");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn gpu_matmul_integer_k_3072_overflow_safety_distilbert_ffn() {
    // GPU oracle for the CPU gating test. a, b all 100 (UINT8/INT8)
    // with K=3072 → cell == 3072 * 100 * 100 = 30_720_000. Verifies the
    // kernel's i32 accumulator handles the full DistilBERT FFN inner dim
    // without truncation or overflow.
    let ctx = IconnxCudaContext::new().expect("ctx");
    let cache = OpsKernelCache::new(&ctx).expect("cache");
    let mut pool = GpuMemoryPool::new();

    let a_host: Vec<u8> = vec![100_u8; 3072];
    let b_host: Vec<i8> = vec![100_i8; 3072];
    let a_gpu = GpuTensor::from_host_u8(&ctx, &a_host, vec![1, 3072]).expect("a");
    let b_gpu = GpuTensor::from_host_i8(&ctx, &b_host, vec![3072, 1]).expect("b");

    let out_gpu = gpu_matmul_integer(&ctx, &cache, &mut pool, &a_gpu, &b_gpu, None, None)
        .expect("matmul_integer");
    let out = dtoh_i32(&ctx, &out_gpu);
    assert_eq!(out_gpu.shape(), &[1, 1]);
    assert_eq!(out[0], 30_720_000);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn gpu_matmul_integer_int8_int8_with_zero_points() {
    // Mirrors the CPU `matmul_integer_int8_int8_with_zero_points` test.
    // Validates the s8s8 kernel + scalar zp handling end-to-end.
    let ctx = IconnxCudaContext::new().expect("ctx");
    let cache = OpsKernelCache::new(&ctx).expect("cache");
    let mut pool = GpuMemoryPool::new();

    let a_host: Vec<i8> = vec![10, 20, 30, 40];
    let b_host: Vec<i8> = vec![1, 2, 3, 4];
    let a_zp_host: Vec<i8> = vec![5];
    let b_zp_host: Vec<i8> = vec![1];
    let a_gpu = GpuTensor::from_host_i8(&ctx, &a_host, vec![2, 2]).expect("a");
    let b_gpu = GpuTensor::from_host_i8(&ctx, &b_host, vec![2, 2]).expect("b");
    let a_zp_gpu = GpuTensor::from_host_i8(&ctx, &a_zp_host, vec![]).expect("a_zp");
    let b_zp_gpu = GpuTensor::from_host_i8(&ctx, &b_zp_host, vec![]).expect("b_zp");

    let out_gpu = gpu_matmul_integer(
        &ctx,
        &cache,
        &mut pool,
        &a_gpu,
        &b_gpu,
        Some(&a_zp_gpu),
        Some(&b_zp_gpu),
    )
    .expect("matmul_integer");
    let out = dtoh_i32(&ctx, &out_gpu);

    let cpu = MatMulInteger::forward(
        &[
            Tensor::from_vec_i8(a_host, vec![2, 2]),
            Tensor::from_vec_i8(b_host, vec![2, 2]),
            Tensor::from_vec_i8(a_zp_host, vec![]),
            Tensor::from_vec_i8(b_zp_host, vec![]),
        ],
        &NodeAttributes::new(),
    );
    assert_eq!(out, cpu_int32(&cpu));
    assert_eq!(out, vec![30_i32, 50, 70, 130]);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn gpu_matmul_integer_rejects_non_int8_uint8_dtype() {
    // f32 × i8 must error out — no silent coercion. Validates the
    // wrapper's dtype gate before kernel dispatch.
    let ctx = IconnxCudaContext::new().expect("ctx");
    let cache = OpsKernelCache::new(&ctx).expect("cache");
    let mut pool = GpuMemoryPool::new();

    let a_gpu =
        GpuTensor::from_host_f32(&ctx, &[1.0_f32, 2.0, 3.0, 4.0], vec![2, 2]).expect("a");
    let b_gpu = GpuTensor::from_host_i8(&ctx, &[1_i8, 2, 3, 4], vec![2, 2]).expect("b");

    let result = gpu_matmul_integer(&ctx, &cache, &mut pool, &a_gpu, &b_gpu, None, None);
    assert!(
        result.is_err(),
        "MatMulInteger with non-Int8/UInt8 a must fail, not silently coerce"
    );
}
