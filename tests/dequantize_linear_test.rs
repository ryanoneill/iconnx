//! WS-4 M4.4 GPU contract tests: `gpu_dequantize_linear`.
//!
//! Each test seeds known integer inputs through the GPU kernel and compares
//! to the CPU forward at exact equality (the math is `(x - zp) * scale`,
//! and our test inputs are chosen so the FP result is representable).

#![cfg(feature = "cuda")]

use iconnx::cuda::{
    gpu_dequantize_linear, GpuMemoryPool, GpuTensor, IconnxCudaContext, OpsKernelCache,
};
use iconnx::operators::dequantize_linear::DequantizeLinear;
use iconnx::{NodeAttributes, Tensor};

/// Helper: collect a Float32 GPU tensor back to host.
fn dtoh_f32(ctx: &IconnxCudaContext, t: &GpuTensor) -> Vec<f32> {
    t.to_host_f32(ctx).expect("to_host_f32")
}

#[test]
#[ignore = "requires CUDA GPU"]
fn gpu_dequantize_linear_per_tensor_int8_no_zero_point() {
    let ctx = IconnxCudaContext::new().expect("ctx");
    let cache = OpsKernelCache::new(&ctx).expect("cache");
    let mut pool = GpuMemoryPool::new();

    // Same scenario as the CPU test: x = [-128, 0, 127], scale = 0.5
    let x_host: Vec<i8> = vec![-128, 0, 127];
    let x_gpu = GpuTensor::from_host_i8(&ctx, &x_host, vec![3]).expect("x");
    let scale_gpu = GpuTensor::from_host_f32(&ctx, &[0.5f32], vec![]).expect("scale");

    let out_gpu = gpu_dequantize_linear(&ctx, &cache, &mut pool, &x_gpu, &scale_gpu, None, 0)
        .expect("dequantize");
    let out = dtoh_f32(&ctx, &out_gpu);
    assert_eq!(out.len(), 3);

    // CPU oracle.
    let cpu_out = DequantizeLinear::forward(
        &[
            Tensor::from_vec_i8(x_host, vec![3]),
            Tensor::from_vec_f32(vec![0.5f32], vec![]),
        ],
        &NodeAttributes::new(),
    );
    assert_eq!(out, cpu_out.as_slice());
}

#[test]
#[ignore = "requires CUDA GPU"]
fn gpu_dequantize_linear_per_channel_int8_axis_0() {
    let ctx = IconnxCudaContext::new().expect("ctx");
    let cache = OpsKernelCache::new(&ctx).expect("cache");
    let mut pool = GpuMemoryPool::new();

    // 2x2 x = [[1,2],[3,4]], scale = [0.1, 0.2] (axis=0).
    let x_host: Vec<i8> = vec![1, 2, 3, 4];
    let x_gpu = GpuTensor::from_host_i8(&ctx, &x_host, vec![2, 2]).expect("x");
    let scale_gpu = GpuTensor::from_host_f32(&ctx, &[0.1f32, 0.2], vec![2]).expect("scale");

    let out_gpu = gpu_dequantize_linear(&ctx, &cache, &mut pool, &x_gpu, &scale_gpu, None, 0)
        .expect("dequantize");
    let out = dtoh_f32(&ctx, &out_gpu);

    let mut attrs = NodeAttributes::new();
    attrs.add_int("axis".into(), 0);
    let cpu_out = DequantizeLinear::forward(
        &[
            Tensor::from_vec_i8(x_host, vec![2, 2]),
            Tensor::from_vec_f32(vec![0.1f32, 0.2], vec![2]),
        ],
        &attrs,
    );
    assert_eq!(out, cpu_out.as_slice());
}

#[test]
#[ignore = "requires CUDA GPU"]
fn gpu_dequantize_linear_per_tensor_uint8_with_zero_point() {
    let ctx = IconnxCudaContext::new().expect("ctx");
    let cache = OpsKernelCache::new(&ctx).expect("cache");
    let mut pool = GpuMemoryPool::new();

    // x = [0, 128, 255], zp = 128, scale = 0.5 → [-64, 0, 63.5].
    let x_host: Vec<u8> = vec![0, 128, 255];
    let x_gpu = GpuTensor::from_host_u8(&ctx, &x_host, vec![3]).expect("x");
    let scale_gpu = GpuTensor::from_host_f32(&ctx, &[0.5f32], vec![]).expect("scale");
    let zp_gpu = GpuTensor::from_host_u8(&ctx, &[128u8], vec![]).expect("zp");

    let out_gpu = gpu_dequantize_linear(
        &ctx,
        &cache,
        &mut pool,
        &x_gpu,
        &scale_gpu,
        Some(&zp_gpu),
        0,
    )
    .expect("dequantize");
    let out = dtoh_f32(&ctx, &out_gpu);

    let cpu_out = DequantizeLinear::forward(
        &[
            Tensor::from_vec_u8(x_host, vec![3]),
            Tensor::from_vec_f32(vec![0.5f32], vec![]),
            Tensor::from_vec_u8(vec![128u8], vec![]),
        ],
        &NodeAttributes::new(),
    );
    assert_eq!(out, cpu_out.as_slice());
}

#[test]
#[ignore = "requires CUDA GPU"]
fn gpu_dequantize_linear_int32_input_for_matmul_integer_output() {
    let ctx = IconnxCudaContext::new().expect("ctx");
    let cache = OpsKernelCache::new(&ctx).expect("cache");
    let mut pool = GpuMemoryPool::new();

    // x = [100, -50, 200] (i32 — MatMulInteger output dtype), scale = 0.01.
    let x_host: Vec<i32> = vec![100, -50, 200];
    let x_gpu = GpuTensor::from_host_i32(&ctx, &x_host, vec![3]).expect("x");
    let scale_gpu = GpuTensor::from_host_f32(&ctx, &[0.01f32], vec![]).expect("scale");

    let out_gpu = gpu_dequantize_linear(&ctx, &cache, &mut pool, &x_gpu, &scale_gpu, None, 0)
        .expect("dequantize");
    let out = dtoh_f32(&ctx, &out_gpu);

    let cpu_out = DequantizeLinear::forward(
        &[
            Tensor::from_vec_i32(x_host, vec![3]),
            Tensor::from_vec_f32(vec![0.01f32], vec![]),
        ],
        &NodeAttributes::new(),
    );
    assert_eq!(out, cpu_out.as_slice());
}

/// Per-axis with a per-axis zero point that is non-zero — exercises the
/// per-element `zp[a]` lookup along with the per-element `scale[a]` lookup
/// in the kernel. With `axis=1` (the spec default), each column gets its
/// own (scale, zp) pair.
#[test]
#[ignore = "requires CUDA GPU"]
fn gpu_dequantize_linear_per_channel_uint8_axis_1_with_zero_points() {
    let ctx = IconnxCudaContext::new().expect("ctx");
    let cache = OpsKernelCache::new(&ctx).expect("cache");
    let mut pool = GpuMemoryPool::new();

    // 2x3 row-major: [[10, 20, 30], [40, 50, 60]]
    // axis=1 (cols): scale = [1.0, 0.5, 0.25], zp = [0, 10, 20]
    //   col 0: y = (x - 0) * 1.0  → [10, 40]
    //   col 1: y = (x - 10) * 0.5 → [5, 20]
    //   col 2: y = (x - 20) * 0.25→ [2.5, 10]
    let x_host: Vec<u8> = vec![10, 20, 30, 40, 50, 60];
    let x_gpu = GpuTensor::from_host_u8(&ctx, &x_host, vec![2, 3]).expect("x");
    let scale_gpu =
        GpuTensor::from_host_f32(&ctx, &[1.0f32, 0.5, 0.25], vec![3]).expect("scale");
    let zp_gpu = GpuTensor::from_host_u8(&ctx, &[0u8, 10, 20], vec![3]).expect("zp");

    let out_gpu = gpu_dequantize_linear(
        &ctx,
        &cache,
        &mut pool,
        &x_gpu,
        &scale_gpu,
        Some(&zp_gpu),
        /*axis=*/ 1,
    )
    .expect("dequantize");
    let out = dtoh_f32(&ctx, &out_gpu);

    let mut attrs = NodeAttributes::new();
    attrs.add_int("axis".into(), 1);
    let cpu_out = DequantizeLinear::forward(
        &[
            Tensor::from_vec_u8(x_host, vec![2, 3]),
            Tensor::from_vec_f32(vec![1.0f32, 0.5, 0.25], vec![3]),
            Tensor::from_vec_u8(vec![0u8, 10, 20], vec![3]),
        ],
        &attrs,
    );
    assert_eq!(out, cpu_out.as_slice());

    // Also lock the literal expectations.
    assert_eq!(out, vec![10.0, 5.0, 2.5, 40.0, 20.0, 10.0]);
}
