//! WS-4 M4.5 GPU contract tests: `gpu_dynamic_quantize_linear`.
//!
//! Each test runs known FP32 inputs through the GPU wrapper and compares
//! the three outputs (y, y_scale, y_zero_point) against the CPU forward
//! at exact equality.

#![cfg(feature = "cuda")]

use iconnx::cuda::{gpu_dynamic_quantize_linear, GpuTensor, IconnxCudaContext};
use iconnx::operators::dynamic_quantize_linear::DynamicQuantizeLinear;
use iconnx::{NodeAttributes, Tensor};

/// Helper: pull the three CPU-side scalars back from the GPU multi-output
/// vector for direct assertion.
fn extract_outputs(
    ctx: &IconnxCudaContext,
    outs: &[GpuTensor],
) -> (Vec<u8>, f32, u8) {
    assert_eq!(outs.len(), 3, "expected 3 outputs (y, y_scale, y_zp)");
    let y = outs[0].to_host_u8(ctx).expect("to_host_u8 y");
    let scale = outs[1].to_host_f32(ctx).expect("to_host_f32 scale");
    let zp = outs[2].to_host_u8(ctx).expect("to_host_u8 zp");
    assert_eq!(scale.len(), 1, "y_scale must be scalar (1-element)");
    assert_eq!(zp.len(), 1, "y_zero_point must be scalar (1-element)");
    (y, scale[0], zp[0])
}

#[test]
#[ignore = "requires CUDA GPU"]
fn gpu_dynamic_quantize_linear_simple_positive() {
    let ctx = IconnxCudaContext::new().expect("ctx");

    let x_host: Vec<f32> = vec![0.0, 2.0, 4.0];
    let x_gpu = GpuTensor::from_host_f32(&ctx, &x_host, vec![3]).expect("x");

    let outs =
        gpu_dynamic_quantize_linear(&ctx, &x_gpu).expect("dynamic_quantize");
    let (y, y_scale, y_zp) = extract_outputs(&ctx, &outs);

    // CPU oracle.
    let cpu_outs = DynamicQuantizeLinear::forward(
        &[Tensor::from_vec_f32(x_host, vec![3])],
        &NodeAttributes::new(),
    );
    let cpu_y: Vec<u8> = match &cpu_outs[0] {
        Tensor::UInt8(arr) => arr.iter().copied().collect(),
        _ => panic!("expected UInt8 y"),
    };
    let cpu_scale: f32 = match &cpu_outs[1] {
        Tensor::Float32(s) => s[[]],
        _ => panic!("expected Float32 scale"),
    };
    let cpu_zp: u8 = match &cpu_outs[2] {
        Tensor::UInt8(z) => z[[]],
        _ => panic!("expected UInt8 zp"),
    };
    assert_eq!(y, cpu_y, "GPU y must match CPU y");
    assert_eq!(y_scale, cpu_scale, "GPU scale must match CPU scale");
    assert_eq!(y_zp, cpu_zp, "GPU zp must match CPU zp");
    // Spot check: known expected per the plan example.
    assert_eq!(y, vec![0_u8, 128, 255]);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn gpu_dynamic_quantize_linear_with_negative_input() {
    // x = [-1, 0, 1] → min=-1, max=1, scale=2/255, zp=128 (round-half-to-even).
    let ctx = IconnxCudaContext::new().expect("ctx");

    let x_host: Vec<f32> = vec![-1.0, 0.0, 1.0];
    let x_gpu = GpuTensor::from_host_f32(&ctx, &x_host, vec![3]).expect("x");
    let outs =
        gpu_dynamic_quantize_linear(&ctx, &x_gpu).expect("dynamic_quantize");
    let (_y, _scale, zp) = extract_outputs(&ctx, &outs);
    assert_eq!(zp, 128_u8, "round-half-to-even at 127.5 should yield 128");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn gpu_dynamic_quantize_linear_constant_input_zero_range() {
    // All-zero input: ORT-convention sentinel — scale=0, zp=0, y=all zeros.
    let ctx = IconnxCudaContext::new().expect("ctx");

    let x_host: Vec<f32> = vec![0.0; 5];
    let x_gpu = GpuTensor::from_host_f32(&ctx, &x_host, vec![5]).expect("x");
    let outs =
        gpu_dynamic_quantize_linear(&ctx, &x_gpu).expect("dynamic_quantize");
    let (y, scale, zp) = extract_outputs(&ctx, &outs);

    assert_eq!(y, vec![0_u8; 5], "all-zero input must yield all-zero y");
    assert_eq!(scale, 0.0, "constant input yields scale=0 sentinel");
    assert_eq!(zp, 0_u8);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn gpu_dynamic_quantize_linear_3d_shape_preserved() {
    // Sanity that the kernel flat-indexes correctly on >1-D shapes.
    // Output shape == input shape, scale and zp are scalar (rank 0).
    let ctx = IconnxCudaContext::new().expect("ctx");

    let x_host: Vec<f32> = (0..24).map(|i| i as f32 * 0.1).collect();
    let x_gpu = GpuTensor::from_host_f32(&ctx, &x_host, vec![2, 3, 4]).expect("x");
    let outs =
        gpu_dynamic_quantize_linear(&ctx, &x_gpu).expect("dynamic_quantize");

    assert_eq!(outs[0].shape(), &[2, 3, 4], "y preserves input shape");
    assert!(outs[1].shape().is_empty(), "scale is scalar (rank 0)");
    assert!(outs[2].shape().is_empty(), "zp is scalar (rank 0)");

    // Cross-check elementwise against CPU.
    let cpu_outs = DynamicQuantizeLinear::forward(
        &[Tensor::from_vec_f32(x_host, vec![2, 3, 4])],
        &NodeAttributes::new(),
    );
    let (gpu_y, _gpu_scale, _gpu_zp) = extract_outputs(&ctx, &outs);
    let cpu_y: Vec<u8> = match &cpu_outs[0] {
        Tensor::UInt8(arr) => arr.iter().copied().collect(),
        _ => panic!("expected UInt8"),
    };
    assert_eq!(gpu_y, cpu_y);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn gpu_dynamic_quantize_linear_rejects_non_float_input() {
    // Non-Float32 input must error out (not silently coerce).
    let ctx = IconnxCudaContext::new().expect("ctx");

    let x_gpu = GpuTensor::from_host_i32(&ctx, &[1_i32, 2, 3], vec![3]).expect("x");
    let result = gpu_dynamic_quantize_linear(&ctx, &x_gpu);
    assert!(
        result.is_err(),
        "DynamicQuantizeLinear on Int32 input must fail, not silently coerce"
    );
}
