use super::*;
use crate::cuda::memory_pool::GpuMemoryPool;

fn setup() -> (IconnxCudaContext, KernelCache, GpuMemoryPool) {
    let ctx = IconnxCudaContext::new().expect("Failed to create CUDA context");
    let kernels = KernelCache::new(&ctx).expect("Failed to compile kernels");
    let pool = GpuMemoryPool::new();
    (ctx, kernels, pool)
}

#[test]
#[ignore] // Requires CUDA GPU
fn test_kernel_compilation() {
    let (ctx, _kernels, _pool) = setup();
    ctx.sync().unwrap();
    println!("All kernels compiled successfully!");
}

#[test]
#[ignore] // Requires CUDA GPU
fn test_gpu_add() {
    let (ctx, kernels, mut pool) = setup();

    let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let b_data = vec![5.0f32, 6.0, 7.0, 8.0];

    let a = GpuTensor::from_host_f32(&ctx, &a_data, vec![4]).unwrap();
    let b = GpuTensor::from_host_f32(&ctx, &b_data, vec![4]).unwrap();

    let c = gpu_add(&ctx, &kernels, &mut pool, &a, &b).unwrap();
    let result = c.to_host_f32(&ctx).unwrap();

    assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
}

#[test]
#[ignore] // Requires CUDA GPU
fn test_gpu_sub() {
    let (ctx, kernels, mut pool) = setup();

    let a_data = vec![5.0f32, 6.0, 7.0, 8.0];
    let b_data = vec![1.0f32, 2.0, 3.0, 4.0];

    let a = GpuTensor::from_host_f32(&ctx, &a_data, vec![4]).unwrap();
    let b = GpuTensor::from_host_f32(&ctx, &b_data, vec![4]).unwrap();

    let c = gpu_sub(&ctx, &kernels, &mut pool, &a, &b).unwrap();
    let result = c.to_host_f32(&ctx).unwrap();

    assert_eq!(result, vec![4.0, 4.0, 4.0, 4.0]);
}

#[test]
#[ignore] // Requires CUDA GPU
fn test_gpu_mul() {
    let (ctx, kernels, mut pool) = setup();

    let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let b_data = vec![2.0f32, 3.0, 4.0, 5.0];

    let a = GpuTensor::from_host_f32(&ctx, &a_data, vec![4]).unwrap();
    let b = GpuTensor::from_host_f32(&ctx, &b_data, vec![4]).unwrap();

    let c = gpu_mul(&ctx, &kernels, &mut pool, &a, &b).unwrap();
    let result = c.to_host_f32(&ctx).unwrap();

    assert_eq!(result, vec![2.0, 6.0, 12.0, 20.0]);
}

#[test]
#[ignore] // Requires CUDA GPU
fn test_gpu_div() {
    let (ctx, kernels, mut pool) = setup();

    let a_data = vec![10.0f32, 20.0, 30.0, 40.0];
    let b_data = vec![2.0f32, 4.0, 5.0, 8.0];

    let a = GpuTensor::from_host_f32(&ctx, &a_data, vec![4]).unwrap();
    let b = GpuTensor::from_host_f32(&ctx, &b_data, vec![4]).unwrap();

    let c = gpu_div(&ctx, &kernels, &mut pool, &a, &b).unwrap();
    let result = c.to_host_f32(&ctx).unwrap();

    assert_eq!(result, vec![5.0, 5.0, 6.0, 5.0]);
}

#[test]
#[ignore] // Requires CUDA GPU
fn test_gpu_div_i64() {
    let (ctx, kernels, mut pool) = setup();

    // Truncating signed division (matches C / ORT semantics):
    //   10 / 3 =  3   (10 - 3*3 = 1 remainder, truncated toward zero)
    //  -10 / 3 = -3   (-10 - 3*(-3) = -1 remainder, truncated toward zero)
    //    7 / 2 =  3
    //    8 / 2 =  4
    let a_data = vec![10i64, -10, 7, 8];
    let b_data = vec![3i64, 3, 2, 2];

    let a = GpuTensor::from_host_i64(&ctx, &a_data, vec![4]).unwrap();
    let b = GpuTensor::from_host_i64(&ctx, &b_data, vec![4]).unwrap();

    let c = gpu_div(&ctx, &kernels, &mut pool, &a, &b).unwrap();
    let result = c.to_host_i64(&ctx).unwrap();

    assert_eq!(result, vec![3, -3, 3, 4]);
}

#[test]
#[ignore] // Requires CUDA GPU
fn test_gpu_div_i64_guards_divide_by_zero() {
    // Hardware signed-integer div-by-zero is UB on CUDA (may trap and
    // corrupt the stream). The kernel guards by emitting 0 for those
    // positions. Shape-arithmetic pipelines never hit this case in
    // practice, but the guard is load-bearing for safety.
    let (ctx, kernels, mut pool) = setup();

    let a_data = vec![10i64, 20, 30];
    let b_data = vec![2i64, 0, 5];

    let a = GpuTensor::from_host_i64(&ctx, &a_data, vec![3]).unwrap();
    let b = GpuTensor::from_host_i64(&ctx, &b_data, vec![3]).unwrap();

    let c = gpu_div(&ctx, &kernels, &mut pool, &a, &b).unwrap();
    let result = c.to_host_i64(&ctx).unwrap();

    // Expect: [10/2=5, 20/0=0 (guard), 30/5=6].
    assert_eq!(result, vec![5, 0, 6]);
}

#[test]
#[ignore] // Requires CUDA GPU
fn test_gpu_sigmoid() {
    let (ctx, kernels, mut pool) = setup();

    let data = vec![0.0f32, 1.0, -1.0, 2.0];
    let input = GpuTensor::from_host_f32(&ctx, &data, vec![4]).unwrap();

    let output = gpu_sigmoid(&ctx, &kernels, &mut pool, &input).unwrap();
    let result = output.to_host_f32(&ctx).unwrap();

    // sigmoid(0) = 0.5
    assert!((result[0] - 0.5).abs() < 1e-5);
    // sigmoid(1) ≈ 0.731
    assert!((result[1] - 0.7310586).abs() < 1e-5);
    // sigmoid(-1) ≈ 0.269
    assert!((result[2] - 0.26894143).abs() < 1e-5);
}

#[test]
#[ignore] // Requires CUDA GPU
fn test_gpu_tanh() {
    let (ctx, kernels, mut pool) = setup();

    let data = vec![0.0f32, 1.0, -1.0];
    let input = GpuTensor::from_host_f32(&ctx, &data, vec![3]).unwrap();

    let output = gpu_tanh(&ctx, &kernels, &mut pool, &input).unwrap();
    let result = output.to_host_f32(&ctx).unwrap();

    assert!((result[0] - 0.0).abs() < 1e-5);
    assert!((result[1] - 0.7615942).abs() < 1e-5);
    assert!((result[2] - (-0.7615942)).abs() < 1e-5);
}

#[test]
#[ignore] // Requires CUDA GPU
fn test_gpu_erf() {
    // Verifies the erff() intrinsic wrapped by gpu_erf matches the
    // CPU A&S reference within 1e-6 across 10k inputs on [-5, 5].
    let (ctx, kernels, mut pool) = setup();

    // Representative range: [-5, 5] at ~10k points.
    let input_vec: Vec<f32> = (0..10_000)
        .map(|i| -5.0 + (i as f32) * 10.0 / 10_000.0)
        .collect();
    let input = GpuTensor::from_host_f32(&ctx, &input_vec, vec![10_000]).unwrap();

    let output = gpu_erf(&ctx, &kernels, &mut pool, &input).unwrap();
    let gpu_vals = output.to_host_f32(&ctx).unwrap();

    for (i, &x) in input_vec.iter().enumerate() {
        let expected = crate::operators::erf::erf_f32(x);
        let got = gpu_vals[i];
        assert!(
            (got - expected).abs() < 1e-6,
            "gpu_erf disagrees with CPU at x={}: got {}, expected {}",
            x,
            got,
            expected,
        );
    }
}

#[test]
#[ignore] // Requires CUDA GPU
fn test_gpu_relu() {
    let (ctx, kernels, mut pool) = setup();

    let data = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let input = GpuTensor::from_host_f32(&ctx, &data, vec![5]).unwrap();

    let output = gpu_relu(&ctx, &kernels, &mut pool, &input).unwrap();
    let result = output.to_host_f32(&ctx).unwrap();

    assert_eq!(result, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
}

#[test]
#[ignore] // Requires CUDA GPU
fn test_gpu_leaky_relu() {
    let (ctx, kernels, mut pool) = setup();

    let data = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let input = GpuTensor::from_host_f32(&ctx, &data, vec![5]).unwrap();

    let output = gpu_leaky_relu(&ctx, &kernels, &mut pool, &input, 0.1).unwrap();
    let result = output.to_host_f32(&ctx).unwrap();

    assert!((result[0] - (-0.2)).abs() < 1e-5);
    assert!((result[1] - (-0.1)).abs() < 1e-5);
    assert!((result[2] - 0.0).abs() < 1e-5);
    assert!((result[3] - 1.0).abs() < 1e-5);
    assert!((result[4] - 2.0).abs() < 1e-5);
}

#[test]
#[ignore] // Requires CUDA GPU
fn test_gpu_exp() {
    let (ctx, kernels, mut pool) = setup();

    let data = vec![0.0f32, 1.0, 2.0];
    let input = GpuTensor::from_host_f32(&ctx, &data, vec![3]).unwrap();

    let output = gpu_exp(&ctx, &kernels, &mut pool, &input).unwrap();
    let result = output.to_host_f32(&ctx).unwrap();

    assert!((result[0] - 1.0).abs() < 1e-5);
    assert!((result[1] - std::f32::consts::E).abs() < 1e-5);
    assert!((result[2] - std::f32::consts::E.powi(2)).abs() < 1e-4);
}

#[test]
#[ignore] // Requires CUDA GPU
fn test_gpu_sqrt() {
    let (ctx, kernels, mut pool) = setup();

    let data = vec![1.0f32, 4.0, 9.0, 16.0];
    let input = GpuTensor::from_host_f32(&ctx, &data, vec![4]).unwrap();

    let output = gpu_sqrt(&ctx, &kernels, &mut pool, &input).unwrap();
    let result = output.to_host_f32(&ctx).unwrap();

    assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
#[ignore] // Requires CUDA GPU
fn test_gpu_clip() {
    let (ctx, kernels, mut pool) = setup();

    let data = vec![-5.0f32, -1.0, 0.5, 2.0, 10.0];
    let input = GpuTensor::from_host_f32(&ctx, &data, vec![5]).unwrap();

    let output = gpu_clip(&ctx, &kernels, &mut pool, &input, 0.0, 1.0).unwrap();
    let result = output.to_host_f32(&ctx).unwrap();

    assert_eq!(result, vec![0.0, 0.0, 0.5, 1.0, 1.0]);
}

#[test]
#[ignore] // Requires CUDA GPU
fn test_gpu_floor_ceil_round() {
    let (ctx, kernels, mut pool) = setup();

    let data = vec![1.2f32, 1.5, 1.8, -1.2, -1.5, -1.8];
    let input = GpuTensor::from_host_f32(&ctx, &data, vec![6]).unwrap();

    let floor_out = gpu_floor(&ctx, &kernels, &mut pool, &input).unwrap();
    let floor_result = floor_out.to_host_f32(&ctx).unwrap();
    assert_eq!(floor_result, vec![1.0, 1.0, 1.0, -2.0, -2.0, -2.0]);

    let ceil_out = gpu_ceil(&ctx, &kernels, &mut pool, &input).unwrap();
    let ceil_result = ceil_out.to_host_f32(&ctx).unwrap();
    assert_eq!(ceil_result, vec![2.0, 2.0, 2.0, -1.0, -1.0, -1.0]);

    let round_out = gpu_round(&ctx, &kernels, &mut pool, &input).unwrap();
    let round_result = round_out.to_host_f32(&ctx).unwrap();
    assert_eq!(round_result, vec![1.0, 2.0, 2.0, -1.0, -2.0, -2.0]);
}

#[test]
#[ignore] // Requires CUDA GPU
fn test_gpu_scalar_ops() {
    let (ctx, kernels, mut pool) = setup();

    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let input = GpuTensor::from_host_f32(&ctx, &data, vec![4]).unwrap();

    let add_out = gpu_add_scalar(&ctx, &kernels, &mut pool, &input, 10.0).unwrap();
    let add_result = add_out.to_host_f32(&ctx).unwrap();
    assert_eq!(add_result, vec![11.0, 12.0, 13.0, 14.0]);

    let mul_out = gpu_mul_scalar(&ctx, &kernels, &mut pool, &input, 2.0).unwrap();
    let mul_result = mul_out.to_host_f32(&ctx).unwrap();
    assert_eq!(mul_result, vec![2.0, 4.0, 6.0, 8.0]);
}

#[test]
#[ignore] // Requires CUDA GPU
fn test_gpu_large_tensor() {
    let (ctx, kernels, mut pool) = setup();

    // Test with 1 million elements
    let n = 1_000_000;
    let a_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..n).map(|i| (i * 2) as f32).collect();

    let a = GpuTensor::from_host_f32(&ctx, &a_data, vec![n]).unwrap();
    let b = GpuTensor::from_host_f32(&ctx, &b_data, vec![n]).unwrap();

    let c = gpu_add(&ctx, &kernels, &mut pool, &a, &b).unwrap();
    let result = c.to_host_f32(&ctx).unwrap();

    // Verify a few elements
    assert_eq!(result[0], 0.0);
    assert_eq!(result[1], 3.0);
    assert_eq!(result[n - 1], (n - 1) as f32 + ((n - 1) * 2) as f32);
}

// =============================================================================
// WS-3 M3.4 sub-2: FP16 elementwise contract tests
//
// All inputs are chosen to be exactly representable in f16 so the bit-level
// expectation matches `f16::from_f32`. For pow / erf / sqrt, results are
// compared with a small tolerance to absorb f32 round-trip and intrinsic
// precision (sub-1-ULP for the chosen inputs).
// =============================================================================

fn h(values: &[f32]) -> Vec<half::f16> {
    values.iter().copied().map(half::f16::from_f32).collect()
}

fn assert_close_f16(actual: &[half::f16], expected: &[f32], tol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{}: length mismatch", label);
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let a_f32 = a.to_f32();
        assert!(
            (a_f32 - e).abs() <= tol,
            "{}: idx {} got {} vs {} (tol {})",
            label,
            i,
            a_f32,
            e,
            tol
        );
    }
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_gpu_add_float16() {
    let (ctx, kernels, mut pool) = setup();
    let a = GpuTensor::from_host_f16(&ctx, &h(&[1.0, 2.0, -3.0, 0.5]), vec![4]).unwrap();
    let b = GpuTensor::from_host_f16(&ctx, &h(&[10.0, -2.0, 3.0, 0.25]), vec![4]).unwrap();

    let out = gpu_add(&ctx, &kernels, &mut pool, &a, &b).unwrap();
    assert_eq!(out.dtype(), super::super::tensor::DType::Float16);
    let result = out.to_host_f16(&ctx).unwrap();
    assert_eq!(result, h(&[11.0, 0.0, 0.0, 0.75]));
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_gpu_sub_float16() {
    let (ctx, kernels, mut pool) = setup();
    let a = GpuTensor::from_host_f16(&ctx, &h(&[5.0, 7.5, -2.0]), vec![3]).unwrap();
    let b = GpuTensor::from_host_f16(&ctx, &h(&[2.0, 0.5, -1.0]), vec![3]).unwrap();

    let out = gpu_sub(&ctx, &kernels, &mut pool, &a, &b).unwrap();
    let result = out.to_host_f16(&ctx).unwrap();
    assert_eq!(result, h(&[3.0, 7.0, -1.0]));
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_gpu_mul_float16() {
    let (ctx, kernels, mut pool) = setup();
    let a = GpuTensor::from_host_f16(&ctx, &h(&[2.0, 3.5, -4.0, 0.5]), vec![4]).unwrap();
    let b = GpuTensor::from_host_f16(&ctx, &h(&[4.0, 2.0, 0.5, 8.0]), vec![4]).unwrap();

    let out = gpu_mul(&ctx, &kernels, &mut pool, &a, &b).unwrap();
    let result = out.to_host_f16(&ctx).unwrap();
    assert_eq!(result, h(&[8.0, 7.0, -2.0, 4.0]));
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_gpu_div_float16() {
    let (ctx, kernels, mut pool) = setup();
    // Choose denominators well above the 1e-8 epsilon so the protect-branch
    // doesn't kick in.
    let a = GpuTensor::from_host_f16(&ctx, &h(&[8.0, 9.0, -6.0, 0.5]), vec![4]).unwrap();
    let b = GpuTensor::from_host_f16(&ctx, &h(&[2.0, 3.0, 4.0, 0.25]), vec![4]).unwrap();

    let out = gpu_div(&ctx, &kernels, &mut pool, &a, &b).unwrap();
    let result = out.to_host_f16(&ctx).unwrap();
    // 4.0, 3.0, -1.5, 2.0 — all exactly representable.
    assert_eq!(result, h(&[4.0, 3.0, -1.5, 2.0]));
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_gpu_div_float16_zero_protected_no_nan() {
    let (ctx, kernels, mut pool) = setup();
    // Both 0/0 and 1/0 cases. The kernel's epsilon guard's job is
    // specifically to prevent 0/0 = NaN; for 1/0 the f32 division
    // 1.0 / 1e-8 overflows f16 to +inf when the result is rounded back,
    // which is the correct behavior — only NaN must be excluded.
    let a = GpuTensor::from_host_f16(&ctx, &h(&[0.0, 1.0]), vec![2]).unwrap();
    let b = GpuTensor::from_host_f16(&ctx, &h(&[0.0, 0.0]), vec![2]).unwrap();

    let out = gpu_div(&ctx, &kernels, &mut pool, &a, &b).unwrap();
    let result = out.to_host_f16(&ctx).unwrap();
    assert!(
        !result[0].to_f32().is_nan(),
        "0/0 produced NaN — epsilon guard regression"
    );
    assert!(
        !result[1].to_f32().is_nan(),
        "1/0 produced NaN — epsilon guard regression"
    );
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_gpu_sqrt_float16() {
    let (ctx, kernels, mut pool) = setup();
    let x = GpuTensor::from_host_f16(&ctx, &h(&[4.0, 9.0, 16.0, 0.25]), vec![4]).unwrap();

    let out = gpu_sqrt(&ctx, &kernels, &mut pool, &x).unwrap();
    let result = out.to_host_f16(&ctx).unwrap();
    // 2.0, 3.0, 4.0, 0.5 — all exact perfect squares in f16 range.
    assert_eq!(result, h(&[2.0, 3.0, 4.0, 0.5]));
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_gpu_sqrt_float16_negative_clamped() {
    let (ctx, kernels, mut pool) = setup();
    // Negative input must clamp to 0 to match the f32 sqrt_kernel guard.
    let x = GpuTensor::from_host_f16(&ctx, &h(&[-4.0, 9.0]), vec![2]).unwrap();

    let out = gpu_sqrt(&ctx, &kernels, &mut pool, &x).unwrap();
    let result = out.to_host_f16(&ctx).unwrap();
    assert_eq!(result, h(&[0.0, 3.0]));
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_gpu_pow_float16() {
    let (ctx, kernels, mut pool) = setup();
    let base = GpuTensor::from_host_f16(&ctx, &h(&[2.0, 3.0, 4.0, 5.0]), vec![4]).unwrap();
    let exp = GpuTensor::from_host_f16(&ctx, &h(&[3.0, 2.0, 0.5, 0.0]), vec![4]).unwrap();

    let out = gpu_pow(&ctx, &kernels, &mut pool, &base, &exp).unwrap();
    let result = out.to_host_f16(&ctx).unwrap();
    // 8.0, 9.0, 2.0, 1.0 — small enough that the f32 round-trip lands on
    // exactly the same f16. Tolerance kept tight (0.01) to catch regressions.
    assert_close_f16(&result, &[8.0, 9.0, 2.0, 1.0], 0.01, "pow_f16");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_gpu_erf_float16() {
    let (ctx, kernels, mut pool) = setup();
    // erf(0) = 0, erf(1) ≈ 0.8427, erf(-1) ≈ -0.8427, erf(2) ≈ 0.9953.
    let x = GpuTensor::from_host_f16(&ctx, &h(&[0.0, 1.0, -1.0, 2.0]), vec![4]).unwrap();

    let out = gpu_erf(&ctx, &kernels, &mut pool, &x).unwrap();
    let result = out.to_host_f16(&ctx).unwrap();
    // f16 has ~3 decimal digits of precision — tolerance set to 1e-3.
    assert_close_f16(
        &result,
        &[0.0, 0.8427, -0.8427, 0.9953],
        1e-3,
        "erf_f16",
    );
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_gpu_add_float16_dtype_mismatch_errors() {
    // Plan-oversight closure: mixing Float16 + Float32 used to silently fall
    // through to the f32 path and crash on `data_f32 called on float16
    // tensor`. With the dispatch made exhaustive on (a.dtype, b.dtype), the
    // mismatch returns a clear error.
    let (ctx, kernels, mut pool) = setup();
    let a = GpuTensor::from_host_f16(&ctx, &h(&[1.0, 2.0]), vec![2]).unwrap();
    let b = GpuTensor::from_host_f32(&ctx, &[1.0f32, 2.0], vec![2]).unwrap();

    match gpu_add(&ctx, &kernels, &mut pool, &a, &b) {
        Ok(_) => panic!("expected dtype mismatch error, got Ok"),
        Err(super::super::context::CudaError::Kernel(msg)) => {
            assert!(
                msg.contains("unsupported dtype combination"),
                "expected typed error, got {}",
                msg
            );
        }
        Err(other) => panic!("expected CudaError::Kernel, got {}", other),
    }
}
