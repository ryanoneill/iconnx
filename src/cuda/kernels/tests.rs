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
