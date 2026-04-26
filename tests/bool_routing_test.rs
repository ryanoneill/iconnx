//! WS-3 M3.5 sub-B — Bool routing end-to-end.
//!
//! Asserts the comparison → Where chain consumes the new
//! `GpuTensor::Bool` representation natively (no f32 1.0/0.0 bounce).
//! Also covers the And/Or/Not + Bool-Cast surface that the migration
//! lit up.

#![cfg(feature = "cuda")]

use iconnx::cuda::{
    gpu_and, gpu_cast, gpu_equal, gpu_greater, gpu_less, gpu_not, gpu_or, gpu_where, DType,
    GpuMemoryPool, GpuTensor, IconnxCudaContext, OpsKernelCache,
};

fn setup() -> (IconnxCudaContext, OpsKernelCache, GpuMemoryPool) {
    let ctx = IconnxCudaContext::new().expect("CUDA context");
    let cache = OpsKernelCache::new(&ctx).expect("kernel cache");
    let pool = GpuMemoryPool::new();
    (ctx, cache, pool)
}

#[test]
#[ignore = "requires CUDA GPU"]
fn comparison_produces_bool_then_where_consumes_bool() {
    let (ctx, cache, mut pool) = setup();

    let a = GpuTensor::from_host_f32(&ctx, &[1.0, 2.0, 3.0], vec![3]).unwrap();
    let b = GpuTensor::from_host_f32(&ctx, &[2.0, 2.0, 2.0], vec![3]).unwrap();

    let mask = gpu_less(&ctx, &cache, &a, &b).unwrap();
    assert_eq!(mask.dtype(), DType::Bool, "comparison must produce Bool");

    let true_v = GpuTensor::from_host_f32(&ctx, &[10.0, 10.0, 10.0], vec![3]).unwrap();
    let false_v = GpuTensor::from_host_f32(&ctx, &[20.0, 20.0, 20.0], vec![3]).unwrap();

    // Where must consume Bool directly (no Bool→Float32 bounce).
    let out = gpu_where(&ctx, &cache, &mut pool, &mask, &true_v, &false_v).unwrap();
    assert_eq!(out.dtype(), DType::Float32);
    let result = out.to_host_f32(&ctx).unwrap();
    // a < b is true at idx 0 only (1<2, 2<2, 3<2).
    assert_eq!(result, vec![10.0, 20.0, 20.0]);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn cast_to_bool_then_where_consumes_bool() {
    let (ctx, cache, mut pool) = setup();

    // Float32 mask through Cast→Bool, then Where consumes the Bool.
    let raw = GpuTensor::from_host_f32(&ctx, &[1.0, 0.0, 5.0, 0.0], vec![4]).unwrap();
    let mask = gpu_cast(&ctx, &cache, &raw, DType::Bool).unwrap();
    assert_eq!(mask.dtype(), DType::Bool);
    assert_eq!(
        mask.to_host_bool(&ctx).unwrap(),
        vec![true, false, true, false]
    );

    let x = GpuTensor::from_host_f32(&ctx, &[10.0, 10.0, 10.0, 10.0], vec![4]).unwrap();
    let y = GpuTensor::from_host_f32(&ctx, &[20.0, 20.0, 20.0, 20.0], vec![4]).unwrap();
    let out = gpu_where(&ctx, &cache, &mut pool, &mask, &x, &y).unwrap();
    assert_eq!(
        out.to_host_f32(&ctx).unwrap(),
        vec![10.0, 20.0, 10.0, 20.0]
    );
}

#[test]
#[ignore = "requires CUDA GPU"]
fn cast_int32_to_bool_round_trip() {
    let (ctx, cache, _pool) = setup();
    let i = GpuTensor::from_host_i32(&ctx, &[0i32, 1, -1, 5, 0], vec![5]).unwrap();
    let b = gpu_cast(&ctx, &cache, &i, DType::Bool).unwrap();
    assert_eq!(b.dtype(), DType::Bool);
    assert_eq!(
        b.to_host_bool(&ctx).unwrap(),
        vec![false, true, true, true, false]
    );
}

#[test]
#[ignore = "requires CUDA GPU"]
fn cast_bool_to_float32_zero_one() {
    let (ctx, cache, _pool) = setup();
    let b = GpuTensor::from_host_bool(&ctx, &[true, false, true, true], vec![4]).unwrap();
    let f = gpu_cast(&ctx, &cache, &b, DType::Float32).unwrap();
    assert_eq!(f.to_host_f32(&ctx).unwrap(), vec![1.0, 0.0, 1.0, 1.0]);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn and_or_not_chain_on_bool() {
    let (ctx, cache, _pool) = setup();

    // Build two masks via comparison, then chain through And/Or/Not.
    let a = GpuTensor::from_host_f32(&ctx, &[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
    let b = GpuTensor::from_host_f32(&ctx, &[2.0, 2.0, 2.0, 2.0], vec![4]).unwrap();

    let lt = gpu_less(&ctx, &cache, &a, &b).unwrap(); // a < b: T F F F
    let eq = gpu_equal(&ctx, &cache, &a, &b).unwrap(); // a == b: F T F F
    let gt = gpu_greater(&ctx, &cache, &a, &b).unwrap(); // a > b: F F T T

    // (a < b) OR (a == b) = a <= b: T T F F
    let leq = gpu_or(&ctx, &cache, &lt, &eq).unwrap();
    assert_eq!(
        leq.to_host_bool(&ctx).unwrap(),
        vec![true, true, false, false]
    );

    // (a > b) AND (a == b) = always false.
    let intersect = gpu_and(&ctx, &cache, &gt, &eq).unwrap();
    assert_eq!(
        intersect.to_host_bool(&ctx).unwrap(),
        vec![false, false, false, false]
    );

    // NOT (a > b) = a <= b: T T F F
    let not_gt = gpu_not(&ctx, &cache, &gt).unwrap();
    assert_eq!(
        not_gt.to_host_bool(&ctx).unwrap(),
        vec![true, true, false, false]
    );
}

#[test]
#[ignore = "requires CUDA GPU"]
fn not_rejects_non_bool_input() {
    // The migration tightens contracts: gpu_not only accepts Bool now.
    // A Float32 input that previously coerced is a typed error today.
    let (ctx, cache, _pool) = setup();
    let f = GpuTensor::from_host_f32(&ctx, &[0.0, 1.0], vec![2]).unwrap();
    match gpu_not(&ctx, &cache, &f) {
        Ok(_) => panic!("Float32 input must produce typed error"),
        Err(iconnx::cuda::CudaError::Kernel(msg)) => assert!(
            msg.contains("must be Bool"),
            "expected typed error, got {}",
            msg
        ),
        Err(other) => panic!("expected CudaError::Kernel, got {}", other),
    }
}
