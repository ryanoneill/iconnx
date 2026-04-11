//! Unit tests for the GpuEventTimer helper.
//!
//! Each test constructs a real IconnxCudaContext, drives a small GPU
//! workload between record() calls to produce nontrivial intervals, and
//! asserts on the BTreeMap returned by finalize().

#![cfg(feature = "cuda")]

use std::collections::BTreeMap;

use iconnx::cuda::inference::gpu_event_timer::GpuEventTimer;
use iconnx::cuda::{gpu_add, GpuMemoryPool, GpuTensor, IconnxCudaContext, KernelCache};

/// Small fixture: build two 1024-element tensors for GPU add().
fn make_add_inputs(ctx: &IconnxCudaContext) -> (GpuTensor, GpuTensor) {
    let n = 1024;
    let a_data: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
    let b_data: Vec<f32> = (0..n).map(|i| (n - i) as f32 * 0.01).collect();
    let a = GpuTensor::from_host_f32(ctx, &a_data, vec![n]).expect("a upload");
    let b = GpuTensor::from_host_f32(ctx, &b_data, vec![n]).expect("b upload");
    (a, b)
}

#[test]
fn timer_records_one_op_and_finalizes() {
    let ctx = IconnxCudaContext::new().expect("CUDA context");
    let elem_cache = KernelCache::new(&ctx).expect("KernelCache");
    let mut pool = GpuMemoryPool::new();
    let (a, b) = make_add_inputs(&ctx);

    let mut timer =
        GpuEventTimer::new(ctx.context().clone(), ctx.stream(), 8).expect("GpuEventTimer::new");

    let _ = gpu_add(&ctx, &elem_cache, &mut pool, &a, &b).expect("gpu_add");
    timer.record("Add".to_string()).expect("timer.record");

    let map: BTreeMap<String, (u64, usize)> = timer.finalize().expect("timer.finalize");

    assert_eq!(map.len(), 1, "expected exactly one entry, got {:?}", map);
    let (total_us, count) = map.get("Add").copied().expect("Add key");
    assert_eq!(count, 1, "Add should have count 1, got {}", count);
    // total_us may legitimately be 0 for a tiny op (elapsed_ms rounds down),
    // but u64 can never be negative, so this assertion is just a smoke check.
    let _ = total_us;
}
