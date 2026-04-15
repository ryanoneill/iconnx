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
        GpuEventTimer::new(ctx.garboard_device(), ctx.garboard_stream(), 8).expect("GpuEventTimer::new");

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

#[test]
fn timer_aggregates_same_label_across_calls() {
    let ctx = IconnxCudaContext::new().expect("CUDA context");
    let elem_cache = KernelCache::new(&ctx).expect("KernelCache");
    let mut pool = GpuMemoryPool::new();
    let (a, b) = make_add_inputs(&ctx);

    let mut timer =
        GpuEventTimer::new(ctx.garboard_device(), ctx.garboard_stream(), 8).expect("GpuEventTimer::new");

    // Three Add operations, all labeled the same -- finalize() should fold
    // them into one entry with count = 3.
    for _ in 0..3 {
        let _ = gpu_add(&ctx, &elem_cache, &mut pool, &a, &b).expect("gpu_add");
        timer.record("Add".to_string()).expect("timer.record");
    }

    let map = timer.finalize().expect("timer.finalize");
    assert_eq!(
        map.len(),
        1,
        "expected one key, got {:?}",
        map.keys().collect::<Vec<_>>()
    );
    let (_total_us, count) = map.get("Add").copied().expect("Add key");
    assert_eq!(count, 3, "Add should have count 3, got {}", count);
}

#[test]
fn timer_zero_elapsed_still_inserted() {
    let ctx = IconnxCudaContext::new().expect("CUDA context");
    let elem_cache = KernelCache::new(&ctx).expect("KernelCache");
    let mut pool = GpuMemoryPool::new();

    // Build the smallest possible tensors -- a shape-[1] f32 add will produce
    // near-zero GPU work, small enough that elapsed_ms rounds to 0 ms,
    // which becomes 0 microseconds after the f32->u64 conversion.
    // The zero-elapsed insert rule guarantees the label is still present
    // in the returned map with count = 1.
    let a = GpuTensor::from_host_f32(&ctx, &[1.0f32], vec![1]).expect("a upload");
    let b = GpuTensor::from_host_f32(&ctx, &[2.0f32], vec![1]).expect("b upload");

    let mut timer =
        GpuEventTimer::new(ctx.garboard_device(), ctx.garboard_stream(), 4).expect("GpuEventTimer::new");

    let _ = gpu_add(&ctx, &elem_cache, &mut pool, &a, &b).expect("gpu_add");
    timer.record("TinyAdd".to_string()).expect("timer.record");

    let map = timer.finalize().expect("timer.finalize");
    let (_total_us, count) = map.get("TinyAdd").copied().unwrap_or_else(|| {
        panic!(
            "TinyAdd key missing from map even though record() was called. \
             finalize() must insert a key whose elapsed rounds to 0. \
             Got keys: {:?}",
            map.keys().collect::<Vec<_>>()
        )
    });
    assert_eq!(
        count, 1,
        "TinyAdd should have count 1 even when total_us may be 0"
    );
}

#[test]
fn timer_multiple_distinct_labels() {
    use iconnx::cuda::gpu_mul;

    let ctx = IconnxCudaContext::new().expect("CUDA context");
    let elem_cache = KernelCache::new(&ctx).expect("KernelCache");
    let mut pool = GpuMemoryPool::new();
    let (a, b) = make_add_inputs(&ctx);

    let mut timer =
        GpuEventTimer::new(ctx.garboard_device(), ctx.garboard_stream(), 4).expect("GpuEventTimer::new");

    let _ = gpu_add(&ctx, &elem_cache, &mut pool, &a, &b).expect("gpu_add");
    timer.record("Add".to_string()).expect("timer.record Add");

    let _ = gpu_mul(&ctx, &elem_cache, &mut pool, &a, &b).expect("gpu_mul");
    timer.record("Mul".to_string()).expect("timer.record Mul");

    let map = timer.finalize().expect("timer.finalize");
    assert_eq!(
        map.len(),
        2,
        "expected two distinct keys, got {:?}",
        map.keys().collect::<Vec<_>>()
    );
    assert_eq!(
        map.get("Add").copied().map(|(_, c)| c),
        Some(1),
        "Add count"
    );
    assert_eq!(
        map.get("Mul").copied().map(|(_, c)| c),
        Some(1),
        "Mul count"
    );
}
