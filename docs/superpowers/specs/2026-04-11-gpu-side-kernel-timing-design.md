# Cycle 1.5a: GPU-Side Kernel Timing — Design

**Date:** 2026-04-11
**Author:** Claude (brainstormed with Ryan O'Neill, reviewed by Leadline Claude)
**Status:** Design approved, plan pending
**Requested by:** Leadline Claude, item #1 of four profiling enhancement requests
**Relates to:** `docs/performance/2026-04-11-cycle1-mul-sin-pow-mul-add-results.md`

---

## Context

After Cycle 1 shipped (MulSinPowMulAdd fusion, 25% speedup on Kokoro), Leadline flagged a diagnostic gap: iconnx's existing `run_with_profiling` uses CPU wall-clock time (`Instant::now()`) which mixes three things into a single per-op number:

1. The CUDA driver API call overhead (kernel launch cost)
2. Any CPU-side bookkeeping inside the dispatch arm
3. The actual GPU compute time of the kernel itself

Without separating (1)+(2) from (3), we cannot answer questions like "is this op launch-bound or compute-bound?" — which is exactly the question that would tell us whether to continue fusing elementwise ops (eliminates launches) or switch to memory-side work like zero-cost reshape (eliminates CPU bookkeeping) or CUDA graph capture (batches launches wholesale).

This cycle adds per-op GPU-side timing by recording CUDA events around every node dispatch in `run_with_profiling`, synchronizing the stream once at the end, and computing elapsed microseconds between event pairs. The result is exposed via a new `ProfileData::gpu_op_times` map whose keys match `op_times` exactly so Leadline can join the two and compute `launch_overhead = cpu_time - gpu_time` per op type.

### Why this is Cycle 1.5a, not Cycle 2

Leadline's original note said "after the current optimization work" — meaning after fusion cycles. But running more fusion cycles blindly wastes effort if the remaining ops turn out to be compute-bound (where fusion doesn't help). A 1–2 hour diagnostic investment unblocks the decision about what Cycle 2 should actually target. Cycle 1.5a also unblocks Leadline to begin parallel work on launch-overhead analysis before later profiling enhancements land.

## Goals

1. Populate `ProfileData::gpu_op_times: BTreeMap<String, (u64, usize)>` in `run_with_profiling`, containing per-op GPU-measured microseconds for every dispatched node (including fused patterns).

2. **Key parity invariant.** `gpu_op_times` must use the *exact same op-type strings* as `op_times`: same fused-pattern labels (`"Fused_MulSinPowMulAdd"`, `"Fused_DivRsqrt"`, etc.) and same per-op labels (`"Conv"`, `"MatMul"`, `"Add"`, etc.). Leadline-bench joins CPU and GPU times by key to compute per-op launch overhead; divergent keys break the join silently. This is enforced by the implementation using the **same label expression** for both maps at each dispatch site, and verified by a test that asserts `gpu_op_times.keys() == op_times.keys()` and `gpu_op_times[k].1 == op_times[k].1` (call counts) for every key.

3. Per-op granularity, not just total, so Leadline can compute `launch_overhead = cpu_time - gpu_time` and GPU utilization per op type.

4. GPU event logic must not serialize execution: allocate events upfront, record async during dispatch, synchronize once at the end.

5. Encapsulate event management in a new `GpuEventTimer` helper. `run_with_profiling` does not grow further.

6. Unit-testable: the helper must work against a real CUDA context in small standalone tests.

7. Error propagation: any CUDA error during event allocation, recording, or elapsed computation returns a `CudaError::Kernel` with a descriptive message. No silent fallback to CPU-only timing.

## Non-Goals

1. Change the behavior or semantics of existing `op_times` (CPU-side timing).
2. Pool events across runs. Events are allocated fresh per profiled run and dropped at the end.
3. Add GPU timing to `run()` (non-profiling) or `run_with_logging()`.
4. Measure sub-op kernels (e.g., the individual kernels inside a fused LSTM). Timing granularity matches existing `op_times` — one measurement per node dispatch.
5. Float precision beyond u64 microseconds. cudarc returns f32 ms; we multiply by 1000 and round down. This matches the precision of the existing CPU `op_times`.
6. Any of Leadline's other requested items (#2 tensor shapes, #3 per-inference pool metrics, #4 per-node shapes in `NodeExecutionLog`). Those are separate cycles (1.5b, 1.5c, 1.5d).

## Approach

**Approach B from brainstorming: extract a `GpuEventTimer` helper.** The helper owns a `Vec<CudaEvent>` (one sentinel event plus one per recorded op), a parallel `Vec<String>` of labels, and a reference to the CUDA context. It exposes three operations:

- `new(ctx, stream, estimated_ops)` — create with an initial start-sentinel event recorded on the stream.
- `record(label)` — allocate one new event, record it on the stream (async), store the label as the label for the interval ending at this event.
- `finalize(self)` — synchronize the stream once, walk event pairs computing elapsed microseconds, accumulate into a `BTreeMap<String, (u64, usize)>`, return the map.

`run_with_profiling` constructs the timer before the execution loop, calls `record(label)` immediately after each dispatch using the **same label expression** already used for `op_times.entry(label).or_insert(...)`, and calls `finalize()` after the loop to populate `profile.gpu_op_times`.

### Why not Approach A (inline) or Approach C (layered context)

- **Approach A** (inline inside `run_with_profiling`) would make the already-long function longer and tangle CPU-timing logic with GPU-event logic. The function is ~200 lines of dispatch logic already; adding event bookkeeping inline makes it harder to review.
- **Approach C** (single `ProfilingContext` carrying CPU + GPU + pool state) would be a significant refactor that would touch every dispatch site and every timing call. High risk of regressing the existing CPU timing, and overkill for a single-item cycle. If later profiling work (item #3, pool metrics) benefits from shared state, we can evolve the `GpuEventTimer` and future helpers into a layered context then.

## File Layout

### New files

```
src/cuda/inference/gpu_event_timer.rs       (~200 lines)
    pub(crate) struct GpuEventTimer {
        ctx: Arc<CudaContext>,
        events: Vec<CudaEvent>,           // sentinel + 1 per record() call
        labels: Vec<String>,              // labels for intervals[i-1..i]
    }

    Public methods:
      pub(crate) fn new(...) -> Result<Self, CudaError>
      pub(crate) fn record(&mut self, label: String) -> Result<(), CudaError>
      pub(crate) fn finalize(self) -> Result<BTreeMap<String, (u64, usize)>, CudaError>

tests/operators/gpu_event_timer_test.rs    (~180 lines)
    Four unit tests (see Testing section below).
```

### Modified files

```
src/cuda/inference/mod.rs
    - Add:   pub(super) mod gpu_event_timer;
    - Add:   use self::gpu_event_timer::GpuEventTimer;
    - Add:   gpu_op_times field to ProfileData
    - Modify: run_with_profiling — create timer, record after each dispatch, finalize at end
    - Modify: ProfileData::print_summary — optionally print gpu_op_times alongside op_times

src/cuda/context.rs  (possibly)
    - Add:   IconnxCudaContext::context() accessor returning &Arc<CudaContext>,
             IF an equivalent doesn't already exist. The plan phase will verify
             before committing to this change.

tests/kokoro_gpu_fusion_test.rs
    - Extend fusion_fires_on_full_kokoro_at_seq_len_5 with key-parity assertions
      and a "top 10 ops by launch overhead" diagnostic printout.
```

### Responsibility split

- `gpu_event_timer.rs` owns event lifecycle, label tracking, and elapsed computation — nothing else. No dependency on any executor state beyond the CUDA context and stream. Unit-testable in isolation.
- `inference/mod.rs` gains exactly three additions: the `use` line, the `ProfileData` field, and a pair of `record()` call sites inside the existing execution loop.

## API Contract

### `GpuEventTimer::new(ctx, stream, estimated_ops)`

```rust
/// Creates a new timer and records a "start sentinel" event on the stream.
///
/// The sentinel marks t=0 for the first operation; the first `record()` call
/// will define an interval from the sentinel up through the end of the first
/// dispatched op.
///
/// `estimated_ops` is used only to size the internal Vec capacity — it's a
/// hint, not a cap. Exceeding it triggers normal Vec growth.
///
/// Important: the event is created with `CU_EVENT_DEFAULT` so that timing
/// is enabled. cudarc's default flag is `CU_EVENT_DISABLE_TIMING`, which
/// would make `elapsed_ms` return an error.
pub(crate) fn new(
    ctx: Arc<CudaContext>,
    stream: &CudaStream,
    estimated_ops: usize,
) -> Result<Self, CudaError>
```

### `GpuEventTimer::record(label)`

```rust
/// Records the end of the operation labeled `label`.
///
/// Allocates one new CUDA event (timing-enabled), records it on the stream
/// (async enqueue), and stores `label` as the label for the interval that
/// ENDS at this event. The interval starts at the previous event (either
/// the start sentinel from `new()`, or the event recorded by the most
/// recent prior `record()` call).
///
/// Call this immediately AFTER dispatching the kernel for the operation.
/// CPU bookkeeping between dispatch and `record()` does not inflate the
/// GPU-measured interval because the event captures the current stream
/// queue position, and CPU work between dispatch and `record()` does not
/// enqueue anything on the stream.
///
/// The same label may be passed multiple times across calls; `finalize()`
/// aggregates repeat labels into a single `(total_us, count)` entry.
pub(crate) fn record(&mut self, label: String) -> Result<(), CudaError>
```

### `GpuEventTimer::finalize(self)`

```rust
/// Synchronizes the stream once, computes per-interval elapsed times via
/// `CudaEvent::elapsed_ms`, and aggregates them by label into a
/// `BTreeMap<String, (total_us, count)>`.
///
/// The returned map will contain exactly as many distinct keys as there
/// were distinct `label` values passed to `record()`. For each label, the
/// `count` is the number of times that label was recorded, and `total_us`
/// is the sum of the elapsed intervals for those recordings in whole
/// microseconds.
///
/// Special case: if an interval's GPU-measured elapsed time rounds down to
/// zero microseconds (common for metadata-only operations like `Unsqueeze`
/// or `Reshape` where the GPU work is effectively instantaneous), the
/// label's entry is still inserted/updated — `total_us` contributes 0 and
/// `count` increments by 1. The entry is never omitted on account of zero
/// elapsed time. This guarantees the key-parity invariant with `op_times`.
///
/// Consumes `self` so the underlying events are dropped after their
/// elapsed values have been read.
pub(crate) fn finalize(self) -> Result<BTreeMap<String, (u64, usize)>, CudaError>
```

### `ProfileData` field addition

```rust
#[derive(Debug, Default)]
pub struct ProfileData {
    // ... existing fields unchanged ...
    pub op_times: BTreeMap<String, (u64, usize)>,

    /// Per-op GPU-side timing (from CUDA events, excludes kernel launch
    /// overhead and CPU-side dispatch bookkeeping).
    ///
    /// Keys match `op_times` exactly: same op-type strings, same fused
    /// pattern labels. For any key present in `op_times` the same key is
    /// present here, and the `count` values match. `total_us` here is the
    /// GPU-measured elapsed time; `total_us` in `op_times` is CPU wall-clock
    /// (includes launch overhead). The difference
    /// `op_times[k].0 - gpu_op_times[k].0` approximates per-op launch
    /// overhead. An entry may have `total_us == 0` when the GPU work is
    /// effectively instantaneous (metadata-only ops).
    pub gpu_op_times: BTreeMap<String, (u64, usize)>,
}
```

## Integration with `run_with_profiling`

The critical invariant is that every dispatch site already computes a label for `op_times`. The implementation must use the **same label expression** for the `GpuEventTimer::record(label)` call at that site, with no duplicate computation or branching.

### Dispatch-site changes

There are two places where `op_times` is updated inside the execution loop:

**1. Fused-pattern dispatch block** (around `inference/mod.rs:4650`). The block resolves a `pattern_name: &str` in a match on `&pattern_info.pattern`, then:

```rust
let entry = op_times.entry(pattern_name.to_string()).or_insert((0, 0));
entry.0 += op_us;
entry.1 += 1;
```

**Change:** Immediately after `entry.1 += 1;`, add:
```rust
gpu_timer.record(pattern_name.to_string())?;
```

**2. Regular-op dispatch block** (around `inference/mod.rs:4775`):

```rust
let entry = op_times.entry(node.op_type.clone()).or_insert((0, 0));
entry.0 += op_us;
entry.1 += 1;
```

**Change:** Immediately after `entry.1 += 1;`, add:
```rust
gpu_timer.record(node.op_type.clone())?;
```

### Loop-boundary changes

**Before the execution loop:**
```rust
let mut gpu_timer = GpuEventTimer::new(
    self.ctx.context().clone(),
    self.ctx.stream(),
    sorted_nodes.len(),
)?;
```

**After the execution loop:**
```rust
profile.gpu_op_times = gpu_timer.finalize()?;
```

### Key-parity enforcement

The structural invariant is: every write to `op_times` is immediately followed by a `gpu_timer.record()` call using the **same variable** (`pattern_name.to_string()` or `node.op_type.clone()`). There is no separate computation of the label for GPU timing. A code reviewer can scan the diff and visually confirm each pair.

This is backed by the integration test assertion that `profile.op_times.keys() == profile.gpu_op_times.keys()`. If a future change to the dispatch sites forgets to update the GPU timer, the test fails immediately.

## Error Handling

No silent fallbacks. Every `CudaError` from the event API propagates via `?`:

- **Event creation failure** (`ctx.new_event`, in `new()` and inside `record()`) → `CudaError::Kernel(format!("GpuEventTimer: failed to create CUDA event: {e}"))`
- **Event record failure** (`event.record(stream)`) → `CudaError::Kernel(format!("GpuEventTimer: failed to record event for op '{label}': {e}"))`
- **Stream sync failure** (inside `finalize`) → `CudaError::Kernel(format!("GpuEventTimer: stream sync failed in finalize: {e}"))`
- **Elapsed computation failure** (`event.elapsed_ms`) → `CudaError::Kernel(format!("GpuEventTimer: elapsed_ms failed at interval {i}: {e}"))`

If `finalize()` is never called (e.g., the run unwinds from an earlier error), the timer's `Vec<CudaEvent>` simply drops. cudarc's `Drop for CudaEvent` handles GPU-side cleanup. No leak.

**Deliberate omissions:**
- No catch-fallback to CPU-only timing. Silent profiling gaps would hide real infrastructure problems.
- No partial `BTreeMap` on error. `finalize()` either returns the full map or an error.

## Testing

### Unit tests: `tests/operators/gpu_event_timer_test.rs`

Four tests, each constructs a real `IconnxCudaContext`, creates a `GpuEventTimer`, performs a tiny amount of real GPU work between `record()` calls to make the intervals non-trivial, and asserts on the returned map.

1. **`timer_records_one_op_and_finalizes`**
   - Create timer, do a small `gpu_add` of two `[1024]` tensors, `record("Add")`, `finalize()`.
   - Assert `map.len() == 1`, `map.get("Add")` is `Some(&(_, 1))`, and the total is a valid `u64` (may be 0 for very short ops).

2. **`timer_aggregates_same_label_across_calls`**
   - Create timer, do three `gpu_add` calls, each followed by `record("Add")`, `finalize()`.
   - Assert `map.get("Add") == Some(&(total_us, 3))` with `count == 3` and `total_us` equal to the sum of the three intervals.

3. **`timer_zero_elapsed_still_inserted`**
   - Create timer, do one call that produces near-zero GPU work (a very small elementwise or the smallest representable op), `record("Shape")`, `finalize()`.
   - Assert the `"Shape"` key is present even if `total_us == 0`, with `count == 1`. Guards the zero-elapsed insert rule.

4. **`timer_multiple_distinct_labels`**
   - Create timer, do `gpu_add` then `gpu_mul`, record `"Add"` then `"Mul"`, finalize.
   - Assert two entries, one per label, both with count 1.

### Integration test: extend `tests/kokoro_gpu_fusion_test.rs`

After the existing fusion-count assertions in `fusion_fires_on_full_kokoro_at_seq_len_5`, add:

```rust
// Verify GPU timing populated
assert!(
    !profile.gpu_op_times.is_empty(),
    "gpu_op_times must be populated when run_with_profiling is called"
);

// Key parity invariant
let cpu_keys: std::collections::HashSet<&String> = profile.op_times.keys().collect();
let gpu_keys: std::collections::HashSet<&String> = profile.gpu_op_times.keys().collect();
assert_eq!(
    cpu_keys, gpu_keys,
    "op_times and gpu_op_times keys must match — divergence breaks \
     Leadline's CPU/GPU join silently"
);

// Count parity and physical sanity (GPU time <= CPU time + small slack)
for (key, (cpu_us, cpu_count)) in &profile.op_times {
    let (gpu_us, gpu_count) = profile.gpu_op_times.get(key).unwrap();
    assert_eq!(
        cpu_count, gpu_count,
        "call count for '{}' diverges: op_times={}, gpu_op_times={}",
        key, cpu_count, gpu_count
    );
    let slack = (*cpu_count as u64).max(1);
    assert!(
        *gpu_us <= *cpu_us + slack,
        "GPU time ({} us) exceeds CPU time ({} us) for '{}' — physically impossible",
        gpu_us, cpu_us, key
    );
}

// Diagnostic printout: top 10 ops by launch overhead
let mut overhead: Vec<(String, u64, u64, u64, usize)> = profile
    .op_times
    .iter()
    .map(|(k, (cpu, count))| {
        let (gpu, _) = profile.gpu_op_times.get(k).copied().unwrap_or((0, 0));
        let launch_overhead = cpu.saturating_sub(gpu);
        (k.clone(), *cpu, gpu, launch_overhead, *count)
    })
    .collect();
overhead.sort_by_key(|(_, _, _, oh, _)| std::cmp::Reverse(*oh));
eprintln!("\n=== Top 10 ops by launch overhead (CPU - GPU, us) ===");
eprintln!(
    "  {:35} {:>10} {:>10} {:>10} {:>6}",
    "op", "cpu_us", "gpu_us", "overhead", "calls"
);
for (k, cpu, gpu, oh, count) in overhead.iter().take(10) {
    eprintln!("  {:35} {:>10} {:>10} {:>10} {:>6}", k, cpu, gpu, oh, count);
}
```

The top-10 printout is the core diagnostic Leadline will use to decide what to optimize next.

### What is deliberately NOT tested

- Absolute numerical accuracy of `elapsed_ms`. We trust cudarc/CUDA to return reasonable values; the `gpu_us <= cpu_us + slack` sanity check is the only correctness assertion.
- Thread safety of `GpuEventTimer`. The timer is `!Sync` by construction (owns a `Vec<CudaEvent>`). It's designed for single-threaded use in a single `run_with_profiling` call.
- Long-running behavior. `elapsed_ms` caps f32 full precision at ~16 seconds of microseconds. Kokoro inference at any expected sequence length is far below this cap.

## Validation Gates

Cycle 1.5a is complete when:

1. `cargo test --features cuda` passes (all existing tests plus the 4 new unit tests and the extended Kokoro integration test).
2. `cargo test` (no cuda) still passes (pre-existing no-cuda issue in `validation_test.rs` is not introduced by this cycle).
3. `cargo clippy --features cuda -- -D warnings` reports zero warnings.
4. `cargo fmt --check` passes.
5. `kokoro_gpu_fusion_test` shows `gpu_op_times` populated with identical keys and call counts to `op_times`, and produces the top-10 launch-overhead table.
6. The top-10 output is captured in the cycle completion commit message, so Leadline can start analysis from the commit log.
7. `src/cuda/inference/mod.rs` grows by fewer than 30 lines net (the two `record()` call sites, the new `use` import, the `gpu_op_times` field, the loop-boundary setup/finalize, the `ProfileData::print_summary` update).
8. `src/cuda/inference/gpu_event_timer.rs` exists at ~200 lines, well under the 1,000-line limit.
9. All commits are signed per CLAUDE.md. If signing fails, the implementer stops and asks the user.

## Risks and Mitigations

**Risk:** cudarc's `new_event(None)` defaults to `CU_EVENT_DISABLE_TIMING`, silently producing events that error out in `elapsed_ms`.
**Mitigation:** `GpuEventTimer::new` and `record` must explicitly pass `Some(sys::CUevent_flags::CU_EVENT_DEFAULT)`. The plan's Phase A step explicitly shows this flag, and one of the unit tests (`timer_records_one_op_and_finalizes`) exercises `elapsed_ms` end-to-end — if the flag is wrong, the test fails at `finalize()`.

**Risk:** Event allocation cost (~2,400 events per Kokoro profiled run) adds measurable overhead to the profiling path, making `cpu_time` appear inflated and skewing the `launch_overhead` calculation.
**Mitigation:** Per-event creation is measured in single-digit microseconds on modern CUDA drivers; 2,400 × ~5us ≈ 12ms of added setup time. The setup happens once at the start of `new()` (just the sentinel) and per-record. Since profiling is opt-in and the purpose is diagnosis rather than end-user latency, this is acceptable. If it becomes a problem, a future cycle can pool events across runs (explicitly rejected in Non-Goals for this cycle).

**Risk:** The key-parity invariant drifts if a future patch adds a new dispatch site but forgets to update the GPU timer.
**Mitigation:** The integration test asserts key-set equality on a real Kokoro run. A forgotten `record()` call would immediately fail the assertion with a specific key listing the divergence.

**Risk:** `self.ctx.context()` accessor may not exist on `IconnxCudaContext`.
**Mitigation:** The plan phase verifies the existence of this accessor before committing to the call. If it doesn't exist, two alternatives: (a) add a trivial accessor, (b) change `GpuEventTimer::new` to take `&IconnxCudaContext` directly and dereference the inner `CudaContext` inside the timer.

**Risk:** Git commit signing fails mid-cycle.
**Mitigation:** The implementer stops and asks the user, never uses `--no-gpg-sign`. This matches the existing Cycle 1 discipline.

## Out of scope (for follow-up cycles)

- **Item #2: `OnnxModel::tensor_shapes()`** — exposes intermediate tensor shapes from `value_info`. Becomes Cycle 1.5b.
- **Item #3: Per-inference pool metrics** — adds `pool_allocations`, `pool_hits`, `pool_hit_rate`, and `peak_intermediate_bytes` to `ProfileData`. Becomes Cycle 1.5c.
- **Item #4: Per-node shapes in `NodeExecutionLog`** — extends the existing log type with runtime-resolved input/output shapes. Becomes Cycle 1.5d.
- **Event pooling across runs** — rejected in Non-Goals for this cycle, revisitable once we have a measurement of whether event allocation is actually a bottleneck.
- **Timing sub-op kernels inside fused patterns** — keeps granularity at node-level, matching `op_times`. If someone later needs finer granularity inside a fused LSTM, they can add it as a debug-inference-feature-gated extension.
