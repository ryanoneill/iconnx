# GPU-Side Kernel Timing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-op GPU-measured timing to `run_with_profiling` via a new `GpuEventTimer` helper, populate a new `ProfileData::gpu_op_times` field with keys that exactly match `op_times`, so Leadline can join the two maps and compute per-op launch overhead.

**Architecture:** A standalone helper in `src/cuda/inference/gpu_event_timer.rs` owns CUDA timing events. `run_with_profiling` creates a timer at the start of the execution loop, calls `record(label)` immediately after each kernel dispatch using the **exact same label expression** already used for `op_times` (so key parity is enforced structurally, not by a second string-formatting step), and calls `finalize()` after the loop to populate `profile.gpu_op_times`.

**Tech Stack:** Rust 2021, `cudarc` 0.17 (`CudaEvent`, `CudaStream`, `sys::CUevent_flags::CU_EVENT_DEFAULT`), existing `IconnxCudaContext` and `ProfileData` infrastructure in `src/cuda/inference/mod.rs`.

**Prerequisites for the implementing agent:**
- Read the spec first: `docs/superpowers/specs/2026-04-11-gpu-side-kernel-timing-design.md`.
- Per `/home/ryano/.claude/CLAUDE.md`: TDD wherever possible, commits after every substantive change, **all commits must be signed** (`git commit -S`), stop and ask if signing fails, warnings never ignored, no file over 1000 lines, no `--no-gpg-sign` under any circumstance.
- `cargo test --features cuda` is the baseline regression gate. Run it after each phase, not just at the end.
- `cargo clippy --features cuda -- -D warnings` must be clean before any commit.
- **Critical cudarc gotcha:** the default flag for `CudaContext::new_event(None)` is `CU_EVENT_DISABLE_TIMING`. You MUST pass `Some(sys::CUevent_flags::CU_EVENT_DEFAULT)` explicitly to get a timing-enabled event. If you forget, `elapsed_ms` fails at runtime with an opaque CUDA driver error. Every event allocation in this plan passes the flag explicitly — do not "simplify" it.
- **Never run `git checkout <sha>`** to inspect commits. Use `git show SHA`, `git show SHA --stat`, `git diff SHA^ SHA`, or `git log --show-signature -1 SHA`. A previous cycle had an implementer accidentally detach HEAD, requiring manual recovery. Before every commit, run `git branch --show-current` and confirm the output.

---

## Scope Check

This plan implements **Cycle 1.5a only** — item #1 from Leadline's four profiling enhancement requests. Items #2 (`OnnxModel::tensor_shapes`), #3 (per-inference pool metrics), and #4 (tensor shapes in `NodeExecutionLog`) are **out of scope** for this plan and will get their own specs and plans in Cycles 1.5b/c/d.

Any work outside the files listed below is a scope violation and should be flagged to the controller before landing.

---

## File Structure

### New files

- `src/cuda/inference/gpu_event_timer.rs` (~200 lines) — owns the `GpuEventTimer` struct and its three public methods (`new`, `record`, `finalize`). No dependency on executor state; testable in isolation against a raw `IconnxCudaContext`.
- `tests/operators/gpu_event_timer_test.rs` (~180 lines) — four unit tests covering basic usage, aggregation, zero-elapsed handling, and multi-label separation.

### Modified files

- `src/cuda/inference/mod.rs`:
  - Add `pub(super) mod gpu_event_timer;`
  - Add `use self::gpu_event_timer::GpuEventTimer;`
  - Add `pub gpu_op_times: BTreeMap<String, (u64, usize)>` to `ProfileData`
  - Modify `run_with_profiling` — construct timer before loop, call `.record(label)` after each dispatch at two sites, call `.finalize()` after loop
  - Modify `ProfileData::print_summary` to include the new map
- `tests/operators/mod.rs`:
  - Add `mod gpu_event_timer_test;` declaration
- `tests/kokoro_gpu_fusion_test.rs`:
  - Extend `fusion_fires_on_full_kokoro_at_seq_len_5` test with key-parity assertions and a top-10 launch-overhead diagnostic printout

### Responsibility split

- `gpu_event_timer.rs` owns **only** event lifecycle, label tracking, and elapsed-to-microsecond aggregation. It does not know about `GpuGraphExecutor`, `ExecutionNode`, `FusedPattern`, or any other executor concept.
- `inference/mod.rs` is the integration point. It translates the existing `pattern_name`/`node.op_type` labels into `GpuEventTimer::record()` calls, preserving key parity with `op_times` by reusing the same variables with no string reformatting.
- Tests are split: unit tests for the helper live under `tests/operators/`, integration tests that exercise the full executor live under `tests/`.

---

## Phase A: Core `GpuEventTimer` (test-first)

### Task 1: Create `GpuEventTimer` with working `new`, `record`, and `finalize` (TDD: one passing test)

**Files:**
- Create: `src/cuda/inference/gpu_event_timer.rs`
- Create: `tests/operators/gpu_event_timer_test.rs`
- Modify: `src/cuda/inference/mod.rs`
- Modify: `tests/operators/mod.rs`

#### Step 1: Create the failing test file

Create `tests/operators/gpu_event_timer_test.rs` with this exact content:

```rust
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

    let mut timer = GpuEventTimer::new(
        ctx.context().clone(),
        ctx.stream(),
        8,
    )
    .expect("GpuEventTimer::new");

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
```

Run the test to confirm it fails at compile time because `GpuEventTimer` does not yet exist:

```bash
cargo test --features cuda --test mod -- operators::gpu_event_timer_test::timer_records_one_op_and_finalizes 2>&1 | tail -20
```

Expected: compile error referencing `iconnx::cuda::inference::gpu_event_timer::GpuEventTimer` not found. This is the red state.

#### Step 2: Register the test module

Add this line to `tests/operators/mod.rs`, in alphabetical position (after `gemm_test`, before `greater_or_equal_test`):

```rust
mod gpu_event_timer_test;
```

Do NOT reorder existing entries.

#### Step 3: Create the `gpu_event_timer.rs` module

Write `src/cuda/inference/gpu_event_timer.rs` with this content:

```rust
//! GPU-side kernel timing via CUDA events.
//!
//! `GpuEventTimer` records a CUDA event before each op dispatch (via a
//! "start sentinel" in `new()`) and after each op dispatch (via `record()`).
//! After all ops have been dispatched, `finalize()` synchronizes the stream
//! once, computes the elapsed time between consecutive event pairs, and
//! aggregates the results into a `BTreeMap<String, (total_us, count)>`
//! keyed by the labels supplied to `record()`.
//!
//! # Invariants
//!
//! - All events are created with `CU_EVENT_DEFAULT` so that `elapsed_ms`
//!   returns a valid measurement. cudarc's default flag is
//!   `CU_EVENT_DISABLE_TIMING`; passing that default would cause
//!   `elapsed_ms` to fail with an opaque CUDA driver error.
//! - `events[0]` is the start sentinel recorded at construction.
//!   `events[i+1]` is the event recorded by the `i`-th call to `record()`.
//!   The label for the interval from `events[i]` to `events[i+1]` is
//!   `labels[i]`, so `labels.len() == events.len() - 1`.
//! - Elapsed time that rounds to zero microseconds still produces a map
//!   entry (with `total_us == 0` and `count` incremented). This preserves
//!   the key-parity invariant with `op_times` in `run_with_profiling`.
//!
//! # Usage
//!
//! ```ignore
//! let mut timer = GpuEventTimer::new(
//!     ctx.context().clone(),
//!     ctx.stream(),
//!     sorted_nodes.len(),
//! )?;
//! for node in sorted_nodes {
//!     dispatch_kernel_for(node)?;
//!     timer.record(label_for(node))?;
//! }
//! let gpu_op_times = timer.finalize()?;
//! ```

use std::collections::BTreeMap;
use std::sync::Arc;

use cudarc::driver::sys::CUevent_flags;
use cudarc::driver::{CudaContext, CudaEvent, CudaStream};

use crate::cuda::context::CudaError;

/// Per-op GPU timing via CUDA events. See module-level docs for invariants.
pub struct GpuEventTimer {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    /// Events recorded on the stream. `events[0]` is the start sentinel;
    /// `events[i+1]` is the event recorded by the i-th `record()` call.
    events: Vec<CudaEvent>,
    /// Label for each recorded interval. `labels[i]` labels the interval
    /// from `events[i]` to `events[i+1]`.
    labels: Vec<String>,
}

impl GpuEventTimer {
    /// Creates a new timer and records a start sentinel event on the stream.
    ///
    /// `estimated_ops` is a capacity hint for the internal `Vec`s — exceeding
    /// it is safe and only triggers normal `Vec` growth.
    pub fn new(
        ctx: Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        estimated_ops: usize,
    ) -> Result<Self, CudaError> {
        let sentinel = ctx
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|e| {
                CudaError::Kernel(format!(
                    "GpuEventTimer: failed to create start sentinel event: {}",
                    e
                ))
            })?;
        sentinel.record(stream).map_err(|e| {
            CudaError::Kernel(format!(
                "GpuEventTimer: failed to record start sentinel event: {}",
                e
            ))
        })?;

        let mut events = Vec::with_capacity(estimated_ops + 1);
        events.push(sentinel);
        Ok(Self {
            ctx,
            stream: stream.clone(),
            events,
            labels: Vec::with_capacity(estimated_ops),
        })
    }

    /// Records the end of the operation labeled `label`.
    ///
    /// Call this immediately AFTER dispatching the kernel for the operation.
    /// CPU-side bookkeeping between dispatch and this call does not affect
    /// the GPU-measured interval because the recorded event captures the
    /// stream's queue position, not the CPU's wall-clock.
    pub fn record(&mut self, label: String) -> Result<(), CudaError> {
        let event = self
            .ctx
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|e| {
                CudaError::Kernel(format!(
                    "GpuEventTimer: failed to create event for op '{}': {}",
                    label, e
                ))
            })?;
        event.record(&self.stream).map_err(|e| {
            CudaError::Kernel(format!(
                "GpuEventTimer: failed to record event for op '{}': {}",
                label, e
            ))
        })?;
        self.events.push(event);
        self.labels.push(label);
        Ok(())
    }

    /// Consumes the timer, synchronizes the stream, and returns the
    /// aggregated `(total_us, count)` map keyed by label.
    ///
    /// See module-level docs for the zero-elapsed insert rule.
    pub fn finalize(self) -> Result<BTreeMap<String, (u64, usize)>, CudaError> {
        // One synchronization point for the whole set of events.
        self.stream.synchronize().map_err(|e| {
            CudaError::Kernel(format!(
                "GpuEventTimer: stream sync failed in finalize: {}",
                e
            ))
        })?;

        let mut map: BTreeMap<String, (u64, usize)> = BTreeMap::new();
        for (i, label) in self.labels.into_iter().enumerate() {
            let start = &self.events[i];
            let end = &self.events[i + 1];
            let elapsed_ms = start.elapsed_ms(end).map_err(|e| {
                CudaError::Kernel(format!(
                    "GpuEventTimer: elapsed_ms failed at interval {} ('{}'): {}",
                    i, label, e
                ))
            })?;
            // elapsed_ms returns f32 milliseconds. Multiply by 1000 and
            // truncate to u64 microseconds. Negative values are not
            // physically possible; clamp to 0 out of paranoia.
            let elapsed_us = (elapsed_ms.max(0.0) * 1000.0) as u64;
            let entry = map.entry(label).or_insert((0, 0));
            entry.0 += elapsed_us;
            entry.1 += 1;
        }
        Ok(map)
    }
}
```

Note on `Arc<CudaStream>`: the spec's API contract showed `stream: &CudaStream` but `IconnxCudaContext::stream()` returns `&Arc<CudaStream>` (confirmed in `src/cuda/context.rs:45`). The implementation uses `&Arc<CudaStream>` to match the actual accessor return type and clones the `Arc` into the struct for later use in `record()` and `finalize()`.

#### Step 4: Wire the module into `inference/mod.rs`

In `src/cuda/inference/mod.rs`, find the module declarations block near the top (just after the `use` statements, around line 44-50). Add:

```rust
pub mod gpu_event_timer;
```

**Visibility choice:** `pub mod` (not `pub(super) mod` or `pub(crate) mod`) because the integration test `tests/operators/gpu_event_timer_test.rs` imports the type as `iconnx::cuda::inference::gpu_event_timer::GpuEventTimer`. For that path to resolve from an external integration test, `inference` must be `pub mod inference;` (which it already is, established in Cycle 1 Task C1+C2), and `gpu_event_timer` must be `pub mod gpu_event_timer;` inside `inference`.

#### Step 5: Build and run the test

```bash
cargo build --features cuda
```

Expected: clean compile. Likely first-attempt errors:
- Import errors on `cudarc::driver::sys::CUevent_flags` — confirm the path is exactly `cudarc::driver::sys::CUevent_flags` and that the `CU_EVENT_DEFAULT` variant exists.
- `CudaError` not found — the correct path is `use crate::cuda::context::CudaError;` (established pattern in the codebase).
- `Arc<CudaStream>` mismatch — `IconnxCudaContext::stream()` returns `&Arc<CudaStream>`; the `new()` signature must accept `&Arc<CudaStream>` (not `&CudaStream`).

```bash
cargo test --features cuda --test mod -- operators::gpu_event_timer_test::timer_records_one_op_and_finalizes
```

Expected: PASS. If the test fails at `timer.finalize()` with a CUDA driver error about event timing, double-check that `new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))` is used (not `None` or a different flag).

#### Step 6: Clippy and format

```bash
cargo clippy --features cuda -- -D warnings
cargo fmt --check
```

Expected: zero warnings, clean format.

#### Step 7: Commit (signed)

```bash
git branch --show-current
# Expected: the branch name set up by the controller (likely cycle1.5a/gpu-side-timing)
```

```bash
git add src/cuda/inference/gpu_event_timer.rs src/cuda/inference/mod.rs tests/operators/gpu_event_timer_test.rs tests/operators/mod.rs
git commit -S -m "$(cat <<'EOF'
feat(cuda): add GpuEventTimer helper for GPU-side per-op timing

Introduces a new helper at src/cuda/inference/gpu_event_timer.rs
that records CUDA events around operator dispatches in
run_with_profiling. Exposes three methods:

  new(ctx, stream, estimated_ops) -> records a start sentinel
  record(label)                   -> records the end of an op
  finalize(self)                  -> sync + aggregate -> BTreeMap

Events are created with CU_EVENT_DEFAULT so elapsed_ms returns
valid measurements (cudarc's default is CU_EVENT_DISABLE_TIMING
which would silently break the measurement).

Includes one unit test (timer_records_one_op_and_finalizes) that
uses gpu_add on a 1024-element tensor to verify the end-to-end
round trip: new -> dispatch -> record -> finalize, asserting
exactly one entry in the returned map with the expected label
and count of 1.

Subsequent tasks add more unit tests, the ProfileData field,
the run_with_profiling integration, and the Kokoro end-to-end
validation.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

If `git commit -S` fails with a signing error, STOP and report BLOCKED. Do NOT use `--no-gpg-sign`.

---

### Task 2: Add the remaining unit tests (aggregation, zero-elapsed, multi-label)

**Files:**
- Modify: `tests/operators/gpu_event_timer_test.rs`

These three tests should all pass against the helper implemented in Task 1. If any fails, there's a bug in `GpuEventTimer::finalize` that Task 1 didn't catch.

#### Step 1: Add the aggregation test

Append to `tests/operators/gpu_event_timer_test.rs`:

```rust
#[test]
fn timer_aggregates_same_label_across_calls() {
    let ctx = IconnxCudaContext::new().expect("CUDA context");
    let elem_cache = KernelCache::new(&ctx).expect("KernelCache");
    let mut pool = GpuMemoryPool::new();
    let (a, b) = make_add_inputs(&ctx);

    let mut timer = GpuEventTimer::new(
        ctx.context().clone(),
        ctx.stream(),
        8,
    )
    .expect("GpuEventTimer::new");

    // Three Add operations, all labeled the same — finalize() should fold
    // them into one entry with count = 3.
    for _ in 0..3 {
        let _ = gpu_add(&ctx, &elem_cache, &mut pool, &a, &b).expect("gpu_add");
        timer.record("Add".to_string()).expect("timer.record");
    }

    let map = timer.finalize().expect("timer.finalize");
    assert_eq!(map.len(), 1, "expected one key, got {:?}", map.keys().collect::<Vec<_>>());
    let (_total_us, count) = map.get("Add").copied().expect("Add key");
    assert_eq!(count, 3, "Add should have count 3, got {}", count);
}
```

#### Step 2: Add the zero-elapsed test

Append:

```rust
#[test]
fn timer_zero_elapsed_still_inserted() {
    use iconnx::cuda::ops::gpu_shape;

    let ctx = IconnxCudaContext::new().expect("CUDA context");
    let ops_cache = iconnx::cuda::ops::OpsKernelCache::new(&ctx).expect("OpsKernelCache");
    let (a, _b) = make_add_inputs(&ctx);

    let mut timer = GpuEventTimer::new(
        ctx.context().clone(),
        ctx.stream(),
        4,
    )
    .expect("GpuEventTimer::new");

    // gpu_shape is a trivial metadata-returning op. Its GPU-side elapsed
    // time is very close to 0 and will usually round down to 0 microseconds.
    // The zero-elapsed insert rule guarantees the "Shape" entry is still
    // present in the returned map with count = 1.
    let _ = gpu_shape(&ctx, &a).expect("gpu_shape");
    timer.record("Shape".to_string()).expect("timer.record");

    let map = timer.finalize().expect("timer.finalize");
    let (_total_us, count) = map.get("Shape").copied().unwrap_or_else(|| {
        panic!(
            "Shape key missing from map even though record() was called. \
             finalize() must insert a key whose elapsed rounds to 0. \
             Got keys: {:?}",
            map.keys().collect::<Vec<_>>()
        )
    });
    assert_eq!(count, 1, "Shape should have count 1 even when total_us is 0");
}
```

Note: if `iconnx::cuda::ops::gpu_shape` or `iconnx::cuda::ops::OpsKernelCache` is not publicly exported, substitute any other op that produces near-zero GPU work and update the imports. `gpu_add` of a very small tensor (e.g. shape `[1]`) is an acceptable fallback — the key assertion is that the label is still inserted when `total_us == 0`, not that it actually IS zero.

#### Step 3: Add the multi-label test

Append:

```rust
#[test]
fn timer_multiple_distinct_labels() {
    use iconnx::cuda::gpu_mul;

    let ctx = IconnxCudaContext::new().expect("CUDA context");
    let elem_cache = KernelCache::new(&ctx).expect("KernelCache");
    let mut pool = GpuMemoryPool::new();
    let (a, b) = make_add_inputs(&ctx);

    let mut timer = GpuEventTimer::new(
        ctx.context().clone(),
        ctx.stream(),
        4,
    )
    .expect("GpuEventTimer::new");

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
    assert_eq!(map.get("Add").copied().map(|(_, c)| c), Some(1), "Add count");
    assert_eq!(map.get("Mul").copied().map(|(_, c)| c), Some(1), "Mul count");
}
```

#### Step 4: Run all four unit tests

```bash
cargo test --features cuda --test mod -- operators::gpu_event_timer_test
```

Expected: 4 tests pass.

#### Step 5: Full regression check

```bash
cargo test --features cuda
```

Expected: existing test counts unchanged plus the new 4 tests. Nothing else should have shifted.

#### Step 6: Clippy and format

```bash
cargo clippy --features cuda -- -D warnings
cargo fmt --check
```

Expected: clean.

#### Step 7: Commit (signed)

```bash
git branch --show-current
```

```bash
git add tests/operators/gpu_event_timer_test.rs
git commit -S -m "$(cat <<'EOF'
test(cuda): add GpuEventTimer unit tests for aggregation and edge cases

Adds three more unit tests for GpuEventTimer covering:

  timer_aggregates_same_label_across_calls -- same label passed 3x
    produces one map entry with count=3 and summed total_us.

  timer_zero_elapsed_still_inserted -- a gpu_shape call produces
    near-zero GPU work; the label must still appear in the map with
    count=1 even when total_us rounds to 0. This guards the
    key-parity invariant with op_times -- every op that was
    dispatched must get an entry, regardless of elapsed magnitude.

  timer_multiple_distinct_labels -- two different labels produce
    two separate entries, each with count=1.

All three pass against the helper as implemented in the previous
commit.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase B: Integrate with `run_with_profiling`

### Task 3: Add `gpu_op_times` field to `ProfileData` and update `print_summary`

**Files:**
- Modify: `src/cuda/inference/mod.rs`

This task adds the field but does not populate it yet. It's a non-behavioral change — existing tests continue to pass because `Default::default()` initializes the field to an empty `BTreeMap`.

#### Step 1: Add the field

In `src/cuda/inference/mod.rs`, find the `ProfileData` struct at line ~92-109 and add the new field at the end, after `pub op_times`:

```rust
/// Profiling data for GPU inference
#[derive(Debug, Default)]
pub struct ProfileData {
    /// Time to upload input tensors (microseconds)
    pub input_upload_us: u64,
    /// Time to insert weight references (microseconds)
    pub weight_ref_us: u64,
    /// Time for topological sort (microseconds)
    pub topo_sort_us: u64,
    /// Total node execution time (microseconds)
    pub node_execution_us: u64,
    /// Time to download outputs (microseconds)
    pub output_download_us: u64,
    /// Total time (microseconds)
    pub total_us: u64,
    /// Per-operator-type timing: (total_us, count)
    pub op_times: BTreeMap<String, (u64, usize)>,
    /// Per-op GPU-side timing from CUDA events (excludes launch overhead
    /// and CPU-side dispatch bookkeeping).
    ///
    /// Keys match `op_times` exactly: same op-type strings, same fused
    /// pattern labels. For any key present in `op_times` the same key is
    /// present here, and the `count` values match. `total_us` here is
    /// the GPU-measured elapsed time; `total_us` in `op_times` is CPU
    /// wall-clock (includes launch overhead). The difference
    /// `op_times[k].0 - gpu_op_times[k].0` approximates per-op launch
    /// overhead. An entry may have `total_us == 0` when the GPU work
    /// is effectively instantaneous (metadata-only ops).
    pub gpu_op_times: BTreeMap<String, (u64, usize)>,
}
```

The `#[derive(Default)]` on the struct handles initializing the new field to `BTreeMap::new()`, so nothing else changes.

#### Step 2: Extend `print_summary`

In the same file, find `ProfileData::print_summary` at line ~113-142 and add a "GPU-Side Timing" section after the existing "Top Operators by Time" block:

```rust
    /// Print a summary of the profiling data
    pub fn print_summary(&self) {
        eprintln!("\n=== GPU Inference Profile ===");
        eprintln!("  Input upload:     {:>8} us", self.input_upload_us);
        eprintln!("  Weight refs:      {:>8} us", self.weight_ref_us);
        eprintln!("  Topo sort:        {:>8} us", self.topo_sort_us);
        eprintln!(
            "  Node execution:   {:>8} us ({:.2} ms)",
            self.node_execution_us,
            self.node_execution_us as f64 / 1000.0
        );
        eprintln!("  Output download:  {:>8} us", self.output_download_us);
        eprintln!(
            "  TOTAL:            {:>8} us ({:.2} ms)",
            self.total_us,
            self.total_us as f64 / 1000.0
        );

        eprintln!("\n=== Top Operators by Time ===");
        let mut sorted_ops: Vec<_> = self.op_times.iter().collect();
        sorted_ops.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));

        for (op, (time_us, count)) in sorted_ops.iter().take(15) {
            let avg = *time_us as f64 / *count as f64;
            eprintln!(
                "  {:20} {:>8} us ({:>4} calls, {:>6.1} us/call)",
                op, time_us, count, avg
            );
        }

        if !self.gpu_op_times.is_empty() {
            eprintln!("\n=== Top Operators by GPU-Side Time (CUDA events) ===");
            let mut sorted_gpu: Vec<_> = self.gpu_op_times.iter().collect();
            sorted_gpu.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));

            for (op, (time_us, count)) in sorted_gpu.iter().take(15) {
                let cpu_us = self.op_times.get(*op).map(|(t, _)| *t).unwrap_or(0);
                let overhead = cpu_us.saturating_sub(*time_us);
                eprintln!(
                    "  {:20} gpu={:>8} us cpu={:>8} us overhead={:>8} us ({:>4} calls)",
                    op, time_us, cpu_us, overhead, count
                );
            }
        }
    }
```

The new section is gated on `!self.gpu_op_times.is_empty()` so callers that get a `ProfileData::default()` (empty map) don't see a confusing empty section.

Do NOT modify any other part of `print_summary`. The existing 15-line top-by-CPU-time section is unchanged.

#### Step 3: Build

```bash
cargo build --features cuda
```

Expected: clean. The `#[derive(Default)]` handles the new field; no other struct callers should need updates.

#### Step 4: Run all tests

```bash
cargo test --features cuda
```

Expected: same test counts as before this task. This is a pure additive change — no behavior should have shifted.

#### Step 5: Clippy

```bash
cargo clippy --features cuda -- -D warnings
```

Expected: zero warnings. If clippy complains about unused `gpu_op_times` (because nothing writes to it yet), that's OK — the next task populates it. If clippy complains with `-D warnings` and would force `#[allow(dead_code)]`, add `#[allow(dead_code)]` to the field for this commit only and remove it in the next commit.

#### Step 6: Commit (signed)

```bash
git branch --show-current
git add src/cuda/inference/mod.rs
git commit -S -m "$(cat <<'EOF'
feat(profiling): add gpu_op_times field to ProfileData

Adds a new BTreeMap<String, (u64, usize)> field on ProfileData that
will hold per-op GPU-side timing collected via CUDA events. The
field is populated in a follow-up commit that wires GpuEventTimer
into run_with_profiling.

print_summary gains a new "GPU-Side Time (CUDA events)" section
that prints GPU times alongside CPU times with a per-op overhead
column. The section is suppressed when the map is empty so legacy
callers (e.g. ProfileData::default()) see unchanged output.

This commit is intentionally non-behavioral: the field starts
empty, no timing logic changes, and all existing tests pass
unchanged.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Wire `GpuEventTimer` into `run_with_profiling`

**Files:**
- Modify: `src/cuda/inference/mod.rs`

This is the integration task. The key-parity invariant is enforced by reusing the same label expressions that `op_times` already uses — no string reformatting, no second match.

#### Step 1: Add the import

In `src/cuda/inference/mod.rs`, near the top imports, add:

```rust
use self::gpu_event_timer::GpuEventTimer;
```

If there is already a `use self::...` block, add the line alphabetically. If not, add it immediately after the `pub mod gpu_event_timer;` line established in Task 1.

#### Step 2: Construct the timer before the execution loop

Find `run_with_profiling` in `src/cuda/inference/mod.rs`. The relevant block starts around line 4148-4154 with the topological sort and op_times initialization:

```rust
        // Topological sort
        let sort_start = Instant::now();
        let sorted_nodes = self.topological_sort()?;
        profile.topo_sort_us = sort_start.elapsed().as_micros() as u64;

        // Execute nodes with per-op-type timing
        let mut op_times: BTreeMap<String, (u64, usize)> = BTreeMap::new();
        let exec_start = Instant::now();
```

Immediately after the `let mut op_times: BTreeMap<...> = BTreeMap::new();` line and BEFORE `let exec_start = Instant::now();`, insert:

```rust
        // GPU-side timing via CUDA events. Allocates events fresh per run.
        let mut gpu_timer = GpuEventTimer::new(
            self.ctx.context().clone(),
            self.ctx.stream(),
            sorted_nodes.len(),
        )?;
```

Leave `let exec_start = Instant::now();` on the next line. The timer construction runs before the CPU-side wall-clock starts so the sentinel event is enqueued before any kernel dispatch.

#### Step 3: Add `record()` call for the fused-pattern dispatch site

Find the fused-pattern dispatch block in `run_with_profiling` around line 4163-4208. It ends with these three lines (around 4204-4206):

```rust
                let entry = op_times.entry(pattern_name.to_string()).or_insert((0, 0));
                entry.0 += op_us;
                entry.1 += 1;
```

Immediately after `entry.1 += 1;` and BEFORE the `continue;` statement that follows, insert:

```rust
                gpu_timer.record(pattern_name.to_string())?;
```

The resulting block should look like:

```rust
                let entry = op_times.entry(pattern_name.to_string()).or_insert((0, 0));
                entry.0 += op_us;
                entry.1 += 1;
                gpu_timer.record(pattern_name.to_string())?;

                continue;
```

**Critical:** Do NOT recompute the label here. Do NOT do `format!("Fused_{}", ...)`. Do NOT add a second match on `&pattern_info.pattern`. The label MUST come from the same `pattern_name` variable that was already used for `op_times`. This is the key-parity invariant.

#### Step 4: Add `record()` call for the regular-op dispatch site

Find the regular-op dispatch block around line 4280-4294. It ends with:

```rust
            // Track per-op-type timing
            let entry = op_times.entry(node.op_type.clone()).or_insert((0, 0));
            entry.0 += op_us;
            entry.1 += 1;
```

Immediately after `entry.1 += 1;`, insert:

```rust
            gpu_timer.record(node.op_type.clone())?;
```

The resulting block should look like:

```rust
            // Track per-op-type timing
            let entry = op_times.entry(node.op_type.clone()).or_insert((0, 0));
            entry.0 += op_us;
            entry.1 += 1;
            gpu_timer.record(node.op_type.clone())?;
```

Again: do NOT recompute the label. Use `node.op_type.clone()`, identical to the op_times write above it.

#### Step 5: Finalize the timer after the loop

Find the loop's closing brace and the `profile.op_times = op_times;` assignment (around line 4317-4319):

```rust
        // Suppress unused warnings (no memory management in profiling mode)
        let _ = (&tensor_groups, &group_remaining);
        profile.node_execution_us = exec_start.elapsed().as_micros() as u64;
        profile.op_times = op_times;
```

Immediately after `profile.op_times = op_times;`, insert:

```rust
        profile.gpu_op_times = gpu_timer.finalize()?;
```

The resulting block should look like:

```rust
        // Suppress unused warnings (no memory management in profiling mode)
        let _ = (&tensor_groups, &group_remaining);
        profile.node_execution_us = exec_start.elapsed().as_micros() as u64;
        profile.op_times = op_times;
        profile.gpu_op_times = gpu_timer.finalize()?;
```

`finalize()` consumes the timer. It calls `stream.synchronize()` internally, then walks all event pairs and returns the aggregated map. Any CUDA error propagates via `?`.

#### Step 6: Build

```bash
cargo build --features cuda
```

Expected: clean. Likely first-attempt errors:
- `self.ctx.context()` not found — check the exact signature in `src/cuda/context.rs:40` (it's `pub fn context(&self) -> &Arc<CudaContext>`, so you need `.clone()` to turn the `&Arc<..>` into an owned `Arc<..>`).
- `self.ctx.stream()` mismatch — it returns `&Arc<CudaStream>` (line 45 of context.rs), matching `GpuEventTimer::new`'s parameter type.
- Unused variable warnings on `gpu_timer` — this means the `record()` calls are missing. Search for `op_times.entry` in `run_with_profiling`; there are exactly two sites that need paired `gpu_timer.record()` calls.

#### Step 7: Run the existing `fusion_detection_test`

```bash
cargo test --features cuda --test fusion_detection_test
```

Expected: 5 tests pass (the tests from Cycle 1). The `executor_runs_mul_sin_pow_mul_add_and_records_profile_entry` test exercises `run_with_profiling` end-to-end with the new timer in the loop — if it fails, the integration is broken.

#### Step 8: Run full test suite

```bash
cargo test --features cuda
```

Expected: all tests pass, including the 4 new unit tests from Task 1+2 and the 5 existing fusion detection tests.

#### Step 9: Clippy

```bash
cargo clippy --features cuda -- -D warnings
```

Expected: clean. If the Task 3 commit added `#[allow(dead_code)]` on the `gpu_op_times` field, remove it now since the field is written.

#### Step 10: Commit (signed)

```bash
git branch --show-current
git add src/cuda/inference/mod.rs
git commit -S -m "$(cat <<'EOF'
feat(profiling): wire GpuEventTimer into run_with_profiling

Constructs a GpuEventTimer before the execution loop, calls
timer.record(label) immediately after each kernel dispatch at
both the fused-pattern site and the regular-op site, and calls
timer.finalize() after the loop to populate profile.gpu_op_times.

Key-parity invariant enforced structurally: each record() call
uses the exact same label expression already used for op_times
(pattern_name.to_string() for fused patterns, node.op_type.clone()
for regular ops) -- no string reformatting, no second match on
FusedPattern. A code reviewer can scan the diff and confirm each
op_times.entry / gpu_timer.record pair visually.

The timer is allocated fresh per run (no pooling) and consumed
by finalize(). Event creation uses CU_EVENT_DEFAULT explicitly
so elapsed_ms returns valid measurements.

Existing executor_runs_mul_sin_pow_mul_add_and_records_profile_entry
test continues to pass, confirming the integration does not
disturb the existing CPU-side op_times path.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase C: Kokoro end-to-end validation

### Task 5: Extend the Kokoro GPU fusion test with key-parity assertions and overhead printout

**Files:**
- Modify: `tests/kokoro_gpu_fusion_test.rs`

This task extends the existing `fusion_fires_on_full_kokoro_at_seq_len_5` test (from Cycle 1) to also verify `gpu_op_times`. The test is `#[ignore]` because it requires the real Kokoro model file.

#### Step 1: Add the assertions and diagnostic printout

Open `tests/kokoro_gpu_fusion_test.rs`. Find the test function `fusion_fires_on_full_kokoro_at_seq_len_5`. It currently ends with an `eprintln!` that prints the assertion-passed message:

```rust
    eprintln!(
        "✅ Fused_MulSinPowMulAdd fired {} times (expected ~48)",
        fused_calls
    );
}
```

Immediately BEFORE that final closing `}`, insert the new block:

```rust

    // === GPU-side timing assertions (Cycle 1.5a) ===

    // The new gpu_op_times map must be populated whenever
    // run_with_profiling is called.
    assert!(
        !profile.gpu_op_times.is_empty(),
        "gpu_op_times must be populated when run_with_profiling is called. \
         Got empty map; check that GpuEventTimer is wired into the execution loop."
    );

    // Key parity invariant: the two maps must have the exact same key set.
    // Divergence breaks Leadline's CPU/GPU join silently.
    let cpu_keys: std::collections::HashSet<&String> =
        profile.op_times.keys().collect();
    let gpu_keys: std::collections::HashSet<&String> =
        profile.gpu_op_times.keys().collect();
    assert_eq!(
        cpu_keys, gpu_keys,
        "op_times and gpu_op_times keys must match exactly. \
         cpu_only: {:?}, gpu_only: {:?}",
        cpu_keys.difference(&gpu_keys).collect::<Vec<_>>(),
        gpu_keys.difference(&cpu_keys).collect::<Vec<_>>()
    );

    // Count parity: every key's call count must match between the two maps.
    // Physical sanity: GPU time must be <= CPU time + small slack (launch
    // overhead cannot be negative; slack allows for clock-granularity noise
    // on very short ops, 1 us per call).
    for (key, (cpu_us, cpu_count)) in &profile.op_times {
        let (gpu_us, gpu_count) = profile
            .gpu_op_times
            .get(key)
            .copied()
            .expect("key in op_times must also be in gpu_op_times (checked above)");
        assert_eq!(
            *cpu_count, gpu_count,
            "call count for '{}' diverges: op_times={}, gpu_op_times={}",
            key, cpu_count, gpu_count
        );
        let slack = (*cpu_count as u64).max(1);
        assert!(
            gpu_us <= *cpu_us + slack,
            "GPU time ({} us) exceeds CPU time ({} us) for '{}' -- \
             physically impossible (launch overhead cannot be negative)",
            gpu_us, cpu_us, key
        );
    }

    // Diagnostic printout: top 10 ops by launch overhead (CPU - GPU).
    // This is the core Leadline diagnostic -- it tells us which ops spend
    // more time in launch/driver overhead than in actual GPU compute,
    // informing whether the next optimization should target fusion
    // (reduces launches), graph capture (batches launches), or kernel
    // micro-optimization (reduces compute).
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
        eprintln!(
            "  {:35} {:>10} {:>10} {:>10} {:>6}",
            k, cpu, gpu, oh, count
        );
    }

    eprintln!(
        "\n✅ gpu_op_times populated with {} keys, key parity verified",
        profile.gpu_op_times.len()
    );
```

The final existing `eprintln!` about `Fused_MulSinPowMulAdd fired N times` stays where it is, right after this new block.

#### Step 2: Run the new assertions locally

The test is `#[ignore]`, so it won't run in the default `cargo test` pass. Run it explicitly with the Kokoro model symlinked in the CWD:

```bash
ls -la kokoro-v1.0.onnx
# If not present, symlink it:
# ln -sf /home/ryano/workspace/ryanoneill/claudio/kokoro-v1.0.onnx kokoro-v1.0.onnx
```

```bash
cargo test --features cuda --test kokoro_gpu_fusion_test -- --ignored --nocapture
```

Expected: the test passes and the output includes:
1. The existing per-operator profile from Cycle 1.
2. The existing `✅ Fused_MulSinPowMulAdd fired 48 times` line.
3. The NEW "Top 10 ops by launch overhead" table showing CPU vs GPU times with an overhead column.
4. The NEW `✅ gpu_op_times populated with N keys, key parity verified` line.

If the key-parity assertion fails with "cpu_only: [...]" or "gpu_only: [...]", a dispatch site is missing a `gpu_timer.record()` call (or has an extra one). Go back to Task 4 Step 3 or 4 and double-check the paired writes.

If the "GPU time exceeds CPU time" assertion fires, the CPU `Instant::now()` measurement is somehow shorter than the GPU elapsed_ms. This should be impossible in practice; if it happens, check that `op_us` is computed from `op_start.elapsed()` AFTER the kernel dispatch (not before) and that the slack tolerance is reasonable.

#### Step 3: Capture the top-10 output

Save the "Top 10 ops by launch overhead" table to include in the final commit message. Pipe the test output to a file:

```bash
cargo test --features cuda --test kokoro_gpu_fusion_test -- --ignored --nocapture 2>&1 | tee /tmp/cycle1.5a-kokoro-output.txt
```

Then extract the top-10 block:

```bash
grep -A 12 "Top 10 ops by launch overhead" /tmp/cycle1.5a-kokoro-output.txt
```

This output goes into the commit message of the final validation task.

#### Step 4: Run the full test suite to ensure no regressions

```bash
cargo test --features cuda
```

Expected: same test counts as the previous task, plus the unchanged default-skipped Kokoro test. No regressions.

#### Step 5: Clippy and format

```bash
cargo clippy --features cuda -- -D warnings
cargo fmt --check
```

Expected: clean.

#### Step 6: Commit (signed)

```bash
git branch --show-current
git add tests/kokoro_gpu_fusion_test.rs
git commit -S -m "$(cat <<'EOF'
test(kokoro): verify gpu_op_times key parity on real Kokoro run

Extends fusion_fires_on_full_kokoro_at_seq_len_5 (from Cycle 1)
with the Cycle 1.5a acceptance criteria:

1. gpu_op_times is non-empty after run_with_profiling
2. gpu_op_times keys exactly match op_times keys (set equality)
3. call counts match between the two maps for every key
4. GPU time <= CPU time + 1us slack per call (physical sanity)

Adds a diagnostic printout of the top 10 ops by launch overhead
(cpu_us - gpu_us) for --nocapture runs. This is the core Leadline
diagnostic that will inform the next cycle's optimization target:
high overhead relative to compute means more fusion or CUDA graph
capture; low overhead (compute-bound) means kernel tuning or a
different algorithm.

The test is #[ignore] by default because it requires the real
Kokoro v1.0 model file. Run manually with:

  cargo test --features cuda --test kokoro_gpu_fusion_test \
      -- --ignored --nocapture

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase D: Final validation and cycle completion

### Task 6: Run final gates, write results doc, prepare for merge

**Files:**
- Create: `docs/performance/2026-04-11-cycle1.5a-gpu-side-timing-results.md` (if the Kokoro results are available)

This task does not modify any source code. It runs the full validation gate set and writes a small results document so the cycle's deliverables are discoverable later.

#### Step 1: Full test suite gate

```bash
cargo test --features cuda 2>&1 | grep -E "^test result" | tail -20
```

Expected: every test binary reports `0 failed`.

#### Step 2: No-cuda gate

```bash
cargo test --no-default-features 2>&1 | tail -10
```

Expected: pre-existing `validation_test.rs` failure is still present (it imports `iconnx::cuda::GpuGraphExecutor` without a `#[cfg]` gate, unrelated to this cycle). Any NEW failure caused by Cycle 1.5a must be fixed before proceeding — `gpu_event_timer.rs` is feature-gated by `src/cuda/inference/mod.rs`'s parent `#[cfg(feature = "cuda")]` on the `pub mod cuda` declaration, so it should not introduce any no-cuda regression.

#### Step 3: Clippy gate

```bash
cargo clippy --features cuda -- -D warnings
```

Expected: zero warnings.

#### Step 4: Format gate

```bash
cargo fmt --check
```

Expected: clean.

#### Step 5: File size check

```bash
wc -l src/cuda/inference/gpu_event_timer.rs src/cuda/inference/mod.rs tests/operators/gpu_event_timer_test.rs tests/kokoro_gpu_fusion_test.rs
```

Expected:
- `gpu_event_timer.rs` under 250 lines (spec target ~200)
- `inference/mod.rs` grew by fewer than 30 lines net (spec validation gate)
- `gpu_event_timer_test.rs` under 250 lines
- `kokoro_gpu_fusion_test.rs` under 300 lines (it grew from ~148 to ~250-ish)

#### Step 6: Kokoro Leadline-friendly final run (optional, requires model)

If the Kokoro model is available in CWD:

```bash
cargo test --features cuda --test kokoro_gpu_fusion_test -- --ignored --nocapture 2>&1 | tee /tmp/cycle1.5a-kokoro-final.txt
```

Extract the two key outputs for the results doc:
1. The existing per-operator profile.
2. The new "Top 10 ops by launch overhead" table.

Both belong in the results document.

#### Step 7: Write the results document

If the Kokoro run succeeded, write `docs/performance/2026-04-11-cycle1.5a-gpu-side-timing-results.md` using this template (fill in the bracketed placeholders with actual values from `/tmp/cycle1.5a-kokoro-final.txt`):

```markdown
# Cycle 1.5a Results: GPU-Side Kernel Timing

**Date:** 2026-04-11
**Spec:** [docs/superpowers/specs/2026-04-11-gpu-side-kernel-timing-design.md](../superpowers/specs/2026-04-11-gpu-side-kernel-timing-design.md)
**Plan:** [docs/superpowers/plans/2026-04-11-gpu-side-kernel-timing.md](../superpowers/plans/2026-04-11-gpu-side-kernel-timing.md)

Adds per-op GPU-measured timing to `ProfileData::gpu_op_times` via a new
`GpuEventTimer` helper. Keys exactly match `op_times`, enabling Leadline
to join the two maps and compute per-op launch overhead.

## Top 10 ops by launch overhead (Kokoro seq_len=5)

```
<paste the top-10 block from /tmp/cycle1.5a-kokoro-final.txt here>
```

## Interpretation

<One paragraph analysis of what the overhead distribution reveals. Common patterns:
 - If Fused_MulSinPowMulAdd has near-zero overhead (gpu_us ~= cpu_us), the
   kernel is compute-bound and further fusion won't help much — future
   cycles should target kernel micro-optimization.
 - If Mul, Add, Conv have high overhead (overhead > gpu_us), they are
   launch-bound and more fusion or CUDA graph capture would help.
 - If the total overhead across all ops is a significant fraction of
   total CPU time, we should prioritize launch-reduction work
   (fusion, graph capture) over compute-optimization work.>

## Tests added

- 4 unit tests in `tests/operators/gpu_event_timer_test.rs`
- Extended assertions + diagnostic printout in `tests/kokoro_gpu_fusion_test.rs`

## Next

The Leadline team now has the data they need to do launch-vs-compute
analysis. Cycle 2 will be chosen based on what the overhead distribution
shows: AddMulAdd detection fix, AddMulAddLeakyRelu fusion, or zero-cost
Reshape, in priority order.

Cycles 1.5b (tensor_shapes), 1.5c (pool metrics), and 1.5d (per-node
shapes in NodeExecutionLog) remain scheduled as separate follow-ups.
```

If the Kokoro run could not be done (no model file available), skip the document and leave a note in the final commit message: "Kokoro validation pending; leadline-bench will run the key-parity gate externally."

#### Step 8: Commit the results doc (if written)

```bash
git branch --show-current
git add docs/performance/2026-04-11-cycle1.5a-gpu-side-timing-results.md
git commit -S -m "$(cat <<'EOF'
docs(perf): Cycle 1.5a results -- GPU-side kernel timing landed

Captures the per-op launch overhead breakdown on the Kokoro
seq_len=5 profile, enabling Leadline to decide which optimization
strategy (more fusion vs graph capture vs kernel tuning) is the
highest-impact next step.

Key numbers from the real-model run:

<paste the top 3-5 lines of the overhead table here>

See docs/performance/2026-04-11-cycle1.5a-gpu-side-timing-results.md
for the full table and interpretation.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Final Validation Gates

Before marking Cycle 1.5a complete, confirm every gate:

1. [ ] `cargo test --features cuda` passes (all existing tests plus 4 new GpuEventTimer unit tests)
2. [ ] `cargo test` (no-cuda) — no NEW failures introduced by this cycle (the pre-existing `validation_test.rs` issue is unchanged)
3. [ ] `cargo clippy --features cuda -- -D warnings` zero warnings
4. [ ] `cargo fmt --check` clean
5. [ ] `tests/kokoro_gpu_fusion_test.rs` extended assertions pass on a real Kokoro run, top-10 overhead table printed
6. [ ] `src/cuda/inference/gpu_event_timer.rs` under 250 lines
7. [ ] `src/cuda/inference/mod.rs` grew by under 30 lines net since the start of this cycle
8. [ ] `src/cuda/inference/mod.rs` still compiles; no unrelated code disturbed
9. [ ] Every commit signed (`git log --show-signature --oneline <base>..HEAD` shows `Good signature` for each)
10. [ ] `git branch --show-current` prints the cycle's branch (not a detached HEAD)
11. [ ] The key-parity invariant is verified by the Kokoro integration test (Task 5 Step 2)
12. [ ] If Kokoro model was available, results doc written at `docs/performance/2026-04-11-cycle1.5a-gpu-side-timing-results.md`

---

## Review Checklist (self-review at cycle end)

- Did the Kokoro overhead numbers reveal a clear signal about which Cycle 2 target is best?
- Is the `GpuEventTimer` code clean enough to reuse in a future pool-metrics cycle (1.5c) if that cycle wants to piggyback on the event lifecycle?
- Did any test inadvertently rely on non-zero GPU elapsed time (would fail on a fast-enough GPU)? If so, the zero-elapsed insert rule is still valid but the test needs relaxing.
- Are there any lingering `#[allow(dead_code)]` or unused imports introduced during Task 3? Clean them up.
- Did `inference/mod.rs` grow more than 30 net lines? If so, something was added beyond the plan's scope — investigate before shipping.
