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
//! ```text
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
