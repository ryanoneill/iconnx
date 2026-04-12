# Cycle 6: cuDNN Conv Algorithm Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace hardcoded cuDNN convolution algorithms with per-shape heuristic selection via `cudnnGetConvolutionForwardAlgorithm_v7` and `cudnnGetConvolutionBackwardDataAlgorithm_v7`, cached across calls. This targets the 88 Conv1D calls (37.5ms GPU time, 32% of total Kokoro inference) and 6 ConvTranspose1D calls that all dispatch through cuDNN today.

**Architecture:** A `ConvAlgoCache` struct holds two `HashMap`s keyed by `ConvAlgoKey` (input shape, kernel shape, stride, padding, dilation, groups, dtype). On the first call for a given shape, `find_best_fwd_algo` / `find_best_bwd_data_algo` calls cuDNN's heuristic algorithm finder, caches the result, and returns it. Subsequent calls for the same shape are a HashMap lookup. The cache lives as a `RefCell<ConvAlgoCache>` field on `GpuGraphExecutor`, passed through to the conv functions as a `&mut ConvAlgoCache` parameter.

**Tech Stack:** Rust 2021, cuDNN 9.x FFI bindings, existing iconnx infrastructure. No new crate dependencies.

**Prerequisites for the implementing agent:**
- Read the spec first: `docs/superpowers/specs/2026-04-12-cudnn-conv-algorithm-selection-design.md`.
- Per `/home/ryano/.claude/CLAUDE.md`: TDD where possible, commits after every substantive change, **all commits must be signed** (`git commit -S`), stop and ask if signing fails, warnings never ignored.
- `cargo test --features cuda` is the regression gate. Run after each task.
- `cargo clippy --features cuda --tests -- -D warnings` must be clean before every commit.
- **NEVER run `git checkout <sha>`** to inspect commits. Use `git show SHA`, `git show SHA --stat`, or `git log --show-signature -1 SHA`.
- **NEVER use unsafe Rust** without asking first. The FFI calls in sys.rs and the `unsafe` blocks in mod.rs follow existing patterns exactly -- replicate those patterns.
- **No files over 1000 lines.** `src/cuda/inference/mod.rs` is already 4963 lines (flag but don't refactor).

---

## Scope Check

This plan implements **Cycle 6 only** -- cuDNN Conv Algorithm Selection. Out of scope:

- `cudnnFindConvolutionForwardAlgorithm` (the benchmarking variant that runs each algorithm -- slower but more accurate; future opt-in)
- Conv2D cuDNN implementation (zero Conv2D calls on Kokoro; still uses im2col)
- Autotuning persistence across sessions (cache is per-executor, reset on construction)
- Changes to the im2col fallback path
- Any changes to fusion patterns (DivRsqrt, Gelu, MulSinPowMulAdd, AddMulAdd, etc.)
- Refactoring `mod.rs` file size

Any work outside the files listed below is a scope violation and should be flagged to the controller before landing.

---

## File Structure

### Modified files

| File | Responsibility | Est. delta |
|---|---|---|
| `src/cuda/cudnn/sys.rs` | Add `cudnnDeterminism_t` enum, `cudnnConvolutionFwdAlgoPerf_t` struct, `cudnnConvolutionBwdDataAlgoPerf_t` struct, `cudnnGetConvolutionForwardAlgorithm_v7` FFI binding, `cudnnGetConvolutionBackwardDataAlgorithm_v7` FFI binding | +55 lines |
| `src/cuda/cudnn/mod.rs` | Add `ConvAlgoKey`, `ConvAlgoCache`, `find_best_fwd_algo`, `find_best_bwd_data_algo`; modify `cudnn_conv_1d` and `cudnn_conv_transpose_1d` signatures and algo selection | +110 lines |
| `src/cuda/inference/mod.rs` | Add `conv_algo_cache: RefCell<ConvAlgoCache>` field, wire through `new_with_profiling` and both Conv/ConvTranspose dispatch sites | +12 lines |

### Files NOT touched

- `src/cuda/cudnn/sys.rs` existing enums and functions -- only additions, no modifications.
- `src/cuda/context.rs` -- no new error variants needed.
- `src/cuda/inference/fusion/` -- no fusion changes.
- `src/cuda/kernels/` -- no kernel changes.
- `tests/kokoro_gpu_fusion_test.rs` -- fusion counts unchanged.

---

## Task 1: Add FFI Bindings for Algorithm Query Functions

**Files:**
- Modify: `src/cuda/cudnn/sys.rs` (currently 465 lines)

This task adds the missing cuDNN types and function bindings needed for algorithm selection. All additions go into `sys.rs` following the existing pattern of `#[repr(C)]` enums/structs and `extern "C"` function declarations.

- [ ] **Step 1: Add `cudnnDeterminism_t` enum**

After the `cudnnMathType_t` enum (line 115), add:

```rust
/// Determinism mode for cuDNN algorithms
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cudnnDeterminism_t {
    CUDNN_NON_DETERMINISTIC = 0,
    CUDNN_DETERMINISTIC = 1,
}
```

This enum is referenced by the performance result structs below.

- [ ] **Step 2: Add `cudnnConvolutionFwdAlgoPerf_t` struct**

After the `cudnnDeterminism_t` enum, add:

```rust
/// Performance results for convolution forward algorithm selection
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct cudnnConvolutionFwdAlgoPerf_t {
    pub algo: cudnnConvolutionFwdAlgo_t,
    pub status: cudnnStatus_t,
    pub time: f32,
    pub memory: usize,
    pub determinism: cudnnDeterminism_t,
    pub mathType: cudnnMathType_t,
    pub reserved: [c_int; 3],
}
```

The field layout matches the cuDNN C header exactly. `memory` is `size_t` in C, which maps to `usize` in Rust. The `reserved` field is 3 `int`s.

- [ ] **Step 3: Add `cudnnConvolutionBwdDataAlgoPerf_t` struct**

Immediately after the forward struct, add the backward-data equivalent:

```rust
/// Performance results for convolution backward data algorithm selection
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct cudnnConvolutionBwdDataAlgoPerf_t {
    pub algo: cudnnConvolutionBwdDataAlgo_t,
    pub status: cudnnStatus_t,
    pub time: f32,
    pub memory: usize,
    pub determinism: cudnnDeterminism_t,
    pub mathType: cudnnMathType_t,
    pub reserved: [c_int; 3],
}
```

- [ ] **Step 4: Add `cudnnGetConvolutionForwardAlgorithm_v7` to the extern block**

Inside the `#[link(name = "cudnn")] extern "C"` block, after the `cudnnGetConvolutionForwardWorkspaceSize` declaration (line 337) and before `cudnnConvolutionForward` (line 340), add a new section:

```rust
    // -------------------------------------------------------------------------
    // Convolution Algorithm Selection
    // -------------------------------------------------------------------------

    /// Get best convolution forward algorithms (heuristic-based, no benchmarking)
    ///
    /// Returns up to `requestedAlgoCount` algorithms ranked by estimated
    /// performance. This is the v7 API that uses heuristics (fast) rather
    /// than actually benchmarking each algorithm (slow).
    pub fn cudnnGetConvolutionForwardAlgorithm_v7(
        handle: cudnnHandle_t,
        xDesc: cudnnTensorDescriptor_t,
        wDesc: cudnnFilterDescriptor_t,
        convDesc: cudnnConvolutionDescriptor_t,
        yDesc: cudnnTensorDescriptor_t,
        requestedAlgoCount: c_int,
        returnedAlgoCount: *mut c_int,
        perfResults: *mut cudnnConvolutionFwdAlgoPerf_t,
    ) -> cudnnStatus_t;
```

- [ ] **Step 5: Add `cudnnGetConvolutionBackwardDataAlgorithm_v7` to the extern block**

In the "Convolution Backward Data" section, after `cudnnGetConvolutionBackwardDataWorkspaceSize` (line 369) and before `cudnnConvolutionBackwardData` (line 372), add:

```rust
    /// Get best backward data algorithms (heuristic-based, no benchmarking)
    pub fn cudnnGetConvolutionBackwardDataAlgorithm_v7(
        handle: cudnnHandle_t,
        wDesc: cudnnFilterDescriptor_t,
        dyDesc: cudnnTensorDescriptor_t,
        convDesc: cudnnConvolutionDescriptor_t,
        dxDesc: cudnnTensorDescriptor_t,
        requestedAlgoCount: c_int,
        returnedAlgoCount: *mut c_int,
        perfResults: *mut cudnnConvolutionBwdDataAlgoPerf_t,
    ) -> cudnnStatus_t;
```

Note the parameter order: `wDesc` comes first for backward data, matching the existing `cudnnGetConvolutionBackwardDataWorkspaceSize` signature.

- [ ] **Step 6: Run clippy to verify**

Run: `cargo clippy --features cuda --tests -- -D warnings`
Expected: 0 errors, 0 warnings. The new types and functions may generate `dead_code` warnings, but the `#![allow(dead_code)]` at the top of `sys.rs` (line 11) suppresses them.

- [ ] **Step 7: Commit**

```bash
git add src/cuda/cudnn/sys.rs
git commit -S -m "Add FFI bindings for cuDNN algorithm selection functions

Bind cudnnGetConvolutionForwardAlgorithm_v7 and
cudnnGetConvolutionBackwardDataAlgorithm_v7 along with their
performance result structs (cudnnConvolutionFwdAlgoPerf_t,
cudnnConvolutionBwdDataAlgoPerf_t) and the cudnnDeterminism_t enum.

These heuristic-based algorithm query functions allow cuDNN to select
the fastest convolution algorithm for a given tensor shape without
benchmarking overhead.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Add ConvAlgoCache and Modify `cudnn_conv_1d` to Use It

**Files:**
- Modify: `src/cuda/cudnn/mod.rs` (currently 795 lines)

This task adds the caching infrastructure and modifies the forward convolution path. The backward data path (ConvTranspose) follows in Task 3.

- [ ] **Step 1: Add imports and re-exports**

At the top of `src/cuda/cudnn/mod.rs`, add `HashMap` to the imports:

```rust
use std::collections::HashMap;
```

Update the `pub use sys` block (line 18-21) to also export the new types:

```rust
pub use sys::{
    cudnnConvolutionBwdDataAlgo_t, cudnnConvolutionFwdAlgo_t, cudnnConvolutionMode_t,
    cudnnDataType_t, cudnnStatus_t, cudnnTensorFormat_t,
};
```

This block does not need to export the perf structs or determinism enum -- those are internal to the algorithm selection helpers.

- [ ] **Step 2: Add `ConvAlgoKey` struct**

After the `Workspace` impl block (after line 405, the `Default` impl), add a new section:

```rust
// =============================================================================
// Algorithm Selection Cache
// =============================================================================

/// Key for caching cuDNN algorithm selections.
///
/// Two conv calls with the same key will select the same algorithm, so we
/// cache the result on first lookup. The key must include dtype because
/// algorithm performance varies between f32 and f16.
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct ConvAlgoKey {
    pub input_shape: Vec<usize>,
    pub kernel_shape: Vec<usize>,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub groups: usize,
    pub dtype: &'static str,
}
```

- [ ] **Step 3: Add `ConvAlgoCache` struct**

Immediately after `ConvAlgoKey`:

```rust
/// Cache for cuDNN algorithm selections, keyed by conv shape signature.
///
/// Populated lazily on first call per unique shape. Single-threaded executor
/// means no internal locking is needed -- the caller wraps this in `RefCell`.
pub struct ConvAlgoCache {
    forward: HashMap<ConvAlgoKey, sys::cudnnConvolutionFwdAlgo_t>,
    backward_data: HashMap<ConvAlgoKey, sys::cudnnConvolutionBwdDataAlgo_t>,
}

impl ConvAlgoCache {
    /// Create an empty algorithm cache.
    pub fn new() -> Self {
        Self {
            forward: HashMap::new(),
            backward_data: HashMap::new(),
        }
    }
}

impl Default for ConvAlgoCache {
    fn default() -> Self {
        Self::new()
    }
}
```

- [ ] **Step 4: Add `find_best_fwd_algo` helper**

After the `ConvAlgoCache` impl block, add:

```rust
/// Query cuDNN for the best forward convolution algorithm for the given
/// descriptors, caching the result by shape key.
///
/// On cache hit, returns immediately. On cache miss, calls
/// `cudnnGetConvolutionForwardAlgorithm_v7` with `requestedAlgoCount=8`
/// (all algorithms), picks the first one with `CUDNN_STATUS_SUCCESS`, and
/// caches it.
///
/// We request all 8 algorithms and take the first successful one because
/// the v7 API returns them sorted by estimated performance. Some algorithms
/// may return a non-success status for certain shapes (e.g., FFT for very
/// small kernels), so we skip those.
fn find_best_fwd_algo(
    handle: &CudnnHandle,
    input_desc: &TensorDescriptor,
    filter_desc: &FilterDescriptor,
    conv_desc: &ConvolutionDescriptor,
    output_desc: &TensorDescriptor,
    cache: &mut ConvAlgoCache,
    key: &ConvAlgoKey,
) -> Result<sys::cudnnConvolutionFwdAlgo_t, CudaError> {
    // Check cache first
    if let Some(&algo) = cache.forward.get(key) {
        return Ok(algo);
    }

    // Query cuDNN for algorithm rankings
    const MAX_ALGOS: usize = 8;
    let mut perf_results = [sys::cudnnConvolutionFwdAlgoPerf_t {
        algo: sys::cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
        status: sys::cudnnStatus_t::CUDNN_STATUS_NOT_INITIALIZED,
        time: 0.0,
        memory: 0,
        determinism: sys::cudnnDeterminism_t::CUDNN_NON_DETERMINISTIC,
        mathType: sys::cudnnMathType_t::CUDNN_DEFAULT_MATH,
        reserved: [0; 3],
    }; MAX_ALGOS];
    let mut returned_count: c_int = 0;

    unsafe {
        check_cudnn(sys::cudnnGetConvolutionForwardAlgorithm_v7(
            handle.raw(),
            input_desc.raw(),
            filter_desc.raw(),
            conv_desc.raw(),
            output_desc.raw(),
            MAX_ALGOS as c_int,
            &mut returned_count,
            perf_results.as_mut_ptr(),
        ))?;
    }

    // Pick the first algorithm with a successful status
    let best = perf_results[..returned_count as usize]
        .iter()
        .find(|r| r.status == sys::cudnnStatus_t::CUDNN_STATUS_SUCCESS)
        .ok_or_else(|| {
            CudaError::Cudnn(
                "cudnnGetConvolutionForwardAlgorithm_v7 returned no successful algorithms".into(),
            )
        })?;

    let algo = best.algo;
    cache.forward.insert(key.clone(), algo);

    Ok(algo)
}
```

**Design notes:**
- We request all 8 algorithms rather than just 1, because the top-ranked algorithm may have `status != SUCCESS` for certain shapes. Taking the first *successful* one is safer.
- The zero-initialized perf_results array uses the default `IMPLICIT_PRECOMP_GEMM` as a placeholder -- it's overwritten by cuDNN. The status field is initialized to `NOT_INITIALIZED` so we can distinguish cuDNN-filled entries from unfilled ones.
- The `key.clone()` for the cache insert is fine: ConvAlgoKey contains small Vecs (3 elements for 1D, 4 for 2D) and a `&'static str`. Kokoro has only 12 unique shapes.

- [ ] **Step 5: Modify `cudnn_conv_1d` to accept and use the cache**

Change the function signature at line 556 to add a `cache` parameter:

```rust
pub fn cudnn_conv_1d(
    handle: &CudnnHandle,
    ctx: &IconnxCudaContext,
    workspace: &mut Workspace,
    cache: &mut ConvAlgoCache,
    input: &GpuTensor,
    kernel: &GpuTensor,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<GpuTensor, CudaError> {
```

Then replace the hardcoded algorithm selection at line 638:

**Before (lines 637-649):**
```rust
    // Get workspace size
    let algo = cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    let mut workspace_size: usize = 0;
    unsafe {
        check_cudnn(sys::cudnnGetConvolutionForwardWorkspaceSize(
            handle.raw(),
            input_desc.raw(),
            filter_desc.raw(),
            conv_desc.raw(),
            output_desc.raw(),
            algo,
            &mut workspace_size,
        ))?;
    }
```

**After:**
```rust
    // Select best algorithm for this shape (cached after first call)
    let key = ConvAlgoKey {
        input_shape: input_shape.to_vec(),
        kernel_shape: kernel_shape.to_vec(),
        stride,
        padding,
        dilation,
        groups,
        dtype: "f32",
    };
    let algo = find_best_fwd_algo(
        handle,
        &input_desc,
        &filter_desc,
        &conv_desc,
        &output_desc,
        cache,
        &key,
    )?;

    // Get workspace size for the selected algorithm
    let mut workspace_size: usize = 0;
    unsafe {
        check_cudnn(sys::cudnnGetConvolutionForwardWorkspaceSize(
            handle.raw(),
            input_desc.raw(),
            filter_desc.raw(),
            conv_desc.raw(),
            output_desc.raw(),
            algo,
            &mut workspace_size,
        ))?;
    }
```

The rest of the function (workspace allocation, output allocation, convolution execution) remains unchanged -- it already uses `algo` and `workspace_size` as variables.

- [ ] **Step 6: Update the test at line 718 to pass the cache**

In `test_cudnn_conv_1d` (line 718), update the call:

**Before:**
```rust
        let output = cudnn_conv_1d(
            &handle,
            &ctx,
            &mut workspace,
            &input,
            &kernel,
            1, // stride
            1, // padding
            1, // dilation
            1, // groups
        )
        .expect("Conv1D failed");
```

**After:**
```rust
        let mut cache = ConvAlgoCache::new();
        let output = cudnn_conv_1d(
            &handle,
            &ctx,
            &mut workspace,
            &mut cache,
            &input,
            &kernel,
            1, // stride
            1, // padding
            1, // dilation
            1, // groups
        )
        .expect("Conv1D failed");
```

- [ ] **Step 7: Run clippy and tests**

Run: `cargo clippy --features cuda --tests -- -D warnings`
Expected: Clippy clean. There will be a compilation error in `src/cuda/inference/mod.rs` because `cudnn_conv_1d` now requires a `cache` parameter -- that's expected and will be fixed in Task 4.

Verify the compilation error is only in `src/cuda/inference/mod.rs`:
```bash
cargo check --features cuda 2>&1 | head -30
```

If you want to verify the test independently, you can temporarily add a `ConvAlgoCache::new()` local in the executor's Conv dispatch. But it's cleaner to wait for Task 4.

- [ ] **Step 8: Commit (may not compile executor -- that's OK)**

```bash
git add src/cuda/cudnn/mod.rs
git commit -S -m "Add ConvAlgoCache and use heuristic algo selection in cudnn_conv_1d

Replace the hardcoded CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
with a per-shape cache that queries cudnnGetConvolutionForwardAlgorithm_v7
on first encounter. The cache stores results in a HashMap keyed by
(input_shape, kernel_shape, stride, padding, dilation, groups, dtype).

Kokoro has 12 unique Conv1D shapes across 88 calls, so only 12 cuDNN
heuristic queries occur; all subsequent calls are HashMap lookups.

The executor wiring follows in a subsequent commit.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Add Backward Data Algorithm Selection for ConvTranspose

**Files:**
- Modify: `src/cuda/cudnn/mod.rs`

This task is symmetric to Task 2's forward path but for `cudnn_conv_transpose_1d`, which currently hardcodes `CUDNN_CONVOLUTION_BWD_DATA_ALGO_1`.

- [ ] **Step 1: Add `find_best_bwd_data_algo` helper**

After the `find_best_fwd_algo` function, add:

```rust
/// Query cuDNN for the best backward-data convolution algorithm for the
/// given descriptors, caching the result by shape key.
///
/// Same caching strategy as `find_best_fwd_algo` but for the backward-data
/// path used by ConvTranspose.
fn find_best_bwd_data_algo(
    handle: &CudnnHandle,
    filter_desc: &FilterDescriptor,
    input_desc: &TensorDescriptor,
    conv_desc: &ConvolutionDescriptor,
    output_desc: &TensorDescriptor,
    cache: &mut ConvAlgoCache,
    key: &ConvAlgoKey,
) -> Result<sys::cudnnConvolutionBwdDataAlgo_t, CudaError> {
    // Check cache first
    if let Some(&algo) = cache.backward_data.get(key) {
        return Ok(algo);
    }

    // Query cuDNN for algorithm rankings
    const MAX_ALGOS: usize = 6;
    let mut perf_results = [sys::cudnnConvolutionBwdDataAlgoPerf_t {
        algo: sys::cudnnConvolutionBwdDataAlgo_t::CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
        status: sys::cudnnStatus_t::CUDNN_STATUS_NOT_INITIALIZED,
        time: 0.0,
        memory: 0,
        determinism: sys::cudnnDeterminism_t::CUDNN_NON_DETERMINISTIC,
        mathType: sys::cudnnMathType_t::CUDNN_DEFAULT_MATH,
        reserved: [0; 3],
    }; MAX_ALGOS];
    let mut returned_count: c_int = 0;

    unsafe {
        check_cudnn(sys::cudnnGetConvolutionBackwardDataAlgorithm_v7(
            handle.raw(),
            filter_desc.raw(),
            input_desc.raw(),
            conv_desc.raw(),
            output_desc.raw(),
            MAX_ALGOS as c_int,
            &mut returned_count,
            perf_results.as_mut_ptr(),
        ))?;
    }

    // Pick the first algorithm with a successful status
    let best = perf_results[..returned_count as usize]
        .iter()
        .find(|r| r.status == sys::cudnnStatus_t::CUDNN_STATUS_SUCCESS)
        .ok_or_else(|| {
            CudaError::Cudnn(
                "cudnnGetConvolutionBackwardDataAlgorithm_v7 returned no successful algorithms"
                    .into(),
            )
        })?;

    let algo = best.algo;
    cache.backward_data.insert(key.clone(), algo);

    Ok(algo)
}
```

**Note:** `MAX_ALGOS` is 6 here (matching the 6 backward-data algorithm variants), vs 8 for forward.

**Note:** Parameter order for the backward-data function is `(filter_desc, input_desc, conv_desc, output_desc)` -- filter comes first, matching `cudnnGetConvolutionBackwardDataWorkspaceSize`. This is different from the forward path where input comes first.

- [ ] **Step 2: Modify `cudnn_conv_transpose_1d` to accept and use the cache**

Change the function signature at line 417 to add a `cache` parameter:

```rust
pub fn cudnn_conv_transpose_1d(
    handle: &CudnnHandle,
    ctx: &IconnxCudaContext,
    workspace: &mut Workspace,
    cache: &mut ConvAlgoCache,
    input: &GpuTensor,
    kernel: &GpuTensor,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<GpuTensor, CudaError> {
```

Then replace the hardcoded algorithm selection at line 501:

**Before (lines 500-513):**
```rust
    // Get workspace size
    let algo = cudnnConvolutionBwdDataAlgo_t::CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
    let mut workspace_size: usize = 0;
    unsafe {
        check_cudnn(sys::cudnnGetConvolutionBackwardDataWorkspaceSize(
            handle.raw(),
            filter_desc.raw(),
            input_desc.raw(),
            conv_desc.raw(),
            output_desc.raw(),
            algo,
            &mut workspace_size,
        ))?;
    }
```

**After:**
```rust
    // Select best algorithm for this shape (cached after first call)
    let key = ConvAlgoKey {
        input_shape: input_shape.to_vec(),
        kernel_shape: kernel_shape.to_vec(),
        stride,
        padding,
        dilation,
        groups,
        dtype: "f32",
    };
    let algo = find_best_bwd_data_algo(
        handle,
        &filter_desc,
        &input_desc,
        &conv_desc,
        &output_desc,
        cache,
        &key,
    )?;

    // Get workspace size for the selected algorithm
    let mut workspace_size: usize = 0;
    unsafe {
        check_cudnn(sys::cudnnGetConvolutionBackwardDataWorkspaceSize(
            handle.raw(),
            filter_desc.raw(),
            input_desc.raw(),
            conv_desc.raw(),
            output_desc.raw(),
            algo,
            &mut workspace_size,
        ))?;
    }
```

The rest of the function remains unchanged.

- [ ] **Step 3: Update the test at line 753 to pass the cache**

In `test_cudnn_conv_transpose_1d` (line 753), update the call:

**Before:**
```rust
        let output = cudnn_conv_transpose_1d(
            &handle,
            &ctx,
            &mut workspace,
            &input,
            &kernel,
            2, // stride
            1, // padding
            0, // output_padding
            1, // dilation
            1, // groups
        )
        .expect("ConvTranspose1D failed");
```

**After:**
```rust
        let mut cache = ConvAlgoCache::new();
        let output = cudnn_conv_transpose_1d(
            &handle,
            &ctx,
            &mut workspace,
            &mut cache,
            &input,
            &kernel,
            2, // stride
            1, // padding
            0, // output_padding
            1, // dilation
            1, // groups
        )
        .expect("ConvTranspose1D failed");
```

- [ ] **Step 4: Run clippy**

Run: `cargo clippy --features cuda --tests -- -D warnings`
Expected: Clippy clean. The executor still won't compile (Task 4 fixes the call sites).

- [ ] **Step 5: Commit**

```bash
git add src/cuda/cudnn/mod.rs
git commit -S -m "Add heuristic algo selection for cudnn_conv_transpose_1d

Mirror the forward algorithm cache for the backward-data path used by
ConvTranspose. The find_best_bwd_data_algo helper queries
cudnnGetConvolutionBackwardDataAlgorithm_v7 on first encounter per shape
and caches the result in ConvAlgoCache::backward_data.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Wire Cache Through Executor and Verify

**Files:**
- Modify: `src/cuda/inference/mod.rs`

This task connects the `ConvAlgoCache` to the executor and updates the Conv/ConvTranspose dispatch sites to pass it through.

- [ ] **Step 1: Add import**

At the import block near line 11, update the cuDNN import line:

**Before:**
```rust
use super::cudnn::{
    cudnn_conv_1d, cudnn_conv_transpose_1d, CudnnHandle, Workspace as CudnnWorkspace,
};
```

**After:**
```rust
use super::cudnn::{
    cudnn_conv_1d, cudnn_conv_transpose_1d, ConvAlgoCache, CudnnHandle,
    Workspace as CudnnWorkspace,
};
```

- [ ] **Step 2: Add `conv_algo_cache` field to `GpuGraphExecutor`**

In the struct definition (line 217), after the `cudnn_workspace` field (line 233), add:

```rust
    /// cuDNN algorithm selection cache (RefCell for interior mutability)
    conv_algo_cache: RefCell<ConvAlgoCache>,
```

- [ ] **Step 3: Initialize the cache in `new_with_profiling`**

In the struct initialization at line 411 (the `Ok(Self { ... })` block), after `cudnn_workspace,` (line 421), add:

```rust
            conv_algo_cache: RefCell::new(ConvAlgoCache::new()),
```

- [ ] **Step 4: Wire cache through Conv dispatch**

At the Conv1D call site (around line 1479-1490), update the call:

**Before:**
```rust
                        let mut workspace = self.cudnn_workspace.borrow_mut();
                        let mut output = cudnn_conv_1d(
                            &self.cudnn_handle,
                            &self.ctx,
                            &mut workspace,
                            inputs[0],
                            inputs[1],
                            stride,
                            padding,
                            dilation,
                            group,
                        )?;
```

**After:**
```rust
                        let mut workspace = self.cudnn_workspace.borrow_mut();
                        let mut cache = self.conv_algo_cache.borrow_mut();
                        let mut output = cudnn_conv_1d(
                            &self.cudnn_handle,
                            &self.ctx,
                            &mut workspace,
                            &mut cache,
                            inputs[0],
                            inputs[1],
                            stride,
                            padding,
                            dilation,
                            group,
                        )?;
```

- [ ] **Step 5: Wire cache through ConvTranspose dispatch**

At the ConvTranspose1D call site (around line 1564-1599), update the call:

**Before:**
```rust
                    let mut workspace = self.cudnn_workspace.borrow_mut();
                    // ... debug printing ...
                    let mut output = cudnn_conv_transpose_1d(
                        &self.cudnn_handle,
                        &self.ctx,
                        &mut workspace,
                        inputs[0],
                        inputs[1],
                        stride,
                        padding,
                        output_pad,
                        dilation,
                        group,
                    )?;
```

**After:**
```rust
                    let mut workspace = self.cudnn_workspace.borrow_mut();
                    let mut cache = self.conv_algo_cache.borrow_mut();
                    // ... debug printing ...
                    let mut output = cudnn_conv_transpose_1d(
                        &self.cudnn_handle,
                        &self.ctx,
                        &mut workspace,
                        &mut cache,
                        inputs[0],
                        inputs[1],
                        stride,
                        padding,
                        output_pad,
                        dilation,
                        group,
                    )?;
```

**Important:** The `let mut cache = self.conv_algo_cache.borrow_mut();` must be on a separate line from the workspace borrow. Both are `RefCell` borrows, but they're on different `RefCell` instances, so this is safe. Do NOT combine them into a single expression.

- [ ] **Step 6: Run clippy and full test suite**

Run:
```bash
cargo clippy --features cuda --tests -- -D warnings
cargo test --features cuda
```

Expected: All existing tests pass. The Conv1D and ConvTranspose1D tests should produce identical numerical results -- cuDNN algorithms are functionally equivalent, differing only in performance.

If any tests fail with cuDNN errors (e.g., `NOT_SUPPORTED` for a specific algorithm on a specific shape), this indicates the `find_best_*_algo` helper is selecting an algorithm that doesn't work for that shape. The fix would be to iterate through `perf_results` and skip entries where `status != SUCCESS`. This is already implemented in the helpers above.

- [ ] **Step 7: Commit**

```bash
git add src/cuda/inference/mod.rs
git commit -S -m "Wire ConvAlgoCache through executor for Conv and ConvTranspose dispatch

Add conv_algo_cache field to GpuGraphExecutor and pass it through to
cudnn_conv_1d and cudnn_conv_transpose_1d at both dispatch sites.

All 88 Kokoro Conv1D calls and 6 ConvTranspose1D calls now use
per-shape heuristic algorithm selection instead of hardcoded algorithms.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 8: Run Kokoro integration test to verify performance improvement**

Run the Kokoro integration test with profiling to check Conv timing:
```bash
cargo test --features cuda --release -- kokoro --nocapture 2>&1 | grep -E "Conv|Total|Fused_AddMulAdd"
```

**Expected results:**
1. Conv GPU time decreases measurably from ~37.5ms baseline (target: 20-50% reduction)
2. `Fused_AddMulAdd: 70` unchanged (no regression on Cycle 5's fusions)
3. All tests pass with identical numerical results

If Conv time does not improve or increases:
- Check which algorithms cuDNN selected by temporarily adding a `eprintln!("Selected algo: {:?} for key: {:?}", algo, key);` in `find_best_fwd_algo` after the cache insert
- Compare against the baseline `IMPLICIT_PRECOMP_GEMM` -- if the heuristic selected the same algorithm, there's no performance change (but also no regression)
- If the heuristic selected a worse algorithm, we can add `cudnnFindConvolutionForwardAlgorithm` (benchmarking variant) as a future follow-up

---

## Verification Checklist

After all 4 tasks are complete, verify:

- [ ] `cargo clippy --features cuda --tests -- -D warnings` -- zero warnings
- [ ] `cargo test --features cuda` -- all tests pass
- [ ] `cargo test --features cuda --release -- kokoro --nocapture` -- Conv time drops, fusion counts unchanged
- [ ] `git log --oneline -4` -- 4 signed commits (1 per task, or 3 if Task 2 and 3 share a commit)
- [ ] `sys.rs` stays under 1000 lines (was 465, adding ~55 = ~520)
- [ ] `mod.rs` stays under 1000 lines (was 795, adding ~110 = ~905)
- [ ] `inference/mod.rs` stays under 5000 lines (was 4963, adding ~12 = ~4975 -- still over 1000 but pre-existing; flag for future refactor)
