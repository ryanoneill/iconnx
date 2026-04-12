# Cycle 6: cuDNN Conv Algorithm Selection — Design Spec

**Date:** 2026-04-12
**Predecessor:** Cycle 5 (per-channel affine kernel)
**Branch:** `cycle6/cudnn-conv-algo-selection`

---

## Goal

Replace the hardcoded `CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM` in `cudnn_conv_1d` with cuDNN's algorithm finder, which selects the fastest algorithm per conv shape at graph-build time. All 88 Kokoro Conv calls are Conv1D and already dispatch through cuDNN — this change only affects which algorithm cuDNN uses internally.

## Current state

`src/cuda/cudnn/mod.rs:638`:
```rust
let algo = cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
```

This is a single algorithm that works for all shapes but isn't the fastest for any specific shape. cuDNN offers ~8 forward algorithms (IMPLICIT_GEMM, IMPLICIT_PRECOMP_GEMM, GEMM, DIRECT, FFT, FFT_TILING, WINOGRAD, WINOGRAD_NONFUSED). The best choice depends on tensor dimensions:
- Large spatial dims (`[1, 128, 11401]`): FFT or FFT_TILING often wins
- Small spatial dims (`[1, 512, 5]`): IMPLICIT_PRECOMP_GEMM or WINOGRAD often wins
- Depthwise/grouped: IMPLICIT_GEMM sometimes wins

## Changes

### 1. Bind `cudnnGetConvolutionForwardAlgorithm_v7` in sys.rs

Add the FFI binding for the cuDNN 7+ algorithm query function. This function returns up to `requestedAlgoCount` algorithms ranked by estimated performance, without running them (heuristic-based, fast).

```rust
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

Also bind the `cudnnConvolutionFwdAlgoPerf_t` struct:
```rust
#[repr(C)]
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

And any missing enum types (`cudnnDeterminism_t`, `cudnnMathType_t`) if not already bound.

### 2. Add algorithm cache to CudnnHandle (or as a sibling struct)

```rust
/// Cache for cuDNN algorithm selections, keyed by conv shape signature.
/// Populated lazily on first call per unique shape. Thread-safe via RefCell
/// (single-threaded executor).
struct ConvAlgoCache {
    forward: HashMap<ConvAlgoKey, cudnnConvolutionFwdAlgo_t>,
}

#[derive(Hash, Eq, PartialEq)]
struct ConvAlgoKey {
    input_shape: Vec<usize>,
    kernel_shape: Vec<usize>,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    dtype: &'static str, // "f32", "f16" — algorithm selection depends on precision
}
```

Store the cache as `RefCell<ConvAlgoCache>` — either on `CudnnHandle`, as a field on the executor alongside `cudnn_handle`, or passed through `cudnn_conv_1d` as a parameter.

### 3. Modify `cudnn_conv_1d` to use the cache

Replace lines 637-649 in `cudnn_conv_1d`:

```rust
// Old:
let algo = cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

// New:
let algo = find_best_conv_fwd_algo(
    handle,
    &input_desc,
    &filter_desc,
    &conv_desc,
    &output_desc,
    cache,  // &mut ConvAlgoCache or &RefCell<ConvAlgoCache>
    &ConvAlgoKey { input_shape, kernel_shape, stride, padding, dilation, groups },
)?;
```

The `find_best_conv_fwd_algo` helper:
1. Checks the cache for the key
2. If hit: returns cached algorithm
3. If miss: calls `cudnnGetConvolutionForwardAlgorithm_v7` with `requestedAlgoCount=1`, caches the result, returns it

### 4. Also apply to ConvTranspose (cudnn_conv_transpose_1d)

The same pattern applies to the backward-data algorithm in `cudnn_conv_transpose_1d` (line 501), which is hardcoded to `CUDNN_CONVOLUTION_BWD_DATA_ALGO_1`. Add `cudnnGetConvolutionBackwardDataAlgorithm_v7` binding and cache. Only 6 ConvTranspose calls on Kokoro, but the change is symmetric and low-risk.

## Scope

**In scope:**
- FFI bindings for `cudnnGetConvolutionForwardAlgorithm_v7` and `cudnnGetConvolutionBackwardDataAlgorithm_v7`
- `ConvAlgoCache` with key-by-shape lookup
- Modified `cudnn_conv_1d` and `cudnn_conv_transpose_1d` to use cached algorithm selection
- Kokoro integration test verification

**Out of scope:**
- `cudnnFindConvolutionForwardAlgorithm` (the benchmarking variant that actually runs each algorithm — slower but more accurate; can be added as a future opt-in)
- Conv2D cuDNN implementation (zero Conv2D calls on Kokoro)
- Autotuning persistence across sessions (cache is per-executor, reset on construction)
- Changes to im2col fallback path (Conv2D still uses it; no Kokoro impact)

## Testing

1. **Existing Conv1D GPU tests must still pass** — the algorithm change should produce identical numerical results (cuDNN algorithms are functionally equivalent, differing only in performance)
2. **Kokoro integration test** — Conv timing should decrease; verify via `op_times["Conv"]` in the profiling output
3. **No new unit tests needed** for the cache itself — it's a trivial HashMap lookup. The correctness gate is the existing conv tests passing with the new algorithm.

## Success criteria

1. Kokoro `Conv` GPU time drops measurably (target: 20-50% reduction from ~37.5ms baseline)
2. All existing tests pass with identical numerical results
3. Clippy clean
4. `Fused_AddMulAdd: 70` unchanged (no regression on Cycle 5's fusions)

## Risk

The main risk is that `cudnnGetConvolutionForwardAlgorithm_v7` returns an algorithm that requires more workspace memory than available. Mitigation: the workspace query (`cudnnGetConvolutionForwardWorkspaceSize`) already takes the algorithm as input and returns the required size; `workspace.ensure_size()` handles reallocation. No change needed.

A secondary risk is that the heuristic-selected algorithm is slower than `IMPLICIT_PRECOMP_GEMM` for some shapes (the heuristic isn't perfect). If Kokoro's Conv time increases, we can fall back to the benchmarking variant (`cudnnFindConvolutionForwardAlgorithm`) in a follow-up, or add a per-shape override mechanism. This is unlikely — ORT uses the same heuristic approach at Level 3 optimization.
