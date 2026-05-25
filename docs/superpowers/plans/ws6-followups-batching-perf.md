# WS-6 Follow-ups: Ragged-Batch OCR + Depthwise-Conv Perf Profiling

**Status:** DEFERRED — correctness-first policy; pdfocr parallelises across
pages at process level in the meantime.

**Tracking requirement:** CLAUDE.md requires a tracking document for deferred
features. This document IS that tracking doc; it is not a collection of
scattered TODOs.

---

## 1. Measured Latency / GPU-memory Figures

**PENDING — requires pdfocr's OCR model fixtures (see WS-6 M6.5).**

To populate this table, place the rapidAI model exports as symlinks in `$PWD`
and run:

```sh
cargo run --release --example ocr_bench --features cuda
```

The benchmark will print wall-clock avg/min/max (ms) and GPU weight-memory
(MiB) for each configuration. Copy the numbers into the table below.

| Config | avg (ms) | min (ms) | max (ms) | GPU weight mem (MiB) |
|--------|----------|----------|----------|----------------------|
| rec `[1, 3, 32, 320]` | PENDING | PENDING | PENDING | PENDING |
| det `[1, 3, 960, 960]` | PENDING | PENDING | PENDING | PENDING |
| det `[1, 3, 1500, 2000]` | PENDING | PENDING | PENDING | PENDING |

*Measurement methodology:* 3 warmup runs, 10 timed runs, `std::time::Instant`
wall-clock (CPU-side; includes GPU synchronization because `run()` returns host
tensors). Platform: see `cargo run --release --example ocr_bench` output header.

---

## 2. Ragged-Batch Plan (Recognition, N > 1)

### Current state

`dispatch_lstm` in `src/cuda/executor/lstm.rs` hardcodes `batch_size=1`:

```rust
// src/cuda/executor/lstm.rs, line ~90
// batch_size is always 1 for Kokoro's LSTMs.
let rnn_plan = prepare_lstm_plan(&self.ctx, packed, 1, seq_length)?;
```

`lstm_forward` in `src/cuda/cudnn/rnn.rs` explicitly rejects `batch_size > 1`:

```rust
if batch_size == 1 {
    // ...
} else {
    return Err(CudaError::Cudnn(format!(
        "cuDNN LSTM with batch_size={} > 1 not yet supported (needs transpose kernel)",
        batch_size
    )));
}
```

`examples/ocr_bench.rs` (`confirm_lstm_batch1_requirement`) documents this
empirically by running rec at N=2 and capturing the error.

### Why it is deferred

The PP-OCRv3 recognition model (`en_number_mobile_rec.onnx`) processes one
character crop at a time. pdfocr's current integration runs the recogniser
once per crop and parallelises across pages using OS-level process
concurrency. That means the batch=1 limit is not a throughput bottleneck for
the current integration. Lifting it requires non-trivial CUDA kernel work
(see below) and is deferred until there is a measured latency gap.

### Width-padding strategy

OCR recognition crops have variable width: a single crop can be 32×W₁ while
the next is 32×W₂, with W₁ ≠ W₂. Batching them into `[N, 3, 32, W_max]`
requires:

1. **Pad-to-max-width**: right-pad each crop to `W_max = max(W_i)` with zeros
   (or mean-pixel value) before stacking. The model's CTC output length `T_i`
   for crop `i` is proportional to `W_i`, not `W_max`, so the model still sees
   padded activations — this is acceptable for CTC because the blank token
   dominates padding regions.
2. **Per-sequence CTC length tracking**: after the batched LSTM forward pass,
   the CTC decoder must know the valid length `T_i` for each sequence to avoid
   decoding the padding tail. The simplest approach is to track `T_i =
   ceil(W_i / stride)` before padding and pass the array to CTC greedy.
3. **Positional bookkeeping**: the caller (pdfocr) must associate decoded
   sequences back to their source crop indices after batched inference.

### Batch > 1 LSTM transpose kernel

`lstm_forward` (cuDNN) produces output of shape `[seq_len, batch_size,
num_directions * hidden_size]` for `batch_size=1`. For `batch_size > 1`, cuDNN
writes a different memory layout that requires a transpose before the
subsequent `Reshape` + `MatMul` in the recognition head:

- cuDNN RNN output layout (batch_size > 1): `[seq_len, batch_size, hidden_size]`
  (not transposed — `batch_size` is the middle dimension).
- The recognition head expects `[batch_size, seq_len, hidden_size]` (batch
  first, needed for the BatchNorm + Linear layers).

The fix is to add a CUDA transpose kernel (or use cuBLAS `geam` with
appropriate strides) after `lstm_forward` for the `batch_size > 1` case.
Concretely:

1. Detect `batch_size > 1` in `lstm_forward`.
2. Allocate `out_transposed` with shape `[seq_len, batch_size, hidden]`.
3. Call a `(seq_len * batch_size * hidden)` CUDA kernel that remaps index
   `(s, b, h) -> (b, s, h)`.
4. Return `out_transposed` as the LSTM output.

The plan cache key `(W_ptr, R_ptr, seq_length)` will need `batch_size` added
once `batch_size > 1` is supported, to avoid reusing a size-1 workspace for a
size-N batch.

### Acceptance criteria (when this work starts)

- `rec [N, 3, 32, W_max]` runs end-to-end for N ∈ {2, 4, 8} without error.
- CTC-greedy decoded sequences for each batch element match the N=1 baseline
  (per-crop correctness is preserved under padding).
- `tests/ocr_parity_test.rs::rec_whole_model_matches_ort` passes for N=2 with
  a multi-crop fixture delivered by pdfocr.
- Clippy `--all-targets --features cuda,debug-inference` clean.
- No regression on the existing `--lib` test suite.

---

## 3. Depthwise-Conv Perf Profile

### Current routing

PP-OCRv3 detection (`en_PP-OCRv3_det.onnx`) uses depthwise-separable
convolutions (`group == in_channels`). As of WS-6 M6.3 Task 8, grouped Conv2D
(`group > 1`) routes through cuDNN's `garboard_conv_2d` (a single grouped-conv
call), not through the in-tree im2col `gpu_conv2d` path:

```
// src/cuda/conv/params.rs, lines 12-14:
// group == in_channels is depthwise. The im2col `gpu_conv2d` path handles
// only `group == 1`; `group > 1` routes through `garboard_conv_2d` (cuDNN).
```

This design was chosen explicitly to avoid the per-group-GEMM dispatch blowup
that a naive loop-over-groups approach would incur. A pre-WS-6 spike confirmed
that per-group GEMMs produce unacceptable overhead for depthwise layers with
many groups (e.g. 32 or 64 channels = 32 or 64 separate cuBLAS calls). The
cuDNN path handles all groups in a single descriptor call.

### What to measure once models are available

Once pdfocr delivers the OCR model fixtures (WS-6 M6.5), the per-Conv-node
GPU-side overhead can be measured using `run_with_profiling`:

```rust
let (_, profile) = exec.run_with_profiling(inputs, output_names)?;
profile.print_summary();
```

The key signal is `gpu_op_times["Conv"]` vs `op_times["Conv"]` — the gap
between GPU kernel time and CPU enqueue time indicates async overhead. For
grouped/depthwise layers specifically:

- A **large CPU-side overhead** relative to GPU kernel time suggests the cuDNN
  descriptor setup cost is non-trivial (solvable by plan caching — already
  implemented via `ConvPlanCache`).
- A **large GPU kernel time** relative to total latency for depthwise layers
  specifically (vs standard Conv layers) would indicate cuDNN is not choosing
  the optimal algorithm for the small kernel / large group count combination.
  cuDNN algorithm selection (`CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM`
  vs `ALGO_WINOGRAD`) can be tuned via the `cudnnGetConvolutionForwardAlgorithm`
  workspace-bounded search path in `garboard`.

### Action items

1. **After pdfocr delivers OCR model fixtures**: run
   `cargo run --release --example ocr_bench --features cuda` and capture the
   `pool_allocation_analysis` output alongside the wall-clock figures. Look for
   allocation sizes that correspond to the depthwise-layer activation maps.

2. **Per-op profiling**: add a `--profile` flag to `ocr_bench` (or a separate
   `ocr_profile` example) that calls `run_with_profiling` and prints the full
   `ProfileData::print_summary()` output, including `gpu_op_times["Conv"]`
   broken down by layer index if possible.

3. **Algorithm selection experiment**: if cuDNN depthwise perf is the bottleneck,
   try `cudnnFindConvolutionForwardAlgorithm` with a time budget of 100 ms and
   cache the result in `ConvPlanCache` alongside the existing descriptor.

4. **Update the latency table** in §1 above with measured numbers.

---

## Related workstream items

- WS-6 M6.5 [`ocr_parity_test.rs`]: whole-model parity gate — **BLOCKED** on
  pdfocr delivering `en_PP-OCRv3_det.onnx` + `en_number_mobile_rec.onnx`
  symlinks and the `ocr_det_input.npy` / `ocr_rec_input.npy` fixtures.
- WS-6 M6.6 [`examples/ocr_bench.rs`]: compile-gated bench (this workstream's
  R5 deliverable). Populates §1 of this doc once models are present.
- DistilBERT-INT8 seq_len divergence — separate tracking doc
  (`project_distilbert_int8_seqlen_divergence.md`), not related to OCR.
- Kokoro audio shape mismatch — separate tracking doc
  (`project_kokoro_audio_shape_mismatch.md`), not related to OCR.
