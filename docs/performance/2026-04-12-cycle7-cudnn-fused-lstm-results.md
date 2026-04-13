# Cycle 7 Results: cuDNN Fused LSTM

**Date:** 2026-04-12
**Spec:** [docs/superpowers/specs/2026-04-12-cudnn-fused-lstm-design.md](../superpowers/specs/2026-04-12-cudnn-fused-lstm-design.md)
**Branch:** `cycle7/cudnn-fused-lstm` (7 signed commits)

---

## Headline

Replaced iconnx's custom multi-kernel LSTM (2 cuBLAS GEMMs + 1 fused CUDA kernel per timestep per direction) with cuDNN's fused `cudnnRNNForward`. All 6 Kokoro LSTM nodes now execute in a single cuDNN call each, down from ~720 kernel launches total.

**LSTM timing (debug-mode profiling, Kokoro seq_len=5):**
- Before: 83,150 us (6 calls) = ~13.9ms per call
- After: 20,220 us (6 calls) = ~3.4ms per call
- **~75% reduction in LSTM time** (debug mode; release-mode improvement will be measured by leadline-bench)

Existing fusions unchanged: `Fused_AddMulAdd: 70`, `Fused_DivRsqrt: 65`, `Fused_Gelu: 12`, `Fused_MulSinPowMulAdd: 48`.

## What shipped

7 signed commits:

1. `65f18a9` — FFI bindings for cuDNN RNN API (7 enums, 3 handles, 15 functions)
2. `09f6b91` — RAII descriptor wrappers (DropoutDescriptor, RnnDescriptor, RnnDataDescriptor) in new `rnn.rs`
3. `315e5a9` — Weight repacking ONNX IOFC → cuDNN IFCO with gate reordering
4. `43d1fc6` — `cudnn_lstm_forward` function using cudnnRNNForward
5. `a706fcc` — GPU integration tests (bidirectional correctness within 6.59e-6, unidirectional sanity)
6. `227028f` — Executor wiring with LstmWeightCache

## Key technical decisions

- **Legacy cuDNN RNN API** (not Graph API) — proven, ORT uses it, simpler FFI surface
- **Lazy weight repacking with pointer-keyed cache** — repacks ONNX weights on first call per unique W/R tensor pair, cached for subsequent inferences
- **IOFC → IFCO gate reordering** via `ONNX_TO_CUDNN_GATE: [i32; 4] = [0, 3, 1, 2]`
- **cuDNN 9 quirk:** `projSize` must equal `hiddenSize` (not 0) when projection is disabled. Discovered during integration testing.
- **Numerical tolerance:** max observed difference between gpu_lstm and cudnn_lstm_forward is 6.59e-6 (well within the 1e-4 bar)
- **batch>1 guard:** cuDNN output `[seq, batch, dirs*hidden]` needs a transpose for batch>1 to match ONNX layout `[seq, dirs, batch, hidden]`. Kokoro is batch=1 throughout, so reshape suffices. batch>1 returns an explicit error.

## Kokoro LSTM profile

All 6 LSTM calls are bidirectional, hidden=256:

| Calls | seq_len | input_size | Before (us) | After (us) |
|---|---|---|---|---|
| 4 | 5 | 640 | ~6,900/call | ~2,800/call |
| 1 | 95 | 640 | ~34,000 | ~7,000 |
| 1 | 5 | 512 | ~6,900 | ~2,800 |

The seq_len=95 call saw the biggest improvement (~5x) because it had the most per-timestep kernel launches to eliminate.

## Follow-ups

1. **Leadline-bench re-run** — release-mode numbers needed for the progression table
2. **batch>1 transpose** — if iconnx ever processes batched inputs, the cuDNN output needs a GPU transpose kernel. Not needed for Kokoro.
3. **cudnnRNNForward with training mode** — if iconnx ever supports fine-tuning
