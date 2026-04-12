# Cycle 7: cuDNN Fused LSTM — Design Spec

**Date:** 2026-04-12
**Predecessor:** Cycle 6 (cuDNN Conv algorithm selection)
**Branch:** `cycle7/cudnn-fused-lstm`

---

## Goal

Replace iconnx's custom multi-kernel LSTM (2 cuBLAS GEMMs + 1 fused CUDA kernel per timestep per direction) with cuDNN's fused `cudnnRNNForward`, which processes the entire sequence in a single call. Kokoro has 6 bidirectional LSTM nodes totaling 12.6ms; the most expensive (seq_len=95) alone accounts for 570 kernel launches. cuDNN's fused kernel should reduce this to 6 calls total.

## Kokoro LSTM profile (empirical, seq_len=5 test)

All 6 LSTM calls are **bidirectional** (num_directions=2), hidden_size=256:

| Calls | seq_len | input_size | W shape | R shape | B | h_init | c_init |
|---|---|---|---|---|---|---|---|
| 4 | 5 | 640 | [2,1024,640] | [2,1024,256] | [2,2048] | [2,1,256] | None |
| 1 | 95 | 640 | [2,1024,640] | [2,1024,256] | [2,2048] | [2,1,256] | None |
| 1 | 5 | 512 | [2,1024,512] | [2,1024,256] | [2,2048] | [2,1,256] | None |

Current: 2 GEMMs + 1 kernel per timestep per direction = 3 × seq_len × 2 ops per call.
Total kernel launches: 6 × avg(5,5,5,5,95,5) × 2 × 3 = ~720 kernel launches.
After: 6 × 1 = 6 cudnnRNNForward calls.

## API choice: cuDNN Legacy RNN (not Graph API)

Using `cudnnSetRNNDescriptor_v8` + `cudnnRNNForward`. This is the stable legacy API that ORT uses. The graph API is more complex and RNN support is less mature.

## Architecture

### New FFI bindings (sys.rs)

New opaque handle types:
- `cudnnRNNDescriptor_t`
- `cudnnRNNDataDescriptor_t`
- `cudnnDropoutDescriptor_t` (required by RNN descriptor even if dropout=0)

New enums:
- `cudnnRNNMode_t` (RELU, TANH, LSTM, GRU)
- `cudnnRNNBiasMode_t` (NO_BIAS, SINGLE_INP_BIAS, DOUBLE_BIAS, SINGLE_REC_BIAS)
- `cudnnDirectionMode_t` (UNIDIRECTIONAL, BIDIRECTIONAL)
- `cudnnRNNInputMode_t` (LINEAR_INPUT, SKIP_INPUT)
- `cudnnForwardMode_t` (INFERENCE, TRAINING)
- `cudnnRNNDataLayout_t` (SEQ_MAJOR_UNPACKED, BATCH_MAJOR_UNPACKED, SEQ_MAJOR_PACKED)
- `cudnnRNNAlgo_t` (STANDARD, PERSIST_STATIC, PERSIST_DYNAMIC, PERSIST_STATIC_SMALL_H)

New functions:
- `cudnnCreateRNNDescriptor` / `cudnnDestroyRNNDescriptor`
- `cudnnSetRNNDescriptor_v8` — configure LSTM mode, bias, direction, etc.
- `cudnnGetRNNWeightSpaceSize` — query packed weight buffer size
- `cudnnGetRNNWeightParams` — get offset/pointer for individual weight matrices within packed buffer
- `cudnnCreateRNNDataDescriptor` / `cudnnDestroyRNNDataDescriptor`
- `cudnnSetRNNDataDescriptor` — set sequence lengths + data layout
- `cudnnGetRNNTempSpaceSizes` — query workspace and reserve space sizes
- `cudnnRNNForward` — the main forward pass
- `cudnnCreateDropoutDescriptor` / `cudnnDestroyDropoutDescriptor`
- `cudnnSetDropoutDescriptor` — configure dropout (set rate=0 for inference)
- `cudnnDropoutGetStatesSize` — query state buffer size for dropout

### Safe Rust wrappers (new file: `src/cuda/cudnn/rnn.rs`)

To avoid growing `mod.rs` further (already 982 lines), the RNN wrappers go in a new file.

Key types:
- `RnnDescriptor` — RAII wrapper around `cudnnRNNDescriptor_t`
- `RnnDataDescriptor` — RAII wrapper around `cudnnRNNDataDescriptor_t`
- `DropoutDescriptor` — RAII wrapper for the dropout state (rate=0 for inference)
- `PackedLstmWeights` — the repacked weight buffer on GPU

Key function:
```rust
pub fn cudnn_lstm_forward(
    handle: &CudnnHandle,
    ctx: &IconnxCudaContext,
    workspace: &mut Workspace,
    packed_weights: &PackedLstmWeights,
    x: &GpuTensor,          // [seq_len, batch, input_size]
    h_init: Option<&GpuTensor>, // [num_directions, batch, hidden_size]
    c_init: Option<&GpuTensor>, // [num_directions, batch, hidden_size]
    hidden_size: usize,
    num_directions: usize,   // 1 or 2
) -> Result<GpuTensor, CudaError>  // Y: [seq_len, batch, num_directions * hidden_size]
```

### Weight repacking: ONNX → cuDNN

**ONNX gate order:** IOFC (input, output, forget, cell)
**cuDNN gate order:** IFCO (input, forget, cell, output)

The repack function:
```rust
pub fn pack_lstm_weights_for_cudnn(
    handle: &CudnnHandle,
    ctx: &IconnxCudaContext,
    w: &GpuTensor,          // [num_directions, 4*hidden, input_size]
    r: &GpuTensor,          // [num_directions, 4*hidden, hidden_size]
    b: Option<&GpuTensor>,  // [num_directions, 8*hidden]
    hidden_size: usize,
    input_size: usize,
    num_directions: usize,
) -> Result<PackedLstmWeights, CudaError>
```

Steps:
1. Create RNN descriptor with LSTM mode, hidden_size, num_layers=1, bidirectional
2. Call `cudnnGetRNNWeightSpaceSize` to get packed buffer size
3. Allocate GPU buffer of that size
4. For each direction and each gate (IFCO order):
   - Call `cudnnGetRNNWeightParams` to get the offset within the packed buffer for that gate's weight/bias
   - Copy the corresponding slice from ONNX's W/R/B tensor (reordered from IOFC) to that offset
5. Return the packed buffer as `PackedLstmWeights`

The IOFC→IFCO reordering:
- ONNX gate 0 (I) → cuDNN gate 0 (I): no change
- ONNX gate 1 (O) → cuDNN gate 3 (O): move to position 3
- ONNX gate 2 (F) → cuDNN gate 1 (F): move to position 1
- ONNX gate 3 (C) → cuDNN gate 2 (C): move to position 2

### Lazy caching

```rust
pub struct LstmWeightCache {
    /// Maps (W_ptr, R_ptr) identity to packed weights.
    /// Using raw device pointer as key — same weight tensor always has same address.
    cache: HashMap<(usize, usize), PackedLstmWeights>,
}
```

Stored as `RefCell<LstmWeightCache>` on the executor. On first LSTM call per unique weight pair, repack and cache. Subsequent calls with the same weights hit the cache.

### Executor wiring

In the LSTM dispatch arm, replace `gpu_lstm(...)` with:

```rust
"LSTM" => {
    let packed = self.get_or_pack_lstm_weights(w, r, b, hidden_size, input_size, num_dirs)?;
    cudnn_lstm_forward(
        &self.cudnn_handle, &self.ctx, &mut workspace,
        &packed, inputs[0], h_init, c_init,
        hidden_size, num_dirs,
    )
}
```

The `get_or_pack_lstm_weights` method checks the cache, repacks on miss.

## Testing

1. **Unit test for weight repacking**: Create known W/R/B tensors in IOFC order, repack, verify the packed buffer contents match cuDNN's expected IFCO layout.
2. **GPU integration test**: Run `cudnn_lstm_forward` on a small synthetic LSTM (seq_len=3, batch=1, hidden=4, bidirectional) and verify output matches the existing `gpu_lstm` implementation.
3. **Kokoro integration**: Verify `LSTM` op_times drops significantly. Existing fusion counts unchanged.
4. **Numerical regression**: Compare `cudnn_lstm_forward` output against `gpu_lstm` output for the same inputs — should match to within f32 epsilon.

## Scope

**In scope:**
- ~15 new FFI bindings in sys.rs
- New `src/cuda/cudnn/rnn.rs` with safe wrappers
- Weight repacking IOFC→IFCO with lazy caching
- Executor dispatch replacement
- Tests

**Out of scope:**
- Training (forward-only, `CUDNN_FWD_MODE_INFERENCE`)
- Dropout (rate=0, but descriptor still needed for API)
- Multi-layer LSTM (Kokoro uses single-layer; the descriptor supports `numLayers=1`)
- Peephole connections (ONNX input P not used by Kokoro)
- GRU or other RNN variants
- Returning Y_h / Y_c separately (current code only returns Y)

## File impact

| File | Change | Est. lines |
|---|---|---|
| `src/cuda/cudnn/sys.rs` | New FFI types + functions | +120 |
| `src/cuda/cudnn/rnn.rs` | New file: safe wrappers + weight repacking | +350 |
| `src/cuda/cudnn/mod.rs` | `pub mod rnn; pub use rnn::*;` | +2 |
| `src/cuda/inference/mod.rs` | LSTM dispatch replacement + LstmWeightCache | +40 |
| Tests | Unit + integration tests | +150 |

## Success criteria

1. Kokoro LSTM time drops from ~12.6ms to ~2-3ms
2. All existing tests pass with identical numerical results
3. `Fused_AddMulAdd: 70` unchanged
4. Clippy clean
