# Cycle 7: cuDNN Fused LSTM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace iconnx's per-timestep LSTM (2 cuBLAS GEMMs + 1 fused CUDA kernel per step per direction) with cuDNN's fused `cudnnRNNForward`, which processes the entire sequence in a single call. Kokoro has 6 bidirectional LSTM nodes totaling ~12.6ms; the most expensive (seq_len=95) alone accounts for ~570 kernel launches. After this change: 6 `cudnnRNNForward` calls total.

**Architecture:** Weight tensors are repacked once from ONNX gate order (IOFC) to cuDNN gate order (IFCO) into a packed weight buffer. A `LstmWeightCache` on the executor maps `(W_ptr, R_ptr)` device pointer pairs to `PackedLstmWeights`, so repacking happens only on first encounter per unique weight pair. The forward pass uses `cudnnRNNForward` in `CUDNN_FWD_MODE_INFERENCE` mode.

**Tech Stack:** Rust 2021, cuDNN 9.x FFI bindings (legacy RNN API, not graph), existing iconnx infrastructure. No new crate dependencies.

**Prerequisites for the implementing agent:**
- Read the spec first: `docs/superpowers/specs/2026-04-12-cudnn-fused-lstm-design.md`.
- Per `/home/ryano/.claude/CLAUDE.md`: TDD where possible, commits after every substantive change, **all commits must be signed** (`git commit -S`), stop and ask if signing fails, warnings never ignored.
- `cargo test --features cuda` is the regression gate. Run after each task.
- `cargo clippy --features cuda --tests -- -D warnings` must be clean before every commit.
- **NEVER run `git checkout <sha>`** to inspect commits. Use `git show SHA`, `git show SHA --stat`, or `git log --show-signature -1 SHA`.
- **NEVER use unsafe Rust** without asking first. The FFI calls in `sys.rs` and the `unsafe` blocks in `mod.rs`/`rnn.rs` follow existing patterns exactly -- replicate those patterns.
- **No files over 1000 lines.** The new `rnn.rs` must stay under 1000 lines. `src/cuda/cudnn/mod.rs` is currently 982 lines -- do NOT add more than a `pub mod rnn; pub use rnn::*;` to it.

---

## Scope Check

This plan implements **Cycle 7 only** -- cuDNN Fused LSTM. Out of scope:

- Training / backward pass (inference only, `CUDNN_FWD_MODE_INFERENCE`)
- Dropout (rate=0 always; the `DropoutDescriptor` is created with rate=0 because the API requires one)
- Multi-layer LSTM (Kokoro uses single-layer; `numLayers=1`)
- Peephole connections (ONNX input P not used by Kokoro)
- GRU or other RNN variants
- Returning Y_h / Y_c separately (current code only returns Y)
- Any changes to fusion patterns (AddMulAdd, Gelu, MulSinPowMulAdd, etc.)
- Refactoring `inference/mod.rs` file size (pre-existing issue)
- Removing the old `gpu_lstm` function (kept as fallback for now)

Any work outside the files listed below is a scope violation and should be flagged to the controller before landing.

---

## File Structure

### New files

| File | Responsibility | Est. lines |
|---|---|---|
| `src/cuda/cudnn/rnn.rs` | RAII descriptor wrappers (RnnDescriptor, RnnDataDescriptor, DropoutDescriptor), PackedLstmWeights, weight repacking, cudnn_lstm_forward | ~450 |

### Modified files

| File | Responsibility | Est. delta |
|---|---|---|
| `src/cuda/cudnn/sys.rs` | New FFI enum types, opaque handle types, ~15 extern "C" function declarations | +120 lines |
| `src/cuda/cudnn/mod.rs` | `pub mod rnn; pub use rnn::*;` | +2 lines |
| `src/cuda/inference/mod.rs` | Add `LstmWeightCache` field, add `get_or_pack_lstm_weights` method, replace `gpu_lstm` call with `cudnn_lstm_forward` | +40 lines |

### Files NOT touched

- `src/cuda/lstm.rs` -- old per-timestep LSTM kept as reference; not deleted.
- `src/cuda/context.rs` -- no new error variants needed (`CudaError::Cudnn(String)` already exists).
- `src/cuda/inference/fusion/` -- no fusion changes.
- `src/cuda/kernels/` -- no kernel changes.
- `tests/kokoro_gpu_fusion_test.rs` -- fusion counts unchanged.

---

## Task 1: FFI Bindings for cuDNN RNN API

**Files:**
- Modify: `src/cuda/cudnn/sys.rs` (currently 527 lines)

This task adds all new enum types, opaque handle types, and extern "C" function declarations needed for the cuDNN RNN (LSTM) API.

- [ ] **Step 1: Add RNN-related enum types**

After the `cudnnActivationMode_t` enum (line 170), add a new section with all RNN enums. Follow the existing pattern of `#[repr(C)]` + `#[derive(Debug, Clone, Copy, PartialEq, Eq)]`:

```rust
// =============================================================================
// RNN Types
// =============================================================================

/// RNN cell mode
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cudnnRNNMode_t {
    CUDNN_RNN_RELU = 0,
    CUDNN_RNN_TANH = 1,
    CUDNN_LSTM = 2,
    CUDNN_GRU = 3,
}

/// RNN bias mode
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cudnnRNNBiasMode_t {
    CUDNN_RNN_NO_BIAS = 0,
    CUDNN_RNN_SINGLE_INP_BIAS = 1,
    CUDNN_RNN_DOUBLE_BIAS = 2,
    CUDNN_RNN_SINGLE_REC_BIAS = 3,
}

/// RNN direction mode
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cudnnDirectionMode_t {
    CUDNN_UNIDIRECTIONAL = 0,
    CUDNN_BIDIRECTIONAL = 1,
}

/// RNN input mode
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cudnnRNNInputMode_t {
    CUDNN_LINEAR_INPUT = 0,
    CUDNN_SKIP_INPUT = 1,
}

/// RNN forward mode
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cudnnForwardMode_t {
    CUDNN_FWD_MODE_INFERENCE = 0,
    CUDNN_FWD_MODE_TRAINING = 1,
}

/// RNN data layout
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cudnnRNNDataLayout_t {
    CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED = 0,
    CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED = 1,
    CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED = 2,
}

/// RNN algorithm
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cudnnRNNAlgo_t {
    CUDNN_RNN_ALGO_STANDARD = 0,
    CUDNN_RNN_ALGO_PERSIST_STATIC = 1,
    CUDNN_RNN_ALGO_PERSIST_DYNAMIC = 2,
    CUDNN_RNN_ALGO_PERSIST_STATIC_SMALL_H = 3,
}
```

- [ ] **Step 2: Add opaque handle types for RNN**

After the existing `cudnnActivationDescriptor_t` definition (line 209), add:

```rust
/// RNN descriptor
#[repr(C)]
pub struct cudnnRNNStruct {
    _private: [u8; 0],
}
pub type cudnnRNNDescriptor_t = *mut cudnnRNNStruct;

/// RNN data descriptor
#[repr(C)]
pub struct cudnnRNNDataStruct {
    _private: [u8; 0],
}
pub type cudnnRNNDataDescriptor_t = *mut cudnnRNNDataStruct;

/// Dropout descriptor
#[repr(C)]
pub struct cudnnDropoutStruct {
    _private: [u8; 0],
}
pub type cudnnDropoutDescriptor_t = *mut cudnnDropoutStruct;
```

- [ ] **Step 3: Add extern "C" function declarations for RNN**

Inside the `#[link(name = "cudnn")] extern "C"` block, after the Activation section (line 497), add a new RNN section:

```rust
    // -------------------------------------------------------------------------
    // RNN Descriptor Management
    // -------------------------------------------------------------------------

    /// Create an RNN descriptor
    pub fn cudnnCreateRNNDescriptor(rnnDesc: *mut cudnnRNNDescriptor_t) -> cudnnStatus_t;

    /// Destroy an RNN descriptor
    pub fn cudnnDestroyRNNDescriptor(rnnDesc: cudnnRNNDescriptor_t) -> cudnnStatus_t;

    /// Configure RNN descriptor (v8 API, cuDNN 8+)
    pub fn cudnnSetRNNDescriptor_v8(
        rnnDesc: cudnnRNNDescriptor_t,
        algo: cudnnRNNAlgo_t,
        cellMode: cudnnRNNMode_t,
        biasMode: cudnnRNNBiasMode_t,
        dirMode: cudnnDirectionMode_t,
        inputMode: cudnnRNNInputMode_t,
        dataType: cudnnDataType_t,
        mathPrec: cudnnDataType_t,
        mathType: cudnnMathType_t,
        inputSize: i32,
        hiddenSize: i32,
        projSize: i32,
        numLayers: i32,
        dropoutDesc: cudnnDropoutDescriptor_t,
        auxFlags: u32,
    ) -> cudnnStatus_t;

    // -------------------------------------------------------------------------
    // RNN Weight Space
    // -------------------------------------------------------------------------

    /// Get the size of the packed weight buffer for an RNN
    pub fn cudnnGetRNNWeightSpaceSize(
        handle: cudnnHandle_t,
        rnnDesc: cudnnRNNDescriptor_t,
        weightSpaceSize: *mut usize,
    ) -> cudnnStatus_t;

    /// Get pointers to individual weight/bias matrices within the packed buffer
    pub fn cudnnGetRNNWeightParams(
        handle: cudnnHandle_t,
        rnnDesc: cudnnRNNDescriptor_t,
        pseudoLayer: i32,
        weightSpaceSize: usize,
        weightSpace: *const c_void,
        linLayerID: i32,
        mDesc: cudnnTensorDescriptor_t,
        mAddr: *mut *mut c_void,
        bDesc: cudnnTensorDescriptor_t,
        bAddr: *mut *mut c_void,
    ) -> cudnnStatus_t;

    // -------------------------------------------------------------------------
    // RNN Data Descriptor
    // -------------------------------------------------------------------------

    /// Create an RNN data descriptor
    pub fn cudnnCreateRNNDataDescriptor(
        rnnDataDesc: *mut cudnnRNNDataDescriptor_t,
    ) -> cudnnStatus_t;

    /// Destroy an RNN data descriptor
    pub fn cudnnDestroyRNNDataDescriptor(
        rnnDataDesc: cudnnRNNDataDescriptor_t,
    ) -> cudnnStatus_t;

    /// Set RNN data descriptor
    pub fn cudnnSetRNNDataDescriptor(
        rnnDataDesc: cudnnRNNDataDescriptor_t,
        dataType: cudnnDataType_t,
        layout: cudnnRNNDataLayout_t,
        maxSeqLength: i32,
        batchSize: i32,
        vectorSize: i32,
        seqLengthArray: *const i32,
        paddingFill: *mut c_void,
    ) -> cudnnStatus_t;

    // -------------------------------------------------------------------------
    // RNN Temp Space and Forward
    // -------------------------------------------------------------------------

    /// Get workspace and reserve space sizes for RNN forward
    pub fn cudnnGetRNNTempSpaceSizes(
        handle: cudnnHandle_t,
        rnnDesc: cudnnRNNDescriptor_t,
        fMode: cudnnForwardMode_t,
        xDesc: cudnnRNNDataDescriptor_t,
        workSpaceSize: *mut usize,
        reserveSpaceSize: *mut usize,
    ) -> cudnnStatus_t;

    /// Execute RNN forward pass
    pub fn cudnnRNNForward(
        handle: cudnnHandle_t,
        rnnDesc: cudnnRNNDescriptor_t,
        fwdMode: cudnnForwardMode_t,
        devSeqLengths: *const i32,
        xDesc: cudnnRNNDataDescriptor_t,
        x: *const c_void,
        yDesc: cudnnRNNDataDescriptor_t,
        y: *mut c_void,
        hDesc: cudnnTensorDescriptor_t,
        hx: *const c_void,
        hy: *mut c_void,
        cDesc: cudnnTensorDescriptor_t,
        cx: *const c_void,
        cy: *mut c_void,
        weightSpaceSize: usize,
        weightSpace: *const c_void,
        workSpaceSize: usize,
        workSpace: *mut c_void,
        reserveSpaceSize: usize,
        reserveSpace: *mut c_void,
    ) -> cudnnStatus_t;

    // -------------------------------------------------------------------------
    // Dropout Descriptor (required for RNN even with rate=0)
    // -------------------------------------------------------------------------

    /// Create a dropout descriptor
    pub fn cudnnCreateDropoutDescriptor(
        dropoutDesc: *mut cudnnDropoutDescriptor_t,
    ) -> cudnnStatus_t;

    /// Destroy a dropout descriptor
    pub fn cudnnDestroyDropoutDescriptor(
        dropoutDesc: cudnnDropoutDescriptor_t,
    ) -> cudnnStatus_t;

    /// Get the size of the state buffer needed for dropout
    pub fn cudnnDropoutGetStatesSize(
        handle: cudnnHandle_t,
        sizeInBytes: *mut usize,
    ) -> cudnnStatus_t;

    /// Configure dropout descriptor
    pub fn cudnnSetDropoutDescriptor(
        dropoutDesc: cudnnDropoutDescriptor_t,
        handle: cudnnHandle_t,
        dropout: f32,
        states: *mut c_void,
        stateSizeInBytes: usize,
        seed: u64,
    ) -> cudnnStatus_t;
```

- [ ] **Step 4: Run clippy to verify**

Run: `cargo clippy --features cuda --tests -- -D warnings`
Expected: 0 errors, 0 warnings. The new types and functions will be used by subsequent tasks. The `#![allow(dead_code)]` at the top of `sys.rs` (line 11) suppresses dead code warnings.

- [ ] **Step 5: Commit**

```bash
git add src/cuda/cudnn/sys.rs
git commit -S -m "Add FFI bindings for cuDNN RNN (LSTM) API

Bind 7 enum types (cudnnRNNMode_t, cudnnRNNBiasMode_t,
cudnnDirectionMode_t, cudnnRNNInputMode_t, cudnnForwardMode_t,
cudnnRNNDataLayout_t, cudnnRNNAlgo_t), 3 opaque handle types
(cudnnRNNDescriptor_t, cudnnRNNDataDescriptor_t,
cudnnDropoutDescriptor_t), and 15 extern C function declarations
for the cuDNN legacy RNN API (v8 descriptor, forward pass, weight
params, dropout).

These bindings support the cudnnRNNForward fused LSTM path that
will replace per-timestep cuBLAS GEMM + custom kernel execution.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: RAII Descriptor Wrappers in New File rnn.rs

**Files:**
- Create: `src/cuda/cudnn/rnn.rs`
- Modify: `src/cuda/cudnn/mod.rs` (add module declaration only)

This task creates the safe Rust wrappers for the three RNN descriptor types. The wrappers follow the exact same RAII pattern as `TensorDescriptor`, `FilterDescriptor`, and `ConvolutionDescriptor` in `mod.rs`: create in `new()`, destroy in `Drop`, expose `raw()` for FFI.

- [ ] **Step 1: Add module declaration to mod.rs**

At the top of `src/cuda/cudnn/mod.rs`, after `mod sys;` (line 11), add:

```rust
pub mod rnn;
pub use rnn::*;
```

This is the only change to `mod.rs`. All RNN code goes in `rnn.rs`.

- [ ] **Step 2: Create `src/cuda/cudnn/rnn.rs` with module header and imports**

```rust
//! cuDNN RNN (LSTM) wrappers for fused sequence-level forward pass
//!
//! Provides safe Rust wrappers around cuDNN's legacy RNN API (v8 descriptor)
//! for LSTM inference. Instead of per-timestep cuBLAS GEMMs + custom CUDA
//! kernels, this module uses a single cudnnRNNForward call per LSTM node.
//!
//! Key types:
//! - RnnDescriptor: RAII wrapper for cudnnRNNDescriptor_t
//! - RnnDataDescriptor: RAII wrapper for cudnnRNNDataDescriptor_t
//! - DropoutDescriptor: RAII wrapper for cudnnDropoutDescriptor_t
//! - PackedLstmWeights: GPU buffer with ONNX weights repacked into cuDNN layout

#![allow(clippy::too_many_arguments)] // cuDNN RNN APIs require many parameters

use std::ffi::c_void;
use std::ptr;

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use super::sys;
use super::{check_cudnn, CudnnHandle, TensorDescriptor, Workspace};
use crate::cuda::context::{CudaError, IconnxCudaContext};
use crate::cuda::tensor::GpuTensor;
```

- [ ] **Step 3: Add DropoutDescriptor**

The dropout descriptor must be created first because it's required by `cudnnSetRNNDescriptor_v8`. We always use rate=0.0 for inference.

```rust
// =============================================================================
// DropoutDescriptor
// =============================================================================

/// RAII wrapper around cudnnDropoutDescriptor_t.
///
/// Always configured with dropout rate 0.0 for inference. The cuDNN RNN API
/// requires a dropout descriptor even when no dropout is applied.
pub struct DropoutDescriptor {
    desc: sys::cudnnDropoutDescriptor_t,
    /// GPU buffer for dropout RNG state (required even at rate=0)
    _states: CudaSlice<u8>,
}

impl DropoutDescriptor {
    /// Create a new dropout descriptor with rate=0.0
    pub fn new(handle: &CudnnHandle, ctx: &IconnxCudaContext) -> Result<Self, CudaError> {
        let mut desc: sys::cudnnDropoutDescriptor_t = ptr::null_mut();
        unsafe {
            check_cudnn(sys::cudnnCreateDropoutDescriptor(&mut desc))?;
        }

        // Query state buffer size
        let mut state_size: usize = 0;
        unsafe {
            check_cudnn(sys::cudnnDropoutGetStatesSize(
                handle.raw(),
                &mut state_size,
            ))?;
        }

        // Allocate state buffer on GPU
        let states = ctx.alloc_zeros::<u8>(state_size.max(1))?;

        // Configure with rate=0.0 (no dropout)
        let stream = ctx.stream();
        let (states_ptr, _guard) = states.device_ptr_mut(stream);
        unsafe {
            check_cudnn(sys::cudnnSetDropoutDescriptor(
                desc,
                handle.raw(),
                0.0, // dropout rate
                states_ptr as *mut c_void,
                state_size,
                0, // seed (arbitrary for rate=0)
            ))?;
        }

        Ok(Self {
            desc,
            _states: states,
        })
    }

    /// Get raw descriptor for FFI calls
    pub(crate) fn raw(&self) -> sys::cudnnDropoutDescriptor_t {
        self.desc
    }
}

impl Drop for DropoutDescriptor {
    fn drop(&mut self) {
        unsafe {
            sys::cudnnDestroyDropoutDescriptor(self.desc);
        }
    }
}
```

**Note:** The `_states` field keeps the GPU allocation alive for the lifetime of the descriptor. cuDNN holds a pointer into this buffer, so it must not be freed while the descriptor exists.

- [ ] **Step 4: Add RnnDescriptor**

```rust
// =============================================================================
// RnnDescriptor
// =============================================================================

/// RAII wrapper around cudnnRNNDescriptor_t, configured for LSTM inference.
pub struct RnnDescriptor {
    desc: sys::cudnnRNNDescriptor_t,
    /// Dropout descriptor must outlive the RNN descriptor
    _dropout: DropoutDescriptor,
}

impl RnnDescriptor {
    /// Create a new RNN descriptor configured for LSTM inference.
    ///
    /// - `hidden_size`: number of hidden units per direction
    /// - `input_size`: input feature dimension
    /// - `num_directions`: 1 for unidirectional, 2 for bidirectional
    pub fn new_lstm(
        handle: &CudnnHandle,
        ctx: &IconnxCudaContext,
        hidden_size: usize,
        input_size: usize,
        num_directions: usize,
    ) -> Result<Self, CudaError> {
        let mut desc: sys::cudnnRNNDescriptor_t = ptr::null_mut();
        unsafe {
            check_cudnn(sys::cudnnCreateRNNDescriptor(&mut desc))?;
        }

        let dropout = DropoutDescriptor::new(handle, ctx)?;

        let dir_mode = if num_directions == 2 {
            sys::cudnnDirectionMode_t::CUDNN_BIDIRECTIONAL
        } else {
            sys::cudnnDirectionMode_t::CUDNN_UNIDIRECTIONAL
        };

        unsafe {
            check_cudnn(sys::cudnnSetRNNDescriptor_v8(
                desc,
                sys::cudnnRNNAlgo_t::CUDNN_RNN_ALGO_STANDARD,
                sys::cudnnRNNMode_t::CUDNN_LSTM,
                sys::cudnnRNNBiasMode_t::CUDNN_RNN_DOUBLE_BIAS,
                dir_mode,
                sys::cudnnRNNInputMode_t::CUDNN_LINEAR_INPUT,
                sys::cudnnDataType_t::CUDNN_DATA_FLOAT,
                sys::cudnnDataType_t::CUDNN_DATA_FLOAT,
                sys::cudnnMathType_t::CUDNN_DEFAULT_MATH,
                input_size as i32,
                hidden_size as i32,
                0,  // projSize (no projection)
                1,  // numLayers (single layer)
                dropout.raw(),
                1,  // auxFlags: CUDNN_RNN_PADDED_IO_ENABLED
            ))?;
        }

        Ok(Self {
            desc,
            _dropout: dropout,
        })
    }

    /// Get raw descriptor for FFI calls
    pub(crate) fn raw(&self) -> sys::cudnnRNNDescriptor_t {
        self.desc
    }
}

impl Drop for RnnDescriptor {
    fn drop(&mut self) {
        unsafe {
            sys::cudnnDestroyRNNDescriptor(self.desc);
        }
    }
}
```

- [ ] **Step 5: Add RnnDataDescriptor**

```rust
// =============================================================================
// RnnDataDescriptor
// =============================================================================

/// RAII wrapper around cudnnRNNDataDescriptor_t.
///
/// Describes the layout of sequence data (input X or output Y) for the RNN.
pub struct RnnDataDescriptor {
    desc: sys::cudnnRNNDataDescriptor_t,
}

impl RnnDataDescriptor {
    /// Create a new RNN data descriptor.
    ///
    /// - `seq_len`: maximum sequence length
    /// - `batch_size`: batch size
    /// - `vector_size`: feature dimension (input_size for X, hidden*num_dirs for Y)
    pub fn new(seq_len: usize, batch_size: usize, vector_size: usize) -> Result<Self, CudaError> {
        let mut desc: sys::cudnnRNNDataDescriptor_t = ptr::null_mut();
        unsafe {
            check_cudnn(sys::cudnnCreateRNNDataDescriptor(&mut desc))?;
        }

        // All sequences in the batch have the same length
        let seq_lengths: Vec<i32> = vec![seq_len as i32; batch_size];

        unsafe {
            check_cudnn(sys::cudnnSetRNNDataDescriptor(
                desc,
                sys::cudnnDataType_t::CUDNN_DATA_FLOAT,
                sys::cudnnRNNDataLayout_t::CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED,
                seq_len as i32,
                batch_size as i32,
                vector_size as i32,
                seq_lengths.as_ptr(),
                ptr::null_mut(), // default padding fill
            ))?;
        }

        Ok(Self { desc })
    }

    /// Get raw descriptor for FFI calls
    pub(crate) fn raw(&self) -> sys::cudnnRNNDataDescriptor_t {
        self.desc
    }
}

impl Drop for RnnDataDescriptor {
    fn drop(&mut self) {
        unsafe {
            sys::cudnnDestroyRNNDataDescriptor(self.desc);
        }
    }
}
```

- [ ] **Step 6: Run clippy to verify**

Run: `cargo clippy --features cuda --tests -- -D warnings`
Expected: Clippy clean. The types are not yet used by any callers (that comes in Tasks 3-4). The `#![allow(clippy::too_many_arguments)]` at the top of `rnn.rs` suppresses warnings on the descriptor constructors.

- [ ] **Step 7: Commit**

```bash
git add src/cuda/cudnn/rnn.rs src/cuda/cudnn/mod.rs
git commit -S -m "Add RAII descriptor wrappers for cuDNN RNN API

Create src/cuda/cudnn/rnn.rs with safe Rust wrappers:
- DropoutDescriptor: rate=0 for inference, manages GPU state buffer
- RnnDescriptor: configured for LSTM, single-layer, STANDARD algo
- RnnDataDescriptor: SEQ_MAJOR_UNPACKED layout for input/output

All wrappers follow the existing Create/Set/Destroy RAII pattern
from TensorDescriptor and ConvolutionDescriptor in mod.rs.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Weight Repacking Function (ONNX IOFC to cuDNN IFCO)

**Files:**
- Modify: `src/cuda/cudnn/rnn.rs`

This is the most complex task. ONNX stores LSTM weights in gate order IOFC (Input, Output, Forget, Cell); cuDNN expects IFCO (Input, Forget, Cell/g, Output). The repacking function queries cuDNN for the exact byte offset of each gate's weight/bias within the packed buffer, then copies slices from the ONNX tensors to those offsets.

**cuDNN linLayerID mapping for LSTM:**
- 0: W_i (input gate, input weights)
- 1: W_f (forget gate, input weights)
- 2: W_g (cell gate, input weights) -- cuDNN calls it "g" not "c"
- 3: W_o (output gate, input weights)
- 4: R_i (input gate, recurrence weights)
- 5: R_f (forget gate, recurrence weights)
- 6: R_g (cell gate, recurrence weights)
- 7: R_o (output gate, recurrence weights)

**ONNX gate order in W tensor:** [I=0, O=1, F=2, C=3]
- rows `[0..hidden]` = I gate
- rows `[hidden..2*hidden]` = O gate
- rows `[2*hidden..3*hidden]` = F gate
- rows `[3*hidden..4*hidden]` = C gate

**ONNX to cuDNN mapping:**
- ONNX I (row 0) -> cuDNN linLayerID 0 (W_i) or 4 (R_i)
- ONNX O (row 1) -> cuDNN linLayerID 3 (W_o) or 7 (R_o)
- ONNX F (row 2) -> cuDNN linLayerID 1 (W_f) or 5 (R_f)
- ONNX C (row 3) -> cuDNN linLayerID 2 (W_g) or 6 (R_g)

Same gate reordering for bias: ONNX B = `[Wb_iofc, Rb_iofc]` = 8*hidden values per direction.

- [ ] **Step 1: Add the PackedLstmWeights struct**

Append to `rnn.rs`:

```rust
// =============================================================================
// Packed LSTM Weights
// =============================================================================

/// GPU buffer containing LSTM weights repacked from ONNX gate order (IOFC) to
/// cuDNN gate order (IFCO).
///
/// Created by `pack_lstm_weights_for_cudnn` and consumed by `cudnn_lstm_forward`.
/// The buffer layout is opaque -- only cuDNN knows the internal offsets. We
/// query offsets via `cudnnGetRNNWeightParams` during packing.
///
/// **Cache pointer invariant:** This struct is valid only while the original
/// ONNX W/R tensors (whose data was copied into this buffer) remain alive.
/// In practice the executor owns both the weight tensors and the cache, so
/// this invariant is trivially satisfied.
pub struct PackedLstmWeights {
    /// Packed weight buffer on GPU
    buffer: CudaSlice<u8>,
    /// Size in bytes
    size: usize,
    /// RNN descriptor that was used to create this packing
    /// (must outlive any cudnnRNNForward call using this buffer)
    rnn_desc: RnnDescriptor,
}

impl PackedLstmWeights {
    /// Size of the packed weight buffer in bytes
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get the RNN descriptor
    pub(crate) fn rnn_desc(&self) -> &RnnDescriptor {
        &self.rnn_desc
    }
}
```

- [ ] **Step 2: Add the gate reorder mapping constant**

```rust
/// Maps ONNX LSTM gate index to cuDNN linLayerID for input weights (W).
///
/// ONNX gate order: I=0, O=1, F=2, C=3
/// cuDNN linLayerID for W: I=0, F=1, G(=C)=2, O=3
///
/// So: ONNX_gate[i] -> cuDNN linLayerID ONNX_TO_CUDNN_GATE[i]
const ONNX_TO_CUDNN_GATE: [i32; 4] = [
    0, // ONNX I -> cuDNN 0 (W_i)
    3, // ONNX O -> cuDNN 3 (W_o)
    1, // ONNX F -> cuDNN 1 (W_f)
    2, // ONNX C -> cuDNN 2 (W_g)
];
```

- [ ] **Step 3: Add the pack_lstm_weights_for_cudnn function**

```rust
/// Repack ONNX LSTM weights into cuDNN's packed weight buffer format.
///
/// ONNX stores weights in IOFC gate order; cuDNN expects IFCO. This function:
/// 1. Creates an RNN descriptor for the given LSTM configuration
/// 2. Queries cuDNN for the packed weight buffer size
/// 3. Allocates the buffer on GPU
/// 4. For each direction and each gate, queries cuDNN for the offset within
///    the buffer and copies the corresponding ONNX weight slice there
///
/// The returned `PackedLstmWeights` owns both the GPU buffer and the RNN
/// descriptor. Both must outlive any `cudnn_lstm_forward` call.
pub fn pack_lstm_weights_for_cudnn(
    handle: &CudnnHandle,
    ctx: &IconnxCudaContext,
    w: &GpuTensor,         // [num_directions, 4*hidden, input_size]
    r: &GpuTensor,         // [num_directions, 4*hidden, hidden_size]
    b: Option<&GpuTensor>, // [num_directions, 8*hidden]
    hidden_size: usize,
    input_size: usize,
    num_directions: usize,
) -> Result<PackedLstmWeights, CudaError> {
    // Create RNN descriptor
    let rnn_desc = RnnDescriptor::new_lstm(handle, ctx, hidden_size, input_size, num_directions)?;

    // Query packed weight buffer size
    let mut weight_space_size: usize = 0;
    unsafe {
        check_cudnn(sys::cudnnGetRNNWeightSpaceSize(
            handle.raw(),
            rnn_desc.raw(),
            &mut weight_space_size,
        ))?;
    }

    // Allocate packed weight buffer (zero-initialized for safety)
    let mut weight_buffer = ctx.alloc_zeros::<u8>(weight_space_size)?;

    // Create temporary tensor descriptors for cudnnGetRNNWeightParams output
    let mut m_desc = TensorDescriptor::new()?;
    let mut b_desc = TensorDescriptor::new()?;

    let stream = ctx.stream();

    for dir in 0..num_directions {
        // pseudoLayer: 0 for forward direction, 1 for backward direction
        let pseudo_layer = dir as i32;

        // Process input weights (W): linLayerID 0-3
        for onnx_gate in 0..4usize {
            let cudnn_lin_id = ONNX_TO_CUDNN_GATE[onnx_gate];

            // Query cuDNN for the destination offset within packed buffer
            let mut m_addr: *mut c_void = ptr::null_mut();
            let mut b_addr: *mut c_void = ptr::null_mut();

            let (wb_ptr, _wb_guard) = weight_buffer.device_ptr_mut(stream);
            unsafe {
                check_cudnn(sys::cudnnGetRNNWeightParams(
                    handle.raw(),
                    rnn_desc.raw(),
                    pseudo_layer,
                    weight_space_size,
                    wb_ptr as *const c_void,
                    cudnn_lin_id,
                    m_desc.raw(),
                    &mut m_addr,
                    b_desc.raw(),
                    &mut b_addr,
                ))?;
            }

            // Copy input weight matrix for this gate
            // Source: W[dir, onnx_gate*hidden..(onnx_gate+1)*hidden, 0..input_size]
            // This is a contiguous block of hidden_size * input_size floats
            let w_offset_floats = dir * 4 * hidden_size * input_size
                + onnx_gate * hidden_size * input_size;
            let w_size_bytes = hidden_size * input_size * std::mem::size_of::<f32>();

            let (w_ptr, _w_guard) = w.data_f32()?.device_ptr(stream);
            let src_ptr = (w_ptr as usize + w_offset_floats * std::mem::size_of::<f32>())
                as *const c_void;

            unsafe {
                let status = cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                    m_addr as u64,
                    src_ptr as u64,
                    w_size_bytes as usize,
                    *stream.cu_stream() as *mut _,
                );
                if status != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                    return Err(CudaError::Transfer(format!(
                        "cuMemcpyDtoDAsync failed for W gate {}: {:?}",
                        onnx_gate, status
                    )));
                }
            }
        }

        // Process recurrence weights (R): linLayerID 4-7
        for onnx_gate in 0..4usize {
            let cudnn_lin_id = ONNX_TO_CUDNN_GATE[onnx_gate] + 4;

            let mut m_addr: *mut c_void = ptr::null_mut();
            let mut b_addr: *mut c_void = ptr::null_mut();

            let (wb_ptr, _wb_guard) = weight_buffer.device_ptr_mut(stream);
            unsafe {
                check_cudnn(sys::cudnnGetRNNWeightParams(
                    handle.raw(),
                    rnn_desc.raw(),
                    pseudo_layer,
                    weight_space_size,
                    wb_ptr as *const c_void,
                    cudnn_lin_id,
                    m_desc.raw(),
                    &mut m_addr,
                    b_desc.raw(),
                    &mut b_addr,
                ))?;
            }

            // Copy recurrence weight matrix for this gate
            // Source: R[dir, onnx_gate*hidden..(onnx_gate+1)*hidden, 0..hidden_size]
            let r_offset_floats = dir * 4 * hidden_size * hidden_size
                + onnx_gate * hidden_size * hidden_size;
            let r_size_bytes = hidden_size * hidden_size * std::mem::size_of::<f32>();

            let (r_ptr, _r_guard) = r.data_f32()?.device_ptr(stream);
            let src_ptr = (r_ptr as usize + r_offset_floats * std::mem::size_of::<f32>())
                as *const c_void;

            unsafe {
                let status = cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                    m_addr as u64,
                    src_ptr as u64,
                    r_size_bytes as usize,
                    *stream.cu_stream() as *mut _,
                );
                if status != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                    return Err(CudaError::Transfer(format!(
                        "cuMemcpyDtoDAsync failed for R gate {}: {:?}",
                        onnx_gate, status
                    )));
                }
            }
        }

        // Process biases if provided
        // ONNX B layout: [Wb_i, Wb_o, Wb_f, Wb_c, Rb_i, Rb_o, Rb_f, Rb_c]
        // = 8*hidden values per direction
        // cuDNN: each linLayerID 0-7 has its own bias vector of size hidden
        if let Some(b_tensor) = b {
            for onnx_gate in 0..4usize {
                // Input bias (Wb): linLayerID 0-3
                let cudnn_lin_id = ONNX_TO_CUDNN_GATE[onnx_gate];

                let mut m_addr: *mut c_void = ptr::null_mut();
                let mut b_addr: *mut c_void = ptr::null_mut();

                let (wb_ptr, _wb_guard) = weight_buffer.device_ptr_mut(stream);
                unsafe {
                    check_cudnn(sys::cudnnGetRNNWeightParams(
                        handle.raw(),
                        rnn_desc.raw(),
                        pseudo_layer,
                        weight_space_size,
                        wb_ptr as *const c_void,
                        cudnn_lin_id,
                        m_desc.raw(),
                        &mut m_addr,
                        b_desc.raw(),
                        &mut b_addr,
                    ))?;
                }

                // Wb[gate]: offset = dir * 8*hidden + onnx_gate * hidden
                let wb_offset_floats = dir * 8 * hidden_size + onnx_gate * hidden_size;
                let bias_size_bytes = hidden_size * std::mem::size_of::<f32>();

                let (b_ptr, _b_guard) = b_tensor.data_f32()?.device_ptr(stream);
                let src_ptr = (b_ptr as usize
                    + wb_offset_floats * std::mem::size_of::<f32>())
                    as *const c_void;

                unsafe {
                    let status = cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                        b_addr as u64,
                        src_ptr as u64,
                        bias_size_bytes as usize,
                        *stream.cu_stream() as *mut _,
                    );
                    if status != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                        return Err(CudaError::Transfer(format!(
                            "cuMemcpyDtoDAsync failed for Wb gate {}: {:?}",
                            onnx_gate, status
                        )));
                    }
                }

                // Recurrence bias (Rb): linLayerID 4-7
                let cudnn_lin_id_r = ONNX_TO_CUDNN_GATE[onnx_gate] + 4;

                let mut m_addr_r: *mut c_void = ptr::null_mut();
                let mut b_addr_r: *mut c_void = ptr::null_mut();

                let (wb_ptr2, _wb_guard2) = weight_buffer.device_ptr_mut(stream);
                unsafe {
                    check_cudnn(sys::cudnnGetRNNWeightParams(
                        handle.raw(),
                        rnn_desc.raw(),
                        pseudo_layer,
                        weight_space_size,
                        wb_ptr2 as *const c_void,
                        cudnn_lin_id_r,
                        m_desc.raw(),
                        &mut m_addr_r,
                        b_desc.raw(),
                        &mut b_addr_r,
                    ))?;
                }

                // Rb[gate]: offset = dir * 8*hidden + (4 + onnx_gate) * hidden
                let rb_offset_floats =
                    dir * 8 * hidden_size + (4 + onnx_gate) * hidden_size;

                let (b_ptr2, _b_guard2) = b_tensor.data_f32()?.device_ptr(stream);
                let src_ptr_r = (b_ptr2 as usize
                    + rb_offset_floats * std::mem::size_of::<f32>())
                    as *const c_void;

                unsafe {
                    let status = cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                        b_addr_r as u64,
                        src_ptr_r as u64,
                        bias_size_bytes as usize,
                        *stream.cu_stream() as *mut _,
                    );
                    if status != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                        return Err(CudaError::Transfer(format!(
                            "cuMemcpyDtoDAsync failed for Rb gate {}: {:?}",
                            onnx_gate, status
                        )));
                    }
                }
            }
        }
    }

    Ok(PackedLstmWeights {
        buffer: weight_buffer,
        size: weight_space_size,
        rnn_desc,
    })
}
```

**Design notes:**
- The `cuMemcpyDtoDAsync_v2` calls are device-to-device async copies on the same stream. This is the same pattern the executor uses for weight uploads elsewhere.
- Each `cudnnGetRNNWeightParams` call returns a pointer *within* the packed weight buffer. We copy the ONNX data to that pointer.
- The weight buffer must be borrowed mutably for `device_ptr_mut` but we also need read-only access to the source tensors. Because these are separate GPU allocations, the borrows don't conflict.
- Checking `cudarc::driver::sys::CUresult` directly follows the same pattern as the rest of the codebase for low-level CUDA calls.

- [ ] **Step 4: Run clippy to verify**

Run: `cargo clippy --features cuda --tests -- -D warnings`
Expected: Clippy clean. The function compiles but is not yet called from anywhere -- that comes in Tasks 4 and 6.

- [ ] **Step 5: Commit**

```bash
git add src/cuda/cudnn/rnn.rs
git commit -S -m "Add LSTM weight repacking from ONNX IOFC to cuDNN IFCO gate order

Implement pack_lstm_weights_for_cudnn that:
1. Creates an RNN descriptor for the LSTM configuration
2. Queries cuDNN for packed weight buffer size
3. Allocates and zero-initializes the GPU buffer
4. For each direction and gate, queries cudnnGetRNNWeightParams for
   the destination offset, then copies the ONNX weight slice using
   async device-to-device memcpy

Gate reordering: ONNX [I,O,F,C] -> cuDNN [I,F,G(=C),O]
Bias handling: ONNX [Wb_iofc, Rb_iofc] split into per-gate biases
for both input (linLayerID 0-3) and recurrence (linLayerID 4-7).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: cudnn_lstm_forward Function

**Files:**
- Modify: `src/cuda/cudnn/rnn.rs`

This task adds the main forward pass function that calls `cudnnRNNForward`.

- [ ] **Step 1: Add the cudnn_lstm_forward function**

Append to `rnn.rs`:

```rust
// =============================================================================
// Forward Pass
// =============================================================================

/// Execute a fused LSTM forward pass using cuDNN.
///
/// Processes the entire sequence in a single cuDNN call, replacing per-timestep
/// cuBLAS GEMMs + custom CUDA kernels. For Kokoro's bidirectional LSTM with
/// seq_len=95, this reduces ~570 kernel launches to 1 call.
///
/// Returns Y with shape [seq_len, batch, num_directions * hidden_size].
/// The output layout matches ONNX LSTM's default seq_major format with
/// directions interleaved in the last dimension.
pub fn cudnn_lstm_forward(
    handle: &CudnnHandle,
    ctx: &IconnxCudaContext,
    workspace: &mut Workspace,
    packed_weights: &PackedLstmWeights,
    x: &GpuTensor,             // [seq_len, batch, input_size]
    h_init: Option<&GpuTensor>, // [num_directions, batch, hidden_size]
    c_init: Option<&GpuTensor>, // [num_directions, batch, hidden_size]
    hidden_size: usize,
    num_directions: usize,
) -> Result<GpuTensor, CudaError> {
    let x_shape = x.shape();

    // Handle 2D input [seq_len, input_size] by treating as batch=1
    let (seq_len, batch_size, input_size) = if x_shape.len() == 2 {
        (x_shape[0], 1, x_shape[1])
    } else if x_shape.len() == 3 {
        (x_shape[0], x_shape[1], x_shape[2])
    } else {
        return Err(CudaError::Cudnn(format!(
            "LSTM input must be 2D or 3D, got {:?}",
            x_shape
        )));
    };

    let output_size = num_directions * hidden_size;

    // Create data descriptors for input X and output Y
    let x_desc = RnnDataDescriptor::new(seq_len, batch_size, input_size)?;
    let y_desc = RnnDataDescriptor::new(seq_len, batch_size, output_size)?;

    // Create tensor descriptors for hidden/cell state: [num_layers=1, num_dirs*batch, hidden]
    // cuDNN expects h/c descriptors as 3D: [numLayers * numDirs, batch, hidden]
    let mut h_desc = TensorDescriptor::new()?;
    let h_dims = [num_directions, batch_size, hidden_size];
    let h_strides = [
        batch_size * hidden_size,
        hidden_size,
        1,
    ];
    h_desc.set_nd(&h_dims, &h_strides, sys::cudnnDataType_t::CUDNN_DATA_FLOAT)?;

    let mut c_desc = TensorDescriptor::new()?;
    c_desc.set_nd(&h_dims, &h_strides, sys::cudnnDataType_t::CUDNN_DATA_FLOAT)?;

    // Query workspace size
    let mut work_space_size: usize = 0;
    let mut reserve_space_size: usize = 0;
    unsafe {
        check_cudnn(sys::cudnnGetRNNTempSpaceSizes(
            handle.raw(),
            packed_weights.rnn_desc().raw(),
            sys::cudnnForwardMode_t::CUDNN_FWD_MODE_INFERENCE,
            x_desc.raw(),
            &mut work_space_size,
            &mut reserve_space_size,
        ))?;
    }

    // Ensure workspace is large enough
    workspace.ensure_size(ctx, work_space_size)?;

    // Allocate output Y: [seq_len, batch, num_directions * hidden_size]
    let mut y = GpuTensor::zeros_f32(ctx, vec![seq_len, batch_size, output_size])?;

    // Create seq_lengths array on device (all same length)
    let seq_lengths_host: Vec<i32> = vec![seq_len as i32; batch_size];
    let seq_lengths_dev: CudaSlice<i32> = ctx
        .stream()
        .htod_copy(seq_lengths_host)
        .map_err(|e| CudaError::Transfer(format!("Failed to copy seq_lengths: {}", e)))?;

    // Execute forward pass
    let stream = ctx.stream();

    let (x_ptr, _x_guard) = x.data_f32()?.device_ptr(stream);
    let (y_ptr, _y_guard) = y.data_f32_mut()?.device_ptr_mut(stream);
    let (w_ptr, _w_guard) = packed_weights.buffer.device_ptr(stream);
    let workspace_ptr = workspace.ptr(stream);
    let (seq_dev_ptr, _seq_guard) = seq_lengths_dev.device_ptr(stream);

    // Get h_init and c_init raw pointers (null if not provided)
    let (hx_ptr, _hx_guard) = if let Some(h) = h_init {
        let (ptr, guard) = h.data_f32()?.device_ptr(stream);
        (ptr as *const c_void, Some(guard))
    } else {
        (ptr::null(), None)
    };

    let (cx_ptr, _cx_guard) = if let Some(c) = c_init {
        let (ptr, guard) = c.data_f32()?.device_ptr(stream);
        (ptr as *const c_void, Some(guard))
    } else {
        (ptr::null(), None)
    };

    unsafe {
        check_cudnn(sys::cudnnRNNForward(
            handle.raw(),
            packed_weights.rnn_desc().raw(),
            sys::cudnnForwardMode_t::CUDNN_FWD_MODE_INFERENCE,
            seq_dev_ptr as *const i32,
            x_desc.raw(),
            x_ptr as *const c_void,
            y_desc.raw(),
            y_ptr as *mut c_void,
            h_desc.raw(),
            hx_ptr,
            ptr::null_mut(), // hy: don't need final hidden state
            c_desc.raw(),
            cx_ptr,
            ptr::null_mut(), // cy: don't need final cell state
            packed_weights.size(),
            w_ptr as *const c_void,
            work_space_size,
            workspace_ptr,
            0,               // reserveSpaceSize: 0 for inference
            ptr::null_mut(), // reserveSpace: null for inference
        ))?;
    }

    // cuDNN output Y is [seq_len, batch, num_directions * hidden_size]
    // but the existing gpu_lstm returns [seq_len, num_directions, batch, hidden_size]
    // Reshape to match the expected output format
    let y_reshaped = y
        .reshape(vec![seq_len, batch_size, num_directions, hidden_size])
        .ok_or_else(|| CudaError::Cudnn("Failed to reshape Y for direction split".into()))?;

    // Transpose from [seq, batch, dirs, hidden] to [seq, dirs, batch, hidden]
    // This matches the ONNX LSTM output layout that gpu_lstm produces
    // We need to produce [seq_len, num_directions, batch, hidden_size]
    //
    // For now, since cuDNN gives us [seq, batch, dirs*hidden], and we need
    // [seq, dirs, batch, hidden], we must rearrange. With batch=1 (Kokoro's
    // case), [seq, 1, dirs, hidden] == [seq, dirs, 1, hidden] so reshape alone
    // suffices. For batch>1 a transpose kernel would be needed.
    if batch_size == 1 {
        Ok(y_reshaped
            .reshape(vec![seq_len, num_directions, batch_size, hidden_size])
            .ok_or_else(|| {
                CudaError::Cudnn("Failed to reshape Y to [seq, dirs, batch, hidden]".into())
            })?)
    } else {
        // General case: need actual data movement for transposition
        // [seq, batch, dirs, hidden] -> [seq, dirs, batch, hidden]
        // This requires a GPU transpose; use the existing gpu_transpose_nd
        // infrastructure. Import and call it here.
        //
        // For Kokoro all LSTMs have batch=1, so this path is not exercised.
        // If reached, fall back to the reshape (which is incorrect for batch>1)
        // and flag it.
        Err(CudaError::Cudnn(format!(
            "cuDNN LSTM with batch_size={} > 1 not yet supported (needs transpose kernel)",
            batch_size
        )))
    }
}
```

**Design notes:**
- The `devSeqLengths` parameter must be a device-side array. We allocate a small `CudaSlice<i32>` for this.
- We pass `hy=NULL` and `cy=NULL` because the ONNX spec's Y output is the full hidden sequence, not the final hidden/cell states. Kokoro doesn't use Y_h or Y_c outputs.
- The output reshape handles Kokoro's batch=1 case efficiently (no data movement). A future task could add a general transpose for batch>1.
- The workspace is shared with convolution operations (same `Workspace` struct). `ensure_size` only reallocates if the new requirement exceeds the current allocation.

- [ ] **Step 2: Run clippy to verify**

Run: `cargo clippy --features cuda --tests -- -D warnings`
Expected: Clippy clean. The function is not yet called from the executor -- that comes in Task 6.

- [ ] **Step 3: Commit**

```bash
git add src/cuda/cudnn/rnn.rs
git commit -S -m "Add cudnn_lstm_forward for fused sequence-level LSTM inference

Implement the main forward pass using cudnnRNNForward that processes
the entire sequence in a single call. Takes packed weights from
pack_lstm_weights_for_cudnn, input X, optional h_init/c_init, and
returns Y in [seq, dirs, batch, hidden] layout matching gpu_lstm.

For batch=1 (all Kokoro LSTMs), the cuDNN output [seq, batch, dirs*H]
is reshaped to [seq, dirs, batch, H] with zero data movement. Returns
an error for batch>1 which would require a transpose kernel.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: GPU Integration Test

**Files:**
- Modify: `src/cuda/cudnn/rnn.rs` (add test module)

This task adds an integration test that runs both `gpu_lstm` and `cudnn_lstm_forward` on the same synthetic inputs and verifies outputs match within tolerance.

- [ ] **Step 1: Add test module to rnn.rs**

Append to the end of `rnn.rs`:

```rust
// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::lstm::LstmKernelCache;
    use crate::cuda::memory_pool::GpuMemoryPool;
    use crate::cuda::ops::OpsKernelCache;

    /// Helper: create a GpuTensor from a Vec<f32> with given shape
    fn gpu_from_vec(
        ctx: &IconnxCudaContext,
        data: Vec<f32>,
        shape: Vec<usize>,
    ) -> GpuTensor {
        GpuTensor::from_host_f32(ctx, &data, shape).expect("Failed to create GPU tensor")
    }

    /// Helper: download GPU tensor to host
    fn to_host(ctx: &IconnxCudaContext, t: &GpuTensor) -> Vec<f32> {
        let stream = ctx.stream();
        let data = t.data_f32().expect("Not f32");
        stream
            .dtoh_sync_copy(data)
            .expect("Failed to download tensor")
    }

    #[test]
    #[ignore = "requires CUDA GPU with cuDNN"]
    fn test_cudnn_lstm_matches_gpu_lstm() {
        // Small synthetic LSTM: seq_len=3, batch=1, input_size=4, hidden=4, bidirectional
        let ctx = IconnxCudaContext::new().expect("Failed to create CUDA context");
        let handle = CudnnHandle::new().expect("Failed to create cuDNN handle");
        handle
            .set_stream(ctx.stream())
            .expect("Failed to set stream");

        let seq_len = 3;
        let batch = 1;
        let input_size = 4;
        let hidden = 4;
        let num_dirs = 2;

        // Create deterministic inputs
        let x_data: Vec<f32> = (0..seq_len * batch * input_size)
            .map(|i| (i as f32) * 0.1 - 0.5)
            .collect();
        let x = gpu_from_vec(&ctx, x_data, vec![seq_len, batch, input_size]);

        // Weights: W [num_dirs, 4*hidden, input_size]
        let w_data: Vec<f32> = (0..num_dirs * 4 * hidden * input_size)
            .map(|i| ((i as f32) * 0.02 - 0.5).sin() * 0.1)
            .collect();
        let w = gpu_from_vec(
            &ctx,
            w_data,
            vec![num_dirs, 4 * hidden, input_size],
        );

        // Recurrence: R [num_dirs, 4*hidden, hidden]
        let r_data: Vec<f32> = (0..num_dirs * 4 * hidden * hidden)
            .map(|i| ((i as f32) * 0.03 + 0.7).cos() * 0.1)
            .collect();
        let r = gpu_from_vec(
            &ctx,
            r_data,
            vec![num_dirs, 4 * hidden, hidden],
        );

        // Bias: B [num_dirs, 8*hidden]
        let b_data: Vec<f32> = (0..num_dirs * 8 * hidden)
            .map(|i| (i as f32) * 0.01 - 0.3)
            .collect();
        let b = gpu_from_vec(&ctx, b_data, vec![num_dirs, 8 * hidden]);

        // Initial hidden state: h_init [num_dirs, batch, hidden]
        let h_data: Vec<f32> = vec![0.0; num_dirs * batch * hidden];
        let h_init = gpu_from_vec(&ctx, h_data, vec![num_dirs, batch, hidden]);

        // Run gpu_lstm (reference implementation)
        let lstm_cache = LstmKernelCache::new(&ctx).expect("Failed to create LSTM cache");
        let ops_cache = OpsKernelCache::new(&ctx).expect("Failed to create ops cache");
        let mut pool = GpuMemoryPool::new();

        let reference_output = crate::cuda::lstm::gpu_lstm(
            &ctx,
            &lstm_cache,
            &ops_cache,
            &mut pool,
            &x,
            &w,
            &r,
            Some(&b),
            Some(&h_init),
            None, // c_init
        )
        .expect("gpu_lstm failed");

        // Run cudnn_lstm_forward
        let mut workspace = Workspace::new();
        let packed = pack_lstm_weights_for_cudnn(
            &handle,
            &ctx,
            &w,
            &r,
            Some(&b),
            hidden,
            input_size,
            num_dirs,
        )
        .expect("Weight packing failed");

        let cudnn_output = cudnn_lstm_forward(
            &handle,
            &ctx,
            &mut workspace,
            &packed,
            &x,
            Some(&h_init),
            None, // c_init
            hidden,
            num_dirs,
        )
        .expect("cudnn_lstm_forward failed");

        // Compare shapes
        assert_eq!(
            reference_output.shape(),
            cudnn_output.shape(),
            "Output shapes must match: ref={:?}, cudnn={:?}",
            reference_output.shape(),
            cudnn_output.shape()
        );

        // Compare values
        let ref_data = to_host(&ctx, &reference_output);
        let cudnn_data = to_host(&ctx, &cudnn_output);

        // Tolerance: 1e-4 because float accumulation order differs between
        // per-timestep cuBLAS GEMMs and cuDNN's fused implementation.
        // On long sequences the divergence grows, but for seq_len=3 it
        // should be well within this bound.
        let tolerance = 1e-4;
        for (i, (r_val, c_val)) in ref_data.iter().zip(cudnn_data.iter()).enumerate() {
            let diff = (r_val - c_val).abs();
            assert!(
                diff < tolerance,
                "Mismatch at index {}: ref={}, cudnn={}, diff={} (tolerance={})",
                i, r_val, c_val, diff, tolerance
            );
        }

        println!(
            "cuDNN LSTM matches gpu_lstm within {} tolerance. Shape: {:?}",
            tolerance,
            cudnn_output.shape()
        );
    }

    #[test]
    #[ignore = "requires CUDA GPU with cuDNN"]
    fn test_cudnn_lstm_unidirectional() {
        // Verify unidirectional LSTM works (num_directions=1)
        let ctx = IconnxCudaContext::new().expect("Failed to create CUDA context");
        let handle = CudnnHandle::new().expect("Failed to create cuDNN handle");
        handle
            .set_stream(ctx.stream())
            .expect("Failed to set stream");

        let seq_len = 3;
        let batch = 1;
        let input_size = 4;
        let hidden = 4;
        let num_dirs = 1;

        let x_data: Vec<f32> = (0..seq_len * batch * input_size)
            .map(|i| (i as f32) * 0.1)
            .collect();
        let x = gpu_from_vec(&ctx, x_data, vec![seq_len, batch, input_size]);

        let w_data: Vec<f32> = (0..num_dirs * 4 * hidden * input_size)
            .map(|i| ((i as f32) * 0.02).sin() * 0.1)
            .collect();
        let w = gpu_from_vec(&ctx, w_data, vec![num_dirs, 4 * hidden, input_size]);

        let r_data: Vec<f32> = (0..num_dirs * 4 * hidden * hidden)
            .map(|i| ((i as f32) * 0.03).cos() * 0.1)
            .collect();
        let r = gpu_from_vec(&ctx, r_data, vec![num_dirs, 4 * hidden, hidden]);

        let mut workspace = Workspace::new();
        let packed = pack_lstm_weights_for_cudnn(
            &handle, &ctx, &w, &r, None, hidden, input_size, num_dirs,
        )
        .expect("Weight packing failed");

        let output = cudnn_lstm_forward(
            &handle, &ctx, &mut workspace, &packed, &x, None, None,
            hidden, num_dirs,
        )
        .expect("cudnn_lstm_forward failed");

        assert_eq!(
            output.shape(),
            &[seq_len, num_dirs, batch, hidden],
            "Unidirectional output shape mismatch"
        );

        // Verify output is not all zeros (sanity check that computation ran)
        let data = to_host(&ctx, &output);
        let max_val = data.iter().cloned().fold(0.0f32, f32::max);
        assert!(
            max_val > 1e-6,
            "Output appears to be all zeros -- computation may not have run"
        );

        println!(
            "Unidirectional cuDNN LSTM OK. Shape: {:?}, max: {}",
            output.shape(),
            max_val
        );
    }
}
```

- [ ] **Step 2: Run the tests**

```bash
cargo test --features cuda -- test_cudnn_lstm --ignored --nocapture
```

Expected: Both tests pass. The bidirectional test verifies numerical equivalence with `gpu_lstm`; the unidirectional test verifies the code path works and produces non-zero output.

If the bidirectional test fails with tolerance issues, increase tolerance to 1e-3 and add a comment explaining the divergence source.

- [ ] **Step 3: Run clippy**

Run: `cargo clippy --features cuda --tests -- -D warnings`

- [ ] **Step 4: Commit**

```bash
git add src/cuda/cudnn/rnn.rs
git commit -S -m "Add GPU integration tests for cuDNN fused LSTM

Two tests verify the cudnn_lstm_forward implementation:
1. test_cudnn_lstm_matches_gpu_lstm: bidirectional LSTM (seq=3, batch=1,
   hidden=4) comparing cuDNN output against gpu_lstm reference within
   1e-4 tolerance (float accumulation order differs between per-timestep
   cuBLAS and cuDNN fused path)
2. test_cudnn_lstm_unidirectional: sanity check that unidirectional
   LSTM produces non-zero output with correct shape

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: LstmWeightCache and Executor Wiring

**Files:**
- Modify: `src/cuda/inference/mod.rs`

This task adds the weight cache to the executor and replaces the `gpu_lstm` call with `cudnn_lstm_forward`.

- [ ] **Step 1: Add import for cuDNN LSTM types**

At the import block near line 11, update the cuDNN import line to include the new types:

**Before:**
```rust
use super::cudnn::{
    cudnn_conv_1d, cudnn_conv_transpose_1d, ConvAlgoCache, CudnnHandle,
    Workspace as CudnnWorkspace,
};
```

**After:**
```rust
use super::cudnn::{
    cudnn_conv_1d, cudnn_conv_transpose_1d, cudnn_lstm_forward, pack_lstm_weights_for_cudnn,
    ConvAlgoCache, CudnnHandle, PackedLstmWeights, Workspace as CudnnWorkspace,
};
```

- [ ] **Step 2: Add LstmWeightCache struct**

After the existing `use` block and before `pub mod fusion;` (line 48), add:

```rust
/// Cache for cuDNN packed LSTM weights, keyed by (W_ptr, R_ptr) device addresses.
///
/// Using raw device pointers as keys: the same ONNX weight tensor always has
/// the same GPU address within an executor's lifetime (weights are loaded once
/// at construction and never moved). This avoids expensive tensor comparisons.
///
/// **Invariant:** The cache is valid only while the original W/R tensors are
/// alive. Since the executor owns both the weights and this cache, the
/// invariant is trivially maintained.
struct LstmWeightCache {
    cache: HashMap<(usize, usize), PackedLstmWeights>,
}

impl LstmWeightCache {
    fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }
}
```

- [ ] **Step 3: Add `lstm_weight_cache` field to `GpuGraphExecutor`**

In the struct definition (line 218), after the `conv_algo_cache` field (line 236), add:

```rust
    /// cuDNN packed LSTM weight cache (RefCell for interior mutability)
    lstm_weight_cache: RefCell<LstmWeightCache>,
```

- [ ] **Step 4: Initialize the cache in `new_with_profiling`**

In the struct initialization at the `Ok(Self { ... })` block, after `conv_algo_cache,` add:

```rust
            lstm_weight_cache: RefCell::new(LstmWeightCache::new()),
```

- [ ] **Step 5: Replace LSTM dispatch**

At the LSTM dispatch site (lines 1676-1706), replace the entire block:

**Before:**
```rust
            "LSTM" => {
                let w = inputs[1]; // Weight
                let r = inputs[2]; // Recurrence
                let b = if inputs.len() > 3 {
                    Some(inputs[3])
                } else {
                    None
                };
                let h_init = if inputs.len() > 5 {
                    Some(inputs[5])
                } else {
                    None
                };
                let c_init = if inputs.len() > 6 {
                    Some(inputs[6])
                } else {
                    None
                };
                gpu_lstm(
                    &self.ctx,
                    &self.lstm_kernels,
                    &self.ops_kernels,
                    &mut pool,
                    inputs[0],
                    w,
                    r,
                    b,
                    h_init,
                    c_init,
                )
            }
```

**After:**
```rust
            "LSTM" => {
                let w = inputs[1]; // Weight
                let r = inputs[2]; // Recurrence
                let b = if inputs.len() > 3 {
                    Some(inputs[3])
                } else {
                    None
                };
                let h_init = if inputs.len() > 5 {
                    Some(inputs[5])
                } else {
                    None
                };
                let c_init = if inputs.len() > 6 {
                    Some(inputs[6])
                } else {
                    None
                };

                let w_shape = w.shape();
                let num_directions = w_shape[0];
                let hidden_size = w_shape[1] / 4;
                let input_size = w_shape[2];

                // Get or create packed weights (cached by W/R device pointer pair)
                let w_ptr = w.device_ptr(&self.ctx) as usize;
                let r_ptr = r.device_ptr(&self.ctx) as usize;
                let cache_key = (w_ptr, r_ptr);

                let mut weight_cache = self.lstm_weight_cache.borrow_mut();
                if !weight_cache.cache.contains_key(&cache_key) {
                    let packed = pack_lstm_weights_for_cudnn(
                        &self.cudnn_handle,
                        &self.ctx,
                        w,
                        r,
                        b,
                        hidden_size,
                        input_size,
                        num_directions,
                    )?;
                    weight_cache.cache.insert(cache_key, packed);
                }
                let packed = weight_cache.cache.get(&cache_key).unwrap();

                let mut workspace = self.cudnn_workspace.borrow_mut();
                cudnn_lstm_forward(
                    &self.cudnn_handle,
                    &self.ctx,
                    &mut workspace,
                    packed,
                    inputs[0],
                    h_init,
                    c_init,
                    hidden_size,
                    num_directions,
                )
            }
```

- [ ] **Step 6: Run clippy and full test suite**

```bash
cargo clippy --features cuda --tests -- -D warnings
cargo test --features cuda
```

Expected: All existing tests pass. The executor now uses cuDNN LSTM instead of per-timestep gpu_lstm.

- [ ] **Step 7: Commit**

```bash
git add src/cuda/inference/mod.rs
git commit -S -m "Wire cuDNN fused LSTM through executor with weight caching

Add LstmWeightCache to GpuGraphExecutor that maps (W_ptr, R_ptr)
device pointer pairs to PackedLstmWeights. On first LSTM call per
unique weight pair, weights are repacked from ONNX IOFC to cuDNN
IFCO and cached. Subsequent calls with the same weights skip repacking.

Replace gpu_lstm dispatch with cudnn_lstm_forward, reducing kernel
launches from 3*seq_len*num_directions per LSTM node to 1.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Kokoro Integration Verification

**Files:** None modified -- this is a verification-only task.

- [ ] **Step 1: Run the Kokoro integration test with profiling**

```bash
cargo test --features cuda --release -- kokoro --nocapture 2>&1 | grep -E "LSTM|Total|Fused_AddMulAdd"
```

**Expected results:**
1. LSTM GPU time drops significantly from ~12.6ms baseline (target: ~2-3ms)
2. `Fused_AddMulAdd: 70` unchanged (no regression on earlier fusion cycles)
3. All tests pass with identical numerical results (the Kokoro test compares output against a reference)

- [ ] **Step 2: Verify no regressions in other operations**

```bash
cargo test --features cuda --release -- kokoro --nocapture 2>&1 | grep -E "Conv|ConvTranspose|STFT|MatMul"
```

Other operation times should be unchanged or within normal variance.

- [ ] **Step 3: Record results**

Record the LSTM timing, total inference time, and fusion counts for Task 8.

If the test fails:
- Check if the numerical output diverged beyond the test's tolerance
- Look for shape mismatches in the error message
- Verify the weight packing is correct by comparing a small LSTM's packed buffer against manually-computed expected values
- If cuDNN returns `NOT_SUPPORTED`, check whether the GPU architecture or cuDNN version supports `CUDNN_RNN_ALGO_STANDARD` for the given LSTM configuration

---

## Task 8: Results Documentation

**Files:**
- Create: `docs/superpowers/results/2026-04-12-cudnn-fused-lstm-results.md`

- [ ] **Step 1: Write results document**

Template:

```markdown
# Cycle 7: cuDNN Fused LSTM Results

**Date:** 2026-04-12
**Branch:** `cycle7/cudnn-fused-lstm`

## Performance

| Metric | Before | After | Improvement |
|---|---|---|---|
| LSTM GPU time | ~12.6ms | TBDms | TBD% |
| Total inference time | TBDms | TBDms | TBD% |
| LSTM kernel launches (6 nodes) | ~720 | 6 | 99.2% reduction |

## Verification

- [ ] `cargo test --features cuda` -- all tests pass
- [ ] `cargo clippy --features cuda --tests -- -D warnings` -- zero warnings
- [ ] Kokoro output matches reference
- [ ] `Fused_AddMulAdd: 70` unchanged
- [ ] New cuDNN LSTM tests pass (`test_cudnn_lstm_matches_gpu_lstm`, `test_cudnn_lstm_unidirectional`)

## Implementation summary

- `src/cuda/cudnn/sys.rs`: +~120 lines (7 enums, 3 handle types, 15 functions)
- `src/cuda/cudnn/rnn.rs`: new file (~450 lines)
- `src/cuda/cudnn/mod.rs`: +2 lines
- `src/cuda/inference/mod.rs`: +~40 lines (LstmWeightCache, dispatch replacement)
```

- [ ] **Step 2: Fill in actual numbers from Task 7 output**

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/results/2026-04-12-cudnn-fused-lstm-results.md
git commit -S -m "Add Cycle 7 cuDNN fused LSTM results

Document performance improvement from replacing per-timestep cuBLAS
GEMM + custom CUDA kernel LSTM with cuDNN fused cudnnRNNForward.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Verification Checklist

After all 8 tasks are complete, verify:

- [ ] `cargo clippy --features cuda --tests -- -D warnings` -- zero warnings
- [ ] `cargo test --features cuda` -- all tests pass
- [ ] `cargo test --features cuda -- test_cudnn_lstm --ignored --nocapture` -- cuDNN LSTM tests pass
- [ ] `cargo test --features cuda --release -- kokoro --nocapture` -- LSTM time drops, fusion counts unchanged
- [ ] `git log --oneline -6` -- signed commits (approximately 1 per task)
- [ ] `sys.rs` stays under 1000 lines (was 527, adding ~120 = ~647)
- [ ] `rnn.rs` is under 1000 lines (~450 lines)
- [ ] `mod.rs` stays under 1000 lines (was 982, adding 2 = 984)
- [ ] `inference/mod.rs` line count tracked (pre-existing > 1000 lines; flag but don't refactor)
- [ ] Old `gpu_lstm` function retained in `src/cuda/lstm.rs` as fallback reference
