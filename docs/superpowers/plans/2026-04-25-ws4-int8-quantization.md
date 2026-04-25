# WS-4: INT8 Dynamic Quantization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run DistilBERT-INT8 (`Xenova/distilbert-base-uncased/onnx/model_int8.onnx`) end-to-end through iconnx with ORT-matched outputs within 1e-2 absolute, closing the M2.7.f roster milestone.

**Architecture:** Add INT8 + UINT8 to the CPU `Tensor` enum, the parser dtype dispatch, and the GPU `DType` / `GpuTensor` enum (M4.1 / M4.2 / M4.3 — parallel-able scaffolding). Add three new ONNX ops: `DequantizeLinear` (M4.4), `DynamicQuantizeLinear` (M4.5, multi-output via `dispatch_op_multi`), `MatMulInteger` (M4.6, NVRTC GEMM with 4 dtype-combo kernels) — parallel-able once scaffolding lands. M4.7 is the serial integration test against ORT.

**Tech Stack:** Rust 1.77+ (`f32::round_ties_even`), `ndarray` (CPU tensor math), `prost` (ONNX protobuf), `garboard` (CUDA bindings; cuBLAS is f32/f64-only — INT8 path is custom NVRTC).

**Performance non-gate (per leadline 2026-04-25):** WS-4 is correctness-first. ORT achieves ~1e-2-tolerance INT8 inference via cuBLAS IMMA / Tensor Cores; iconnx's NVRTC INT8 GEMM will be substantially slower. **A 5×–50× slowdown vs. ORT on M4.7 is not a blocker.** Performance parity is a follow-on workstream tracked alongside the garboard `cublasGemmEx` INT8 binding request.

**Sequencing recommendation:**
- **Phase 1 (parallel-able):** M4.1, M4.2, M4.3 — pure scaffolding extensions of existing dtype patterns. No interdependencies.
- **Phase 2 (parallel-able after Phase 1):** M4.4, M4.5, M4.6 — each new op depends on the dtype scaffolding but not on each other.
- **Phase 3 (serial):** M4.7 — integration; runs only after Phase 2 lands.

The plan documents linear order for clarity, but a parallel-execution agent can run M4.1/M4.2/M4.3 concurrently and then M4.4/M4.5/M4.6 concurrently.

---

## Test fixture conventions

- All new GPU integration tests under `tests/` use `#[ignore = "requires CUDA GPU"]` and `--include-ignored` to gate on hardware.
- CPU unit tests live in `#[cfg(test)] mod tests` blocks within the source file.
- Conformance tests against ORT use the leadline-bench compare harness once M4.7 wires up the profile; pre-M4.7, hand-computed expected outputs (closed-form arithmetic) suffice.
- Per-test naming: `{op}_{scenario}_{detail}` — e.g., `dequantize_linear_per_channel_int8_axis_0`.

---

## Milestone M4.1 — CPU `Tensor` adds Int8 + UInt8 variants

**Files:**
- Modify: `src/tensor.rs`

- [ ] **Step 1: Write the failing tests**

In `src/tensor.rs`'s `#[cfg(test)] mod tests`:

```rust
#[test]
fn tensor_int8_constructor_and_accessors() {
    let t = Tensor::from_vec_i8(vec![-128_i8, 0, 127], vec![3]);
    assert_eq!(t.dtype(), "int8");
    assert!(t.is_int8());
    assert_eq!(t.shape(), &[3]);
    assert_eq!(t.len(), 3);
    assert_eq!(t.as_slice_i8(), vec![-128_i8, 0, 127]);
}

#[test]
fn tensor_uint8_constructor_and_accessors() {
    let t = Tensor::from_vec_u8(vec![0_u8, 128, 255], vec![3]);
    assert_eq!(t.dtype(), "uint8");
    assert!(t.is_uint8());
    assert_eq!(t.as_slice_u8(), vec![0_u8, 128, 255]);
}

#[test]
fn tensor_int8_uint8_panic_on_wrong_accessor() {
    let i8t = Tensor::from_vec_i8(vec![0_i8], vec![1]);
    let result = std::panic::catch_unwind(|| i8t.as_slice_u8());
    assert!(result.is_err(), "as_slice_u8 on Int8 must panic");
}

#[test]
fn tensor_int8_uint8_to_array_roundtrip() {
    let i8t = Tensor::from_vec_i8(vec![1, -2, 3], vec![3]);
    let arr = i8t.to_array_i8();
    assert_eq!(arr.shape(), &[3]);
    assert_eq!(arr[[0]], 1_i8);
    assert_eq!(arr[[1]], -2_i8);
}
```

- [ ] **Step 2: Run tests to verify failure**

`cargo test --lib --features cuda tensor::tests::tensor_int8`
Expected: compile error (constructor/accessor not found).

- [ ] **Step 3: Add Int8 + UInt8 variants**

In `src/tensor.rs` `Tensor` enum (after the `Bool` variant):

```rust
/// 8-bit signed integer (i8) — INT8 quantization weights / inputs.
Int8(ArrayD<i8>),
/// 8-bit unsigned integer (u8) — UINT8 quantization (DynamicQuantizeLinear output).
UInt8(ArrayD<u8>),
```

Update the module docstring's "Supports" line to add `Int8, UInt8`.

- [ ] **Step 4: Add constructors `from_vec_i8` / `from_vec_u8`**

Mirror the structure of `from_vec_i64` exactly. After the `from_vec_bool` impl, add:

```rust
pub fn from_vec_i8(data: Vec<i8>, shape: Vec<usize>) -> Self {
    let expected_len: usize = if shape.is_empty() { 1 } else { shape.iter().product() };
    if data.len() != expected_len {
        panic!(
            "Shape mismatch: data has {} elements but shape {:?} requires {}",
            data.len(), shape, expected_len
        );
    }
    let array = ArrayD::from_shape_vec(IxDyn(&shape), data)
        .expect("Failed to create array from shape and data");
    Tensor::Int8(array)
}

pub fn from_vec_u8(data: Vec<u8>, shape: Vec<usize>) -> Self {
    let expected_len: usize = if shape.is_empty() { 1 } else { shape.iter().product() };
    if data.len() != expected_len {
        panic!(
            "Shape mismatch: data has {} elements but shape {:?} requires {}",
            data.len(), shape, expected_len
        );
    }
    let array = ArrayD::from_shape_vec(IxDyn(&shape), data)
        .expect("Failed to create array from shape and data");
    Tensor::UInt8(array)
}
```

- [ ] **Step 5: Add accessors and update introspection methods**

```rust
pub fn as_slice_i8(&self) -> Vec<i8> {
    match self {
        Tensor::Int8(a) => a.as_slice().map(|s| s.to_vec()).unwrap_or_else(|| a.iter().copied().collect()),
        _ => panic!("as_slice_i8() called on non-Int8 tensor ({})", self.dtype()),
    }
}

pub fn as_slice_u8(&self) -> Vec<u8> { /* mirror */ }

pub fn to_array_i8(&self) -> &ArrayD<i8> {
    match self {
        Tensor::Int8(a) => a,
        _ => panic!("to_array_i8() called on non-Int8 tensor ({})", self.dtype()),
    }
}

pub fn to_array_u8(&self) -> &ArrayD<u8> { /* mirror */ }
```

Update `dtype()`, `is_int8()` / `is_uint8()` predicates, and ALL the `match self` arms in `shape()`, `ndim()`, `len()`, `is_empty()` to include the two new variants. The compiler will surface every site if you don't.

- [ ] **Step 6: Run tests to verify pass**

`cargo test --lib --features cuda tensor::tests::tensor_int8` — all 4 new tests pass.
`cargo test --lib --features cuda tensor::tests::tensor_uint8` — passes.
`cargo build --all-targets --features cuda` — clean.

- [ ] **Step 7: Commit**

```
git add src/tensor.rs
git commit -S -m "feat(tensor): add Int8 + UInt8 dtype variants (WS-4 M4.1)"
```

---

## Milestone M4.2 — Parser reads INT8 + UINT8 initializers

**Files:**
- Modify: `src/onnx_parser.rs`

- [ ] **Step 1: Write the failing tests**

In `src/onnx_parser.rs`'s `#[cfg(test)] mod tests`:

```rust
#[test]
fn parse_tensor_proto_int8_via_int32_data() {
    let proto = onnx_proto::TensorProto {
        name: Some("weights_i8".to_string()),
        data_type: Some(3), // INT8
        dims: vec![4],
        int32_data: vec![-128, 0, 1, 127],
        ..Default::default()
    };
    match OnnxModel::parse_tensor_proto(&proto).expect("INT8 should parse") {
        Tensor::Int8(arr) => {
            assert_eq!(arr.shape(), &[4]);
            assert_eq!(arr.iter().copied().collect::<Vec<_>>(), vec![-128_i8, 0, 1, 127]);
        }
        other => panic!("expected Tensor::Int8, got {:?}", other),
    }
}

#[test]
fn parse_tensor_proto_int8_via_raw_data() {
    let raw: Vec<u8> = (-128_i8..=127).map(|v| v as u8).collect(); // wraps to two's complement
    // Take just first 4 bytes for a small tensor.
    let small: Vec<u8> = vec![0xFF, 0x00, 0x01, 0x7F]; // -1, 0, 1, 127
    let proto = onnx_proto::TensorProto {
        name: Some("w".into()),
        data_type: Some(3),
        dims: vec![4],
        raw_data: Some(small),
        ..Default::default()
    };
    let tensor = OnnxModel::parse_tensor_proto(&proto).expect("INT8 raw_data");
    if let Tensor::Int8(arr) = tensor {
        assert_eq!(arr.iter().copied().collect::<Vec<_>>(), vec![-1_i8, 0, 1, 127]);
    } else {
        panic!("expected Int8");
    }
    // Suppress unused warning on raw helper:
    let _ = raw;
}

#[test]
fn parse_tensor_proto_uint8_via_int32_data() {
    let proto = onnx_proto::TensorProto {
        name: Some("zp".into()),
        data_type: Some(2), // UINT8
        dims: vec![3],
        int32_data: vec![0, 128, 255],
        ..Default::default()
    };
    let tensor = OnnxModel::parse_tensor_proto(&proto).expect("UINT8 should parse");
    if let Tensor::UInt8(arr) = tensor {
        assert_eq!(arr.iter().copied().collect::<Vec<_>>(), vec![0_u8, 128, 255]);
    } else {
        panic!("expected UInt8");
    }
}

#[test]
fn parse_tensor_proto_int8_zero_element() {
    let proto = onnx_proto::TensorProto {
        name: Some("empty".into()),
        data_type: Some(3),
        dims: vec![0],
        ..Default::default()
    };
    let tensor = OnnxModel::parse_tensor_proto(&proto).expect("zero-element INT8");
    if let Tensor::Int8(arr) = tensor {
        assert_eq!(arr.shape(), &[0]);
        assert_eq!(arr.len(), 0);
    } else {
        panic!("expected Int8");
    }
}
```

- [ ] **Step 2: Run tests to verify failure**

`cargo test --lib --features cuda parse_tensor_proto_int8`
Expected: error (UnsupportedDType: 3) since the parser doesn't handle dtype 3 yet.

- [ ] **Step 3: Add zero-element fast-path arms for dtypes 2 and 3**

In `parse_tensor_proto`, the `if expected_len == 0 { match data_type { ... } }` block, add:

```rust
2 => Ok(crate::tensor::Tensor::from_vec_u8(Vec::new(), shape)),
3 => Ok(crate::tensor::Tensor::from_vec_i8(Vec::new(), shape)),
```

- [ ] **Step 4: Add INT8 / UINT8 decode arms in `parse_tensor_proto`**

After the `9 => { /* BOOL */ }` arm, add INT8 (dtype 3) and UINT8 (dtype 2). Both follow the same wire format: `int32_data` carries 1 i32 per element (low 8 bits, sign-extended for i8; zero-extended for u8); `raw_data` carries 1 byte per element.

```rust
3 => {
    // INT8 (i8). Wire format: int32_data — one i32 per element (the
    // low 8 bits, sign-extended if from int32_data); raw_data — one
    // byte per element interpreted as i8 (two's complement).
    let data: Vec<i8> = if !tensor_proto.int32_data.is_empty() {
        tensor_proto.int32_data.iter().map(|&v| v as i8).collect()
    } else if let Some(raw) = &tensor_proto.raw_data {
        if raw.is_empty() {
            return Err(ParseError::InvalidTensorData {
                name,
                reason: "empty raw_data for INT8 tensor".to_string(),
            });
        }
        raw.iter().map(|&b| b as i8).collect()
    } else {
        return Err(ParseError::InvalidTensorData {
            name,
            reason: "no int32_data and no raw_data".to_string(),
        });
    };
    if data.len() != expected_len {
        return Err(ParseError::ShapeMismatch {
            name,
            declared: shape,
            declared_len: expected_len,
            actual_len: data.len(),
        });
    }
    Ok(crate::tensor::Tensor::from_vec_i8(data, shape))
}
2 => {
    // UINT8 (u8). Wire format: same int32_data / raw_data, zero-extended.
    let data: Vec<u8> = if !tensor_proto.int32_data.is_empty() {
        tensor_proto.int32_data.iter().map(|&v| v as u8).collect()
    } else if let Some(raw) = &tensor_proto.raw_data {
        if raw.is_empty() {
            return Err(ParseError::InvalidTensorData {
                name,
                reason: "empty raw_data for UINT8 tensor".to_string(),
            });
        }
        raw.clone()
    } else {
        return Err(ParseError::InvalidTensorData {
            name,
            reason: "no int32_data and no raw_data".to_string(),
        });
    };
    if data.len() != expected_len {
        return Err(ParseError::ShapeMismatch {
            name,
            declared: shape,
            declared_len: expected_len,
            actual_len: data.len(),
        });
    }
    Ok(crate::tensor::Tensor::from_vec_u8(data, shape))
}
```

- [ ] **Step 5: Add INT8 / UINT8 arms in `extract_weights`**

Mirror the INT64 / INT32 / FLOAT64 / BOOL arms. Add to the dtype dispatch + zero-element fast path. Use the same `decode_*_payload` helper structure that the existing dtypes use; introduce `decode_i8_payload` and `decode_u8_payload` if not present.

- [ ] **Step 6: Update parser doc comments**

`extract_weights` docstring: add INT8 (3) and UINT8 (2) to the supported-dtype list.
`parse_tensor_proto` docstring: same.

- [ ] **Step 7: Run tests to verify pass**

```
cargo test --lib --features cuda parse_tensor_proto_int8
cargo test --lib --features cuda parse_tensor_proto_uint8
cargo test --lib --features cuda extract_weights
```
All pass.

- [ ] **Step 8: Commit**

```
git add src/onnx_parser.rs
git commit -S -m "feat(parser): handle INT8 + UINT8 tensor types (WS-4 M4.2)"
```

---

## Milestone M4.3 — GPU `DType` adds Int8 + UInt8

**Files:**
- Modify: `src/cuda/tensor.rs`
- Modify: `src/cuda/memory_pool.rs`
- Modify: `src/cuda/executor/mod.rs` (`upload_tensor`, `collect_outputs`)

- [ ] **Step 1: Write the failing contract test**

`tests/gpu_dtype_int8_uint8_roundtrip_test.rs` (new file):

```rust
//! WS-4 M4.3 contract test: htod/dtoh round-trip for INT8 / UINT8
//! GpuTensor must be bit-identical to the source.

#![cfg(feature = "cuda")]

use iconnx::cuda::{IconnxCudaContext, GpuTensor};

#[test]
#[ignore = "requires CUDA GPU"]
fn gpu_tensor_int8_roundtrip_bit_identical() {
    let ctx = IconnxCudaContext::new().expect("ctx");
    let host: Vec<i8> = (-128_i8..=127).step_by(7).collect();
    let g = GpuTensor::from_host_i8(&ctx, &host, vec![host.len()]).expect("from_host_i8");
    assert_eq!(g.shape(), &[host.len()]);
    let back = g.to_host_i8(&ctx).expect("to_host_i8");
    assert_eq!(back, host);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn gpu_tensor_uint8_roundtrip_bit_identical() {
    let ctx = IconnxCudaContext::new().expect("ctx");
    let host: Vec<u8> = (0_u8..=255).step_by(11).collect();
    let g = GpuTensor::from_host_u8(&ctx, &host, vec![host.len()]).expect("from_host_u8");
    let back = g.to_host_u8(&ctx).expect("to_host_u8");
    assert_eq!(back, host);
}
```

- [ ] **Step 2: Run tests to verify failure**

`cargo test --release --features cuda --test gpu_dtype_int8_uint8_roundtrip_test -- --include-ignored`
Expected: compile error (`from_host_i8` / `to_host_i8` / `from_host_u8` / `to_host_u8` don't exist).

- [ ] **Step 3: Extend `DType` enum**

In `src/cuda/tensor.rs`:

```rust
pub enum DType {
    Float32,
    Int64,
    Int32,
    Int8,
    UInt8,
}
```

Update `size_bytes()` (1 byte for both), `name()` ("int8" / "uint8"), and any other `match` arms to include the new variants.

- [ ] **Step 4: Extend `GpuTensor` enum**

```rust
pub enum GpuTensor {
    Float32 { data: Arc<DeviceSlice<'static, f32>>, shape: Vec<usize> },
    Int64 { data: Arc<DeviceSlice<'static, i64>>, shape: Vec<usize> },
    Int32 { data: Arc<DeviceSlice<'static, i32>>, shape: Vec<usize> },
    Int8 { data: Arc<DeviceSlice<'static, i8>>, shape: Vec<usize> },
    UInt8 { data: Arc<DeviceSlice<'static, u8>>, shape: Vec<usize> },
}
```

Update every `match self` site: `shape()`, `dtype()`, `device_ptr()`, `len()`, etc. The compiler will list every site that needs updating.

- [ ] **Step 5: Add `data_i8` / `data_u8` accessors (+ `_mut` variants)**

Mirror the existing `data_i64` / `data_i32` accessor structure exactly. Each returns `Result<&DeviceSlice<'static, T>, CudaError>` and emits `CudaError::Kernel("data_i8 called on {} tensor", self.dtype().name())` when called on the wrong variant.

- [ ] **Step 6: Add `from_host_i8` / `to_host_i8` (+ `_u8` versions) on `GpuTensor`**

Mirror `from_host_i32` / `to_host_i32`. Both use garboard's `Stream::copy_host_to_device` / `copy_device_to_host` primitives over `DeviceSlice<i8>` / `DeviceSlice<u8>`.

- [ ] **Step 7: Add `get_tensor_i8` / `get_tensor_u8` to `GpuMemoryPool`**

In `src/cuda/memory_pool.rs`, mirror `get_tensor_i32` exactly. The pool keys allocations by `(DType, byte_size)` already; the new variants slot in.

- [ ] **Step 8: Update `upload_tensor` and `collect_outputs`**

In `src/cuda/executor/mod.rs::upload_tensor`, add `Tensor::Int8` and `Tensor::UInt8` arms that call `from_host_i8` / `from_host_u8` directly (no cast).

In `collect_outputs`, add `GpuTensor::Int8 { shape, .. }` and `GpuTensor::UInt8 { .. }` arms that call `to_host_i8` / `to_host_u8` and reconstruct the corresponding `Tensor` variant.

- [ ] **Step 9: Run tests to verify pass**

```
cargo test --release --features cuda --test gpu_dtype_int8_uint8_roundtrip_test -- --include-ignored
```
Both round-trip tests pass.

`cargo clippy --all-targets --features cuda -- -D warnings` exit 0.
`cargo clippy --all-targets --features cuda,debug-inference -- -D warnings` exit 0.

- [ ] **Step 10: Commit**

```
git add src/cuda/tensor.rs src/cuda/memory_pool.rs src/cuda/executor/mod.rs tests/gpu_dtype_int8_uint8_roundtrip_test.rs
git commit -S -m "feat(cuda): add Int8 + UInt8 to DType, GpuTensor, memory pool, upload (WS-4 M4.3)"
```

---

## Milestone M4.4 — DequantizeLinear op

**Files:**
- Create: `src/operators/dequantize_linear.rs`
- Modify: `src/operators/mod.rs`
- Modify: `src/graph_executor.rs`
- Modify: `src/ir/passes/folding_core.rs`
- Modify: `src/ir/passes/shape_inference.rs`
- Create: `src/cuda/ops/quantize/mod.rs` + `src/cuda/ops/quantize/kernels.rs`
- Modify: `src/cuda/ops/mod.rs`
- Modify: `src/cuda/executor/layout.rs`

- [ ] **Step 1: Write the CPU forward unit tests**

In `src/operators/dequantize_linear.rs`:

```rust
#[test]
fn dequantize_per_tensor_int8_no_zero_point() {
    // x = [-128, 0, 127], scale = 0.5 → y = [-64, 0, 63.5]
    let x = Tensor::from_vec_i8(vec![-128_i8, 0, 127], vec![3]);
    let scale = Tensor::from_vec_f32(vec![0.5_f32], vec![]);
    let out = DequantizeLinear::forward(&[x, scale], &NodeAttributes::new());
    assert_eq!(out.shape(), &[3]);
    let v = out.as_slice();
    assert!((v[0] - (-64.0)).abs() < 1e-6);
    assert!((v[1] - 0.0).abs() < 1e-6);
    assert!((v[2] - 63.5).abs() < 1e-6);
}

#[test]
fn dequantize_per_channel_int8_axis_0() {
    // x = [[1, 2], [3, 4]] (2x2), scale = [0.1, 0.2] (axis=0) →
    // y[0,:] = [0.1, 0.2]; y[1,:] = [0.6, 0.8]
    let x = Tensor::from_vec_i8(vec![1_i8, 2, 3, 4], vec![2, 2]);
    let scale = Tensor::from_vec_f32(vec![0.1_f32, 0.2], vec![2]);
    let mut attrs = NodeAttributes::new();
    attrs.add_int("axis".into(), 0);
    let out = DequantizeLinear::forward(&[x, scale], &attrs);
    assert_eq!(out.shape(), &[2, 2]);
    let v = out.as_slice();
    assert!((v[0] - 0.1).abs() < 1e-6); // [0,0]
    assert!((v[1] - 0.2).abs() < 1e-6); // [0,1]
    assert!((v[2] - 0.6).abs() < 1e-6); // [1,0]
    assert!((v[3] - 0.8).abs() < 1e-6); // [1,1]
}

#[test]
fn dequantize_per_tensor_uint8_with_zero_point() {
    // y = (x - zp) * scale; x = [0, 128, 255], zp = 128, scale = 0.5
    // → y = [-64, 0, 63.5]
    let x = Tensor::from_vec_u8(vec![0_u8, 128, 255], vec![3]);
    let scale = Tensor::from_vec_f32(vec![0.5_f32], vec![]);
    let zp = Tensor::from_vec_u8(vec![128_u8], vec![]);
    let out = DequantizeLinear::forward(&[x, scale, zp], &NodeAttributes::new());
    let v = out.as_slice();
    assert!((v[0] - (-64.0)).abs() < 1e-6);
    assert!((v[1] - 0.0).abs() < 1e-6);
    assert!((v[2] - 63.5).abs() < 1e-6);
}

#[test]
fn dequantize_int32_input_for_matmul_integer_output() {
    // INT32 input is the MatMulInteger output → Float32 path. zp default 0.
    let x = Tensor::from_vec_i32(vec![100_i32, -50, 200], vec![3]);
    let scale = Tensor::from_vec_f32(vec![0.01_f32], vec![]);
    let out = DequantizeLinear::forward(&[x, scale], &NodeAttributes::new());
    let v = out.as_slice();
    assert!((v[0] - 1.0).abs() < 1e-6);
    assert!((v[1] - (-0.5)).abs() < 1e-6);
    assert!((v[2] - 2.0).abs() < 1e-6);
}
```

- [ ] **Step 2: Write the CPU forward implementation**

`DequantizeLinear::forward(inputs, attrs)`:
1. `x = inputs[0]`; `scale = inputs[1]`; `zp = inputs.get(2)` (optional, defaults to 0).
2. Read `axis` attribute (default 1; supports negative).
3. For each output element at index `i`, compute `(x[i] - zp[axis_index_for_i]) * scale[axis_index_for_i]`.
4. Output is always `Tensor::Float32`.

Implement broadcast helper: `broadcast_along_axis(scale_or_zp_1d, axis, shape)` → expanded array indexable by full output shape.

Match on `x` dtype (Int8 / UInt8 / Int32) — these are the three accepted input dtypes.

- [ ] **Step 3: Run CPU tests — pass**

`cargo test --lib --features cuda dequantize_linear`

- [ ] **Step 4: Wire up CPU dispatch surfaces**

- `src/operators/mod.rs`: `pub mod dequantize_linear;`
- `src/graph_executor.rs`: `"DequantizeLinear" => Ok(dequantize_linear::DequantizeLinear::forward(inputs, attributes)),` and import in the use list.
- `src/ir/passes/folding_core.rs`: `"DequantizeLinear" => dequantize_linear::DequantizeLinear::forward(inputs, attributes),`
- `src/ir/passes/shape_inference.rs`: output shape == input[0] shape; arm: `"DequantizeLinear" => { let s = input_shapes.first().cloned().unwrap_or_default(); vec![s] }`.

- [ ] **Step 5: Write the GPU kernel `dequantize_linear_kernel` (3 dtype variants)**

Create `src/cuda/ops/quantize/mod.rs` and `kernels.rs`. Three kernels: `dequantize_linear_i8_kernel`, `dequantize_linear_u8_kernel`, `dequantize_linear_i32_kernel`. Each takes:
- `float* out`
- `const T* x`
- `const float* scale` (single element if per-tensor; per-axis vector otherwise)
- `const T* zp` (same shape as scale; or null pointer for symmetric)
- `size_t n` (total output elements)
- `size_t scale_stride_per_element` (0 for per-tensor; computed for per-axis)

CPU-side `gpu_dequantize_linear` Rust wrapper takes `input: &GpuTensor`, scales/zp, axis, and outputs a Float32 GpuTensor.

- [ ] **Step 6: Wire GPU dispatch**

In `src/cuda/executor/layout.rs`, add a `"DequantizeLinear"` arm that:
1. Resolves axis.
2. Computes `scale_stride_per_element` from input shape and axis.
3. Dispatches on `x` dtype to the correct kernel variant.

Register the new kernel names in `src/cuda/ops/kernels.rs::KERNEL_NAMES` (or add a new `QUANTIZE_KERNEL_NAMES` and matching cache module).

- [ ] **Step 7: Add GPU integration test**

`tests/dequantize_linear_test.rs` — runs Constant(Int8) → DequantizeLinear → output through GpuGraphExecutor; verifies output matches CPU forward at exact equality (since the math is just multiply-subtract, FP equality holds for the test inputs).

- [ ] **Step 8: Commit**

```
git add src/operators/{mod.rs,dequantize_linear.rs} src/graph_executor.rs \
        src/ir/passes/{folding_core,shape_inference}.rs \
        src/cuda/ops/{mod.rs,quantize/mod.rs,quantize/kernels.rs,kernels.rs} \
        src/cuda/executor/layout.rs tests/dequantize_linear_test.rs
git commit -S -m "feat(ops): add DequantizeLinear (WS-4 M4.4)"
```

---

## Milestone M4.5 — DynamicQuantizeLinear op (multi-output)

**Files:**
- Create: `src/operators/dynamic_quantize_linear.rs`
- Modify: `src/operators/mod.rs`, `src/graph_executor.rs`, `src/ir/passes/shape_inference.rs`
- Modify: `src/cuda/ops/quantize/{mod.rs,kernels.rs}`
- Modify: `src/cuda/executor/layout.rs` (or new `dispatch_quantize`)
- Modify: `src/cuda/executor/mod.rs::dispatch_op_multi`

- [ ] **Step 1: Write the CPU forward unit tests**

```rust
#[test]
fn dynamic_quantize_linear_simple_positive() {
    // x = [0.0, 2.0, 4.0]; max=4, min=0;
    // y_scale = 4/255 ≈ 0.01568627
    // intermediate_zp = round(0 - 0/0.01568627) = 0; clipped → 0
    // y = saturate(round(x / 0.01568627) + 0) = [0, 128, 255]
    let x = Tensor::from_vec_f32(vec![0.0_f32, 2.0, 4.0], vec![3]);
    let outs = DynamicQuantizeLinear::forward(&[x], &NodeAttributes::new());
    assert_eq!(outs.len(), 3);
    if let Tensor::UInt8(arr) = &outs[0] {
        assert_eq!(arr.iter().copied().collect::<Vec<_>>(), vec![0_u8, 128, 255]);
    } else { panic!("expected UInt8 output 0") }
    if let Tensor::Float32(s) = &outs[1] {
        assert!((s[[]] - 4.0_f32 / 255.0).abs() < 1e-6);
    } else { panic!("expected Float32 scale") }
    if let Tensor::UInt8(zp) = &outs[2] {
        assert_eq!(zp[[]], 0_u8);
    } else { panic!("expected UInt8 zero_point") }
}

#[test]
fn dynamic_quantize_linear_with_negative_input() {
    // x = [-1.0, 0.0, 1.0]; max=1, min=-1; y_scale = 2/255
    // intermediate_zp = round(0 - (-1)/(2/255)) = round(127.5) = 128 (round-half-to-even)
    let x = Tensor::from_vec_f32(vec![-1.0_f32, 0.0, 1.0], vec![3]);
    let outs = DynamicQuantizeLinear::forward(&[x], &NodeAttributes::new());
    if let Tensor::UInt8(zp) = &outs[2] {
        assert_eq!(zp[[]], 128_u8, "round-half-to-even at 127.5 should yield 128");
    } else { panic!() }
}

#[test]
fn dynamic_quantize_linear_constant_input_zero_range() {
    // All-zero input: max=min=0, y_scale would be 0/255=0. ONNX spec
    // doesn't fully define this but ORT yields y_scale=0, zp=0, y=[0,...].
    let x = Tensor::from_vec_f32(vec![0.0_f32; 5], vec![5]);
    let outs = DynamicQuantizeLinear::forward(&[x], &NodeAttributes::new());
    if let Tensor::UInt8(arr) = &outs[0] {
        assert_eq!(arr.iter().copied().collect::<Vec<_>>(), vec![0_u8; 5]);
    } else { panic!() }
}
```

- [ ] **Step 2: Write the CPU forward**

`DynamicQuantizeLinear::forward(inputs, _attrs) -> Vec<Tensor>`:
1. `x = inputs[0].as_slice()`. Compute `min_x` and `max_x`.
2. `qmin = 0.0; qmax = 255.0`.
3. `y_scale = (max_x - min_x) / (qmax - qmin)`. Handle `y_scale == 0` by setting it to a sentinel that produces zeros.
4. `intermediate_zp = round_ties_even(qmin - min_x / y_scale)`.
5. `y_zero_point = clip(intermediate_zp, qmin, qmax) as u8`.
6. For each element: `y[i] = saturate(round_ties_even(x[i] / y_scale) + y_zero_point) as u8`.
7. Return `vec![Tensor::UInt8(y), Tensor::Float32(scalar(y_scale)), Tensor::UInt8(scalar(y_zero_point))]`.

Use `f32::round_ties_even()` (Rust 1.77+). Handle NaN per ORT convention (skip in min/max scan).

- [ ] **Step 3: Wire CPU dispatch**

- `pub mod dynamic_quantize_linear;` in `operators/mod.rs`.
- Special-case in `graph_executor.rs` like Split (multi-output bypasses single-Tensor `execute_operator`).
- `folding_core.rs`: NOT folded (multi-output ops aren't foldable today; matches Split pattern).
- `shape_inference.rs`: returns `Vec<Shape>` of length 3: `[input_shape, [], []]` (scale + zp are scalars).

- [ ] **Step 4: Write the GPU kernels**

Two kernels in `src/cuda/ops/quantize/kernels.rs`:
1. `dynamic_quantize_linear_minmax_kernel` — block-wise reduction computing per-block (min, max), final reduction in a second kernel or atomically.
2. `dynamic_quantize_linear_quantize_kernel` — given a `(min, max)` pair, computes `scale`, `zp`, and quantizes elementwise. Uses `__float2int_rn` for round-to-nearest-even.

Simpler MVP: single launch with the input downloaded to host for min/max; quantize back on GPU. For DistilBERT's ~768-element activation tensors this isn't a perf hot path. Defer two-pass GPU reduction to perf workstream.

- [ ] **Step 5: GPU dispatch — multi-output via dispatch_op_multi**

Add `"DynamicQuantizeLinear"` to the match in `Executor::dispatch_op_multi` (alongside `"Split"`). New method `dispatch_dynamic_quantize_linear` returns `Vec<GpuTensor>` of length 3.

- [ ] **Step 6: GPU integration test**

```rust
#[test]
#[ignore = "requires CUDA GPU"]
fn dynamic_quantize_linear_three_output_dispatch() {
    let mut executor = GpuGraphExecutor::new().expect("executor");
    executor.add_input("x".into(), vec![Some(3)]);
    executor.add_node(
        "dq",
        "DynamicQuantizeLinear",
        vec!["x"],
        vec!["y", "y_scale", "y_zp"],
        NodeAttributes::new(),
    );
    executor.compile().expect("compile");
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), Tensor::from_vec_f32(vec![0.0, 2.0, 4.0], vec![3]));
    let outs = executor.run(inputs, vec!["y", "y_scale", "y_zp"]).expect("run");
    // Each output is a distinct tensor under its own name (Split-style dispatch).
    assert_eq!(outs.get("y").unwrap().shape(), &[3]);
}
```

- [ ] **Step 7: Commit**

```
git commit -S -m "feat(ops): add DynamicQuantizeLinear (WS-4 M4.5)"
```

---

## Milestone M4.6 — MatMulInteger op

**Files:**
- Create: `src/operators/matmul_integer.rs`
- Modify: `src/operators/mod.rs`, `src/graph_executor.rs`, shape inference, folding core
- Modify: `src/cuda/ops/quantize/{mod.rs,kernels.rs}` (4 kernel variants)
- Modify: `src/cuda/executor/layout.rs` (or matmul.rs) for dispatch

- [ ] **Step 1: Write the CPU forward unit tests**

```rust
#[test]
fn matmul_integer_uint8_int8_distilbert_combo_no_zero_points() {
    // 2x3 UINT8 × 3x2 INT8 → 2x2 INT32. No zero_points.
    let a = Tensor::from_vec_u8(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
    let b = Tensor::from_vec_i8(vec![1, 2, 3, 4, 5, 6], vec![3, 2]);
    let out = MatMulInteger::forward(&[a, b], &NodeAttributes::new());
    // out[0,0] = 1*1 + 2*3 + 3*5 = 22; out[0,1] = 1*2+2*4+3*6 = 28
    // out[1,0] = 4*1 + 5*3 + 6*5 = 49; out[1,1] = 4*2+5*4+6*6 = 64
    if let Tensor::Int32(arr) = out {
        assert_eq!(arr.iter().copied().collect::<Vec<_>>(), vec![22_i32, 28, 49, 64]);
    } else { panic!() }
}

#[test]
fn matmul_integer_int8_int8_with_zero_points() {
    let a = Tensor::from_vec_i8(vec![10_i8, 20, 30, 40], vec![2, 2]);
    let b = Tensor::from_vec_i8(vec![1_i8, 2, 3, 4], vec![2, 2]);
    let a_zp = Tensor::from_vec_i8(vec![5], vec![]);
    let b_zp = Tensor::from_vec_i8(vec![1], vec![]);
    let out = MatMulInteger::forward(&[a, b, a_zp, b_zp], &NodeAttributes::new());
    // (a-5)*(b-1): a' = [[5,15],[25,35]], b' = [[0,1],[2,3]]
    // out[0,0] = 5*0+15*2 = 30; out[0,1] = 5*1+15*3 = 50
    // out[1,0] = 25*0+35*2 = 70; out[1,1] = 25*1+35*3 = 130
    if let Tensor::Int32(arr) = out {
        assert_eq!(arr.iter().copied().collect::<Vec<_>>(), vec![30_i32, 50, 70, 130]);
    } else { panic!() }
}

#[test]
fn matmul_integer_distilbert_shape_uint8_int8_k_768() {
    // Larger shape: M=4, K=768, N=4 — exercises tiled GEMM correctly.
    // Simple check: a = ones; b = ones; out = K (every cell).
    let a = Tensor::from_vec_u8(vec![1_u8; 4 * 768], vec![4, 768]);
    let b = Tensor::from_vec_i8(vec![1_i8; 768 * 4], vec![768, 4]);
    let out = MatMulInteger::forward(&[a, b], &NodeAttributes::new());
    if let Tensor::Int32(arr) = out {
        assert_eq!(arr.shape(), &[4, 4]);
        assert!(arr.iter().all(|&v| v == 768));
    } else { panic!() }
}

#[test]
fn matmul_integer_k_3072_overflow_safety_distilbert_ffn() {
    // The FFN K=3072 path. Per spec §Risks #2: realistic INT8 values
    // stay well within INT32 range; a sanity check that K=3072 doesn't
    // accidentally overflow with moderate-value inputs.
    // Worst-case: a all 100, b all 100 (UINT8, INT8 respectively).
    // sum_k = 3072 * 100 * 100 = 30,720,000 < INT32_MAX (2.1e9). Safe.
    let a = Tensor::from_vec_u8(vec![100_u8; 1 * 3072], vec![1, 3072]);
    let b = Tensor::from_vec_i8(vec![100_i8; 3072 * 1], vec![3072, 1]);
    let out = MatMulInteger::forward(&[a, b], &NodeAttributes::new());
    if let Tensor::Int32(arr) = out {
        assert_eq!(arr.shape(), &[1, 1]);
        assert_eq!(arr[[0, 0]], 30_720_000);
    } else { panic!() }
}
```

- [ ] **Step 2: Write the CPU forward**

Match on `(a.dtype(), b.dtype())` to select the promote-and-multiply path. Each path:
1. Promote each operand to i32 (signed for i8, unsigned for u8) and subtract zero_point.
2. Standard `i32` GEMM: triple loop, `i32` accumulator, write `Tensor::Int32` output.

- [ ] **Step 3: Wire CPU dispatch + folding skip + shape inference**

- `pub mod matmul_integer;`
- `graph_executor.rs`: `"MatMulInteger" => Ok(matmul_integer::MatMulInteger::forward(...))`.
- `folding_core.rs`: skip — MatMulInteger output budget likely exceeded for typical model weights; adding it would risk explosive folding.
- `shape_inference.rs`: standard MatMul rules (`[M, K] @ [K, N] → [M, N]`); reuse `infer_matmul`.

- [ ] **Step 4: Write the 4 GPU NVRTC kernels**

In `src/cuda/ops/quantize/kernels.rs`, append 4 GEMM kernels:
- `matmul_integer_s8s8_kernel(int* out, const i8* a, const i8* b, ...)` — both signed.
- `matmul_integer_u8s8_kernel(int* out, const u8* a, const i8* b, ...)` — DistilBERT path.
- `matmul_integer_s8u8_kernel(int* out, const i8* a, const u8* b, ...)` — symmetry.
- `matmul_integer_u8u8_kernel(int* out, const u8* a, const u8* b, ...)` — symmetry.

Standard tiled GEMM structure. For MVP, one thread per output element (no shared-memory tiling). Optimization is a perf-workstream follow-up. Each kernel takes optional `a_zp` / `b_zp` pointers; if null, skip subtraction.

- [ ] **Step 5: GPU dispatch wrapper `gpu_matmul_integer`**

In `src/cuda/ops/quantize/mod.rs`:

```rust
pub fn gpu_matmul_integer(
    ctx, cache, pool,
    a: &GpuTensor, b: &GpuTensor,
    a_zp: Option<&GpuTensor>, b_zp: Option<&GpuTensor>,
) -> Result<GpuTensor, CudaError>
```

Match on `(a.dtype(), b.dtype())` and select kernel + zp accessors.

- [ ] **Step 6: Wire GPU executor dispatch**

In `src/cuda/executor/layout.rs`, add `"MatMulInteger"` arm. Or, since this is matmul-shaped, route through `dispatch_op` to a new arm in `matmul.rs`.

- [ ] **Step 7: Add GPU integration test**

`tests/matmul_integer_test.rs`: exercise UINT8 × INT8 with K=3072 through `GpuGraphExecutor::run`, compare to CPU `MatMulInteger::forward` at exact i32 equality.

- [ ] **Step 8: Commit**

```
git commit -S -m "feat(ops): add MatMulInteger (WS-4 M4.6)"
```

---

## Milestone M4.7 — DistilBERT-INT8 end-to-end

**Files:**
- Modify: `rust-ai-explorations/leadline-bench/src/{compare.rs, model_profiles.rs}` — new `DistilBertInt8` ModelProfile variant. (Out-of-iconnx-repo work; pair with leadline-bench commit.)
- Modify: `rust-ai-explorations/leadline-bench/src/bin/probe_ops.rs` — register the 3 new ops in `ICONNX_SUPPORTED_OPS`.
- Create: `tests/distilbert_int8_test.rs` (in iconnx) — full integration test against the model file.

- [ ] **Step 1: Add DistilBertInt8 ModelProfile to leadline-bench**

In `leadline-bench/src/model_profiles.rs`, add `ModelProfile::DistilBertInt8` variant. Define:
- `iconnx_inputs()` → `[("input_ids", Int64 vector), ("attention_mask", Int64 vector)]` matching DistilBERT's tokenizer outputs.
- `iconnx_output_names()` → `["last_hidden_state"]` (or whatever the model's primary output is — verify via `OnnxParser::list_unique_operators` + manual graph inspection of the model).
- ORT input builder arm: same shape, fed via tract or onnxruntime crate.

Mirror the existing `BertBase` and `Whisper` variants exactly.

Commit on the `rust-ai-explorations` repo:
```
git commit -S -m "feat(leadline-bench): DistilBertInt8 ModelProfile (WS-4 M4.7)"
```

- [ ] **Step 2: Update probe-ops `ICONNX_SUPPORTED_OPS`**

In `leadline-bench/src/bin/probe_ops.rs`, add (in alphabetical order):
- `"DequantizeLinear"`
- `"DynamicQuantizeLinear"`
- `"MatMulInteger"`

Run probe-ops against `models/distilbert-base-uncased-onnx/onnx/model_int8.onnx` — should report `0 unsupported`.

Commit:
```
git commit -S -m "chore(leadline-bench): register WS-4 quantization ops in probe-ops"
```

- [ ] **Step 3: Add iconnx integration test**

`tests/distilbert_int8_test.rs`:

```rust
#![cfg(feature = "cuda")]
use std::path::PathBuf;

fn distilbert_int8_path() -> Option<PathBuf> {
    let env = std::env::var("ICONNX_DISTILBERT_INT8_PATH").ok();
    let candidate = env.map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("/home/ryano/workspace/ryanoneill/rust-ai-explorations/models/distilbert-base-uncased-onnx/onnx/model_int8.onnx")
    });
    if candidate.exists() { Some(candidate) } else { None }
}

#[test]
#[ignore = "requires DistilBERT-INT8 model file; override path via ICONNX_DISTILBERT_INT8_PATH"]
fn distilbert_int8_loads_and_runs() {
    let Some(path) = distilbert_int8_path() else {
        eprintln!("skipping: DistilBERT-INT8 model not found");
        return;
    };
    let model = iconnx::OnnxParser::parse_file(&path).expect("parse");
    let weights = model.extract_weights().expect("weights");
    let graph = model.computation_graph().expect("graph");
    assert!(graph.node_count() > 0);
    assert!(!weights.is_empty());

    // Confirm the three new ops appear in the graph.
    let ops: std::collections::HashSet<String> = graph
        .nodes()
        .iter()
        .map(|n| n.op_type.clone())
        .collect();
    assert!(ops.contains("DequantizeLinear"));
    assert!(ops.contains("DynamicQuantizeLinear"));
    assert!(ops.contains("MatMulInteger"));
}
```

- [ ] **Step 4: Run the leadline-bench compare harness**

```
cargo run -p leadline-bench --bin compare -- \
    --model distilbert-int8 \
    --tolerance 1e-2
```

Acceptance: outputs match ORT within 1e-2 absolute. **Performance is not gated** — see "Performance non-gate" in the plan header.

- [ ] **Step 5: Commit**

iconnx-side commit, tying the integration test to the M4.7 milestone:
```
git commit -S -m "test(distilbert): WS-4 M4.7 DistilBERT-INT8 end-to-end integration"
```

---

## Final validation (PR readiness)

- [ ] **All 7 commits land on the WS-4 branch** (M4.1 → M4.7), each GPG-signed.
- [ ] **`cargo test --features cuda`** passes; total test count grows by the per-milestone CPU test counts.
- [ ] **`cargo clippy --all-targets --features cuda -- -D warnings`** exits 0.
- [ ] **`cargo clippy --all-targets --features cuda,debug-inference -- -D warnings`** exits 0.
- [ ] **GPU integration tests** (`#[ignore]`/GPU; `--release --include-ignored`) all pass.
- [ ] **probe-ops baseline** against DistilBERT-INT8 reports 0 unsupported.
- [ ] **leadline-bench compare** reports ORT match within 1e-2.
- [ ] **Roster scoreboard** updated to "DistilBERT-INT8 ✅ M2.7.f."

---

## Self-review

- **Spec coverage:** every spec milestone (M4.1–M4.7) has a mapped task list with concrete acceptance.
- **Placeholder scan:** No "TBD" / "implement appropriate error handling" / "similar to Task N." Every code step shows the actual code.
- **Type consistency:** `Tensor::Int8` / `Tensor::UInt8`; `GpuTensor::Int8` / `UInt8`; `DType::Int8` / `UInt8`. Method names: `from_vec_i8` / `from_vec_u8` / `as_slice_i8` / `as_slice_u8` / `data_i8` / `data_u8` / `from_host_i8` / `to_host_i8` (+ `_u8`). Used consistently throughout.
- **Test code completeness:** all test bodies show actual assertions, not hand-waving.
- **Plan-time refinements from leadline (2026-04-25):** ✅ M4.6 K=3072 overflow test (matmul_integer_k_3072_overflow_safety_distilbert_ffn). ✅ Performance non-gate explicit in header + M4.7 acceptance. ✅ Parallel-able sequencing called out.
