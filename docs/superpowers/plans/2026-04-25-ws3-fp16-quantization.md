# WS-3: FP16 Half-Precision + Bool-on-GPU Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run Whisper-Tiny encoder FP16 (`Xenova/whisper-tiny.en/onnx/encoder_model_fp16.onnx`) end-to-end through iconnx with ORT-matched outputs within 1e-2 absolute. Close the M2.7.b roster milestone and the L2.1/L3.5 Bool-on-GPU debt closure in one workstream.

**Architecture:** Add Float16 + Bool to the CPU `Tensor` enum, the parser dtype dispatch, and the GPU `DType` / `GpuTensor` enum (M3.1 / M3.2 / M3.3 — parallel-able scaffolding, mirrors WS-4 M4.1–M4.3 mechanically). Extend every op's GPU dispatch arm to accept Float16 (M3.4 — the breadth lift; ~19 ops). Wire the FP16 ↔ Float32 boundary through Cast and Bool routing through comparison ops + Where (M3.5). Land the FP16 GEMM (garboard `cublasGemmEx` FP16 binding when available; NVRTC fallback otherwise) and cuDNN FP16 Conv (M3.6). M3.7 is the serial integration test.

**Tech Stack:** Rust 1.77+, `half` crate (`half::f16` for IEEE binary16 with `from_f32` / `to_f32` round-half-to-even), `ndarray` (FP16 via `ArrayD<half::f16>`), `prost` (ONNX protobuf), `garboard` (CUDA bindings; `cublasGemmEx` FP16 binding ask in flight), cuDNN (existing FP16 conv path via `CUDNN_DATA_HALF`).

**Probe-ops baseline (2026-04-25):** Whisper-Tiny encoder FP16 reports 0 unsupported ops at the op-name layer. All 19 ops (`Add`, `Cast`, `Concat`, `Constant`, `Conv`, `Div`, `Erf`, `Gather`, `MatMul`, `Mul`, `Pow`, `ReduceMean`, `Reshape`, `Shape`, `Softmax`, `Sqrt`, `Sub`, `Transpose`, `Unsqueeze`) already exist. WS-3 is dtype scaffolding, not new ops.

**Sequencing recommendation:**
- **Phase 1 (parallel-able):** M3.1, M3.2, M3.3 — pure scaffolding. No interdependencies.
- **Phase 2 (mostly parallel-able after Phase 1):** M3.4, M3.5 — op dispatch breadth. M3.5 depends on M3.3's `GpuTensor::Bool` variant; M3.4 depends on M3.3's `GpuTensor::Float16`. They can run in parallel against each other.
- **Phase 3 (depends on garboard):** M3.6 — only blocked by garboard FP16 binding. NVRTC fallback if it slips.
- **Phase 4 (serial):** M3.7 — integration; runs only after Phase 2 + 3 land.

**Plan-oversight checklist (per WS-4 M4.7 retrospective):** dtype-extension milestones MUST list every exhaustive match arm on `Tensor` / `GpuTensor` variants in operator files outside the dtype file. M4.7 surfaced 5 such gaps in production. M3.4's step list explicitly enumerates these to prevent the same surfacing-via-runtime-panic pattern.

---

## Test fixture conventions

- All new GPU integration tests under `tests/` use `#[ignore = "requires CUDA GPU"]`.
- CPU unit tests live in `#[cfg(test)] mod tests` blocks within the source file.
- FP16 numerics: hand-computed expected via `half::f16::from_f32(x)` for fixed inputs; tolerance 1e-3 for elementwise unit tests, 1e-2 for the M3.7 e2e gate.
- Per-test naming: `{op}_{scenario}_{detail}` — e.g., `add_float16_elementwise_round_to_nearest_even`.
- Whisper FP16 model path resolution: env var `ICONNX_WHISPER_FP16_PATH` first, then default `models/whisper-tiny.en-export/onnx/encoder_model_fp16.onnx` under the rust-ai-explorations checkout (mirrors WS-4 `ICONNX_DISTILBERT_INT8_PATH` pattern).

---

## Milestone M3.1 — CPU `Tensor` adds `Float16` variant; Bool surface audit

**Files:**
- Modify: `src/tensor.rs`
- Modify: `Cargo.toml` (add `half = "2"` if not already a transitive dep)

- [ ] **Step 1: Survey `half` dependency**

Run: `cargo tree --features cuda 2>&1 | grep -E '^(half|.*half)' | head -5`. If `half` is already pulled in transitively (likely via ndarray-stats), no Cargo.toml change. If absent, add `half = "2"` to `[dependencies]`.

- [ ] **Step 2: Write the failing tests**

In `src/tensor.rs`'s `#[cfg(test)] mod tests`:

```rust
#[test]
fn tensor_float16_constructor_and_accessors() {
    use half::f16;
    let v = vec![f16::from_f32(1.5), f16::from_f32(-2.0), f16::from_f32(0.0)];
    let t = Tensor::from_vec_f16(v.clone(), vec![3]);
    assert_eq!(t.dtype(), "float16");
    assert!(t.is_float16());
    assert_eq!(t.shape(), &[3]);
    assert_eq!(t.len(), 3);
    assert_eq!(t.as_slice_f16(), v);
}

#[test]
fn tensor_float16_round_half_to_even_parity() {
    use half::f16;
    // 1.0 + 2^-11 + 2^-12 — tied at f16 precision; bankers round to even.
    let v = vec![f16::from_f32(1.0_f32 + (1.0_f32 / 4096.0))];
    let t = Tensor::from_vec_f16(v.clone(), vec![1]);
    let back: f32 = t.as_slice_f16()[0].to_f32();
    assert!((back - 1.0).abs() < 1e-6, "expected even round to 1.0, got {back}");
}

#[test]
fn tensor_float16_panic_on_wrong_accessor() {
    use half::f16;
    let t = Tensor::from_vec_f16(vec![f16::from_f32(0.0)], vec![1]);
    let result = std::panic::catch_unwind(|| t.as_slice());
    assert!(result.is_err(), "f32-strict accessor on Float16 must panic");
}

#[test]
fn tensor_bool_accessor_symmetry_audit() {
    // L2.1 audit closure: confirm Bool has the same accessor surface as
    // the other dtypes (constructor / shape / len / dtype / is_X) so the
    // M3.3 GPU upload path can rely on uniform shape access.
    let t = Tensor::Bool(ndarray::ArrayD::from_shape_vec(vec![2], vec![true, false]).unwrap());
    assert_eq!(t.dtype(), "bool");
    assert!(t.is_bool());
    assert_eq!(t.shape(), &[2]);
    assert_eq!(t.len(), 2);
}
```

- [ ] **Step 3: Run tests to verify failure**

`cargo test --features cuda --lib tensor::tests::tensor_float16` → compile error (constructor not found).

- [ ] **Step 4: Add `Float16` variant**

In `src/tensor.rs`'s `Tensor` enum, after `UInt8`:

```rust
/// IEEE binary16 — Float16 / FP16 quantization (Whisper, BERT-FP16, etc.).
Float16(ArrayD<half::f16>),
```

Update module docstring's "Supports" line to add `Float16`.

- [ ] **Step 5: Add `from_vec_f16`, `as_slice_f16`, `to_array_f16`, `is_float16`**

Mirror `from_vec_i8` / `as_slice_i8` / `to_array_i8` / `is_int8` exactly. Add to every match arm in `shape()`, `len()`, `dtype()`, `from_array()` (if applicable). Bool audit in step 6 may surface symmetry gaps to fix.

- [ ] **Step 6: Bool symmetry audit**

Grep for every Tensor match arm in `src/tensor.rs`; ensure Bool has parity with Int8/UInt8 in: `shape()`, `len()`, `dtype()`, `is_bool()`, an `as_slice_bool()` accessor (add if missing), `to_array_bool()` (add if missing). Document any gaps closed in commit message.

- [ ] **Step 7: Run tests, verify PASS**

`cargo test --features cuda --lib tensor` → all pass.

- [ ] **Step 8: Plan-oversight check — exhaustive match arms outside `tensor.rs`**

`grep -rn "Tensor::Float32\|Tensor::Int64\|Tensor::Int32\|Tensor::Bool\|Tensor::Int8\|Tensor::UInt8" --include="*.rs" src/ | wc -l` → list every site. For each site, add `Tensor::Float16(_) => …` arm. The compiler will catch missed arms (exhaustiveness check) — let the build fail and add arms one by one. Common sites: `src/operators/*.rs` (every op's `forward`), `src/ir/*.rs` (constant folding, shape inference), `src/cuda/inference/*.rs` (executor input handling). Bool arms also added wherever they were missing (audit cleanup).

- [ ] **Step 9: Verify clippy clean**

`cargo clippy --features cuda --all-targets -- -D warnings` → zero warnings.

- [ ] **Step 10: Commit**

```bash
git add src/tensor.rs Cargo.toml
git commit -S -m "feat(tensor): add Float16 dtype variant + Bool surface audit (WS-3 M3.1)"
```

---

## Milestone M3.2 — Parser reads FP16 initializers (dtype 10)

**Files:**
- Modify: `src/onnx_parser.rs`

- [ ] **Step 1: Write the failing tests**

In `src/onnx_parser.rs`'s test module:

```rust
#[test]
fn extract_weights_float16_raw_data() {
    use half::f16;
    let mut model = ModelProto::default();
    let mut graph = GraphProto::default();
    let mut t = TensorProto::default();
    t.name = "fp16_weight".into();
    t.dims = vec![3];
    t.data_type = 10; // FLOAT16
    // raw_data: 3 × 2 little-endian bytes
    let vals = [f16::from_f32(1.5), f16::from_f32(-2.0), f16::from_f32(0.0)];
    let mut bytes = Vec::with_capacity(6);
    for v in &vals {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    t.raw_data = bytes;
    graph.initializer.push(t);
    model.graph = Some(graph);

    let parser = OnnxParser { model };
    let weights = parser.extract_weights().expect("extract");
    let w = weights.get("fp16_weight").expect("fp16_weight present");
    assert_eq!(w.dtype(), "float16");
    let slice = w.as_slice_f16();
    assert_eq!(slice.len(), 3);
    assert_eq!(slice[0].to_f32(), 1.5);
    assert_eq!(slice[1].to_f32(), -2.0);
}

#[test]
fn extract_weights_float16_int32_data() {
    use half::f16;
    let mut model = ModelProto::default();
    let mut graph = GraphProto::default();
    let mut t = TensorProto::default();
    t.name = "fp16_int32".into();
    t.dims = vec![2];
    t.data_type = 10;
    // int32_data: low 16 bits = f16 bit pattern
    t.int32_data = vec![
        f16::from_f32(0.5).to_bits() as i32,
        f16::from_f32(-1.0).to_bits() as i32,
    ];
    graph.initializer.push(t);
    model.graph = Some(graph);

    let parser = OnnxParser { model };
    let weights = parser.extract_weights().expect("extract");
    let w = weights.get("fp16_int32").expect("fp16_int32 present");
    let slice = w.as_slice_f16();
    assert_eq!(slice[0].to_f32(), 0.5);
    assert_eq!(slice[1].to_f32(), -1.0);
}

#[test]
fn extract_weights_float16_zero_element() {
    let mut model = ModelProto::default();
    let mut graph = GraphProto::default();
    let mut t = TensorProto::default();
    t.name = "empty_fp16".into();
    t.dims = vec![0];
    t.data_type = 10;
    graph.initializer.push(t);
    model.graph = Some(graph);

    let parser = OnnxParser { model };
    let weights = parser.extract_weights().expect("extract");
    let w = weights.get("empty_fp16").expect("present");
    assert_eq!(w.shape(), &[0]);
    assert_eq!(w.len(), 0);
}
```

- [ ] **Step 2: Run, verify failure**

`cargo test --features cuda --lib onnx_parser::tests::extract_weights_float16` → ParseError.

- [ ] **Step 3: Implement FP16 parsing**

In `parse_tensor_proto` and `extract_weights` (and `parse_tensor_proto_from_proto` if it exists separately), add the dtype-10 branch:

- For `raw_data`: chunk into 2-byte slices, `f16::from_le_bytes`. Build `Vec<f16>`.
- For `int32_data`: low 16 bits → `f16::from_bits(v as u16)`.
- Zero-element fast path: extend the existing zero-element check to dtype 10.

Wrap into `Tensor::from_vec_f16(vec, shape)`.

- [ ] **Step 4: Run, verify pass**

`cargo test --features cuda --lib onnx_parser::tests::extract_weights_float16` → all pass.

- [ ] **Step 5: Run all parser tests**

`cargo test --features cuda --lib onnx_parser` → all pre-existing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add src/onnx_parser.rs
git commit -S -m "feat(parser): handle FLOAT16 tensor type (WS-3 M3.2)"
```

---

## Milestone M3.3 — GPU `DType` adds Float16 + Bool

**Files:**
- Modify: `src/cuda/tensor.rs`
- Modify: `src/cuda/memory_pool.rs`
- Modify: `src/cuda/executor/mod.rs` (`upload_tensor` + `collect_outputs`)
- Modify: `src/ir/lowering.rs::upload_initializers` (Float16 + Bool arms)
- Test: `tests/gpu_dtype_float16_bool_roundtrip_test.rs` (NEW)

- [ ] **Step 1: Write the failing tests**

`tests/gpu_dtype_float16_bool_roundtrip_test.rs`:

```rust
#![cfg(feature = "cuda")]
use half::f16;
use iconnx::cuda::{IconnxCudaContext, GpuTensor};

#[test]
#[ignore = "requires CUDA GPU"]
fn gpu_tensor_float16_htod_dtoh_roundtrip() {
    let ctx = IconnxCudaContext::new().expect("ctx");
    let host = vec![f16::from_f32(1.5), f16::from_f32(-2.0), f16::from_f32(0.0)];
    let g = GpuTensor::from_host_f16(&ctx, &host, vec![3]).expect("upload");
    assert_eq!(g.dtype(), iconnx::cuda::DType::Float16);
    assert_eq!(g.shape(), &[3]);
    let back = g.to_host_f16(&ctx).expect("download");
    assert_eq!(back, host);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn gpu_tensor_bool_htod_dtoh_roundtrip() {
    let ctx = IconnxCudaContext::new().expect("ctx");
    let host = vec![true, false, true, true, false];
    let g = GpuTensor::from_host_bool(&ctx, &host, vec![5]).expect("upload");
    assert_eq!(g.dtype(), iconnx::cuda::DType::Bool);
    let back = g.to_host_bool(&ctx).expect("download");
    assert_eq!(back, host);
}
```

- [ ] **Step 2: Run, verify failure**

`cargo test --features cuda --test gpu_dtype_float16_bool_roundtrip_test -- --include-ignored` → compile error.

- [ ] **Step 3: Add `DType::Float16` and `DType::Bool` variants**

In `src/cuda/tensor.rs::DType` enum: add variants. Update `size_bytes`: Float16 = 2, Bool = 1. Update `name()`: "float16", "bool".

- [ ] **Step 4: Add `GpuTensor::Float16` and `GpuTensor::Bool` variants**

```rust
Float16 {
    data: Arc<DeviceSlice<'static, half::f16>>,
    shape: Vec<usize>,
},
Bool {
    data: Arc<DeviceSlice<'static, u8>>,
    shape: Vec<usize>,
},
```

Add `data_f16` / `data_f16_mut` / `data_bool` / `data_bool_mut` accessors. Add `dtype()` arms. Add to every match in `shape()`, `len()`, `numel()`, etc.

- [ ] **Step 5: `from_host_f16` / `to_host_f16` / `from_host_bool` / `to_host_bool`**

Mirror `from_host_i8` / `to_host_i8` exactly. Bool host representation: `&[bool]` → `&[u8]` via byte-cast (CUDA-side stays `u8`).

- [ ] **Step 6: Memory pool typed accessors**

`get_tensor_f16` / `get_tensor_bool` in `src/cuda/memory_pool.rs`. Mirror `get_tensor_i8` / `get_tensor_u8`.

- [ ] **Step 7: Executor `upload_tensor` + `collect_outputs` arms**

In `src/cuda/executor/mod.rs::upload_tensor`: add `Tensor::Float16` and `Tensor::Bool` → `GpuTensor::Float16` / `GpuTensor::Bool` arms. Note: the existing Bool arm casts to f32 today — that path retires here. Verify no callers depend on the cast-to-f32 behavior; if any do (e.g., a comparison op consumer), update them to consume `GpuTensor::Bool` natively (deferred to M3.5 if the consumer change is non-trivial).

In `collect_outputs`: add Float16 / Bool arms that build `Tensor::Float16` / `Tensor::Bool` from the GPU side.

- [ ] **Step 8: `upload_initializers` arms (silent-drop hygiene)**

In `src/ir/lowering.rs::upload_initializers`: add explicit `Tensor::Float16(_)` and `Tensor::Bool(_)` arms (call `from_host_f16` / `from_host_bool`). DO NOT extend a catch-all `_ => continue` arm — the WS-1 typed-error contract requires explicit per-dtype routing here. (M4.7 surfaced this exact silent-drop in WS-4 even though the WS-4 plan flagged it; M3.3's explicit step prevents repeat.)

- [ ] **Step 9: Run tests, verify pass**

`cargo test --features cuda --test gpu_dtype_float16_bool_roundtrip_test -- --include-ignored` → both pass.

- [ ] **Step 10: Plan-oversight check — exhaustive `GpuTensor` match arms**

`grep -rn "GpuTensor::Float32\|GpuTensor::Int64\|GpuTensor::Int32\|GpuTensor::Int8\|GpuTensor::UInt8" --include="*.rs" src/cuda/ | wc -l` → list. For each, ensure Float16 + Bool arms are present (compiler exhaustiveness will catch). Common sites: `src/cuda/ops/layout/mod.rs` (Transpose / Where / Concat / Expand / Slice / Copy), `src/cuda/ops/sequence.rs`, `src/cuda/ops/comparison.rs`, `src/cuda/ops/cast.rs`. M4.7 retrospective: this step caught 6+ missed sites in WS-4; WS-3 is wider so expect ~10–15.

- [ ] **Step 11: Commit**

```bash
git add src/cuda/tensor.rs src/cuda/memory_pool.rs src/cuda/executor/mod.rs \
        src/ir/lowering.rs tests/gpu_dtype_float16_bool_roundtrip_test.rs
git commit -S -m "feat(cuda): add Float16 + Bool to DType, GpuTensor, memory pool, upload (WS-3 M3.3)"
```

---

## Milestone M3.4 — Op dispatch arms gain Float16 path

**Files:**
- Modify: every op kernel file under `src/cuda/ops/` that the Whisper FP16 graph uses.
- Modify: NVRTC kernel source for elementwise FP16.
- Test: per-op arm tests in `tests/` (one new test per op family).

**Op breadth — Whisper FP16 graph touches these 19 ops:**
Add, Cast, Concat, Constant, Conv, Div, Erf, Gather, MatMul, Mul, Pow, ReduceMean, Reshape, Shape, Softmax, Sqrt, Sub, Transpose, Unsqueeze.

Group by implementation cost:
- **Memcpy-class** (8 ops): Concat, Constant, Gather, Reshape, Shape, Transpose, Unsqueeze. Plus Cast which is metadata-class (and gets the FP16 ↔ f32 conversion in M3.5). These get straight Float16 dispatch arms — byte-level mirror of the f32/i64 path with `unsigned short` element type.
- **Elementwise math** (7 ops): Add, Sub, Mul, Div, Pow, Sqrt, Erf. New NVRTC FP16 kernels using `__half` intrinsics (`__hadd`, `__hmul`, `__hpow`, etc.). One kernel file `src/cuda/ops/kernels_fp16.rs` (NEW) holding the FP16 elementwise NVRTC source, mirroring `quantize_kernels.rs` from WS-4.
- **Reduction-class** (2 ops): ReduceMean, Softmax. FP16 IO with f32 accumulator inside the kernel (read FP16 → promote to f32 → reduce → write FP16).
- **Heavy ops** (2 ops): Conv (cuDNN dtype change), MatMul (M3.6 — separate milestone).

- [ ] **Step 1: Survey + ordering decision**

Run a build with M3.3 landed but no op arms: `cargo build --features cuda 2>&1 | grep "error\[" | wc -l`. Count is your scoreboard. The compiler exhaustiveness checker enumerates every site needing a Float16 arm — work down the list one op family at a time.

Suggested sub-step ordering:
1. Memcpy-class (lowest risk; pure dtype dispatch wired to identical kernel as f32).
2. Elementwise (new NVRTC kernels — biggest implementation cost in this milestone).
3. Reduction (Softmax / ReduceMean — accumulator-promotion pattern).
4. Conv (cuDNN dtype descriptor change).

Each sub-step gets its own commit. M3.4's commit count: ~4.

- [ ] **Step 2: Memcpy-class arms (sub-commit 1)**

For each of: Concat, Gather, Reshape, Shape, Transpose, Unsqueeze, Constant — add Float16 arm to the dispatch in `src/cuda/ops/layout/mod.rs` (and individual op files where applicable). Each arm calls the same kernel as f32 with `unsigned short` element type substituted (the kernels are dtype-agnostic at the byte level — same memcpy_general / transpose_general pattern WS-4 used for UInt8).

Add unit test: `tests/transpose_float16_test.rs::transpose_float16_2d` — deterministic FP16 input, expected FP16 output.

```bash
git add ...
git commit -S -m "feat(cuda): add Float16 dispatch to memcpy-class layout ops (WS-3 M3.4 sub-1)"
```

- [ ] **Step 3: Elementwise FP16 NVRTC kernels (sub-commit 2)**

Create `src/cuda/ops/kernels_fp16.rs` holding the FP16 elementwise NVRTC source. Pattern matches `quantize_kernels.rs`:

```rust
pub const FP16_KERNEL_NAMES: &[&str] = &[
    "add_fp16_kernel",
    "sub_fp16_kernel",
    "mul_fp16_kernel",
    "div_fp16_kernel",
    "pow_fp16_kernel",
    "sqrt_fp16_kernel",
    "erf_fp16_kernel",
];

pub const FP16_KERNELS: &str = r#"
#include <cuda_fp16.h>

extern "C" __global__ void add_fp16_kernel(__half* out, const __half* a, const __half* b, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __hadd(a[i], b[i]);
}
// ... mirrors for sub/mul/div/pow/sqrt/erf
"#;
```

Concatenate `FP16_KERNELS` into `OpsKernelCache::new` alongside the existing kernel sources (mirroring how `QUANTIZE_KERNELS` was added in WS-4 M4.4).

For each elementwise op: in `src/cuda/ops/kernels.rs::dispatch_*` add a Float16 dispatch arm calling the new kernel. The wrapper signature follows the existing typed_kernel pattern.

Tests: per-op `tests/{op}_float16_test.rs` (e.g., `tests/add_float16_test.rs`) — one elementwise op per file with shape `[8]`, hand-computed FP16 expected output.

Erf is a special case — CUDA doesn't have a `__half` erf intrinsic. Cast to f32, call `erff`, cast back: `__float2half_rn(erff(__half2float(x[i])))`. Document the f32 detour in the kernel comment.

```bash
git add ...
git commit -S -m "feat(cuda): add FP16 NVRTC elementwise kernels (WS-3 M3.4 sub-2)"
```

- [ ] **Step 4: Reduction FP16 arms (sub-commit 3)**

ReduceMean and Softmax: add FP16 dispatch arms in `src/cuda/ops/reduction.rs` (or wherever they live — search `dispatch_reduction`). Each kernel reads FP16, promotes to f32 for the accumulator, reduces, then converts back to FP16 on write:

```cuda
extern "C" __global__ void softmax_fp16_kernel(__half* out, const __half* x, size_t n, ...) {
    // f32 accumulator, FP16 IO
    float max_v = -INFINITY;
    for (...) max_v = fmaxf(max_v, __half2float(x[...]));
    float sum = 0.0f;
    for (...) sum += expf(__half2float(x[...]) - max_v);
    for (...) out[...] = __float2half_rn(expf(__half2float(x[...]) - max_v) / sum);
}
```

Tests: `tests/softmax_float16_test.rs::softmax_float16_matches_ort_within_1e_3`. ReduceMean: `tests/reduce_mean_float16_test.rs`.

```bash
git commit -S -m "feat(cuda): add FP16 reduction (Softmax/ReduceMean) with f32 accumulator (WS-3 M3.4 sub-3)"
```

- [ ] **Step 5: Conv FP16 path (sub-commit 4)**

cuDNN supports FP16 conv natively. In `src/cuda/cudnn/conv.rs`: extend the descriptor builder to accept `DType::Float16` → `CUDNN_DATA_HALF`. Algorithm selection should still work; cuDNN handles FP16 algos transparently via the descriptor.

Test: `tests/conv_float16_test.rs::conv_float16_3x3_matches_ort_within_1e_3` — small Whisper-shape conv (e.g., `[1, 80, 3000] × [1500, 80, 3]` style).

```bash
git commit -S -m "feat(cuda): add Float16 to cuDNN Conv descriptor (WS-3 M3.4 sub-4)"
```

- [ ] **Step 6: M3.4 acceptance — graph-level shape inference passes**

Black-box test in `tests/whisper_fp16_lower_test.rs`:

```rust
#[test]
fn whisper_fp16_lowers_without_panic() {
    let path = whisper_fp16_path().expect("model present");
    let model = iconnx::OnnxParser::parse_file(&path).expect("parse");
    let weights = model.extract_weights().expect("weights");
    let graph = model.computation_graph().expect("graph");
    assert!(graph.node_count() > 0);
    // Confirms: parser handles dtype 10, IR shape inference walks Float16
    // tensor flow without panicking, ops are all recognized.
}
```

(The `#[ignore]` integration test against ORT lives in M3.7.)

- [ ] **Step 7: Verify clippy + build clean**

`cargo build --features cuda` (no errors), `cargo clippy --features cuda --all-targets -- -D warnings` (clean).

---

## Milestone M3.5 — Cast op + Bool routing for comparison/Where

**Files:**
- Modify: `src/operators/cast.rs` (CPU forward)
- Modify: `src/cuda/ops/cast.rs` (GPU dispatch)
- Modify: `src/cuda/ops/comparison.rs` (Bool output)
- Modify: `src/cuda/ops/sequence.rs` (Where consumes Bool)
- Test: `tests/cast_float16_test.rs` (NEW)
- Test: `tests/bool_routing_test.rs` (NEW)

- [ ] **Step 1: Write the failing tests**

`tests/cast_float16_test.rs`:

```rust
#[test]
#[ignore = "requires CUDA GPU"]
fn cast_float16_to_float32_round_trip() {
    use half::f16;
    let ctx = ...;
    let g_in = GpuTensor::from_host_f16(&ctx, &[f16::from_f32(1.5)], vec![1])?;
    let g_out = gpu_cast(&ctx, &g_in, DType::Float32)?;
    assert_eq!(g_out.to_host_f32(&ctx)?, vec![1.5_f32]);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn cast_float32_to_float16_round_half_to_even() {
    let ctx = ...;
    let g_in = GpuTensor::from_host_f32(&ctx, &[1.0_f32 + 1.0/4096.0], vec![1])?;
    let g_out = gpu_cast(&ctx, &g_in, DType::Float16)?;
    let back = g_out.to_host_f16(&ctx)?[0].to_f32();
    assert!((back - 1.0).abs() < 1e-6);
}
```

`tests/bool_routing_test.rs`:

```rust
#[test]
#[ignore = "requires CUDA GPU"]
fn comparison_produces_bool_then_where_consumes_bool() {
    let ctx = ...;
    let a = GpuTensor::from_host_f32(&ctx, &[1.0, 2.0, 3.0], vec![3])?;
    let b = GpuTensor::from_host_f32(&ctx, &[2.0, 2.0, 2.0], vec![3])?;
    let mask = gpu_less(&ctx, &a, &b)?; // GpuTensor::Bool
    assert_eq!(mask.dtype(), DType::Bool);
    let true_v = GpuTensor::from_host_f32(&ctx, &[10.0, 10.0, 10.0], vec![3])?;
    let false_v = GpuTensor::from_host_f32(&ctx, &[20.0, 20.0, 20.0], vec![3])?;
    let out = gpu_where(&ctx, &mask, &true_v, &false_v)?;
    assert_eq!(out.to_host_f32(&ctx)?, vec![10.0, 20.0, 20.0]);
}
```

- [ ] **Step 2: Run, verify failure**

- [ ] **Step 3: Cast op extends — CPU forward**

In `src/operators/cast.rs::forward`: add `Tensor::Float16` arms to the source-dtype match (cast-from-FP16) and target-dtype branches (cast-to-FP16). Use `half::f16::from_f32` / `half::f16::to_f32` for the arithmetic.

- [ ] **Step 4: Cast op extends — GPU dispatch**

In `src/cuda/ops/cast.rs::dispatch_cast`: add the `(Float16, Float32)`, `(Float32, Float16)`, `(Float16, Int32)`, `(Int32, Float16)` etc. dtype-pair arms. NVRTC kernels: `cast_f16_to_f32_kernel` (`__half2float`), `cast_f32_to_f16_kernel` (`__float2half_rn`), and the integer pairs analogously.

- [ ] **Step 5: Comparison ops produce Bool natively**

In `src/cuda/ops/comparison.rs`: change kernel output dtype from `f32` (1.0/0.0) to `u8` (1/0); wrap in `GpuTensor::Bool`. Update kernels' output pointer type. Existing f32-Bool consumers (only Where today, per audit) get updated in step 6.

- [ ] **Step 6: Where consumes Bool**

In `src/cuda/ops/sequence.rs::dispatch_where`: change condition input handling from `data_f32()?` to `data_bool()?`. Kernel reads the condition as `u8` (nonzero = true).

- [ ] **Step 7: Run all tests**

`cargo test --features cuda --test cast_float16_test --test bool_routing_test -- --include-ignored` → pass.
`cargo test --features cuda` (full suite) → no regressions.

- [ ] **Step 8: Commit**

```bash
git commit -S -m "feat(ops): Cast FP16↔Float32 + Bool routing for comparison/Where (WS-3 M3.5)"
```

---

## Milestone M3.6 — MatMul + Conv FP16 GPU paths

**Files:**
- Modify: `src/cuda/ops/matmul.rs` (FP16 GEMM dispatch)
- Modify: `src/cuda/cudnn/conv.rs` (already updated in M3.4 sub-4; double-check)
- Test: `tests/matmul_float16_test.rs` (NEW)

**Branch decision — garboard FP16 binding status:**
- IF garboard ships `cublasGemmEx` FP16 binding (FP16 IO + FP32 accumulate) before M3.6 entry: use it. cuBLAS handles dispatch / Tensor Cores transparently.
- IF garboard binding slips: write NVRTC FP16 GEMM. Naive tiled `__half` × `__half` → FP32 accumulator → `__half` output. Adequate for correctness; perf falls behind cuBLAS but this is correctness-only per leadline.
- DECISION POINT: at M3.6 entry, `cargo update -p garboard && grep -n "gemm_ex\|fp16\|f16" $(find ~/.cargo/registry/src -path '*garboard*' -name '*.rs' 2>/dev/null) | head -20`. If FP16 gemm function visible, take Path A (cuBLAS). Else Path B (NVRTC).

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
#[ignore = "requires CUDA GPU"]
fn matmul_float16_2x2_matches_ort_within_1e_3() {
    use half::f16;
    let ctx = ...;
    let a = GpuTensor::from_host_f16(&ctx, &[
        f16::from_f32(1.0), f16::from_f32(2.0),
        f16::from_f32(3.0), f16::from_f32(4.0),
    ], vec![2, 2])?;
    let b = ... // 2x2 fp16 weights
    let out = gpu_matmul(&ctx, &a, &b)?;
    assert_eq!(out.dtype(), DType::Float16);
    // Hand-computed expected, tolerance 1e-3.
}

#[test]
#[ignore = "requires CUDA GPU"]
fn matmul_float16_whisper_attention_shape() {
    // [1, 1500, 384] × [384, 384] → [1, 1500, 384] (Whisper-Tiny attention)
    // Cross-check vs ORT to within 1e-2.
    ...
}
```

- [ ] **Step 2: Run, verify failure**

- [ ] **Step 3a (Path A — garboard FP16 ready): cuBLAS dispatch**

Wire `gpu_matmul` FP16 arm through garboard's FP16 GEMM binding. Pattern mirrors the existing f32 cuBLAS dispatch.

- [ ] **Step 3b (Path B — garboard slip): NVRTC FP16 GEMM**

Add `matmul_fp16_kernel` to `src/cuda/ops/kernels_fp16.rs`. Naive tiled GEMM, 16×16 tiles, FP32 accumulator, FP16 IO. Document as "perf placeholder, replace once garboard FP16 binding lands."

Both paths land the same `dispatch_matmul` Float16 arm. Subsequent garboard upgrade is a swap, not a rewrite.

- [ ] **Step 4: 3-D × 2-D batched FP16**

WS-4 M4.7 retrospective: MatMulInteger's 3-D × 2-D batched case was missed in M4.6 and surfaced via runtime panic in M4.7. Whisper attention uses `[1, T, D] × [D, D]` patterns. Apply the same wrapper-level reshape pattern: `[B, M, K] × [K, N]` → reshape `[B*M, K]`, run 2-D GEMM, reshape result back to `[B, M, N]`.

- [ ] **Step 5: Run tests**

- [ ] **Step 6: Commit**

```bash
git commit -S -m "feat(ops): add Float16 MatMul (cuBLAS / NVRTC fallback) (WS-3 M3.6)"
```

---

## Milestone M3.7 — Whisper-Tiny encoder FP16 end-to-end

**Files:**
- Test: `tests/whisper_encoder_fp16_test.rs` (NEW)
- Cross-repo: leadline-bench `WhisperEncoderFp16` ModelProfile (chore commit on rust-ai-explorations).

- [ ] **Step 1: Write the structural integration test**

`tests/whisper_encoder_fp16_test.rs`:

```rust
#![cfg(feature = "cuda")]
use std::path::PathBuf;

fn whisper_fp16_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("ICONNX_WHISPER_FP16_PATH") {
        let path = PathBuf::from(p);
        if path.exists() { return Some(path); }
    }
    let default = PathBuf::from(
        "/home/ryano/workspace/ryanoneill/rust-ai-explorations/\
         models/whisper-tiny.en-export/onnx/encoder_model_fp16.onnx",
    );
    if default.exists() { return Some(default); }
    None
}

#[test]
#[ignore = "requires Whisper-Tiny encoder FP16 model file; override path via ICONNX_WHISPER_FP16_PATH"]
fn whisper_encoder_fp16_loads_and_lowers() {
    let Some(path) = whisper_fp16_path() else {
        eprintln!("skipping: Whisper-Tiny FP16 model not found");
        return;
    };
    let model = iconnx::OnnxParser::parse_file(&path).expect("parse");
    let weights = model.extract_weights().expect("weights");
    let graph = model.computation_graph().expect("graph");
    assert!(graph.node_count() > 0);
    assert!(!weights.is_empty());

    // Confirm at least one Float16 weight surfaced.
    let any_fp16 = weights.values().any(|t| matches!(t, iconnx::Tensor::Float16(_)));
    assert!(any_fp16, "Whisper FP16 must contain at least one Float16 initializer");
}
```

- [ ] **Step 2: Run the structural test**

`cargo test --features cuda --test whisper_encoder_fp16_test -- --ignored --nocapture` → pass.

- [ ] **Step 3: leadline-bench prep**

Cross-repo work in `rust-ai-explorations`:
- Add `WhisperEncoderFp16` variant to `ModelProfile` in `leadline-bench/src/lib.rs` (mirror `DistilBertInt8` structure). Default tolerance 1e-2.
- Probe-ops list already covers the 19 ops; no probe-ops change needed.

These are signed chore commits on rust-ai-explorations main; leadline relay 2026-04-25 confirms the same prep cadence as WS-4 M4.7.

- [ ] **Step 4: Run leadline-bench compare gate**

```bash
cd ~/workspace/ryanoneill/rust-ai-explorations
cargo run --release -p leadline-bench --bin compare -- \
    /home/ryano/workspace/ryanoneill/rust-ai-explorations/models/whisper-tiny.en-export/onnx/encoder_model_fp16.onnx \
    --model whisper-encoder-fp16 \
    --tolerance 1e-2
```

Acceptance: `Tolerance gate: PASS (max_abs_diff <= 0.01)`.

If FAIL, expect surfaced gaps mirroring M4.7's pattern (silent-drop in upload_initializers, missing dispatch arms, fusion matchers needing tighter shape contracts). The plan-oversight checklist in M3.4 step 8 should have caught most of these; remaining ones get patches in this milestone with explicit "M3.7 surfaced X" entries in commit messages.

- [ ] **Step 5: Update WS-3 memory file post-merge**

Memory file: `project_ws3_in_progress.md` → `project_ws3_merged.md` upon merge to main, mirroring the WS-4 transition.

- [ ] **Step 6: Commit**

```bash
git commit -S -m "feat(ws3): land Whisper-Tiny encoder FP16 end-to-end (WS-3 M3.7)"
```

---

## Plan self-review

- **Spec coverage:** Every section of the WS-3 spec (M3.1–M3.7, dtype scaffold, Bool bundling, Cast extension, MatMul backend choice) has at least one task in the plan.
- **Placeholder scan:** No "TBD"s. Garboard FP16 binding contingency has explicit Path A / Path B branches. Bool comparison-op output migration has explicit step (M3.5 step 5).
- **Type consistency:** `Tensor::Float16` / `GpuTensor::Float16` / `DType::Float16` named consistently throughout. Bool surface pinned as `GpuTensor::Bool { data: Arc<DeviceSlice<u8>>, shape }` everywhere.
- **Plan-oversight steps:** M3.1 step 8 (Tensor match arms), M3.3 step 10 (GpuTensor match arms), M3.4 step 1 (compiler-driven exhaustiveness audit), M3.6 step 4 (3-D × 2-D batched MatMul) all explicitly added because WS-4 M4.7 surfaced their absence as runtime panics.
- **Garboard contingency:** M3.6 has Path A / Path B; first 3 milestones don't depend on garboard. Per leadline relay, ~3 milestones of runway is exactly what's expected.

## Execution handoff

Plan complete. Recommend executing via superpowers:subagent-driven-development — fresh subagent per milestone (or per sub-commit for M3.4), two-stage review (spec compliance then code quality) after each.

Phase 1 (M3.1–M3.3) is mechanically identical to WS-4 M4.1–M4.3; should land in 2–3 subagent dispatches each.

Phase 2 (M3.4–M3.5) is the breadth lift — M3.4 has 4 sub-commits (memcpy / elementwise / reduction / conv), each its own subagent dispatch.

Phase 3 (M3.6) gates on garboard. Path A vs Path B decided at entry.

Phase 4 (M3.7) is the integration test — should be brief if M3.4's plan-oversight check did its job.
