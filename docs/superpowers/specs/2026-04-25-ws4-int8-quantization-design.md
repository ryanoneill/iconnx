# WS-4: INT8 Dynamic Quantization Design

> Status: spec, pre-plan
> Roster target: M2.7.f DistilBERT-INT8 (`Xenova/distilbert-base-uncased/onnx/model_int8.onnx`)
> Sequencing: lands before WS-3 FP16; establishes the dtype-extension scaffolding subsequent dtype workstreams reuse.

## Goal

Dynamically-quantized INT8 ONNX models run end-to-end through iconnx with ORT-matched outputs within INT8 tolerance (~1e-2 absolute). Pilot against DistilBERT-INT8; close the M2.7.f roster milestone.

## Probe-ops finding (2026-04-25)

`probe-ops` against `models/distilbert-base-uncased-onnx/onnx/model_int8.onnx` reports **3 unsupported ops**: `DequantizeLinear`, `DynamicQuantizeLinear`, `MatMulInteger`. All three quantization ops are present — the model uses Xenova's standard dynamic-quantization pattern (`DynamicQuantizeLinear` produces UINT8 activations at runtime; INT8 weights are static initializers; `MatMulInteger` consumes UINT8 × INT8 → INT32; `DequantizeLinear` brings results back to Float32).

**This expands the dtype scope:** WS-4 must add both **INT8** (for static weights) and **UINT8** (for dynamic activations). The two variants follow identical scaffolding patterns; bundling them into one workstream avoids fork — for the same architectural reason Bool was deferred to its own slot, INT8 and UINT8 are inseparable for DistilBERT.

## Why this workstream now

- 5 of 6 active roster models already run end-to-end on iconnx (post-WS-2). The only uncovered model class is INT8-quantized.
- **Architectural sequencing** (per the 2026-04-25 plan amendment): WS-4 establishes the dtype-extension surface — Tensor / parser / GpuTensor::DType / upload / ops dispatch all gain a new dtype variant and a new quantization-op family. WS-3 (FP16), WS-3.5 (BF16), and the Bool-on-GPU debt closure all reuse this pattern. Doing INT8 first lets later dtypes batch onto a known-good scaffold rather than re-architecting it three times.

## Scope decisions taken before this spec

| Decision | Verdict | Rationale |
|---|---|---|
| Bool-on-GPU bundling | **Defer** | Architectural overlap is small (Tensor::Bool already exists; parser already handles BOOL). INT8 is intrinsically required (M2.7.f); Bool is debt closure (audit L2.1 + L3.5). Mixing dilutes review/revert scope. Bool batches cleanly into a later dtype-architecture sweep. |
| MatMulInteger backend (M4.6) | **Custom NVRTC INT8 GEMM for MVP** | Garboard's `BlasDataType` only impls f32/f64; INT8 cuBLAS via `cublasGemmEx` is not exposed. WS-4's bar is correctness within 1e-2, not performance. NVRTC keeps WS-4 self-contained. cuBLAS INT8 (Tensor Cores / IMMA) tracked as a perf-workstream follow-up. Mirrors the MaxPool precedent (NVRTC even though cuDNN has pooling). |
| Quantization scope | **Per-tensor + per-channel** | Per the M4.4 acceptance criteria. Per-channel = per-axis scale/zero_point broadcast along the quantization axis. |
| Symmetric vs asymmetric | **Both — `zero_point` parameter respected** | DequantizeLinear's `zero_point` input is optional in the ONNX spec; symmetric (zero_point=0) and asymmetric paths share one kernel that just passes `zp = zero_point.unwrap_or(0)`. |
| INT8 GPU representation | **Native `i8` in `DeviceSlice`** | Matches the existing dtype pattern (Float32 / Int64 / Int32 each have native `DeviceSlice<T>` storage). Packed Int32 representation rejected — it adds shift/mask overhead on every access and complicates kernel parameterization. |
| INT8 element type for CPU `Tensor` | **`i8`** | Direct mapping. ndarray supports `ArrayD<i8>`. |
| UINT8 inclusion (post-probe-ops finding) | **In scope** | `DynamicQuantizeLinear` outputs UINT8; `MatMulInteger` in DistilBERT consumes UINT8 × INT8. Adding UINT8 alongside INT8 in this workstream is the only way to reach M2.7.f. Both follow identical dtype-extension scaffolding. |
| UINT8 element type for CPU `Tensor` | **`u8`** | Direct mapping. ndarray supports `ArrayD<u8>`. |

## Architecture

### Dtype extension surface

Five iconnx surfaces gain Int8 + UInt8 variants/paths in lockstep:

1. **`Tensor` enum** (`src/tensor.rs`) — new `Int8(ArrayD<i8>)` and `UInt8(ArrayD<u8>)` variants. Constructors `from_vec_i8` / `from_vec_u8`, accessors `as_slice_i8` / `as_slice_u8` / `to_array_i8` / `to_array_u8`, `dtype()` returns `"int8"` / `"uint8"`, `is_int8()` / `is_uint8()`. Mirrors the existing Int32 / Int64 pattern.
2. **`OnnxParser`** — `parse_tensor_proto` and `extract_weights` decode ONNX dtypes 3 (INT8) and 2 (UINT8). Wire format for both: `int32_data` field carries 1 i32 per element (low 8 bits, sign-extended for INT8); `raw_data` field carries 1 byte per element. Mirrors the WS-1 M1.1 / M1.2 dtype-extension pattern.
3. **`GpuTensor` + `DType`** (`src/cuda/tensor.rs`) — new `DType::Int8` and `DType::UInt8` variants; new `GpuTensor::Int8 { data: Arc<DeviceSlice<'static, i8>>, shape }` and `GpuTensor::UInt8 { data: Arc<DeviceSlice<'static, u8>>, shape }`; `data_i8` / `data_u8` (+ `_mut` variants) accessors. Memory pool gains `get_tensor_i8` / `get_tensor_u8`. `from_host_i8` / `to_host_i8` (+ `_u8`). `upload_tensor` arms upload natively (no cast).
4. **Three new ONNX ops** — DequantizeLinear, DynamicQuantizeLinear, MatMulInteger. Each follows the existing op pattern (CPU forward + GPU dispatch + shape inference + folding). Details below.
5. **Probe-ops list** — leadline-bench's `ICONNX_SUPPORTED_OPS` gains the three new op names.

### Quantization op semantics

#### DequantizeLinear (M4.4)

ONNX spec: `y = (x - zero_point) * scale`. Inputs:
- `x` — INT8 **or** UINT8 **or** INT32 tensor (DistilBERT uses all three: UINT8 activations dequantize for elementwise ops, INT32 MatMulInteger results dequantize back to Float32).
- `x_scale` — Float32 scalar (per-tensor) or 1-D vector (per-axis).
- `x_zero_point` — same dtype as `x`, optional. Defaults to 0 if absent.

Attributes:
- `axis` — quantization axis when `x_scale` is per-channel. Defaults to 1.

Output: Float32 tensor, same shape as `x`.

Implementation: single elementwise NVRTC kernel parameterized over per-tensor / per-axis scale via a stride argument (stride 0 for per-tensor scalar broadcast). CPU forward via `dispatch_multi`-style per-dtype expansion.

#### DynamicQuantizeLinear (M4.5)

ONNX spec: scales a Float32 tensor to UINT8 with dynamically-computed per-tensor scale and zero_point:

```
y_scale = (max(x) - min(x)) / 255
intermediate_zero_point = round(0.0 - min(x) / y_scale)
y_zero_point = clip(intermediate_zero_point, 0, 255)
y = saturate(round(x / y_scale) + y_zero_point)
```

This is iconnx's **first multi-output op with three distinct outputs** — `(y: UINT8, y_scale: Float32, y_zero_point: UINT8)`. Reuses the `dispatch_op_multi` infrastructure that landed for Split (`127d763`). Each output gets its own GpuTensor; `store_outputs` distributes via the existing distinct-multi-output path.

Implementation: two-pass GPU computation —
1. **Reduce pass:** scan input for `min` and `max` into pinned scalar staging (single kernel launch with shared-memory reduction).
2. **Quantize pass:** elementwise kernel reads `(min, max)`, computes `scale` / `zero_point` inline, writes UINT8 output and the two scalar outputs.

CPU forward: straight-line iteration computing min/max then quantize. Round-half-to-even semantics — Rust's `f32::round_ties_even()` is the right primitive (added Rust 1.77).

#### MatMulInteger (M4.6)

ONNX spec: matrix multiplication with 8-bit inputs and INT32 accumulator output:
```
y[i, j] = sum_k (a[i, k] - a_zp) * (b[k, j] - b_zp)
```

Inputs:
- `A` — INT8 **or** UINT8 tensor (typically rank-2 matrix, but supports broadcasting per ONNX spec).
- `B` — INT8 **or** UINT8 tensor.
- `a_zero_point` — same dtype as `A`, scalar or per-row vector. Optional; defaults to 0.
- `b_zero_point` — same dtype as `B`, scalar or per-column vector. Optional; defaults to 0.

Output: INT32 tensor.

DistilBERT's specific combo is **UINT8 × INT8 → INT32**: dynamic UINT8 activations multiplied by static INT8 weights. M4.6's MVP supports this combo plus the symmetric INT8 × INT8 path; UINT8 × UINT8 and INT8 × UINT8 fall out of the same kernel parameterization at no extra cost (just dtype dispatch in the executor arm).

Implementation: **NVRTC 8-bit GEMM kernel** writing to INT32 accumulator. Standard tiled-GEMM structure with 8-bit → i32 promotion before the multiply-accumulate. Promotion path:
- INT8 input: signed promote (`(int32_t)a_i8`).
- UINT8 input: unsigned promote (`(int32_t)a_u8`, no sign-extension).

Per-row / per-column zero-point subtraction handled inline (subtract before promote). For DistilBERT's 768×768 / 768×3072 / 3072×768 matrix shapes the naive tiled-GEMM is sufficient; performance optimization (Tensor Cores via cuBLAS IMMA) is a perf-workstream follow-up.

**Symmetric quantization fast path:** when both zero_points are 0, skip the subtraction step. Detected at executor dispatch time.

**Kernel parameterization:** four kernels (signed/unsigned × signed/unsigned for the two operands) versus one kernel with runtime dtype branches — the four-kernel approach is simpler to reason about and matches the existing dtype-specific kernel pattern (gpu_add has separate add_kernel and add_i64_kernel; same shape here). Total: `matmul_integer_s8s8_kernel`, `matmul_integer_u8s8_kernel`, `matmul_integer_s8u8_kernel`, `matmul_integer_u8u8_kernel`. The executor dispatches based on input dtypes.

### Execution flow for a DistilBERT-INT8 attention block

```
Float32 activations
  → DynamicQuantizeLinear (3 outputs: UINT8 quantized, Float32 scale, UINT8 zero_point)
  → MatMulInteger (UINT8 activations × INT8 static weights → INT32 output)
  → DequantizeLinear (INT32 → Float32, combined scale = a_scale * b_scale)
  → continues as Float32 through LayerNorm / softmax / etc.
```

The 8-bit dtype boundary is bounded — UINT8 / INT8 live only between the quantize and dequantize endpoints. iconnx's existing executor + memory pool infrastructure handles the new dtypes as peers of Int32 / Int64.

## Milestones

Table mirrors leadline's relay verbatim, with implementation pointers added.

| M | Title | Acceptance | Files touched |
|---|---|---|---|
| **M4.1** | CPU Tensor adds Int8 + UInt8 variants | 4 unit tests (2 per dtype); `Tensor::from_vec_i8` / `from_vec_u8`, accessors, `dtype()` strings | `src/tensor.rs` |
| **M4.2** | Parser reads INT8 + UINT8 initializers | 4 unit tests (`extract_weights` and `parse_tensor_proto` × 2 dtypes); zero-element fast path extended | `src/onnx_parser.rs` |
| **M4.3** | GPU DType adds Int8 + UInt8 | 2 contract tests (1 per dtype); htod/dtoh round-trip bit-identical | `src/cuda/tensor.rs`, `src/cuda/memory_pool.rs`, `src/cuda/executor/mod.rs::upload_tensor` + `collect_outputs` |
| **M4.4** | DequantizeLinear op | 4 unit tests (per-tensor INT8, per-channel INT8, UINT8 with zero_point, INT32 input from MatMulInteger output) | `src/operators/dequantize_linear.rs`, `src/cuda/ops/quantize/mod.rs` (NEW), executor dispatch |
| **M4.5** | DynamicQuantizeLinear op | 3 unit tests (ORT-conformance on fixed inputs); 1 multi-output dispatch test through GpuGraphExecutor | `src/operators/dynamic_quantize_linear.rs`, GPU min/max + quantize kernels, multi-output dispatch via `dispatch_op_multi` |
| **M4.6** | MatMulInteger op | 4 unit tests (UINT8 × INT8 small matmul, INT8 × INT8 symmetric, with zero_points, larger DistilBERT-shape) → INT32 output | `src/operators/matmul_integer.rs`, 4 NVRTC 8-bit GEMM kernels (s8s8 / u8s8 / s8u8 / u8u8) |
| **M4.7** | DistilBERT-INT8 end-to-end | `cargo run -p leadline-bench --bin compare ... --model distilbert-int8`; ORT-matched within 1e-2 | leadline-bench `DistilBertInt8` ModelProfile, integration test in iconnx `tests/distilbert_int8_test.rs` |

### Test design notes

- M4.1–M4.3 follow the existing dtype-extension test pattern (Int32 / Int64 precedents).
- M4.4–M4.6 use ORT as the conformance reference where feasible. For pure-numerical ops with closed-form expected output (e.g., DequantizeLinear with hand-computed scales), unit tests assert direct equality at f32 precision. For DynamicQuantizeLinear's `round-half-to-even` semantics, fix the input precisely so ORT and iconnx produce bit-identical outputs.
- M4.7 is the integration test. Tolerance 1e-2 reflects INT8 quantization's intrinsic accuracy floor — DistilBERT's logits may differ from the f32 reference within that band even when iconnx and ORT agree perfectly with each other.

## Cross-repo handoffs

| Repo | Item | Status |
|---|---|---|
| **leadline-bench** | New `DistilBertInt8` ModelProfile variant | Standard pattern (mirrors recent additions). Single chore commit alongside M4.7. |
| **leadline-bench** | probe-ops `ICONNX_SUPPORTED_OPS` adds `DequantizeLinear`, `DynamicQuantizeLinear`, `MatMulInteger` | Three lines, single chore commit alongside the relevant feat commits. |
| **garboard** | `cublasGemmEx` INT8 binding | **Deferred — not on WS-4's critical path.** Tracked as a perf-workstream follow-up. WS-4 uses NVRTC for M4.6. |

## Risks & open questions

1. ~~**DistilBERT-INT8 actually uses which ops?**~~ — **Resolved 2026-04-25:** probe-ops confirms all three quantization ops are in the graph. UINT8 is in scope.
2. **INT32 accumulator in MatMulInteger overflow** — for large K (DistilBERT FFN: K=3072), INT8×INT8 → INT32 sums can exceed INT32 range only if every element is ±127. Realistic quantized weights stay well within range. Document the assumption; add a sanity check in debug builds.
3. **Per-channel quantization axis** — ONNX DequantizeLinear's `axis` attribute defaults to 1, but per-channel quantized Conv weights typically use axis 0. Test fixtures must cover both.
4. **Zero-element INT8/UINT8 tensors** — apply the WS-1 M1.2 zero-element fix symmetrically. Parser zero-element fast path adds dtypes 2 and 3.
5. **`as_slice_i8` / `as_slice_u8` panic shape** — extends the existing f32-strict accessor pattern. Audit L2.1 broader sweep stays out of WS-4 scope; new accessors follow existing conventions, future polymorphism happens in WS-1 M1.4 surfacing.
6. **`round_ties_even` MSRV check** — Rust 1.77 added `f32::round_ties_even()`. iconnx's MSRV must accommodate (likely already does given other 1.77+ uses).
7. **`DynamicQuantizeLinear` constant folding** — when input is a constant tensor, the op can be folded at compile time. M4.5's `forward` works in `folding_core::dispatch_cpu_forward`'s single-tensor return convention how? **Decision:** like Split, exclude DynamicQuantizeLinear from constant folding (multi-output ops aren't foldable in folding_core today). DistilBERT's DynamicQuantizeLinear inputs are never compile-time constant in practice — they're dynamic activations.

## Spec self-review

- **Placeholder scan:** No "TBD"s. All milestones have concrete acceptance criteria.
- **Internal consistency:** M4.5 / M4.6 / M4.4 all reference UINT8 consistently. The dtype-extension pattern is uniform across M4.1–M4.3.
- **Scope check:** 7 milestones, single workstream, focused on one roster model. Does not bundle Bool, FP16, or BF16 — those are separate workstreams per the project plan. The UINT8 inclusion is forced by DistilBERT's actual ops, not speculative scope creep.
- **Ambiguity check:** "Per-tensor" vs "per-channel" defined explicitly; "symmetric" / "asymmetric" defined via zero_point semantics; "round-half-to-even" called out for DynamicQuantizeLinear; signed vs unsigned promotion in MatMulInteger explicit.

## Next step

User reviews the spec. Once approved, draft the implementation plan covering all 7 milestones. Plan structure follows the established `docs/superpowers/plans/YYYY-MM-DD-<feature>.md` shape.

Probe-ops baseline confirms the entry path is clean: only the three quantization ops are unsupported on DistilBERT-INT8; everything else (Add / Cast / Concat / Div / Equal / Erf / Expand / Gather / MatMul / Mul / Pow / ReduceMean / Reshape / Shape / Slice / Softmax / Sqrt / Sub / Transpose / Unsqueeze / Where) is already covered.
