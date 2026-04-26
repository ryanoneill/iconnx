# WS-3: FP16 Half-Precision + Bool-on-GPU Design

> Status: spec, pre-plan
> Roster target: M2.7.b Whisper-Tiny encoder FP16 (`Xenova/whisper-tiny.en/onnx/encoder_model_fp16.onnx`)
> Sequencing: lands after WS-4 INT8; reuses the dtype-extension scaffolding pattern WS-4 just battle-tested.

## Goal

FP16 ONNX models run end-to-end through iconnx with ORT-matched outputs within FP16 tolerance (~1e-2 absolute). Pilot against Whisper-Tiny encoder FP16; close the M2.7.b roster milestone. Bundle the long-deferred Bool-on-GPU debt closure into the same workstream so the dtype-extension scaffold is exercised against two new dtypes simultaneously instead of being re-walked twice.

## Probe-ops finding (2026-04-25)

`probe-ops` against `models/whisper-tiny.en-export/onnx/encoder_model_fp16.onnx` reports **0 unsupported ops** at the op-name layer. All 19 ops (`Add`, `Cast`, `Concat`, `Constant`, `Conv`, `Div`, `Erf`, `Gather`, `MatMul`, `Mul`, `Pow`, `ReduceMean`, `Reshape`, `Shape`, `Softmax`, `Sqrt`, `Sub`, `Transpose`, `Unsqueeze`) are already covered by the existing iconnx op set.

The challenge in WS-3 is therefore **not new ops** — it's pure dtype propagation: every initializer and every intermediate tensor in the Whisper FP16 graph is dtype 10 (FLOAT16), and iconnx has no Float16 representation today. Existing op kernels are f32-only; getting the model to run is "the dispatch arms accept FP16 and call the right kernel," not "write new ops from scratch."

This makes WS-3 substantially smaller than WS-4. M3.4 is the only milestone with non-trivial breadth (one FP16 dispatch arm per op family); everything else is dtype scaffolding identical to WS-4 M4.1–M4.3.

## Why this workstream now

- WS-4 just landed (commit `802ec8b`) and proved the dtype-extension scaffold works. Repeating it for Float16 + Bool while the pattern is fresh in the codebase keeps the friction at near-mechanical levels.
- **Architectural sequencing** (per the 2026-04-25 plan amendment): WS-3 reuses everything WS-4 added — Tensor variants, parser dtype dispatch, GpuTensor / DType variants, memory-pool typed accessors, upload-tensor arms, dispatch-op-multi (not needed here but available), the Constant-folding plan-oversight checklist that surfaced 5 plan gaps in M4.7. Doing FP16 second amortizes the scaffolding cost.
- **Bool-on-GPU is debt** (audit L2.1 + L3.5): `Tensor::Bool` already exists and the parser already handles dtype 9, but GPU upload casts to f32 and there is no `GpuTensor::Bool`. WS-3 is the natural rendezvous to close this — same scaffolding milestones, same review surface.

## Scope decisions

| Decision | Verdict | Rationale |
|---|---|---|
| Bool bundling | **In scope** | Parallels WS-4's UINT8 inclusion: the dtype-scaffolding milestones (M3.1–M3.3) extend two variants in lockstep at near-zero marginal cost. CPU Bool already exists; only the GPU side and parser GPU-upload arm are missing. |
| Float16 element type for CPU `Tensor` | **`half::f16`** | The `half` crate (already a transitive dep via `ndarray-stats` / similar) provides `f16` with full IEEE-754 binary16 semantics. ndarray supports `ArrayD<half::f16>`. |
| Float16 GPU representation | **Native `f16` in `DeviceSlice`** | Matches the existing dtype pattern (Float32 / Int64 / Int32 / Int8 / UInt8 each have native `DeviceSlice<T>` storage). Garboard's CUDA bindings already type-parameterize `DeviceSlice<T: DeviceRepr>`; `f16` will plug in once the FP16 BLAS binding lands. |
| FP16 MatMul backend (M3.6) | **Garboard `cublasGemmEx` FP16 binding** | Garboard relay 2026-04-25: FP16 binding ask is going out in parallel and is expected ready before M3.6. Falls back to NVRTC if the binding slips. Mirrors the M4.6 contingency pattern. |
| FP16 Conv backend | **cuDNN FP16 path** | cuDNN's existing `convolution_forward` already supports FP16 via descriptor dtype = `CUDNN_DATA_HALF`. M3.4's Conv arm exposes this; no new kernel work. |
| FP16 reduction accumulator | **Float32 accumulator, FP16 input/output** | IEEE binary16's 3-decimal-digit precision overflows for ReduceMean / Softmax denominators. cuDNN, ORT, and PyTorch all use Float32 accumulators internally for FP16 reductions. Document the contract; verify against ORT in the integration test. |
| Bool element type (CPU) | **`bool`** | Already in place. `Tensor::Bool(ArrayD<bool>)` exists. |
| Bool element type (GPU) | **`u8` via `DeviceSlice<u8>`** | CUDA has no native `bool` type; conventional choice is byte-per-element. iconnx's existing comparison ops already produce f32-1.0/0.0 ad-hoc; M3.3's `GpuTensor::Bool { data: Arc<DeviceSlice<'static, u8>>, shape }` formalizes the representation. |
| FP16 ↔ Float32 boundary | **`Cast` op extends** | Whisper graphs typically have explicit Cast nodes at FP16 boundaries. The existing `Cast` op gains FP16 ↔ Float32 (and FP16 ↔ Int*) arms. No new op. |
| FP16 round-half-to-even | **In-scope; same parity contract as M4.5** | FP16 quantization (`x.to_f16()`) round semantics matter for ORT conformance. Use `half::f16::from_f32` which uses IEEE round-half-to-even. CUDA's `__float2half_rn` is the matching intrinsic. M3.5 verifies parity. |

## Architecture

### Dtype extension surface

Five iconnx surfaces gain Float16 + Bool variants/paths in lockstep, mirroring WS-4:

1. **`Tensor` enum** (`src/tensor.rs`) — new `Float16(ArrayD<half::f16>)` variant. Constructors `from_vec_f16` / `from_array_f16`, accessors `as_slice_f16` / `to_array_f16`, `dtype()` returns `"float16"`, `is_float16()`. Bool already exists; M3.1 audits the accessor surface for symmetry. Mirrors the existing Int8 / UInt8 pattern from WS-4 M4.1.
2. **`OnnxParser`** — `parse_tensor_proto` and `extract_weights` decode ONNX dtype 10 (FLOAT16). Wire format: `int32_data` field carries 1 i32 per element (low 16 bits = the f16 bit pattern); `raw_data` field carries 2 bytes per element little-endian. Bool (dtype 9) parser path already in place; M3.2's parser scope is FP16-only.
3. **`GpuTensor` + `DType`** (`src/cuda/tensor.rs`) — new `DType::Float16` and `DType::Bool` variants; new `GpuTensor::Float16 { data: Arc<DeviceSlice<'static, half::f16>>, shape }` and `GpuTensor::Bool { data: Arc<DeviceSlice<'static, u8>>, shape }`; `data_f16` / `data_bool` (+ `_mut` variants) accessors. Memory pool gains `get_tensor_f16` / `get_tensor_bool`. `from_host_f16` / `to_host_f16` (+ Bool). `upload_tensor` arms upload natively (no cast — Bool today casts to f32; that path retires).
4. **Op dispatch arms — FP16 across 19 ops** (M3.4) — every op the Whisper FP16 graph uses gains an FP16 dispatch arm. The bulk of the work: NVRTC kernel rewrites for elementwise ops (Add / Sub / Mul / Div / Pow / Erf / Sqrt) using `__half` intrinsics; cuDNN dtype-descriptor change for Conv; FP16-aware ReduceMean / Softmax with f32 accumulator; memcpy-class ops (Concat / Reshape / Shape / Transpose / Unsqueeze / Gather) gain Float16 arms (byte-level — straight `unsigned short` mirror).
5. **Probe-ops list** — leadline-bench's `ICONNX_SUPPORTED_OPS` already covers all 19 ops; M3.7 adds `WhisperEncoderFp16` ModelProfile.

### FP16 / Bool semantics through the existing op surface

The Whisper-Tiny encoder forward pass:

```
FP16 input
  → Conv (FP16 weights, cuDNN CUDNN_DATA_HALF)
  → Add (FP16 bias)
  → GELU (Mul/Add/Pow chain — FP16 elementwise)
  → MatMul (FP16 × FP16 → FP16, garboard cublasGemmEx FP16 IO)
  → Softmax (FP16 IO with f32 accumulator)
  → ReduceMean / LayerNorm (FP16 IO with f32 accumulator)
  → ...continues purely in FP16 until the model output.
```

Bool flows narrower — most graphs use Bool only for the comparison-op outputs that feed into `Where` and similar masking ops. Whisper's encoder doesn't use Bool itself, but bundling it lets future encoders/decoders that DO use Bool (Whisper decoder, BERT-with-attention-mask, etc.) work without another scaffolding pass.

### Mixed-precision conventions

- **Constant folding** — folding-core's CPU forward path for FP16 ops uses `half::f16` directly. Constant-folded FP16 tensors stay FP16 through the IR (no f32 promotion).
- **Accumulator dtype for reductions** — ReduceMean / Softmax / LayerNorm use Float32 accumulators internally (kernel-level) and write FP16 output. This matches cuDNN / ORT / PyTorch.
- **Cast op** — gains FP16 ↔ Float32 (the canonical bridge) plus FP16 ↔ Int8/UInt8/Int32/Int64 for completeness. CPU forward uses `half::f16::from_f32` and `half::f16::to_f32` (IEEE round-half-to-even). GPU uses `__float2half_rn` / `__half2float`.

## Milestones

Table mirrors leadline's relay verbatim, with implementation pointers added.

| M | Title | Acceptance | Files touched |
|---|---|---|---|
| **M3.1** | CPU Tensor adds `Float16` variant; Bool surface audit | 4 unit tests (Float16 constructors / accessors / dtype string / round-trip via `from_f32` / `to_f32` parity); Bool accessor symmetry verified against the Int8/UInt8 pattern | `src/tensor.rs` |
| **M3.2** | Parser reads FP16 initializers (dtype 10) | 3 unit tests (`extract_weights` for FP16 raw_data + int32_data paths; zero-element FP16 fast path) | `src/onnx_parser.rs` |
| **M3.3** | GPU DType adds Float16 + Bool | 2 contract tests (Float16 htod/dtoh round-trip bit-identical; Bool htod/dtoh round-trip bit-identical) | `src/cuda/tensor.rs`, `src/cuda/memory_pool.rs`, `src/cuda/executor/mod.rs::upload_tensor` + `collect_outputs` |
| **M3.4** | Existing op dispatch arms gain Float16 path | Op-by-op unit tests (one Float16 path test per op family touched: elementwise / reduction / matmul / conv / memcpy / cast); plus a black-box test that loads the Whisper FP16 graph through `ir::lower` and asserts shape inference flows through without a panic | All 19 op kernel files; CUDA dispatch arms in `src/cuda/ops/{kernels.rs,layout/mod.rs,reduction.rs,...}` |
| **M3.5** | Cast op + comparison/Where Bool dispatch | 5 unit tests (Cast Float16↔Float32, Cast Float16↔Int32, comparison op produces Bool GpuTensor, Where consumes Bool, end-to-end Bool routing via comparison→Where chain) | `src/cuda/ops/cast.rs`, `src/cuda/ops/comparison.rs`, `src/cuda/ops/sequence.rs` (Where) |
| **M3.6** | MatMul GPU FP16 path | 4 unit tests (FP16 × FP16 → FP16 small matmul; aligned FP16 matmul against ORT for Whisper attention shapes [1, 1500, 384] × [384, 384] etc.; tolerance 1e-2; cuDNN Conv FP16 path tested against ORT in same milestone) | `src/cuda/ops/matmul.rs`, garboard FP16 GEMM consumption (or NVRTC fallback if binding slips), `src/cuda/cudnn/conv.rs` for FP16 descriptor |
| **M3.7** | Whisper-Tiny encoder FP16 end-to-end | `cargo run -p leadline-bench --bin compare ... --model whisper-encoder-fp16 --tolerance 1e-2`; ORT-matched within 1e-2 (FP16 precision floor) | leadline-bench `WhisperEncoderFp16` ModelProfile (chore commit), integration test in iconnx `tests/whisper_encoder_fp16_test.rs` (structural + tolerance gate via cross-repo bench) |

### Test design notes

- M3.1–M3.3 follow the dtype-extension test pattern from WS-4 M4.1–M4.3 (and Int32 / Int64 precedents before that).
- M3.4 is breadth-first: one test per op family. The integration test in M3.7 exercises the combinations.
- M3.5 specifically targets the Bool-on-GPU debt closure. Producing Bool from comparison ops, consuming Bool in Where, round-tripping via the Cast op all need explicit coverage to retire the audit-L2.1 flag.
- M3.6 may stage the cuDNN Conv FP16 work into M3.4 (under the Conv op family arm) or into M3.6 (alongside MatMul) — implementation-time choice. The plan resolves this once the M3.4 breadth survey identifies the actual cuDNN-side delta.
- M3.7 tolerance 1e-2 reflects FP16's 3-decimal-digit precision floor — even ORT vs PyTorch on the same Whisper encoder differ by ~5e-3 over 1500 frames.

## Cross-repo handoffs

| Repo | Item | Status |
|---|---|---|
| **leadline-bench** | New `WhisperEncoderFp16` ModelProfile variant | Standard pattern (mirrors recent `DistilBertInt8` addition). Single chore commit alongside M3.7. leadline relay 2026-04-25 confirms the prep cadence will mirror WS-4 M4.7. |
| **leadline-bench** | probe-ops `ICONNX_SUPPORTED_OPS` already covers Whisper's 19 ops | No probe-ops change required. |
| **garboard** | `cublasGemmEx` FP16 binding (FP16 IO + FP32 accumulate) | leadline relay 2026-04-25: ask going out in parallel with WS-3 kickoff. Expected ready before M3.6. M3.4 (~3 milestones runway) does not depend on it. |
| **garboard** | `f16` in `DeviceRepr` impls | Required for `DeviceSlice<f16>` to compile. May ship in the same garboard PR as the BLAS binding. |

## Risks & open questions

1. **`half::f16` already in iconnx's transitive deps?** — likely yes via `ndarray-stats` or similar; if not, an explicit `half = "2"` dependency is added in M3.1. No risk; well-established crate.
2. **FP16 elementwise NVRTC kernels — `__half` ABI** — CUDA's `__half` type is a 16-bit struct on the device; intrinsics like `__hadd` / `__hmul` / `__hpow` need `cuda_fp16.h` included. M3.4 adds the include and verifies via a smoke kernel.
3. **FP16 reduction accumulator dtype** — confirmed Float32 above. The cuDNN / ORT / PyTorch precedent is unambiguous; no design ambiguity, just an implementation note.
4. **Garboard FP16 binding slipping** — fallback is NVRTC FP16 GEMM (analogous to WS-4 M4.6). Naive tiled GEMM with `__half` operands and `float` accumulator is straightforward; performance-wise it's worse than IMMA / Tensor Cores but adequate for correctness gating. Plan documents both paths; resolves at M3.6 entry.
5. **Bool comparison-op output today** — comparison ops currently materialize Bool as Float32 1.0/0.0 on GPU (`comparison.rs`). M3.5 must migrate to producing `GpuTensor::Bool` natively, then update Where (the only consumer) to read from Bool dtype. Backward-compat: no public API breaks since Bool was never a documented GPU dtype.
6. **Whisper decoder is not in scope** — Whisper-Tiny's full encoder-decoder pair would need cross-attention KV-cache + autoregressive decoding. WS-3's pilot is the encoder only (a single forward pass, audio-features in → encoder-states out). Decoder integration tracked as a follow-on.
7. **`half::f16` literals in tests** — Rust's `1.5_f16` literal requires nightly. Tests use `half::f16::from_f32(1.5)` for stable Rust. Documented as a one-line note in the M3.1 step.
8. **Mixed-precision activation models** — Whisper FP16 has a model_input cast at the boundary (Float32 audio features → FP16 internals). The Cast extension in M3.5 handles this; no special-casing needed.

## Spec self-review

- **Placeholder scan:** No "TBD"s. All milestones have concrete acceptance criteria.
- **Internal consistency:** M3.1–M3.3 mirror M4.1–M4.3 dtype-extension structure; M3.4 / M3.6 reflect the WS-3 specifics (op-arm breadth + GEMM backend). Bool inclusion is consistent across M3.1 (audit), M3.3 (GPU variant), M3.5 (op coverage).
- **Scope check:** 7 milestones, single workstream, focused on one roster model (Whisper-Tiny encoder). Bundled Bool is debt closure not scope creep. No FP16-only ops introduced (probe-ops finding); the work is dispatch arms across existing infrastructure.
- **Ambiguity check:** "Float16 representation" pinned to `half::f16` and `DeviceSlice<half::f16>`. "FP16 reduction accumulator" pinned to Float32. "Bool GPU" pinned to `DeviceSlice<u8>`. Round-half-to-even contract pinned to `half::f16::from_f32` (CPU) and `__float2half_rn` (GPU).

## Next step

User reviews the spec. Once approved, draft the implementation plan covering all 7 milestones. Plan structure follows the established `docs/superpowers/plans/YYYY-MM-DD-<feature>.md` shape.

Probe-ops baseline confirms the entry path is clean: zero unsupported ops on Whisper-Tiny encoder FP16 — every op the model uses already exists in iconnx. WS-3 is dtype scaffolding, not new ops.
