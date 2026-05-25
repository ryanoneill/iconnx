# WS-6 GPU OCR Enablement — Design

- **Date:** 2026-05-24
- **Status:** Design rev.2 (brainstorm complete; M6.0 de-risk spike + two adversarial reviews done; pending plan)
- **Driver:** pdfocr requires iconnx's CUDA path to run two pinned Apache-2.0 OCR models. Source requirements: `../pdfocr/docs/superpowers/specs/2026-05-24-iconnx-gpu-ocr-requirements.md` (rev.3) and `../pdfocr/docs/superpowers/specs/2026-05-24-iconnx-op-diff.md`.
- **Prior-art cadence:** WS-3.5 (`2026-04-26-ws3.5-bf16-design.md`) and WS-5 (`2026-04-29-ws5-observability-design.md`) — milestone-per-signed-commit, subagent-driven execution with two-stage review, sibling worktree, GPG-signed.

---

## 1. Goal & context

Make iconnx's CUDA `GpuGraphExecutor` run both pinned models end-to-end with ONNX-Runtime-validated correctness, consumable by pdfocr through a one-call API:

- **REC** `en_number_mobile_v2.0_rec` — MobileNetV3-Small + BiLSTM×2 + CTC; opset 12; input `[N,3,H,W]` (dynamic), output `(N,T,C)` C=97; SHA-256 `e679ba62…11c01`.
- **DET** `en_PP-OCRv3_det` — DBNet + MobileNetV3-Large; opset 14; input `[N,3,H,W]` (dynamic), output prob-map `(N,1,H,W)`; SHA-256 `f139598b…a866b`.

fp32 only (iconnx's cuDNN conv/LSTM bindings are f32; FP16 has documented gaps). pdfocr owns preprocessing, CTC decode, and box postprocess; iconnx owns the model outputs (rec logits, det probability map).

This workstream corrects a critical misconception in the op-diff: **op-name coverage ≠ semantic coverage.** An op appearing in `OP_SUPPORT_TABLE` does not mean every attribute/opset-form the OCR models use is handled. Adversarial review + the M6.0 spike found two model-critical gaps behind "supported" ops (grouped Conv2D, input-form Clip) that no roster model had ever exercised.

---

## 2. M6.0 de-risk spike — outcome (foundational, already executed)

A throwaway spike ran on the RTX 4070 (cuDNN) with `ort` (CPU EP) as oracle, loading 1-node ONNX graphs identically into both engines. No upstream-contract blockers found.

- **Grouped Conv2D → cuDNN routing (approach a), bit-exact.** `garboard::ConvForwardConfig` (`dnn_config.rs:44-60`) takes `input_dims [N,C,H,W]`, `filter_dims [Cout, Cin/groups, kH, kW]`, 2D `padding`/`stride`/`dilation`, and `groups: i32`, calling `cudnnSetConvolutionGroupCount`. iconnx's `garboard_conv_1d` (`src/cuda/cudnn/mod.rs:~29`) already drives this for 1D; the 2D restriction is iconnx's wrapper, not garboard's. Probe parity vs ORT: `max_abs_diff = 0.000e0` at group=8 (depthwise `[1,8,16,16]`,W`[8,1,3,3]`) and group=2 (`[1,8,16,16]`,W`[16,4,3,3]`). The per-group im2col+GEMM alternative is rejected (group=480 → 480 tiny GEMMs, dispatch-bound).
- **rec BiLSTM with nonzero initial states → PASS.** `lstm_forward` (`src/cuda/cudnn/rnn.rs:~447`) forwards `h_init`/`c_init` to garboard; the prior bidirectional test only used zero states. Probe (bidirectional, hidden=48, nonzero seeded `initial_h/c`): output layout `[seq,2,batch,hidden]` matches ORT, parity `max_abs_diff = 1.04e-4`; zero-vs-nonzero sensitivity = 0.24 confirms initial states are consumed, not ignored. Batch=1 works; `dispatch_lstm` hardcodes batch=1 and `lstm_forward` errors on batch>1 ("needs transpose kernel").
- **Live ORT is net-new infra.** `ort` is a dev-dependency but has **zero live `ort::` call sites today** — all existing parity uses pre-recorded fixtures (`tests/fixtures/ort_reference`, `src/cuda/checkpoints.rs`). WS-6's parity gates are the first live use; the harness is new infrastructure.

---

## 3. Locked design decisions

1. **R2 dynamic shapes — error guard always; verify the dynamic-load path; re-plan only for concrete-load.** Today `ensure_compiled_with_outputs` (`src/cuda/inference/mod.rs:~415-428`) keys the plan cache **only** on output-name coverage and never receives the input tensors; `executor.run` (`src/cuda/executor/mod.rs:~127`) never compares input shapes to the plan → silent miscompute. Two distinct cases:
   - **Dynamic-loaded models (pdfocr's stated path; input dims `None`):** lowering leaves dynamic dims unresolved and Reshape/Slice resolve at runtime against the actual tensors (`src/ir/passes/precompute.rs:~91-195`), so a single plan is *likely already* correct across shapes. This is pdfocr's MUST — **verify it empirically, don't assume.**
   - **Concrete-loaded models (declared `dim_value`s):** lowering bakes those dims via `tensor_shapes_from_graph` (`src/ir/lowering.rs:~79`, seeded from declared input dims in `src/ir/passes/shape_inference.rs:~54`), so a later differing-shape input is silently miscomputed. A correct re-plan needs **new plumbing** (not a cache-key tweak): thread runtime input shapes into `ensure_compiled`/`run` and overwrite the lowering input dims before re-lowering. Re-lowering re-uploads initializers — **cheap (sub-ms, ~2 MB)**; NVRTC kernels are cached on the `Executor`, not the plan, so re-planning does **not** recompile kernels. Keep one most-recent plan (multi-entry shape-keyed cache deferred).
   - **Error guard (always-on floor):** before executing a cached plan, compare each runtime input shape to the plan's recorded input dims; mismatch → new `CudaError::ShapeMismatch` (loud), never a wrong result. This guard runs *before* the baked plan is used, so any Reshape/Slice params baked from those dims are covered by construction — there is no separate baked-param check to design (corrects an earlier over-specification).
   - **Scope:** the always-on guard + verified dynamic-load correctness are the **MUST** (and satisfy pdfocr's R2). Concrete-load re-plan is a **SHOULD** — the guard is the floor that prevents any silent miscompute if it is deferred.
2. **R3/R6 parity metric — REALISTIC input, not random.** Hard tensor gate `atol=1e-3, rtol=1e-3` (fp32; spike headroom — conv 0, LSTM 1e-4) on both models' raw outputs, **computed on a realistic preprocessed input** (§3.4). Empirically (ORT run during spec review), seeded-random `N(0,1)` input is **degenerate and makes the whole-model gate vacuous**: REC argmaxes to CTC-blank at *every* timestep (decoded sequence empty → "exact match" trivially asserts `[]==[]`) and DET's sigmoid map is ≈0 everywhere (mask all-zeros → IoU undefined; `atol` vs ≈0 non-discriminating). So the whole-model gate **requires a realistic input** that drives the models into their discriminative regime. Semantic gates: **rec** CTC-greedy decoded index/string sequence exact-match vs ORT (in-test ~30-line argmax→collapse-repeats→drop-blank-0) **with a non-degeneracy guard** (assert the ORT reference has ≥1 non-blank timestep, else fail); **det** probability-map closeness + binary-mask IoU@0.3 **with the empty-mask case defined as failure** (assert ORT mask has foreground). We test engine *agreement* on a realistic input, not OCR accuracy. Full det box-IoU (contours/min-area-rect/unclip) stays pdfocr's. **Per-op tests (M6.3/M6.4) keep seeded-random fixtures** — fine in isolation, where each kernel is exercised directly. **R6 Resize parity** folds into the det test; verified non-issue — all 6 DET Resize are `nearest/asymmetric/floor`, exactly what GPU `ResizeCoordMode` supports.
3. **R5 batching — deferred; characterize only.** WS-6 ships correct one-crop-at-a-time rec (batch=1). R5 = characterization (single-crop rec + full-page det latency, GPU-mem at large `(H,W)`, confirm batch=1 precondition) + a documented ragged-batch plan (width-padding + CTC length tracking + the batch>1 transpose-kernel) as a future workstream. No batching kernels now.
4. **Fixtures — kokoro-style symlink + SHA-256 verify, for models AND inputs.** Parity tests read the two models via symlinks (e.g. `./en_PP-OCRv3_det.onnx`, `./en_number_mobile_rec.onnx`) and, per §3.2, a **realistic preprocessed input tensor per model** (e.g. `./ocr_det_input.npy`, `./ocr_rec_input.npy`) — **provided by pdfocr** (which owns preprocessing) so iconnx does not reimplement PP-OCR normalization. Each fixture's SHA-256 is verified against a pinned value before running; tests **skip with a printed reason** if a symlink is absent or no GPU (kokoro precedent + WS-5 explicit-skip discipline). Nothing is committed to iconnx. **Cross-repo dependency:** M6.5 is blocked on pdfocr delivering the two preprocessed input fixtures (a real text-line crop tensor `[1,3,48,W]` + a real page tensor `[1,3,H,W]`) and their SHA-256s; M6.1–M6.4 are unblocked.
5. **Grouped Conv2D — route `group>1` (and any dilated) Conv2D through garboard cuDNN; leave `group==1` on the existing validated im2col+GEMM path.** Avoids re-validating the whole roster through a new conv path. Unifying all 2D conv onto cuDNN is a possible future simplification, out of scope here.
6. **Milestone shape — Approach A (dependency-ordered, model-independent first).** Unblocks pdfocr's API wiring earliest.

---

## 4. Milestones

Each milestone = one GPG-signed feat commit (split only if subsystems separate cleanly), TDD (failing test → run-to-fail → minimal impl → run-to-pass → commit), two-stage review (spec compliance, then code quality).

### M6.1 — GPU API + live-ORT test harness + carry-forward hygiene *(model-independent)*
- `GpuGraphExecutor::from_model(&OnnxModel) -> Result<Self>`: `extract_weights()` → `add_initializer(name, &tensor)`; iterate `computation_graph()` nodes → `add_node_with_attributes(...)`. Note `GraphNode` (`src/onnx_parser.rs:~1304`) carries **no `name` field** while `add_node` requires one → synthesize index-based names (e.g. `n{i}`). Output-equivalence test vs hand-wiring on an existing roster model.
- GPU support-check **reusing WS-5**: add a `&OnnxModel`-taking entry to `validate_model` (`src/validate/mod.rs:~116`; its post-parse body already operates on the parsed `model`/`graph` and emits `ModelIncompatibility::UnsupportedOp { op_type, count, sample_node_names }` by diffing graph ops against `op_support::supported_ops()`). Do **not** fork a parallel `validate_gpu_support`/`UnsupportedOp` type. **Caveat to document:** this is an **op-NAME presence check only** — it does *not* guarantee runnability (it would have reported `Conv` "supported" before the C1 grouped-conv fix; `DispatchContext::Gpu` lives in `op_support::lookup()`, which `validate_model` does not call). Word the API/docs accordingly; `from_model` may call it for a fail-fast name check.
- **Live-ORT parity helper** (foundational; used by M6.2–M6.5): a test-side helper that runs a model/graph through `ort` and returns outputs for comparison (first live `ort::` use; the M6.0 spike's `ort_session` pattern was thrown away with its worktree, so this is built fresh under TDD). Lives in a shared test module.
- Carry-forward doc fixes: rewrite README quickstart so it **compiles** (garboard not cudarc, correct `add_initializer(name, &Tensor)` signature, note `GpuGraphExecutor: !Sync`); fix the stale `lib.rs` doc header — "Opset 17+" (real bounds 11–20), "CPU fallback — Full CPU implementation", and "Concurrent inference — Multiple streams share weights" (executor is `!Sync`, not exposed); add a compiling `examples/gpu_infer.rs`.
- Pin `garboard` + `garboard-sys` in `Cargo.toml` from `branch="main"` to `rev="75a1f66071a700c97e60070c39d89d6d04e15e90"` (current `Cargo.lock` rev) — reproducibility for pdfocr as a downstream consumer.
- **Acceptance:** `from_model` round-trips; support-check returns the existing failure surface; `cargo build --example gpu_infer`; README/lib.rs accurate; garboard pinned.

### M6.2 — R2 dynamic-shape correctness + error guard *(model-independent)*
- **(MUST) Always-on guard:** add `CudaError::ShapeMismatch`; record the plan's input dims and compare each runtime input shape before executing a cached plan; mismatch → loud error. Runs before the baked plan is used → baked Reshape/Slice params covered by construction (§3.1).
- **(MUST) Dynamic-load correctness:** verify a model loaded with dynamic input dims (`None`) produces correct output across ≥2 differing shapes with a single plan (Reshape/Slice resolve at runtime — confirm empirically). This is pdfocr's actual path.
- **(SHOULD) Concrete-load re-plan:** thread runtime input shapes into `ensure_compiled`/`run`, overwrite the lowering input dims, and re-lower on shape change so a concrete-loaded model rebuilds a correct plan (initializers re-upload — cheap; kernels stay cached). Keep one most-recent plan. If deferred, the guard above prevents any silent miscompute.
- **Acceptance:** (a) a **dynamically-loaded** synthetic conv model run at shape A then B is numerically correct vs ORT at **both** shapes with no re-plan; (b) a **concrete-loaded** model at a differing shape either re-plans to a correct result OR raises `ShapeMismatch` — never a silent wrong result; (c) a forced shape/plan mismatch raises `ShapeMismatch`. Oracle = the M6.1 live-ORT helper, **not** the CPU `GraphExecutor` (its 2D Conv hardcodes stride=1/pad=0, `src/operators/conv.rs:~222`).

### M6.3 — Conv correctness: grouped Conv2D + input-form Clip *(model-independent, per-op ORT parity)*
- **Grouped Conv2D via cuDNN:** add `group` (and `dilation`) fields to `Conv2dParams` (`src/cuda/conv/params.rs`); add a `garboard_conv_2d` wrapper mirroring `garboard_conv_1d` but passing real H/W + groups; in the 2D dispatch (`src/cuda/executor/conv.rs`), route `group>1` (or dilated) to cuDNN and `group==1` to the existing im2col+GEMM `gpu_conv2d`; remove the `in_channels==kernel_in_channels` assert on the cuDNN path. **Bias seam:** garboard's `conv_forward` does **not** fold bias (it takes only input/kernel/output), so the cuDNN path must apply bias via the existing `add_bias` follow-up (`src/cuda/executor/conv.rs:~170`), unlike the inline-bias im2col path. Both models' grouped convs are bias-free (bias rides the following BatchNorm — verified DET 0/15, REC 0/11), but the per-op test must include a **biased** grouped conv to cover `add_bias`. Per-op parity vs ORT at depthwise + general grouping + biased (spike: bit-exact for the bias-free cases).
- **ConvTranspose-2D per-op parity (verification, no new code):** 2D ConvTranspose is already dispatched via the in-tree `gpu_conv_transpose_2d` col2im kernel (`src/cuda/executor/conv.rs:~299`) and DET's nodes (2×2 kernel, stride 2, group 1) are its supported form — but no roster model has exercised it end-to-end and the whole-model gate (M6.5) is insufficient alone. Add a standalone ORT parity test for the DBNet upsample form (stride=2).
- **Input-form Clip fix:** GPU `Clip` (`src/cuda/executor/elementwise.rs:~273`) reads `min`/`max` from attributes only; opset-11+ passes them as `inputs[1]/inputs[2]`. REC has **18/18 Clip nodes in input-form** (ReLU6). Fix GPU Clip to read the optional input tensors (precedent: Pad/Expand via `get_or_fetch_i64`/host fetch), falling back to attributes for the legacy form. Parity test on input-form Clip.
- **Acceptance:** grouped/depthwise + dilated-path Conv2D and input-form Clip each match ORT within tolerance; drift-fence + `op_support` table consistent.

### M6.4 — Missing ops: BatchNormalization + HardSigmoid + HardSwish + Tile *(model-independent, per-op ORT parity)*
- **BatchNormalization** (inference): precompute per-channel `a = scale/√(var+eps)`, `b = B − mean·a` from the 5 inputs + `epsilon` attr (ignore `momentum`/training attrs), then apply via a **new standalone per-channel-affine kernel/wrapper** over NCHW (the existing channel-affine code is buried in `gpu_fused_add_mul_add` Path 3 with a `[1,C,T]` gate — **not** reusable as-is; the math is correct, the "reuse" is not).
- **HardSigmoid:** elementwise `clamp(α·x+β, 0, 1)` — **read α/β from node attributes** (`src/attributes.rs` `get_float`). Both OCR models use `α=0.2, β=0.5` (the ONNX defaults — *not* the 1/6 MobileNetV3 value the op-diff stated). Do not hardcode; the per-op parity test pins the attribute-read.
- **HardSwish:** fixed `x·clamp((x+3)/6, 0, 1)` (no attributes).
- **Tile:** general `repeats`-driven repeat (np.tile semantics); read the `repeats` input at execute time (precedent: Expand/Pad, `src/cuda/executor/layout.rs:~667`).
- Each op: kernel/dispatch arm + `op_support.rs::OP_SUPPORT_TABLE` entry + `mod.rs` type-tag map entry + drift-fence passes + per-op GPU↔ORT parity on seeded-random f32 fixtures.
- **Acceptance:** all four ops match ORT within tolerance; both feature combos `-D warnings` clean; drift-fence green.

### M6.5 — R3/R6 whole-model parity *(needs symlinked models + input fixtures)*
- Build the whole-model GPU↔ORT compare on the M6.1 live-ORT helper (run on `GpuGraphExecutor`, run same input on ORT, compare). New `tests/ocr_parity_test.rs`; **do not conflate with the `run_dataset` conformance stub** (stays as-is — it is the ONNX node-suite harness, a different shape).
- **Realistic preprocessed input per §3.2/§3.4 — NOT random** (random is degenerate: empty rec sequence / all-zero det mask). Metrics per §3.2, **including the non-degeneracy guards** (ORT reference must have ≥1 non-blank rec timestep / det foreground, else the test fails rather than passing vacuously). Run det at 2 `(H,W)` and rec at 2 `W` — also exercises M6.2 and is the end-to-end gate for the BiLSTM + BatchNorm + HardSwish + grouped-conv + ConvTranspose stacks. Det test asserts Resize parity (R6).
- Fixtures per §3.4 (symlink + SHA-256 + skip-with-reason); **blocked on pdfocr's input fixtures.**
- **Acceptance:** both models run end-to-end with no unsupported-op error; tensor `atol/rtol` met on a **non-degenerate** reference; rec CTC-decoded sequence matches ORT exactly **and is non-empty**; det prob-map closeness + mask-IoU met **with a non-empty mask**; all at ≥2 input shapes.

### M6.6 — R5 characterization + batching/perf plan
- Benchmark single-crop rec + full-page det (~960² and ~1500×2000) wall-clock; GPU-mem behavior at large `(H,W)`; confirm batch=1 precondition. Document the ragged-batch plan + the depthwise-via-cuDNN perf profile (note any per-Conv-node dispatch overhead) as a future workstream.
- **Acceptance:** documented latency + GPU-mem figures; batching/perf follow-up doc written.

---

## 5. Op-coverage truth (verified against the real model graphs)

| Op | Status on GPU before WS-6 | WS-6 action |
|---|---|---|
| Conv `group==1` | supported (im2col+GEMM) | unchanged |
| Conv `group>1`/depthwise | **panics** (`in==kernel_in` assert; no `group` field) — ~15 DET + ~11 REC nodes, groups 8…480 | M6.3 cuDNN route |
| Clip (input-form, opset 11+) | **silent no-op** (reads attrs only) — 18 REC nodes | M6.3 read inputs |
| BatchNormalization | recognized-but-not-runnable (cost-table only) — 49 DET + 35 REC | M6.4 standalone kernel |
| HardSigmoid | missing — 8 DET + 9 REC | M6.4 |
| HardSwish | missing — 20 DET | M6.4 |
| Tile | missing — 2 REC | M6.4 |
| Resize (nearest/asymmetric/floor) | supported | M6.5 parity confirm (no code) |
| LSTM bidirectional + initial_h/c | supported (batch=1) — spike-validated | M6.5 parity confirm (no code) |
| Conv dilation | absent on 2D | not needed (models use `dilations=(1,1)`); free via the cuDNN path |
| ConvTranspose-2D | supported (in-tree col2im) | M6.3 adds a standalone per-op parity test (was only implicitly covered) |

All other ops in both models (Add, Concat, Constant, GlobalAveragePool, MatMul, MaxPool, Mul, Relu, Reshape, Shape, Slice, Squeeze, Transpose, Unsqueeze, Sigmoid, Div, Identity) are present in the GPU dispatch and handle the real-model forms (verified node-by-node: MaxPool 2×2/stride-2 no-dilation, Reshape 0/-1 const-fed, Slice step=1 const-fed, Concat axis-normalized, Squeeze/Unsqueeze axes-as-attr opset-12 form, Constant `value`-tensor). **Verified caveats:** **Softmax** ignores the `axis` attribute and always operates on the last axis (`src/cuda/executor/elementwise.rs:~285`); REC's `axis=2` on rank-3 = the last axis, so correct *for these models only* — a non-terminal-axis Softmax would silently miscompute (note it; out of scope to fix). **LSTM** ignores `sequence_lens` (slot 4) — harmless for batch=1 full-length OCR sequences.

---

## 6. Out of scope (intentional, to keep the workstream focused)

- The broken `cargo build --no-default-features` / `ir`-module gating / `fftw`-optional CPU-build fix (CPU path dropped by pdfocr; orthogonal to GPU OCR; tracked in the carry-forward memory).
- CPU `lstm.rs` stale-comment/test debt (GPU uses cuDNN RNN; M6.5 validates the GPU BiLSTM instead).
- Batched (`N>1`) rec inference and the batch>1 LSTM transpose kernel (R5 deferred; characterized + planned only).
- Unifying all 2D conv onto cuDNN (only `group>1`/dilated routes there).
- Training/autograd; transformer/SVTR ops (`Gelu`/`InstanceNormalization`) — pdfocr Slice-4 / SVTR contingency only.

---

## 7. CI gates & acceptance

Per-commit (mirroring WS-5's gate set):
1. `cargo build` (default `cuda`) + `cargo build --features cuda,debug-inference`.
2. `cargo clippy` both combos, `-D warnings`.
3. lib + integration tests (GPU where available).
4. `op_support_table_consistency` drift-fence + `opset_bounds`.
5. `tests/ocr_parity_test.rs` — GPU-gated, skips with printed reason without GPU/model symlinks.

Excluded from sign-off (inherited deferral): `cargo build --no-default-features` (CPU build is §6 out-of-scope).

**Workstream acceptance:** both models execute end-to-end on `GpuGraphExecutor` with the §3.2 metrics met at ≥2 input shapes each, consumable via `from_model`, with a pre-run support check; carry-forward docs accurate; garboard pinned.

---

## 8. Build facts to communicate to pdfocr

- fp32 conv/LSTM/MatMul are the validated paths; FP16 has known gaps (no FP16 Slice kernel; FP16 conv groups≠1 unsupported).
- Minimum CUDA toolkit / compute capability: the fp32 conv/LSTM floor (not the BF16 `sm_80` floor — to be stated precisely in M6.1 docs). WSL2 is supported (development platform).
- Batch=1 is a precondition for rec (batch>1 LSTM not yet supported).
- garboard is pinned to rev `75a1f66`; the real CUDA-version contract lives in garboard.
- Live det/rec parity in CI requires a GPU runner + the model symlinks; without them, det/rec CI is parser-only (`DispatchContext::ParserOnlyCi`).

---

## 9. Sequencing, worktree, cadence

- Sibling worktree `iconnx-ws6` on branch `ws6-gpu-ocr` (WS-3.5/WS-5 pattern). All commits GPG-signed (key `0F0266D010EE6736`).
- Subagent-driven execution: fresh implementer per milestone + two-stage review; escalate upstream-contract issues via BLOCKED, never work around in tests (per `feedback_subagent_escalation`).
- M6.1–M6.4 are model-independent and can proceed immediately; M6.5 needs the two model symlinks from pdfocr's `vendor/ocr-models/` **and** the two preprocessed input fixtures from pdfocr (§3.4). M6.1 alone gives pdfocr the API to begin wiring.
- PR-via-GitHub merge at the end = shared-state action surfaced to the user, **not** autonomous (mirrors WS-3.5/WS-5 protocol).
