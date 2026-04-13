# iconnx Optimization Journey: Cycles 1–7

**Date:** 2026-04-12
**Benchmark:** Kokoro v1.0 TTS, RTX 4070 Laptop, WSL2, CUDA 12, seq_len=5
**Comparison:** ORT (ONNX Runtime) with Level 3 optimization
**Tool:** leadline-bench (in ryanoneill/claudio repo)

---

## The numbers

| Milestone | Avg (ms) | vs Baseline | vs ORT (26ms) |
|---|---|---|---|
| Baseline (pre-Cycle 1) | 187 | — | 7.2x |
| Post-Cycle 1 (MulSinPowMulAdd) | 141 | -25% | 5.4x |
| Post-Cycle 5 (AddMulAdd) | 117 | -37% | 4.5x |
| Post-Cycle 6 (Conv algo heuristic) | ~115 | -38% | 4.4x |
| Post-Cycle 7 (cuDNN LSTM) | ~109 | -42% | 4.2x |

**78ms saved. 42% faster. From 7.2x ORT gap to 4.2x.**

---

## Cycle 1: MulSinPowMulAdd fusion (the biggest single win)

**Problem:** Kokoro's vocoder has 48 instances of a 5-op pattern: `Mul → Sin → Pow → Mul → Add` (periodic activation). Each runs as 5 separate kernel launches.

**Fix:** Custom fused CUDA kernel that does all 5 ops in one launch. Walker detects the pattern at graph-build time, marks interior nodes for skipping.

**Result:** 187ms → 141ms (-46ms, -25%). 48 fusions, 192 kernel launches eliminated.

**Lesson learned:** Walker ordering matters. The 5-op pattern must be detected BEFORE shorter patterns (MulAdd, AddMul) that would claim its suffix. A realistic-weights integration test caught this ordering bug that unit tests missed.

**Also in Cycle 1:** Per-channel-scale broadcast kernel variant for Kokoro's `[1,256,1] × [1,256,1900]` weight shape. The vocoder's periodic activation has a trailing-dim per-channel broadcast that the elementwise kernel doesn't handle.

---

## Cycle 1.5a: GPU-side profiling (infrastructure)

**Problem:** CPU wall-clock timing only measures enqueue time, not actual GPU kernel time. Can't tell if an op is GPU-bound or launch-overhead-bound.

**Fix:** Added `GpuEventTimer` using `CU_EVENT_DEFAULT` (not `CU_EVENT_DISABLE_TIMING`) CUDA events. Each op gets a start/stop event pair. `gpu_op_times` map added to `ProfileData`.

**Key discovery:** The GPU event timer itself has a per-call overhead floor of ~10-30us on WSL2 + RTX 4070. For near-no-op ops (Unsqueeze with precomputed axes), this floor dominates the measured time. Hit-rate assertions are more reliable than GPU-time assertions for these ops.

---

## Cycle 2: Zero-cost Unsqueeze/Reshape/Squeeze (precomputation)

**Problem:** 207 D2H driver calls per inference for shape/axes parameters (Unsqueeze axes, Reshape shapes, Squeeze axes). Each call is a `cuMemcpyDtoH` + stream sync.

**Fix:** Precompute shape/axes parameters at `add_node` time from the static Constant-Int64 cache. Store in `Option<Vec<i64>>` fields on `ExecutionNode`. At runtime, check precomputed field first; fall through to `get_or_fetch_i64` on miss.

**Result:** 192/192 Unsqueeze hits (100%), 5/5 Squeeze hits (100%), 10/61 Reshape hits (16%). 207 D2H calls eliminated.

**Lesson:** The remaining 51 Reshape misses come from runtime-computed shapes (`Shape → Cast → Gather` chains). A symbolic shape-inference pass (Approach C) would close this gap. Deferred as too large for this cycle.

**Testing pattern established:** Hit-rate assertions (primary gate) + loose GPU-time regression guards (secondary). The regression guards need generous headroom — Cycle 3 later widened the Unsqueeze guard from 6000us to 25000us after discovering the measurement-floor noise range.

---

## Cycle 3: AddMulAdd detection gap — the diagnostic pivot

**Hypothesis:** Leadline's chain analysis found 51 `Add → Mul → Add` instances. The walker's `is_static` check was too strict (initializer-only). Relaxing it to recognize `Constant` node outputs should unlock 51 fusions.

**What actually happened:** The walker fix shipped and unit tests proved it works on synthetic Constant-sourced graphs. But on real Kokoro: **zero matches**. A one-shot diagnostic (`eprintln` at the walker rejection point) revealed all 73 candidate chains are **Adaptive Instance Normalization (AdaIN)** blocks where `b` (scale) comes from a runtime `Div` and `c` (bias) comes from a runtime `Slice`. Plus 3 LSTM layer-norm chains with `b=LayerNormalization`, `c=Transpose`.

**Result:** 0ms performance improvement. But the diagnostic data was invaluable — it directly informed Cycles 4 and 5.

**Lesson:** **Diagnostic-first development.** Every fusion cycle since has started with an empirical shape/structure diagnostic before writing any kernel code. The cost is ~30 minutes; the payoff is avoiding wasted implementation cycles based on wrong assumptions. Cycle 3 proved this methodology by catching the assumption error before any kernel was written.

**Also shipped:** `is_static` closure (reusable for future models), Kokoro `add_mul_add == 0` reality-check assertion, Unsqueeze guard widening.

---

## Cycle 4 + 4.5: Dynamic AddMulAdd infrastructure

**Problem:** The 73 AdaIN chains have runtime-computed `b` and `c`. The existing `AddMulAdd` walker only registers static patterns. Need infrastructure for "try fusion at runtime, fall back on shape mismatch."

**Cycle 4 delivered:**
- `CudaError::FusionShapeMismatch` sentinel for fallback
- Broadcast-c CUDA kernel: `c[(i/stride)%c_size]` handles both AdaIN and LSTM patterns
- Walker registers non-static chains as `dynamic_candidates` (separate from static `fused_patterns`)
- Executor dispatch with per-run `dynamically_fused` HashSet + input-readiness pre-check
- 16 new tests

**Cycle 4 discovery:** 73 candidates detected, **0 fused at runtime.** The head Add1 node executes before Div/Slice produce `b` and `c` — topological ordering gap.

**Cycle 4.5 fix:** Scheduling edges in the topological sort. For each dynamic candidate, Add1 gains artificial dependencies on `b` and `c` tensor names. Forces the sort to place Div/Slice before Add1.

**Cycle 4.5 discovery:** Scheduling edges work (all 73 candidates reach dispatcher with inputs available), but **all 73 fail on shape mismatch.** The shapes are:

```
x = [1, C, 1]   ← per-channel scale (NOT the full-shape tensor)
a = []           ← empty (Add1 is identity)
b = [1, C, T]   ← the FULL-SHAPE normalized input
c = [1, C, 1]   ← per-channel bias
```

The broadcast-c kernel assumes `x/a/b` are all full-shape and only `c` broadcasts. Reality: `b` is the only full-shape tensor; `x` and `c` are both per-channel.

**Lesson:** **Three diagnostic rounds caught three wrong assumptions:**
1. Cycle 3: "b/c are Constant-sourced" → actually runtime Div/Slice
2. Cycle 4: "shapes match at head position" → topo ordering gap
3. Cycle 4.5: "x/a/b are full-shape" → x is per-channel, b is full-shape

Each round cost ~30 minutes and prevented a wasted kernel implementation.

---

## Cycle 5: Per-channel affine kernel (the AddMulAdd payoff)

**Kernel design (from Leadline):**
```cuda
out[i] = scale[i/T] * input[i] + bias[i/T]
```

Where `input` is the walker's `b` (Div output, `[1,C,T]`), `scale` is `x` (`[1,C,1]`), and `bias` is `c` (`[1,C,1]`). The wrapper detects "b is 3D `[1,C,T]`, x and c are `[1,C,1]`, a is empty" and routes to this kernel.

**Result:** 141ms → 117ms (-24ms, -17%). 70 of 73 candidates fuse at runtime. 140 kernel launches eliminated. The 3 LSTM layer-norm chains fall back (different shape pattern: `x=[1,1,C]` not `[1,C,1]`).

**Lesson:** The 4-cycle infrastructure investment (Cycles 3, 4, 4.5, 5) paid off in one kernel. The existing dynamic-candidate dispatch, scheduling edges, input-readiness check, and shape-mismatch fallback all worked correctly once the right kernel was wired in.

---

## Clippy baseline cleanup (between Cycles 3 and 4)

88 pre-existing `cargo clippy --features cuda --tests` errors resolved across ~20 files in 5 focused commits. Float precision truncation, math constants, collection idioms, iterator conversions, dead code gating. Zero behavior changes. The `--tests` gate is now clean for all future cycles.

---

## Cycle 6: cuDNN Conv algorithm selection

**Hypothesis:** Replacing hardcoded `CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM` with cuDNN's heuristic algorithm finder would pick FFT/Winograd for large spatial dims.

**Result:** ~5% Conv improvement (~2ms). The heuristic returns the same algorithm for most Kokoro shapes.

**Cycle 6b attempt:** Switching to `cudnnFindConvolutionForwardAlgorithm` (benchmarking variant) works but adds ~1s first-inference overhead (12 shapes × ~70ms per benchmark). The test's regression guard tripped because the benchmark overhead polluted the GPU profiling. **Reverted.** Needs warm-up infrastructure (benchmark at model-load time, not during timed inference).

**What shipped:** `ConvAlgoCache` with per-shape HashMap, `find_best_fwd_algo` + `find_best_bwd_data_algo` helpers, `cudnnConvolutionFwdAlgoPerf_t` FFI bindings. The cache infrastructure is reusable if/when warm-up is added.

**Lesson:** The heuristic is "good enough" for most shapes. The real Conv speedup requires the benchmarking variant with warm-up plumbing — a small targeted follow-up.

---

## Cycle 7: cuDNN fused LSTM

**Problem:** 6 bidirectional LSTM nodes, each running 2 cuBLAS GEMMs + 1 fused kernel per timestep per direction = ~720 total kernel launches. The seq_len=95 call alone has 570 launches.

**Fix:** Replace with cuDNN's `cudnnRNNForward` — one call per LSTM node.

**Key challenges:**
1. **No cuDNN RNN bindings existed** — 15 FFI functions, 7 enums, 3 opaque handle types bound from scratch
2. **Weight format mismatch** — ONNX stores gates in IOFC order; cuDNN expects IFCO. The `ONNX_TO_CUDNN_GATE: [i32; 4] = [0, 3, 1, 2]` mapping handles reordering via `cudnnGetRNNWeightParams` offset queries + D2D memcpy
3. **cuDNN 9 quirk** — `projSize` must equal `hiddenSize` (not 0) when LSTM projection is disabled. This caused a crash (SIGSEGV from an unmapped `cudnnStatus_t` value of 2000) until discovered during integration testing

**Result:** LSTM 13ms → 5.25ms (-60%, -7.75ms). 6 cuDNN calls replace ~720 kernel launches. Numerical agreement with gpu_lstm within 6.59e-6.

**Testing:** Bidirectional correctness test (cuDNN vs gpu_lstm output comparison within 1e-4 tolerance) + unidirectional sanity check. The tolerance accounts for float accumulation order differences between per-timestep cuBLAS and cuDNN's fused implementation.

---

## What didn't work (or didn't work yet)

| Attempt | Why it didn't land | Status |
|---|---|---|
| Cycle 3's Constant-node is_static fix | Kokoro's chains are AdaIN (runtime b/c), not Constant-sourced | Correct code, zero Kokoro impact. Kept for future models. |
| Cycle 4's broadcast-c kernel | Designed for wrong shape assumption (x/a/b full-shape) | Correct kernel, never dispatched on Kokoro. Infrastructure reusable. |
| Cycle 6b Conv benchmarking variant | ~1s first-inference overhead, needs warm-up | Reverted. FFI bindings ready; needs warm-up plumbing. |
| Approach C symbolic shape inference | Too large for any single cycle | Deferred. Would close Reshape 51/61 gap + enable broader graph opt. |

---

## Methodology: what worked well

1. **Diagnostic-first development.** Every cycle starts with an empirical measurement before writing kernel code. The cost is 30 minutes; the payoff is avoiding wrong-assumption wasted cycles. Cycles 3, 4, and 4.5 each caught a different wrong assumption this way.

2. **Subagent-driven development with two-stage review.** Fresh subagent per task preserves controller context. Spec compliance review catches over/under-building. Code quality review catches style issues. The Task 7 bug (cross-walker claiming of dynamic candidate interior nodes) was caught by a unit test that the review prompted.

3. **Git worktrees for cycle isolation.** Each cycle works in its own worktree, merges to main with `--no-ff` for clean history. No conflicts between cycles. Easy to discard abandoned work (Cycle 6b).

4. **Leadline-bench as external oracle.** iconnx's internal profiling (debug mode, single inference, GPU event timing) has noise and overhead. Leadline-bench provides controlled release-mode measurements with multiple iterations. The split avoids optimizing for the wrong metric.

5. **TDD for fusion detection.** Synthetic graph unit tests catch walker logic bugs (ordering, cross-pattern claiming, single-use violations) that integration tests miss. The Kokoro integration test catches shape/runtime issues that unit tests can't reach.

---

## What's left

**Remaining gap: ~109ms vs 26ms ORT (4.2x).** The optimization program has moved from "launch overhead" territory (kernel fusion) through "kernel quality" territory (cuDNN algorithm selection, fused LSTM) into "graph-level optimization" territory. The remaining improvements require architectural changes:

1. **Conv algorithm benchmarking warm-up** (~10-15ms potential) — the Cycle 6 infrastructure just needs a warm-up mechanism to run `cudnnFindConvolutionForwardAlgorithm` at model-load time. Estimated 1-2 tasks.

2. **Approach C symbolic shape inference** — closes Reshape 51/61 precompute gap from Cycle 2 and enables broader graph optimization (operator elimination, layout optimization). Multi-cycle effort.

3. **Transpose elimination** (~11ms, 89 calls) — many transposes exist because ops output in one layout and the next op expects another. A layout-aware approach tracking preferred format per tensor could eliminate most of these.

4. **Remaining elementwise fusion** — 164 Add + 68 Sub calls still unfused. Diminishing returns vs. the graph-level targets above.

5. **AddMulAddLeakyRelu** — 22 instances from Leadline's chain analysis. Separate fusion pattern, separate cycle. Moderate impact.
