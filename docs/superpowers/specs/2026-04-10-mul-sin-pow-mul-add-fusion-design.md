# Cycle 1: MulSinPowMulAdd Fusion Pattern — Design

**Date:** 2026-04-10
**Author:** Claude (brainstormed with Ryan O'Neill)
**Status:** Design, awaiting implementation plan
**Supersedes/extends:** `docs/performance/2026-04-09-leadline-bench-findings.md`

---

## Context

iconnx is ~5x slower than ONNX Runtime on Kokoro v1.0 TTS at short sequence lengths, primarily because of CUDA kernel launch overhead from ~2,400 kernel dispatches per inference. The Leadline performance analysis document identifies five optimization priorities; this spec covers the first work cycle.

A refined Leadline analysis (received 2026-04-10) reorders the original priorities. It identified three high-value elementwise fusion patterns that collectively eliminate 360 kernel launches, plus a revised picture of the shape-only ops where Unsqueeze/Squeeze are already nearly free but Reshape and Gather-on-constants remain worthwhile. The top single win is a 5-op chain `Mul → Sin → Pow → Mul → Add` that appears 48 times in the Kokoro vocoder, eliminating 192 launches when fused.

### Sequence-length sweep (2026-04-10)

A second Leadline run swept sequence lengths 5 → 200. Three findings sharpen the picture and affect what success looks like for this cycle:

| seq_len | iconnx | ORT    | ratio |
|---------|--------|--------|-------|
| 5       | 142.65 | 28.88  | 0.20× |
| 10      | 137.11 | 30.00  | 0.22× |
| 20      | 282.47 | 32.81  | 0.12× |
| 50      | 315.58 | 64.97  | 0.21× |
| 100     | 466.41 | 125.92 | 0.27× |
| 200     | OOM    | —      | —     |

The ratio stays near 0.2× across the entire sweep. This means iconnx is not just fixed-overhead-bound at short sequences — the overhead scales with tensor size, because the per-launch cost grows as intermediate tensors get larger (more DRAM traffic, more memory-pool pressure, more time spent in kernel setup for larger grids). Fusing a 5-op chain at seq_len=100 saves 20× more memory bandwidth per occurrence than at seq_len=5.

iconnx OOMs at seq_len=200 while ORT would not. This strongly suggests iconnx allocates a fresh GPU buffer for every intermediate tensor in the un-fused chain. Fusing `Mul → Sin → Pow → Mul → Add` eliminates 4 intermediate tensors per occurrence × 48 occurrences = **192 fewer intermediate allocations per inference**. That is not merely a compute optimization — at long sequences it may be what makes the model fit at all.

The seq_len=20 outlier (first run 507 ms, then settling to ~186 ms) exposes memory-pool thrashing: the first inference at a new tensor size misses the pool and allocates fresh, then later runs reuse those allocations. Reducing the number of distinct intermediate shapes (which fusion does, because the fused output has the final shape instead of 4 intermediate shapes) should ease the thrashing.

This cycle implements one fusion pattern end-to-end. It is deliberately narrow so the change can be benchmarked and validated in isolation before moving to the next pattern. But the sweep data means success must be measured across multiple sequence lengths and must include memory-side metrics, not only wall-clock latency at seq_len=5.

## Goals

1. Eliminate 192 CUDA kernel launches on the Kokoro forward pass by fusing the `Mul → Sin → Pow → Mul → Add` chain into a single kernel that fires ~48 times.
2. Extract fusion code from `src/cuda/inference/mod.rs` (currently 4,985 lines) into a dedicated `src/cuda/inference/fusion/` module as a targeted refactor scoped to the code this cycle touches.
3. Preserve bit-compatible numerical output on Kokoro compared to the current implementation (and thus vs ONNX Runtime within existing test tolerances).
4. Establish a repeatable TDD workflow for adding future fusion patterns — so cycles 2 and 3 (AddMulAddLeakyRelu and the AddMulAdd detection fix) can reuse the same structure.
5. Show measurable improvement on the `leadline-bench` Kokoro comparison binary across multiple sequence lengths: the new fused pattern must appear in the per-operator breakdown with ~48 calls, total inference latency must decrease at both seq_len=5 and seq_len=50, and memory pool allocation counts must drop by approximately 192 per inference.

## Non-Goals

1. Broader elementwise fusion beyond this one pattern (cycles 2–3 of the refined Option 1 plan).
2. Zero-cost Reshape or graph-time Shape/Gather evaluation (later cycles).
3. cuDNN LSTM replacement (P3).
4. Constant folding (P4).
5. Layout-aware operator dispatch (P5).
6. A full refactor of `src/cuda/inference/mod.rs` — only the fusion-related code moves in this cycle. Other modules extract when a later cycle naturally touches them.
7. Changes to the public `GpuGraphExecutor` API, the `GpuTensor` type, or the profiling API.
8. Any optimization to the existing fused kernels (e.g., making DivRsqrt faster). Those remain as-is.

## Background: the pattern in context

### What it is mathematically

The pattern expands, in graph order, as:

```
t0 = Mul(x, w0)           // pre-sin scaling (w0 is typically a weight)
t1 = Sin(t0)              // periodic activation input
t2 = Pow(t1, p)           // p is a constant scalar exponent (integer in practice)
t3 = Mul(t2, w1)           // post-pow scaling (w1 is a weight)
y  = Add(t3, b)            // bias add (b is a weight or a computed tensor)
```

As a single expression:

```
y = sin(x * w0)^p * w1 + b
```

This is a variant of the "snake" / SiLU-family periodic activation common in modern vocoders. In Kokoro's decoder path it fires once per residual block and contributes 4 of the 5 elementwise-op launches per call.

### Why it's fuseable

All five ops are elementwise (modulo broadcasting) and each intermediate tensor (`t0`, `t1`, `t2`, `t3`) is consumed exactly once by the next op in the chain. No intermediate escapes the chain, so a fused kernel can keep every intermediate in registers and write only the final `y` to global memory. This eliminates:

- 4 kernel launch events per occurrence (5 ops → 1 fused op)
- 4 intermediate allocations (no `t0..t3` buffers)
- 8 redundant global-memory round trips (read+write for each intermediate)

At 48 occurrences per inference: **192 kernel launches saved** and significant bandwidth reduction.

### What we don't know yet

The exact graph shape in Kokoro — specifically:
- Whether `w0` is a scalar, per-channel broadcast, or full-shape tensor
- Whether `w1` and `b` broadcast along the same or different axes
- Whether `p` is always the same value (likely 2, worth verifying)
- Whether `x` and `w0` come in a fixed operand order or sometimes swapped

The design accommodates broadcasting generically. Before implementation lands, Phase E validates the detection code against the real Kokoro graph — if the assumed pattern shape differs, detection is updated to match.

## File Layout

### New files

```
src/cuda/inference/fusion/
├── mod.rs          (~50 lines)   Re-exports, module glue
├── patterns.rs     (~200 lines)  FusedPattern enum, FusedPatternInfo struct
└── detection.rs    (~400 lines)  detect_fused_patterns() as a free function
```

**Rationale for extraction:** The existing `FusedPattern` / `FusedPatternInfo` / `detect_fused_patterns()` code lives mid-file inside `src/cuda/inference/mod.rs`. Adding the new `MulSinPowMulAdd` variant plus its detection walker would push the file further past the 1000-line limit. Extracting it now (before adding more patterns in cycles 2–3) means every future fusion pattern lands cleanly.

**Why a free function for detection:** Detection only needs to read the node list, the weights map, and return pattern metadata. Making it a free function rather than a method on `GpuGraphExecutor` decouples it from the executor's internal state, which makes it trivially unit-testable without constructing a full CUDA context.

```
tests/operators/fused_mul_sin_pow_mul_add_test.rs  (~250 lines)
    Unit tests for the new fused kernel:
      - naive_vs_fused_matches (same-shape case)
      - broadcasting_w1_and_b (per-channel weights)
      - odd_exponent_p3 (sign handling)

tests/fusion_detection_test.rs  (~150 lines, if a fresh file is cleaner)
    Graph-level tests for the new pattern:
      - detects_mul_sin_pow_mul_add_chain
      - rejects_multi_use_intermediate
      - registers_correct_skip_set
```

### Modified files

```
src/cuda/inference/mod.rs
    Removes:
      - FusedPattern enum definition (~60 lines)
      - FusedPatternInfo struct (~10 lines)
      - detect_fused_patterns() method body (~250 lines)
      (GpuGraphExecutor::detect_fused_patterns becomes a thin wrapper
       that calls the free function and populates self.fused_patterns)
    Adds:
      - use crate::cuda::inference::fusion::{...}
      - Dispatch case for MulSinPowMulAdd pattern (~20 lines)
    Net change: approximately −300 lines

src/cuda/kernels/fused.rs  (currently 872 lines)
    Adds:
      - fused_mul_sin_pow_mul_add_kernel CUDA source
      - fused_mul_sin_pow_mul_add_broadcast_kernel (for per-channel weights)
      - gpu_fused_mul_sin_pow_mul_add() host wrapper
      - Two entries in FUSED_KERNEL_NAMES
    Net change: approximately +100 lines (stays under 1000)
```

## TDD Implementation Plan

Five phases, each ending in a green commit. Every step names the red/green state explicitly so the implementation agent cannot skip a test step and "quickly verify later."

### Phase A: Refactor the fusion module

No behavioral change. The existing test suite is the regression gate — no test code is modified in this phase.

**A1.** Create empty `src/cuda/inference/fusion/mod.rs`, wire into `inference/mod.rs`. Build green.

**A2.** Move `FusedPattern` and `FusedPatternInfo` into `fusion/patterns.rs`. Mark `pub(crate)`. Update imports in `inference/mod.rs`. `cargo test --features cuda` green.

**A3.** Move `detect_fused_patterns` and its helpers into `fusion/detection.rs` as a free function with the signature:

```rust
pub(crate) fn detect_fused_patterns(
    nodes: &[ExecutionNode],
    weights: &HashMap<String, GpuTensor>,
) -> (HashMap<String, FusedPatternInfo>, HashSet<String>)
```

`cargo test --features cuda` green.

**A4.** Turn `GpuGraphExecutor::detect_fused_patterns` into a thin wrapper that calls the free function and populates `self.fused_patterns` and `self.nodes_to_skip`. `cargo test --features cuda` green, `cargo clippy --features cuda -- -D warnings` clean.

**Commit A:** `refactor(inference): extract fusion detection into dedicated module`

### Phase B: New fused kernel (test-first)

**B1 (red).** Write `tests/operators/fused_mul_sin_pow_mul_add_test.rs` with three tests calling `iconnx::cuda::kernels::gpu_fused_mul_sin_pow_mul_add`, which does not yet exist:

```rust
#[test]
fn naive_vs_fused_matches() {
    // same-shape case: a, w0, w1, b all [4, 8]; p = 2
    // compute naively via gpu_mul -> gpu_sin -> gpu_pow -> gpu_mul -> gpu_add
    // compute via gpu_fused_mul_sin_pow_mul_add
    // assert element-wise equality within 1e-5
}

#[test]
fn broadcasting_w1_and_b() {
    // a is [N, C]; w0, w1, b are [1, C]; p = 2
    // verify broadcast-correct output
}

#[test]
fn odd_exponent_p3() {
    // a designed to produce negative sin values; p = 3
    // verify sign handling matches naive computation
}
```

Tests fail to compile. Red confirmed.

**B2 (green, case 1).** Add `fused_mul_sin_pow_mul_add_kernel` CUDA source to `src/cuda/kernels/fused.rs` — same-shape only. Add `gpu_fused_mul_sin_pow_mul_add` host wrapper. Register in `FUSED_KERNEL_NAMES`. Test 1 passes; tests 2 and 3 still fail.

**B3 (green, broadcasting).** Add `fused_mul_sin_pow_mul_add_broadcast_kernel` CUDA variant (mirroring `fused_div_rsqrt_broadcast_kernel`'s shape convention — broadcast along the leading dimension). Host wrapper picks the broadcast kernel when shapes differ. Test 2 passes.

**B4 (green, odd exponent).** CUDA `powf(negative, 3.0)` returns NaN by default. For integer exponents (the common case), use `pown(x, int_p)` or manual expansion. Detect integer-valued `p` at kernel-launch time (attribute lookup) and pass an `int_exp` flag to the kernel. Test 3 passes.

**Commit B:** `feat(cuda): add fused_mul_sin_pow_mul_add kernel`

### Phase C: Pattern detection (test-first)

**C1 (red).** Create `tests/fusion_detection_test.rs`:

```rust
#[test]
fn detects_mul_sin_pow_mul_add_chain() {
    // build a Vec<ExecutionNode> with the 5-op chain
    // build a dummy weights HashMap
    // call fusion::detection::detect_fused_patterns(&nodes, &weights)
    // assert the head Mul is in the returned pattern map
    // assert the intermediate node names are in the skip set
}

#[test]
fn rejects_multi_use_intermediate() {
    // build the same chain but add a second consumer of the Pow output
    // assert the pattern is NOT registered
}
```

These fail because the detection code doesn't know the pattern. Red confirmed.

**C2 (green).** Add `MulSinPowMulAdd` variant to `FusedPattern`. Add a detection walker in `fusion/detection.rs` that:
- Starts from `Mul` nodes
- Follows single-use chain: `Mul → Sin → Pow → Mul → Add`
- Verifies the `Pow` exponent is a constant scalar
- Collects the skip set and builds `FusedPatternInfo`

Test 1 passes. Test 2 passes if the single-use check is correctly implemented.

**C3 (guard against regressions in other patterns).** Run the full test suite. The existing `DivRsqrt`, `Gelu`, `AddMulAdd`, and other patterns must still detect correctly — the new detection walker must not consume nodes that belonged to another pattern.

**Commit C:** `feat(fusion): detect MulSinPowMulAdd pattern`

### Phase D: Executor dispatch (test-first)

**D1 (red).** Extend `tests/fusion_detection_test.rs` or add a separate test that:
- Builds a minimal synthetic graph with the pattern
- Creates a real `GpuGraphExecutor`, runs inference
- Asserts `profile.op_times` contains `Fused_MulSinPowMulAdd` with the expected call count
- Asserts the output numerically matches a naive CPU ground truth

Red confirmed — dispatch doesn't know the pattern.

**D2 (green).** Add the `MulSinPowMulAdd` dispatch case in the executor's fused-pattern execution path. Call `gpu_fused_mul_sin_pow_mul_add`. Record under the name `Fused_MulSinPowMulAdd` in the profiling map. Test passes.

**Commit D:** `feat(inference): execute MulSinPowMulAdd fused pattern`

### Phase E: Kokoro validation

**E1.** `cargo test --features cuda` — full suite green.

**E2.** `cargo clippy --features cuda -- -D warnings` — zero warnings.

**E3.** Run `tests/kokoro_validation_test.rs` — iconnx's output on Kokoro must still match ORT within existing tolerance.

**E4.** Rebuild and run `leadline-bench compare` against the Kokoro model at **two sequence lengths** — `seq_len=5` (matches the original baseline) and `seq_len=50` (exposes tensor-size-dependent bandwidth effects). Use the reproduction steps in `docs/performance/2026-04-09-leadline-bench-findings.md`. Required outcomes at **each** sequence length:

- The `Fused_MulSinPowMulAdd` line appears in the iconnx operator breakdown
- Call count is ~48 (within ±5; some patterns may legitimately not match)
- `Mul`, `Sin`, `Pow`, `Add` counts in the profile each drop by approximately 48
- Variance is no worse than current (currently 46%)

Sequence-length-specific targets:

- **seq_len=5:** total iconnx latency decreases by ≥2 ms (target); ≥5 ms (stretch) from the 142.65 ms baseline
- **seq_len=50:** total iconnx latency decreases by ≥10 ms (target); ≥25 ms (stretch) from the 315.58 ms baseline, reflecting the expected larger bandwidth savings at larger tensor sizes

The seq_len=50 gains should be proportionally larger than seq_len=5. If they are not, the fusion is not actually keeping intermediates in registers — investigate the kernel implementation before declaring success.

**E5. Memory-side validation.** Record `executor.pool_stats()` (total allocations, cache hits, hit rate) before and after the optimization at both sequence lengths. Required outcomes:

- Allocation count drops by approximately 192 per inference (4 eliminated intermediates × 48 occurrences)
- Pool hit rate at seq_len=50 does not regress
- **Stretch goal:** rerun at seq_len=200. If it still OOMs, that is expected — cycles 2 and 3 plus likely memory-pool work are needed to fit that length. Report how far it gets. If it no longer OOMs, that is a strong signal and worth a commit message callout.

**E6.** If the real Kokoro pattern shape differs from the design's assumption — for example, `w0` is a scalar rather than a tensor, or the operand order is flipped — update the detection walker to match and rerun E4 and E5. Do not declare completion until the 48 detected instances number is verified. The verification-before-completion discipline from the superpowers skill applies: evidence before assertions, always.

**Commit E:** `perf(kokoro): validate MulSinPowMulAdd fusion across seq lengths`

## Validation Gates

Cycle 1 is complete when all of these are true:

1. `cargo test --features cuda` passes (all existing tests plus the new ones).
2. `cargo test` passes without the `cuda` feature (CPU fallback still works; the new kernel is behind the cuda feature gate).
3. `cargo clippy --features cuda -- -D warnings` reports zero warnings.
4. `cargo fmt --check` passes.
5. `tests/kokoro_validation_test.rs` passes — Kokoro output still matches ORT.
6. `leadline-bench compare` on Kokoro v1.0 at **both seq_len=5 and seq_len=50** shows:
   - New `Fused_MulSinPowMulAdd` line with ~48 calls at each sequence length
   - `Mul`, `Sin`, `Pow`, `Add` counts each decreased by ~48 at each sequence length
   - Total latency improved: ≥2 ms at seq_len=5, ≥10 ms at seq_len=50
   - Variance no worse than current baseline
7. Memory-side evidence from `executor.pool_stats()`:
   - Allocation count per inference drops by approximately 192
   - Pool hit rate does not regress
   - seq_len=200 stretch-goal result reported (still-OOMs-but-further OR no-longer-OOMs)
8. `src/cuda/inference/mod.rs` is ~300 lines smaller than at the start of this cycle.
9. New `src/cuda/inference/fusion/` module exists with `patterns.rs`, `detection.rs`, `mod.rs`, each under 1000 lines.
10. `src/cuda/kernels/fused.rs` remains under 1000 lines.
11. The latency improvement and memory pool stats are reported in the Phase E commit message (actual numbers from the benchmark, not the expected numbers from this spec).
12. Commits are signed per CLAUDE.md. If signing fails, stop and ask the user.

## Risks and Mitigations

**Risk:** The real Kokoro graph doesn't have the exact `Mul → Sin → Pow → Mul → Add` shape I'm assuming (operand order, broadcasting, `Pow` exponent source).
**Mitigation:** Phase E6 explicitly re-validates detection against the real graph. Before implementing C2, instrument the existing fusion detection temporarily to log nodes that look close to the pattern.

**Risk:** Adding the new pattern conflicts with an existing pattern's detection (e.g., the inner `Mul` at the start of the chain could also be the start of a different pattern).
**Mitigation:** C3 runs the full test suite to catch regressions in existing patterns. Detection walks in a specific order, and any `Mul` already claimed by an earlier-detected pattern is in the skip set and won't be rewalked.

**Risk:** `pown` or integer-exponent handling introduces numerical divergence from the original chain.
**Mitigation:** B1's `naive_vs_fused_matches` uses the unfused GPU kernels as the ground truth, not a CPU reference — so any divergence is caught against the very path we're replacing. Tolerance is `1e-5`.

**Risk:** The `leadline-bench` tool requires modifications or environment setup not captured here.
**Mitigation:** `docs/performance/2026-04-09-leadline-bench-findings.md` has a full reproduction recipe including `LD_LIBRARY_PATH` and the git-to-path dependency switch. Phase E4 follows that recipe.

**Risk:** File signing fails per CLAUDE.md — "only commit code with signing."
**Mitigation:** The implementation agent stops and asks the user to resolve signing rather than committing unsigned. No `--no-gpg-sign` fallback is permitted.

## Out of scope but worth noting for next cycles

- The refined Leadline analysis says `AddMulAdd` already exists as a pattern but finds 51 unfused `Add → Mul → Add` instances. Cycle 2 should first verify whether the existing detection is missing these, then decide whether the fix is a detection tweak or a broadened pattern.
- `AddMulAddLeakyRelu` is Cycle 3.
- Zero-cost Reshape + graph-time Gather-on-constants + Shape-at-build-time become Cycle 4 (the refined shape-ops work, now lighter than originally scoped).
- The broader refactor of `src/cuda/inference/mod.rs` continues organically: each future cycle extracts the sections it naturally touches, avoiding a single big-bang refactor.
