# Cycles 4 / 4.5 / 5 Results: Dynamic AddMulAdd Fusion

**Date:** 2026-04-12
**Specs:** Cycle 4, Cycle 4.5, Cycle 5 design specs in `docs/superpowers/specs/`
**Branches:** `cycle4/dynamic-addmuladd-fusion`, `cycle4.5/addmuladd-scheduling-edges`, `cycle5/channel-affine-kernel`

---

## Headline

**70 `Fused_AddMulAdd` calls fire on Kokoro at runtime**, eliminating 140 kernel launches per inference. `Add` calls drop from 304 to 164; `Mul` calls drop from 171 to 101. Each fusion replaces a 3-op chain (Add→Mul→Add) with a single `fused_channel_affine_kernel` call.

## The journey: three cycles of diagnostics

This feature required three cycles because each diagnostic round revealed that the previous assumption was wrong:

**Cycle 3** discovered the 73 `Add→Mul→Add` chains are AdaIN blocks with runtime-computed b/c (Div/Slice outputs), not static constants. The `is_static` walker fix shipped but had zero Kokoro impact.

**Cycle 4** built the complete dynamic fusion infrastructure: broadcast-c kernel, FusionShapeMismatch fallback, dynamic candidate walker, executor dispatch with `dynamically_fused` per-run tracking. 73 candidates detected, 0 fused at runtime — topological ordering placed b/c producers after the head.

**Cycle 4.5** added scheduling edges to the topo sort, forcing b/c producers before the head. All 73 candidates reached the dispatcher with inputs available. But all 73 fell through on shape mismatch — the shapes were different from what we expected.

**The diagnostic revealed:** `x=[1,C,1], a=[], b=[1,C,T], c=[1,C,1]`. The full-shape tensor is `b` (the normalized input from Div), not `x`. The pattern is a per-channel affine transform, not a general 4-operand fusion.

**Cycle 5** added the `fused_channel_affine_kernel` matching the actual pattern: `out[i] = scale[i/T] * input[i] + bias[i/T]`. 70 of 73 candidates fuse successfully at runtime.

## Kokoro seq_len=5 results

```
  Add                                    10222 us (  164 calls)   [was 304]
  Fused_AddMulAdd                         4466 us (   70 calls)   [was 0]
  Mul                                     3313 us (  101 calls)   [was 171]
```

| Metric | Before (Cycle 3) | After (Cycle 5) | Change |
|---|---|---|---|
| Add calls | 304 | 164 | -140 |
| Mul calls | 171 | 101 | -70 |
| Fused_AddMulAdd calls | 0 | 70 | +70 |
| Net kernel launches saved | — | 140 | — |

3 of the 73 candidates don't fuse — these are the LSTM layer-norm chains where the shapes are `x=[1,1,512], a=[], b=[1,5,512], c=[1,1,512]`. The per-channel affine kernel requires `T > 1` in `b`'s trailing dim and `x.shape() == c.shape() == [1,C,1]`. The LSTM pattern has `T=512, C=5` with `x=[1,1,512]` — the channel and time dims are swapped relative to the AdaIN pattern. These 3 chains correctly fall back to unfused execution.

## Pattern counts (full snapshot)

```
=== Cycle 4 fused pattern counts ===
  DivRsqrt:            65   [unchanged]
  AddMulAdd:           73   [detected — 70 fuse, 3 fall back]
  Gelu:                12   [unchanged]
  MulSinPowMulAdd:     48   [unchanged]
  MulAdd:               0
  AddMul:               0
  SubMul:               0
  DivMul:               0
```

## What shipped (across all 3 cycles)

| Cycle | Commits | Key change |
|---|---|---|
| 4 | 8 | Dynamic candidate infrastructure: kernel, wrapper, walker, executor dispatch, 16 tests |
| 4.5 | 1 | Scheduling edges in topo sort |
| 5 | 1 | Per-channel affine kernel + wrapper dispatch |

### Key files

- `src/cuda/kernels/fused/mod.rs` — two new kernel sources: `fused_add_mul_add_bcast_c_kernel` and `fused_channel_affine_kernel`
- `src/cuda/kernels/fused/wrappers.rs` — three-way wrapper dispatch + `compute_c_broadcast_params` helper + per-channel affine detection
- `src/cuda/inference/fusion/detection.rs` — dynamic candidate registration with `dynamic_heads`/`dynamic_interiors` guard sets
- `src/cuda/inference/mod.rs` — `dynamic_candidates` field, scheduling edges in topo sort, dynamic dispatch in all 4 run methods, `dynamic_candidate_inputs_ready` pre-check
- `src/cuda/context.rs` — `CudaError::FusionShapeMismatch` sentinel

## Methodology lesson

**Diagnostic-first development pays compounding returns.** Every cycle in this series started with a shape or structure assumption that turned out to be wrong:
- Cycle 3: "b/c come from Constant nodes" → actually Div/Slice (runtime)
- Cycle 4: "x/a/b are full-shape, only c broadcasts" → actually b is full-shape, x/c are per-channel
- Cycle 4.5: "scheduling edges make b/c available" → they do, but shapes still don't match

Each diagnostic round cost ~30 minutes and prevented a wasted implementation cycle. The compounding effect: without diagnostics, we would have shipped a kernel that matched zero real chains (Cycle 4's broadcast-c kernel) and declared victory based on detection-time counts.

## Follow-ups

1. **LSTM layer-norm fusion (3 chains)** — different shape pattern `x=[1,1,C], b=[1,T,C]`. Would need a second kernel variant or the general broadcast-c approach with axis-swapped parameters. Low priority (3 chains).
2. **Leadline-bench re-run** — the 140 fewer kernel launches should show measurable improvement. The big question: does the launch-overhead reduction dominate, or is the memory bandwidth savings from fewer intermediate stores the bigger win?
3. **AddMulAddLeakyRelu** (22 instances from Leadline's chain analysis) — separate fusion pattern, separate cycle.
