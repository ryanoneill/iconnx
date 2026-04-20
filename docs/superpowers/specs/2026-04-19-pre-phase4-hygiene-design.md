# Pre-Phase-4 hygiene: silent-skip fixes + Erf operator

**Status:** finalized 2026-04-19 after brainstorming dialogue with leadline.
**Scope label:** "C" (per Phase 4 scope decomposition in `2026-04-17-phase3-graph-optimizer.md` §Phase 3 outcomes).
**Landing:** one PR with two signed commits on branch `pre-phase4-c` (or similar). Phase 4 proper (B then A) starts from the post-C main tip.

## Motivation

Phase 3 merged with a zero direct-per-run-benefit finding on Kokoro. Two threads need clearing before Phase 4 proper lands:

1. **Silent-skip tests masked a real Gather correctness bug in Phase 3 Commit 3** until the Kokoro model was symlinked into the ir-phase3 worktree. Eight test files still use the silent-skip pattern (`if !model_path.exists() { return; }`). Phase 4's memory planner has the same failure mode — a silent-skip could hide a wrongly-sized buffer for months. Fix before Phase 4 lands.

2. **Whisper (HuggingFace ONNX export) is the planned second benchmark.** An op-coverage probe (`leadline-bench/src/bin/probe_ops.rs`) found iconnx's 50-op dispatch table covers 17/18 of Whisper-Tiny encoder's unique ops; `Erf` is the sole missing op. Adding it unblocks the second benchmark. `Erf`-chain GELU is the first non-trivial `FusedPattern::GeneralChain` fire we'd expect in the wild, validating Phase 3 Commit 4's generic-infrastructure frame.

Out of iconnx scope (parallel leadline-bench PR):
- Benchmark provenance recording (ORT + CUDA driver + cuDNN versions + GPU name). Lives in `leadline-bench/src/bin/compare.rs` and a new `environment.json` sidecar; later extended into `SessionProto::EnvironmentProto` as a follow-up.

## Scope

One PR, two commits:

| # | Commit | Files | Landing criteria |
|---|---|---|---|
| 1 | `test(ir): fail loud on missing Kokoro model across 8 tests` | `tests/common/kokoro_model.rs` (new shared helper), 8 test files | All previously-silent-skip tests panic with the shared diagnostic when `kokoro-v1.0.onnx` is missing. Existing tests that run (with the model present) still pass. |
| 2 | `feat(ops): add Erf unary operator (CPU + GPU + fusion coverage)` | `src/operators/erf.rs` (new), 11 integration-point files | Full unary-op integration: CPU forward, GPU `gpu_erf`, constant-folding dispatch, shape inference pass-through, elementwise fusion allowlists, codegen arm, executor dispatch, CPU spot + property tests, GPU correctness test, Erf-chain GELU equivalence test demonstrating `GeneralChain` fires. |

## Commit 1: silent-skip fixes

### Affected files

Eight files currently use `if !model_path.exists() { return; }` or `println!("Model not found, skipping")` + early return:

- `tests/kokoro_inference_test.rs`
- `tests/kokoro_progressive_test.rs`
- `tests/kokoro_debug_test.rs`
- `tests/kokoro_validation_test.rs`
- `tests/validation_test.rs` (8 skip sites)
- `tests/onnx_parser_test.rs` (5 skip sites)
- `tests/weight_extraction_test.rs`
- `tests/end_to_end_test.rs`

Already-fixed in Phase 3 (out of scope here): `tests/kokoro_gpu_fusion_test.rs`, `tests/kokoro_shape_trace_test.rs`.

### Shared helper

```rust
// tests/common/kokoro_model.rs

use std::path::{Path, PathBuf};

/// Resolve the Kokoro model path from the test working directory.
/// Panics loudly if missing — silent-skip masked the Phase 3 Commit 3
/// Gather shape-inference correctness bug for weeks. Future silent
/// defects in the memory planner or elsewhere would be hidden the
/// same way; we make the failure mode explicit.
pub fn kokoro_model_path() -> PathBuf {
    let path = PathBuf::from("kokoro-v1.0.onnx");
    assert!(
        path.exists(),
        "kokoro-v1.0.onnx must be reachable from the working directory \
         (symlink or copy it in). A silent skip here would hide real \
         correctness regressions — see Phase 3 Commit 3's Gather bug \
         for an example of what silent-skipping can mask."
    );
    path
}
```

Each affected test file replaces its local `if !model_path.exists() { return; }` block with a call to `common::kokoro_model::kokoro_model_path()` (or a `use` of the helper fn directly). Parsing now uses `.expect("parse kokoro-v1.0.onnx")` instead of `.ok()?`.

### Tests remaining as `#[ignore]`

Every test that was already `#[ignore]`'d stays `#[ignore]`'d. The silent-skip fix is about loud failure when the test DOES run, not about whether it runs by default. Long-running GPU integration tests continue to require explicit `--ignored` invocation.

### Non-Kokoro model tests

Out of scope: `tests/end_to_end_test.rs` has one Kokoro-gated test and also synthetic-ONNX tests. Only the Kokoro-gated block is fixed. Synthetic tests that create ephemeral ONNX files in `tempdir` have different failure modes (permissions, tempdir allocation) and stay as-is.

## Commit 2: Erf operator

### CPU implementation (`src/operators/erf.rs`)

```rust
/// Erf activation operator for Iconnx.
///
/// erf(x) = (2/√π) · ∫₀ˣ e^(-t²) dt
/// ONNX spec: https://onnx.ai/onnx/operators/onnx__Erf.html
///
/// Rust `std` does not expose erf on stable. Implemented here via the
/// Abramowitz & Stegun 7.1.26 polynomial approximation. Max absolute
/// error ~1.5e-7, well within iconnx's 1e-5 fp32 correctness budget.
/// iconnx has zero math-crate dependencies; introducing libm for a
/// single op would be speculative infrastructure (see Phase 3 Commit 2
/// for the same "no crate for one use" rule applied to fp64).
use crate::tensor::Tensor;

pub struct Erf;

impl Erf {
    pub fn forward(
        inputs: &[Tensor],
        _attributes: &crate::attributes::NodeAttributes,
    ) -> Tensor {
        assert_eq!(inputs.len(), 1, "Erf requires exactly 1 input");
        let data = inputs[0].to_array();
        let result = data.mapv(erf_f32);
        Tensor::from_array(result)
    }
}

/// Abramowitz & Stegun 7.1.26. |error| ≤ 1.5e-7 on all finite inputs.
fn erf_f32(x: f32) -> f32 {
    let sign = x.signum();
    let a = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-a * a).exp();
    sign * y
}
```

### GPU implementation (`src/cuda/kernels/mod.rs`)

New `gpu_erf` free function alongside `gpu_tanh`, `gpu_sigmoid`, etc. Launches a single-source elementwise kernel whose body is `out[i] = erff(in[i]);`. Follows the same pattern as existing unary elementwise kernels — no new kernel-cache entry type, no new dispatch machinery.

Integration sites (total 11 for Commit 2):

| # | File | Change |
|---|---|---|
| 1 | `src/operators/erf.rs` | NEW — CPU forward + A&S polynomial |
| 2 | `src/operators/mod.rs` | `pub mod erf;` |
| 3 | `src/graph_executor.rs` | Dispatch arm `"Erf" => Ok(erf::Erf::forward(inputs, attributes))` |
| 4 | `src/ir/passes/constant_folding.rs` | Dispatch arm `"Erf" => erf::Erf::forward(inputs, attributes)` |
| 5 | `src/ir/passes/shape_inference.rs:107` | Add `"Erf"` to the unary pass-through list |
| 6 | `src/ir/passes/elementwise_fusion/mod.rs` | Add `"Erf"` to both `ELEMENTWISE_OPS` and `UNARY_ELEMENTWISE_OPS` |
| 7 | `src/ir/passes/elementwise_fusion_codegen.rs` | Add `"Erf" => "    v = erff(v);\n".into()` arm |
| 8 | `src/cuda/kernels/mod.rs` | NEW `pub fn gpu_erf(...)` |
| 9 | `src/cuda/mod.rs` | Add `gpu_erf` to re-exports (line near :72) |
| 10 | `src/cuda/executor/elementwise.rs` | Dispatch arm `"Erf" => gpu_erf(...)` |
| 11 | `src/cuda/executor/mod.rs:241` | Add `"Erf"` to the elementwise op classifier |

### Tests

CPU tests in `src/operators/erf.rs::tests` module:

1. **Spot values:**
   - `erf(0.0) == 0.0`
   - `(erf(1.0) - 0.8427008).abs() < 1e-6`
   - `(erf(-1.0) + 0.8427008).abs() < 1e-6`
   - `erf(f32::INFINITY) == 1.0`
   - `erf(f32::NEG_INFINITY) == -1.0`
   - `erf(f32::NAN).is_nan()` (a polynomial eval times `exp(-NaN*NaN)` = NaN; verify the sign propagation doesn't turn it into a finite)

2. **Property tests (hand-rolled grid over [-6, 6], step 0.001, ~12k inputs):**
   - **Odd symmetry:** `(erf(-x) + erf(x)).abs() < 1e-6` for every grid point.
   - **Boundedness:** `erf(x).abs() <= 1.0 + 1e-6` for every grid point (tolerance absorbs polynomial error).
   - **Monotonicity:** for every adjacent pair `(x_n, x_{n+1})`, `erf(x_{n+1}) >= erf(x_n) - 1e-6`. The tolerance absorbs polynomial noise; true strict monotonicity holds mathematically.

3. **A&S tolerance check:** against high-precision reference values at ±0.5 and ±2.5, `|erf_f32(x) - reference| < 2e-7`. Locks the polynomial coefficients; any transcription drift fails loud. Reference values (sourced from high-precision references; truncated to f32 in the test):
   - `erf(0.5) ≈ 0.5204998778130465`
   - `erf(-0.5) ≈ -0.5204998778130465`
   - `erf(2.5) ≈ 0.9995930479825550`
   - `erf(-2.5) ≈ -0.9995930479825550`

GPU test in `src/cuda/kernels/tests.rs`:

- `test_gpu_erf`: random f32 input tensor on [-5, 5], launch `gpu_erf`, download, compare against CPU `Erf::forward` with `1e-6` tolerance.

Fusion equivalence test in `src/ir/passes/elementwise_fusion/equivalence_tests.rs`:

- `erf_chain_gelu_prefix_fuses_into_general_chain_matches_sequential`: build a GELU-via-Erf graph `x → Mul(1/√2) → Erf → Add(1.0) → Mul(0.5) → Mul(x)`. Note that the final `Mul(x)` is a two-input op (its second input is the original `x`), which breaks the single-input invariant `FusedPattern::GeneralChain` requires. The fuseable piece is therefore the 4-op prefix `Mul(1/√2) → Erf → Add(1.0) → Mul(0.5)`; the final `Mul(x)` stays separate. Lower the graph, verify a `FusedPattern::GeneralChain` annotation appears for the 4-op prefix (signature includes `Erf` verbatim), run it on GPU, compare `fused(prefix) → Mul(x)` against the fully-sequential unfused chain at 1e-5 tolerance.

## Non-goals

- `libm`, `statrs`, or any math-crate dependency.
- `proptest` or `quickcheck` dev-dependency.
- A dedicated `FusedPattern` variant for GELU-via-Erf. `GeneralChain` handles it.
- Benchmark provenance recording (moves to parallel leadline-bench PR).
- Pre-Phase-4 fixes to non-Kokoro silent-skip tests or to `.rs` test files that don't reference Kokoro.
- Converting any `#[ignore]`'d test to a non-ignored test.

## Open questions

None. All design questions resolved in brainstorming dialogue with leadline.

## Testing strategy summary

| Layer | Coverage |
|---|---|
| CPU unit | Spot values (5) + property tests (3 invariants × 12k inputs) + A&S reference locks (4) |
| GPU unit | `test_gpu_erf` comparing GPU kernel to CPU at 1e-6 |
| Integration | Erf-chain GELU fusion equivalence test demonstrating `GeneralChain` fires correctly |
| Regression guards | 8 Kokoro-loading tests now panic on missing model instead of silently skipping |

## Landing criteria (checklist-style)

- [ ] `cargo build --features cuda --release` clean on both `cuda` and `cuda,debug-inference`.
- [ ] Zero warnings on both feature combos.
- [ ] All existing tests still pass (443+ on `cuda --release`).
- [ ] 4 new CPU spot-value tests pass.
- [ ] 3 new property tests pass (odd-symmetry, boundedness, monotonicity).
- [ ] A&S tolerance test passes.
- [ ] GPU `test_gpu_erf` passes on CUDA hardware.
- [ ] Erf-chain GELU fusion equivalence test passes on CUDA hardware.
- [ ] All 8 previously-silent-skip tests now panic loudly when `kokoro-v1.0.onnx` is missing (manually verify by running them without the symlink present, confirm panic with the new diagnostic).
- [ ] All commits GPG-signed.

## Post-C → Phase 4 handoff

Once C merges, Phase 4 proper starts. Scope split:
- **B:** const-fold extension for Shape → Cast → Gather runtime-shape chains (unblocks 23/51 Reshape precompute misses on Kokoro — the stacked-gate verdict's resolution path).
- **A:** memory planner (static lifetime analysis + pre-allocated arena, consuming `tensor_shapes` + `tensor_last_use` from Phase 3).
- B lands before A. Both squashed into Phase 4's single PR.

Phase 4 brainstorming is a separate cycle starting after C lands.
