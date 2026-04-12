# Cycle 4: Dynamic AddMulAdd Fusion — Design Spec

**Date:** 2026-04-11
**Predecessor:** Cycle 3 (AddMulAdd detection gap — AdaIN diagnosis)
**Branch:** `cycle4/dynamic-addmuladd-fusion`

---

## Goal

Fuse Kokoro's ~73 `Add → Mul → Add` chains that Cycle 3 identified as Adaptive Instance Normalization (AdaIN) and LSTM layer-norm blocks. These chains have runtime-computed `b` (scale) and `c` (bias) operands — the existing `gpu_fused_add_mul_add` kernel requires all four tensors at the same shape and cannot handle broadcasting. Cycle 4 adds a new broadcast-c kernel variant, wires it through the walker and executor with shape-check fallback to unfused execution on mismatch.

## Empirical shape data (from Cycle 4 diagnostic round)

Shapes observed on Kokoro v1.0 at seq_len=5:

**AdaIN chains (~70 of 73):**
- `x`, `a`, `b`: all at full shape `[1, C, T]` — e.g., `[1, 256, 1900]`, `[1, 1024, 95]`
- `c`: `[1, C, 1]` — broadcast over trailing dim `T`
- `b` = Div_1_output (instance-norm scale, runtime), `c` = Slice_1_output (style projection bias, runtime)

**LSTM layer-norm chains (3 of 73):**
- `x`, `a`, `b`: all at `[1, 5, 512]`
- `c`: `[1, 1, 512]` — broadcast over middle dim (sequence dim of 5)
- `b` = LayerNormalization_output, `c` = Transpose_1_output

**Key constraint: only `c` broadcasts.** In every observed chain, `x`, `a`, `b` share the full output shape. Only `c` has a size-1 dimension that needs broadcasting. This significantly simplifies the kernel design — no broadcasting logic needed for three of the four operands.

## Kernel design — one kernel, two parameters

Both broadcast patterns collapse into a single CUDA kernel using a generalized index formula:

```cuda
extern "C" __global__ void fused_add_mul_add_bcast_c_kernel(
    float* out,
    const float* x,
    const float* a,
    const float* b,
    const float* c,
    size_t n,
    size_t c_inner_stride,
    size_t c_size
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        size_t c_idx = (i / c_inner_stride) % c_size;
        out[i] = (x[i] + a[i]) * b[i] + c[c_idx];
    }
}
```

- **AdaIN** (`c = [1, C, 1]`, `x = [1, C, T]`): `c_inner_stride = T`, `c_size = C`. Gives `c[(i/T) % C]` = `c[i/T]`.
- **LSTM** (`c = [1, 1, C]`, `x = [1, T, C]`): `c_inner_stride = 1`, `c_size = C`. Gives `c[(i/1) % C]` = `c[i % C]`.

Same kernel, two integer launch parameters derived from runtime shape inspection. The existing same-shape `fused_add_mul_add_kernel` stays for the case when all four shapes match (static affine transforms from other models); the new kernel handles the broadcast-c case only.

## Walker modification — dynamic candidates

The walker's `detect_fused_patterns` currently rejects `Add → Mul → Add` chains where `b` or `c` fails `is_static`. Cycle 4 extends this: instead of skipping non-static chains, the walker registers them as **dynamic candidates** in a separate data structure.

```rust
pub(crate) struct FusedPatternResult {
    pub patterns: HashMap<String, FusedPatternInfo>,       // static (unconditional)
    pub nodes_to_skip: HashSet<String>,                     // static (unconditional)
    pub dynamic_candidates: HashMap<String, FusedPatternInfo>, // dynamic (try at runtime)
}
```

Static-fusible chains (`is_static(b) && is_static(c)`) go into `patterns` + `nodes_to_skip` as before — no change to existing behavior.

Dynamic chains (where one or both operands fail `is_static`) go into `dynamic_candidates` only. Their interior nodes are NOT added to `nodes_to_skip` — they must remain executable for the fallback path.

The walker only considers `AddMulAdd` chains for dynamic registration. Other pattern types (DivRsqrt, Gelu, MulSinPowMulAdd, etc.) are unaffected.

## Executor dispatch — shape-check with fallback

The execution loop in `run_with_profiling` (and `run`) gains a second check after the existing static-pattern dispatch:

```
for each node in sorted_nodes:
    if nodes_to_skip.contains(node):     → skip (static fusion, existing)
    if fused_patterns.contains(node):    → execute fused (static, existing)
    if dynamic_candidates.contains(node):
        → try_execute_dynamic_fused(pattern, values)
        → if Ok(output):  store result, mark interior nodes as "dynamically fused"
        → if Err(FusionShapeMismatch): fall through to normal execution
    if dynamically_fused.contains(node): → skip (dynamic fusion succeeded earlier)
    else:                                → normal node execution
```

The `dynamically_fused: HashSet<String>` tracks interior nodes whose pattern head was successfully fused at runtime. On shape mismatch, the head node executes normally and its interior nodes also execute normally (they're not in `dynamically_fused`).

**Fallback cost:** running 3 individual ops (Add, Mul, Add) instead of 1 fused op. Minor performance regression, zero correctness impact. This is the "precompute hit or miss" pattern from Cycle 2: try the fast path, fall back on miss.

## Wrapper broadcast detection

The Rust wrapper `gpu_fused_add_mul_add` (or a new wrapper function) gains shape-inspection logic:

1. If `x.shape() == a.shape() == b.shape() == c.shape()` → dispatch existing same-shape kernel.
2. Else if `x.shape() == a.shape() == b.shape()` AND `c` broadcast-compatible with `x`:
   - Compute `(c_inner_stride, c_size)` from the shape pair.
   - Dispatch `fused_add_mul_add_bcast_c_kernel`.
3. Else → return `Err(CudaError::FusionShapeMismatch)` (the sentinel).

**Broadcast compatibility check for `c`:** c is broadcast-compatible with x if:
- c and x have the same number of dimensions.
- For each dimension, c's size is either 1 or equals x's size.
- At least one dimension in c is size 1 (otherwise it's the same-shape case).

**Computing `(c_inner_stride, c_size)`:** Given x shape `[d0, d1, d2]` and c shape `[c0, c1, c2]`:
- Find the contiguous non-broadcast block in c (the dimensions where `ci == di`). Call this block's product `c_size`.
- `c_inner_stride` is the product of all x dimensions AFTER the broadcast block.
- For AdaIN: c = [1, C, 1], x = [1, C, T]. Non-broadcast block is dim 1 (size C). `c_size = C`. `c_inner_stride = T` (product of dims after the block).
- For LSTM: c = [1, 1, C], x = [1, T, C]. Non-broadcast block is dim 2 (size C). `c_size = C`. `c_inner_stride = 1` (no dims after the block).

This logic generalizes to any 3D broadcast pattern where `c` has exactly one contiguous non-broadcast block. Patterns with non-contiguous non-broadcast blocks (e.g., `c = [C, 1, C]`) are rejected as incompatible.

## Testing strategy

### Unit tests (synthetic graphs)

1. **Dynamic AddMulAdd with trailing-dim broadcast c** — synthetic graph with x/a/b at `[1, 4, 8]`, c at `[1, 4, 1]`. Verify walker registers as dynamic candidate. Verify execution produces correct output. Verify fused-pattern count increments.

2. **Dynamic AddMulAdd with middle-dim broadcast c** — synthetic graph with x/a/b at `[1, 3, 6]`, c at `[1, 1, 6]`. Same verifications.

3. **Dynamic AddMulAdd with same-shape (no broadcast)** — synthetic graph where all four are `[1, 4, 8]` but b/c are non-static (Div/Slice outputs). Verify walker registers as dynamic candidate. Verify existing same-shape kernel is dispatched.

4. **Shape mismatch fallback** — synthetic graph where c's shape is incompatible (e.g., `[2, 4, 8]` vs x's `[1, 4, 8]`). Verify fallback to unfused execution. Verify output is still correct. Verify no error propagation.

5. **Static AddMulAdd regression** — existing tests from Cycle 3 (`add_mul_add_detects_with_initializer_bc`, etc.) still pass unchanged.

### Kokoro integration assertion

Update `tests/kokoro_gpu_fusion_test.rs::fusion_fires_on_full_kokoro_at_seq_len_5`:
- Change `pattern_counts.add_mul_add == 0` assertion (from Cycle 3) to `pattern_counts.add_mul_add >= 40`.
- Add a separate dynamic-fusion count diagnostic: how many of the 73 candidates were fused successfully vs. fell back.
- Add the `PatternCount` to count both static AND dynamic fusions (both go through the AddMulAdd kernel, so both should increment `add_mul_add`).

The threshold `>= 40` absorbs:
- Single-use exclusions (the walker's existing requirement)
- Future pattern subsumption (AddMulAddLeakyRelu, not yet implemented)
- Shape-mismatch fallbacks on chains we haven't characterized

## Scope boundaries

**In scope:**
- New `fused_add_mul_add_bcast_c_kernel` CUDA kernel source
- Rust wrapper with broadcast detection + same-shape fallback + shape-mismatch sentinel
- Walker: register non-static `AddMulAdd` chains as dynamic candidates
- Executor: dynamic-candidate dispatch with fallback
- `FusedPatternResult` struct to carry both static patterns and dynamic candidates
- Unit tests for all four cases above
- Kokoro integration assertion update

**Out of scope:**
- Broadcasting `b` or `a` (only `c` broadcasts in Cycle 4)
- Tensor ranks other than 3 (all observed chains are 3D)
- Non-contiguous broadcast blocks in `c` (e.g., `c = [C, 1, C]`)
- `AddMulAddLeakyRelu` pattern (separate cycle)
- Symbolic shape inference at walker time (not needed — runtime shape check is the mechanism)
- Any changes to other fusion patterns (DivRsqrt, Gelu, MulSinPowMulAdd, etc.)

## File impact estimate

| File | Change | Est. lines |
|---|---|---|
| `src/cuda/kernels/fused/mod.rs` | New kernel source | +30 |
| `src/cuda/kernels/fused/wrappers.rs` | Broadcast detection + dispatch | +80 |
| `src/cuda/inference/fusion/detection.rs` | Dynamic candidate registration | +20 |
| `src/cuda/inference/fusion/patterns.rs` | `FusedPatternResult` struct | +15 |
| `src/cuda/inference/mod.rs` | Dynamic dispatch + fallback in execution loop | +60 |
| `src/cuda/inference/testing.rs` | Test helper updates for dynamic counts | +20 |
| `tests/fusion_detection_test.rs` | 4-5 new unit tests | +200 |
| `tests/kokoro_gpu_fusion_test.rs` | Assertion update | +20 |

Total: ~445 new lines. `mod.rs` grows by ~60 lines (4,722 → ~4,782). Still well above the 1,000-line soft cap but Cycle 4 doesn't make it significantly worse.

## New error variant

Add `FusionShapeMismatch` to the `CudaError` enum (in `src/cuda/mod.rs` or wherever `CudaError` is defined). This is the sentinel that the wrapper returns when shapes don't match any dispatch path. The executor catches this specific variant and falls through to unfused execution; all other `CudaError` variants propagate as hard errors.

## Error handling

| Scenario | Behavior |
|---|---|
| Shape mismatch at dispatch time | Wrapper returns `CudaError::FusionShapeMismatch`. Executor falls through to unfused execution. Interior nodes execute normally. |
| `b` needs broadcasting (unexpected) | Same as shape mismatch — treated as incompatible. Falls through to unfused. |
| `c` has non-contiguous broadcast dims | Same as shape mismatch. |
| Kernel launch failure | Propagates as `CudaError::Kernel(...)` — hard error, inference fails. This is a real CUDA error, not a shape issue. |
| Tensor not found in `values` at dispatch time | Hard error (`CudaError::Kernel("... not found")`). Same as existing behavior. |

## Success criteria

1. `cargo clippy --features cuda --tests -- -D warnings` = 0 errors
2. `cargo test --features cuda` = same pass/fail/ignore as baseline
3. Kokoro integration test passes with `pattern_counts.add_mul_add >= 40`
4. Diagnostic printout shows ~70 dynamic fusions + 0 fallbacks on Kokoro seq_len=5
5. Existing Cycle 1-3 fusion counts (DivRsqrt 65, Gelu 12, MulSinPowMulAdd 48) unchanged
6. All commits signed
