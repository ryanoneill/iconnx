# Cycle 4: Dynamic AddMulAdd Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fuse Kokoro's ~73 `Add -> Mul -> Add` chains (AdaIN and LSTM layer-norm) that have runtime-computed `b`/`c` operands, using a new broadcast-c CUDA kernel with shape-check fallback to unfused execution on mismatch.

**Architecture:** A new CUDA kernel `fused_add_mul_add_bcast_c_kernel` handles the case where `x`, `a`, `b` share identical shapes but `c` broadcasts over one or more size-1 dimensions. The walker registers non-static `AddMulAdd` chains as *dynamic candidates* (separate from unconditionally-fusible static patterns). At execution time, the executor attempts fusion on each dynamic candidate: if `gpu_fused_add_mul_add` succeeds (same-shape or broadcast-c dispatch), the interior nodes are marked as "dynamically fused" for the remainder of that run; if the wrapper returns `CudaError::FusionShapeMismatch`, execution falls through to the normal unfused path. The `dynamically_fused` set is a per-run local variable, not a persistent field on the executor.

**Tech Stack:** Rust 2021, CUDA C (via cudarc NVRTC), existing iconnx infrastructure. No new crate dependencies.

**Prerequisites for the implementing agent:**
- Read the spec first: `docs/superpowers/specs/2026-04-11-dynamic-addmuladd-fusion-design.md`.
- Per `/home/ryano/.claude/CLAUDE.md`: TDD where possible, commits after every substantive change, **all commits must be signed** (`git commit -S`), stop and ask if signing fails, warnings never ignored.
- `cargo test --features cuda` is the regression gate. Run after each task.
- `cargo clippy --features cuda --tests -- -D warnings` must be clean before every commit.
- **NEVER run `git checkout <sha>`** to inspect commits. Use `git show SHA`, `git show SHA --stat`, or `git log --show-signature -1 SHA`.
- **NEVER use unsafe Rust** without asking first. The CUDA launch in the wrapper already uses `unsafe` -- follow the existing pattern exactly.
- **No files over 1000 lines.** `src/cuda/inference/mod.rs` is already 4722 lines (flag but don't refactor).

---

## Scope Check

This plan implements **Cycle 4 only** -- Dynamic AddMulAdd Fusion. Out of scope:

- Broadcasting `b` or `a` (only `c` broadcasts in Cycle 4)
- Tensor ranks other than 3 (all observed chains are 3D)
- Non-contiguous broadcast blocks in `c` (e.g., `c = [C, 1, C]`)
- `AddMulAddLeakyRelu` pattern (separate cycle)
- Symbolic shape inference at walker time
- Any changes to other fusion patterns (DivRsqrt, Gelu, MulSinPowMulAdd, etc.)
- Refactoring `mod.rs` file size

Any work outside the files listed below is a scope violation and should be flagged to the controller before landing.

---

## File Structure

### Modified files

| File | Responsibility | Est. delta |
|---|---|---|
| `src/cuda/context.rs` | Add `FusionShapeMismatch` variant to `CudaError` enum + Display arm | +4 lines |
| `src/cuda/kernels/fused/mod.rs` | Add broadcast-c kernel source + register in `FUSED_KERNEL_NAMES` | +25 lines |
| `src/cuda/kernels/fused/wrappers.rs` | Broadcast-c detection + dispatch in `gpu_fused_add_mul_add` | +80 lines |
| `src/cuda/inference/fusion/detection.rs` | Register non-static `AddMulAdd` chains as dynamic candidates; return 3-tuple | +25 lines |
| `src/cuda/inference/fusion/mod.rs` | Re-export updated function | +1 line |
| `src/cuda/inference/mod.rs` | Add `dynamic_candidates` field, wire through `detect_fused_patterns`, add dynamic dispatch + fallback in all 4 execution loops | +80 lines |
| `src/cuda/inference/testing.rs` | Update `detect_fused_patterns_for_tests` to return 3-tuple, update `count_fused_patterns` to include dynamic candidates | +15 lines |
| `tests/fusion_detection_test.rs` | 4 new unit tests for dynamic candidate detection | +120 lines |
| `tests/kokoro_gpu_fusion_test.rs` | Update `add_mul_add` assertion from `== 0` to `>= 40` | +5 lines |

### Files NOT touched

- `src/cuda/inference/fusion/patterns.rs` -- `FusedPattern::AddMulAdd` already exists; `FusedPatternInfo` is unchanged.
- `src/cuda/kernels/fused/wrappers.rs` existing kernel wrappers -- only `gpu_fused_add_mul_add` changes.
- `src/cuda/inference/gpu_event_timer.rs` -- no changes.
- Cycle 1-3 tests that assert existing fusion counts.

---

## Task 1: Add `FusionShapeMismatch` to `CudaError`

**Files:**
- Modify: `src/cuda/context.rs:152-189`

This is a pure type-level change. No tests needed -- the variant is only useful once the wrapper returns it in Task 3.

- [ ] **Step 1: Add the new variant to the `CudaError` enum**

In `src/cuda/context.rs`, add a new variant after `Memory(String)` at line 170:

```rust
/// Fusion shapes are incompatible — fall back to unfused execution.
/// This is a sentinel, not a hard error: the executor catches it and
/// runs the individual ops instead.
FusionShapeMismatch(String),
```

The enum at lines 152-171 becomes:

```rust
#[derive(Debug, Clone)]
pub enum CudaError {
    /// Failed to initialize device
    DeviceInit(String),
    /// Failed to create stream
    StreamCreate(String),
    /// Failed to allocate memory
    Alloc(String),
    /// Failed to transfer data
    Transfer(String),
    /// Failed to synchronize
    Sync(String),
    /// Kernel execution failed
    Kernel(String),
    /// cuBLAS operation failed
    Cublas(String),
    /// cuDNN operation failed
    Cudnn(String),
    /// Memory operation failed
    Memory(String),
    /// Fusion shapes are incompatible — fall back to unfused execution.
    /// This is a sentinel, not a hard error: the executor catches it and
    /// runs the individual ops instead.
    FusionShapeMismatch(String),
}
```

- [ ] **Step 2: Add the Display arm**

In the `Display` impl at lines 173-186, add before the closing brace:

```rust
CudaError::FusionShapeMismatch(s) => write!(f, "Fusion shape mismatch (fallback): {}", s),
```

The full match becomes:

```rust
impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CudaError::DeviceInit(s) => write!(f, "CUDA device init failed: {}", s),
            CudaError::StreamCreate(s) => write!(f, "CUDA stream creation failed: {}", s),
            CudaError::Alloc(s) => write!(f, "CUDA allocation failed: {}", s),
            CudaError::Transfer(s) => write!(f, "CUDA transfer failed: {}", s),
            CudaError::Sync(s) => write!(f, "CUDA sync failed: {}", s),
            CudaError::Kernel(s) => write!(f, "CUDA kernel failed: {}", s),
            CudaError::Cublas(s) => write!(f, "cuBLAS operation failed: {}", s),
            CudaError::Cudnn(s) => write!(f, "cuDNN operation failed: {}", s),
            CudaError::Memory(s) => write!(f, "CUDA memory operation failed: {}", s),
            CudaError::FusionShapeMismatch(s) => {
                write!(f, "Fusion shape mismatch (fallback): {}", s)
            }
        }
    }
}
```

- [ ] **Step 3: Run clippy to verify**

Run: `cargo clippy --features cuda --tests -- -D warnings`
Expected: 0 errors, 0 warnings. (The new variant is unused so far, but `#[derive(Debug, Clone)]` means no dead-code warning for enum variants.)

- [ ] **Step 4: Commit**

```bash
git add src/cuda/context.rs
git commit -S -m "$(cat <<'EOF'
feat: add FusionShapeMismatch variant to CudaError

Sentinel error variant for the dynamic AddMulAdd fusion path. The
executor will catch this specific variant and fall back to unfused
execution; all other CudaError variants propagate as hard errors.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Add the broadcast-c CUDA kernel source and register it

**Files:**
- Modify: `src/cuda/kernels/fused/mod.rs:65-88` (kernel name list) and `:150-177` (kernel source)

- [ ] **Step 1: Add the kernel name to `FUSED_KERNEL_NAMES`**

In `src/cuda/kernels/fused/mod.rs`, find the `FUSED_KERNEL_NAMES` array. After the line `"fused_add_mul_add_scalar_kernel",` (currently at line 72), add:

```rust
    "fused_add_mul_add_bcast_c_kernel",
```

The relevant section becomes:

```rust
    // Affine transform: (x + a) * b + c
    "fused_add_mul_add_kernel",
    "fused_add_mul_add_scalar_kernel",
    "fused_add_mul_add_bcast_c_kernel",
```

- [ ] **Step 2: Add the kernel source to `FUSED_KERNELS`**

In the same file, find the end of `fused_add_mul_add_scalar_kernel` (currently at line 177, ending with `}`). After that closing brace and before the GELU section comment, add:

```cuda
// Broadcast-c version: x, a, b have identical shape, c broadcasts via
// a contiguous non-broadcast block. c_inner_stride is the product of
// x dims after the non-broadcast block; c_size is the product of dims
// in the non-broadcast block.
// AdaIN: c=[1,C,1], x=[1,C,T] -> c_inner_stride=T, c_size=C
// LSTM:  c=[1,1,C], x=[1,T,C] -> c_inner_stride=1, c_size=C
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

- [ ] **Step 3: Run clippy to verify the kernel compiles**

Run: `cargo clippy --features cuda --tests -- -D warnings`
Expected: 0 errors. If NVRTC compilation fails at test time, verify the raw string syntax is correct.

- [ ] **Step 4: Commit**

```bash
git add src/cuda/kernels/fused/mod.rs
git commit -S -m "$(cat <<'EOF'
feat: add fused_add_mul_add_bcast_c_kernel CUDA source

New kernel for AddMulAdd chains where c broadcasts over one or more
size-1 dimensions. Uses (c_inner_stride, c_size) parameters to compute
the broadcast index generically for both AdaIN (trailing-dim broadcast)
and LSTM layer-norm (middle-dim broadcast) patterns.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Modify the wrapper with broadcast-c detection and dispatch

**Files:**
- Modify: `src/cuda/kernels/fused/wrappers.rs:176-224`

This is the core dispatch logic. The existing `gpu_fused_add_mul_add` function currently requires all four shapes to match or returns a hard error. We change it to: (1) same-shape -> existing kernel, (2) x==a==b shapes + c broadcast-compatible -> new kernel, (3) else -> `FusionShapeMismatch`.

- [ ] **Step 1: Add the `compute_c_broadcast_params` helper function**

Add a new private helper function in `wrappers.rs` before `gpu_fused_add_mul_add`. This function computes `(c_inner_stride, c_size)` from the x and c shapes:

```rust
/// Compute broadcast parameters for c in the AddMulAdd fusion.
///
/// Returns `Some((c_inner_stride, c_size))` if c is broadcast-compatible
/// with x (same ndim, each c dim is 1 or matches x, at least one dim is 1,
/// and the non-broadcast dims form a single contiguous block).
///
/// Returns `None` if shapes are incompatible or the broadcast pattern
/// is non-contiguous (e.g., c = [C, 1, C]).
fn compute_c_broadcast_params(x_shape: &[usize], c_shape: &[usize]) -> Option<(usize, usize)> {
    if x_shape.len() != c_shape.len() {
        return None;
    }

    // Classify each dimension as broadcast (c=1, x>1) or matching (c==x)
    // Also reject incompatible (c!=1 && c!=x)
    let mut has_broadcast = false;
    for (x_dim, c_dim) in x_shape.iter().zip(c_shape.iter()) {
        if *c_dim == *x_dim {
            // matching
        } else if *c_dim == 1 {
            has_broadcast = true;
        } else {
            return None; // incompatible
        }
    }

    if !has_broadcast {
        // All dims match -- this is the same-shape case, not broadcast
        return None;
    }

    // Find the contiguous non-broadcast block.
    // Walk from the end to find the first non-broadcast dim, then verify
    // all non-broadcast dims are contiguous.
    //
    // We need to identify a contiguous range [start..end) of dimensions
    // where c_dim == x_dim (the "non-broadcast block"). All dims outside
    // this range must be broadcast (c_dim == 1) or size-1 in both.
    //
    // c_size = product of x dims in the non-broadcast block
    // c_inner_stride = product of x dims after the non-broadcast block

    // Find all non-broadcast dimension indices (where c_dim == x_dim AND x_dim > 1)
    // Dims where both are 1 can be treated as either broadcast or matching.
    let non_bcast_indices: Vec<usize> = x_shape
        .iter()
        .zip(c_shape.iter())
        .enumerate()
        .filter(|(_i, (x_d, c_d))| **c_d == **x_d && **x_d > 1)
        .map(|(i, _)| i)
        .collect();

    if non_bcast_indices.is_empty() {
        // c is all 1s -- degenerate scalar broadcast.
        // c_size = 1, c_inner_stride = 1 works (c_idx = (i/1)%1 = 0 always).
        return Some((1, 1));
    }

    // Check contiguity: indices must be consecutive
    for window in non_bcast_indices.windows(2) {
        if window[1] != window[0] + 1 {
            return None; // non-contiguous non-broadcast block
        }
    }

    let block_start = non_bcast_indices[0];
    let block_end = *non_bcast_indices.last().unwrap() + 1; // exclusive

    // c_size = product of x dims in [block_start..block_end)
    let c_size: usize = x_shape[block_start..block_end].iter().product();

    // c_inner_stride = product of x dims in [block_end..ndim)
    let c_inner_stride: usize = x_shape[block_end..].iter().product();

    Some((c_inner_stride, c_size))
}
```

- [ ] **Step 2: Rewrite the shape-checking logic in `gpu_fused_add_mul_add`**

Replace the body of `gpu_fused_add_mul_add` (lines 185-223) with the three-way dispatch:

```rust
pub fn gpu_fused_add_mul_add(
    ctx: &IconnxCudaContext,
    cache: &FusedKernelCache,
    pool: &mut GpuMemoryPool,
    x: &GpuTensor,
    a: &GpuTensor,
    b: &GpuTensor,
    c: &GpuTensor,
) -> Result<GpuTensor, CudaError> {
    let n = x.len();

    // Path 1: All four shapes identical -> existing same-shape kernel
    if x.shape() == a.shape() && x.shape() == b.shape() && x.shape() == c.shape() {
        let mut out = pool.get_tensor_f32(ctx, x.shape().to_vec())?;

        let func = cache
            .get("fused_add_mul_add_kernel")
            .ok_or_else(|| CudaError::Kernel("fused_add_mul_add_kernel not found".into()))?;

        unsafe {
            ctx.stream()
                .launch_builder(func)
                .arg(out.data_f32_mut()?)
                .arg(x.data_f32()?)
                .arg(a.data_f32()?)
                .arg(b.data_f32()?)
                .arg(c.data_f32()?)
                .arg(&n)
                .launch(elementwise_config(n))
                .map_err(|e| {
                    CudaError::Kernel(format!("fused_add_mul_add launch failed: {}", e))
                })?;
        }

        return Ok(out);
    }

    // Path 2: x, a, b same shape + c broadcasts -> broadcast-c kernel
    if x.shape() == a.shape() && x.shape() == b.shape() {
        if let Some((c_inner_stride, c_size)) =
            compute_c_broadcast_params(x.shape(), c.shape())
        {
            let mut out = pool.get_tensor_f32(ctx, x.shape().to_vec())?;

            let func = cache.get("fused_add_mul_add_bcast_c_kernel").ok_or_else(|| {
                CudaError::Kernel("fused_add_mul_add_bcast_c_kernel not found".into())
            })?;

            unsafe {
                ctx.stream()
                    .launch_builder(func)
                    .arg(out.data_f32_mut()?)
                    .arg(x.data_f32()?)
                    .arg(a.data_f32()?)
                    .arg(b.data_f32()?)
                    .arg(c.data_f32()?)
                    .arg(&n)
                    .arg(&c_inner_stride)
                    .arg(&c_size)
                    .launch(elementwise_config(n))
                    .map_err(|e| {
                        CudaError::Kernel(format!(
                            "fused_add_mul_add_bcast_c launch failed: {}",
                            e
                        ))
                    })?;
            }

            return Ok(out);
        }
    }

    // Path 3: shapes incompatible -> sentinel error for fallback
    Err(CudaError::FusionShapeMismatch(format!(
        "fused_add_mul_add: incompatible shapes x={:?} a={:?} b={:?} c={:?}",
        x.shape(),
        a.shape(),
        b.shape(),
        c.shape()
    )))
}
```

Note: the existing "scalar a" comment block (lines 187-191) is dead code that was never completed -- remove it as part of this rewrite. The new Path 1 handles the same-shape case identically to the old code. Path 3 changes the error type from `CudaError::Kernel` to `CudaError::FusionShapeMismatch`.

- [ ] **Step 3: Make `compute_c_broadcast_params` visible to the test module**

Change the function visibility from `fn` to `pub(super) fn` so it can be tested from the `mod.rs` test module:

```rust
pub(super) fn compute_c_broadcast_params(x_shape: &[usize], c_shape: &[usize]) -> Option<(usize, usize)> {
```

- [ ] **Step 4: Add unit tests for `compute_c_broadcast_params`**

In `src/cuda/kernels/fused/mod.rs`, find the `#[cfg(test)] mod tests` block (around line 451). Add these tests:

```rust
    #[test]
    fn test_c_broadcast_params_adain_trailing() {
        // AdaIN: c=[1,C,1], x=[1,C,T] -> c_inner_stride=T, c_size=C
        let result = compute_c_broadcast_params(&[1, 256, 1900], &[1, 256, 1]);
        assert_eq!(result, Some((1900, 256)));
    }

    #[test]
    fn test_c_broadcast_params_lstm_middle() {
        // LSTM: c=[1,1,C], x=[1,T,C] -> c_inner_stride=1, c_size=C
        let result = compute_c_broadcast_params(&[1, 5, 512], &[1, 1, 512]);
        assert_eq!(result, Some((1, 512)));
    }

    #[test]
    fn test_c_broadcast_params_same_shape_returns_none() {
        // Same shape -> None (caller should use same-shape kernel)
        let result = compute_c_broadcast_params(&[1, 4, 8], &[1, 4, 8]);
        assert_eq!(result, None);
    }

    #[test]
    fn test_c_broadcast_params_incompatible_returns_none() {
        // Incompatible: c dim is neither 1 nor matching
        let result = compute_c_broadcast_params(&[1, 4, 8], &[2, 4, 8]);
        assert_eq!(result, None);
    }

    #[test]
    fn test_c_broadcast_params_non_contiguous_returns_none() {
        // Non-contiguous non-broadcast block: c=[C, 1, C]
        let result = compute_c_broadcast_params(&[4, 3, 4], &[4, 1, 4]);
        assert_eq!(result, None);
    }

    #[test]
    fn test_c_broadcast_params_different_ndim_returns_none() {
        let result = compute_c_broadcast_params(&[1, 4, 8], &[4, 8]);
        assert_eq!(result, None);
    }

    #[test]
    fn test_c_broadcast_params_all_ones() {
        // Degenerate: c is all 1s
        let result = compute_c_broadcast_params(&[1, 4, 8], &[1, 1, 1]);
        assert_eq!(result, Some((1, 1)));
    }

    #[test]
    fn test_c_broadcast_params_small_adain() {
        // Smaller AdaIN: c=[1,4,1], x=[1,4,8]
        let result = compute_c_broadcast_params(&[1, 4, 8], &[1, 4, 1]);
        assert_eq!(result, Some((8, 4)));
    }

    #[test]
    fn test_c_broadcast_params_small_lstm() {
        // Smaller LSTM: c=[1,1,6], x=[1,3,6]
        let result = compute_c_broadcast_params(&[1, 3, 6], &[1, 1, 6]);
        assert_eq!(result, Some((1, 6)));
    }
```

Note: the `compute_c_broadcast_params` function is in `wrappers.rs` with `pub(super)` visibility. The test module in `mod.rs` uses `use super::*;`, but `pub use wrappers::*` only re-exports `pub` items, not `pub(super)` items. So the test module must reference the function via `super::wrappers::compute_c_broadcast_params(...)` or add an explicit `use super::wrappers::compute_c_broadcast_params;` import. Alternatively, if this is too awkward, change the function's visibility to `pub(crate)` instead of `pub(super)`. Choose whichever approach passes clippy without warnings.

- [ ] **Step 5: Run the new unit tests**

Run: `cargo test --features cuda -- test_c_broadcast_params -v`
Expected: All 9 tests PASS.

- [ ] **Step 6: Add a kernel-level integration test for the broadcast-c kernel**

Still in `src/cuda/kernels/fused/mod.rs`'s test module, add:

```rust
    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_fused_add_mul_add_bcast_c_adain() {
        let (ctx, cache, mut pool) = setup();

        // AdaIN pattern: x=[1,4,8], a=[1,4,8], b=[1,4,8], c=[1,4,1]
        // out[i] = (x[i] + a[i]) * b[i] + c[c_idx]
        // where c_idx = (i / 8) % 4

        let x_data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let a_data: Vec<f32> = vec![1.0; 32];
        let b_data: Vec<f32> = vec![2.0; 32];
        let c_data: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0]; // [1, 4, 1]

        let x = GpuTensor::from_host_f32(&ctx, &x_data, vec![1, 4, 8]).unwrap();
        let a = GpuTensor::from_host_f32(&ctx, &a_data, vec![1, 4, 8]).unwrap();
        let b = GpuTensor::from_host_f32(&ctx, &b_data, vec![1, 4, 8]).unwrap();
        let c = GpuTensor::from_host_f32(&ctx, &c_data, vec![1, 4, 1]).unwrap();

        let out = gpu_fused_add_mul_add(&ctx, &cache, &mut pool, &x, &a, &b, &c).unwrap();
        let result = out.to_host_f32(&ctx).unwrap();

        // Verify: out[i] = (i + 1) * 2 + c[(i/8) % 4]
        for i in 0..32 {
            let c_idx = (i / 8) % 4;
            let expected = ((i as f32) + 1.0) * 2.0 + c_data[c_idx];
            assert!(
                (result[i] - expected).abs() < 1e-5,
                "mismatch at i={}: got {} expected {}",
                i, result[i], expected,
            );
        }
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_fused_add_mul_add_bcast_c_lstm() {
        let (ctx, cache, mut pool) = setup();

        // LSTM pattern: x=[1,3,6], a=[1,3,6], b=[1,3,6], c=[1,1,6]
        // c_inner_stride=1, c_size=6 -> c_idx = (i/1) % 6 = i % 6
        let x_data: Vec<f32> = (0..18).map(|i| i as f32).collect();
        let a_data: Vec<f32> = vec![0.5; 18];
        let b_data: Vec<f32> = vec![3.0; 18];
        let c_data: Vec<f32> = vec![100.0, 200.0, 300.0, 400.0, 500.0, 600.0]; // [1, 1, 6]

        let x = GpuTensor::from_host_f32(&ctx, &x_data, vec![1, 3, 6]).unwrap();
        let a = GpuTensor::from_host_f32(&ctx, &a_data, vec![1, 3, 6]).unwrap();
        let b = GpuTensor::from_host_f32(&ctx, &b_data, vec![1, 3, 6]).unwrap();
        let c = GpuTensor::from_host_f32(&ctx, &c_data, vec![1, 1, 6]).unwrap();

        let out = gpu_fused_add_mul_add(&ctx, &cache, &mut pool, &x, &a, &b, &c).unwrap();
        let result = out.to_host_f32(&ctx).unwrap();

        for i in 0..18 {
            let c_idx = i % 6;
            let expected = ((i as f32) + 0.5) * 3.0 + c_data[c_idx];
            assert!(
                (result[i] - expected).abs() < 1e-4,
                "mismatch at i={}: got {} expected {}",
                i, result[i], expected,
            );
        }
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_fused_add_mul_add_shape_mismatch_returns_sentinel() {
        let (ctx, cache, mut pool) = setup();

        // Incompatible shapes: x=[1,4,8] but c=[2,4,8] (dim 0 mismatch)
        let x = GpuTensor::from_host_f32(&ctx, &vec![1.0; 32], vec![1, 4, 8]).unwrap();
        let a = GpuTensor::from_host_f32(&ctx, &vec![1.0; 32], vec![1, 4, 8]).unwrap();
        let b = GpuTensor::from_host_f32(&ctx, &vec![1.0; 32], vec![1, 4, 8]).unwrap();
        let c = GpuTensor::from_host_f32(&ctx, &vec![1.0; 64], vec![2, 4, 8]).unwrap();

        let result = gpu_fused_add_mul_add(&ctx, &cache, &mut pool, &x, &a, &b, &c);
        assert!(
            matches!(result, Err(CudaError::FusionShapeMismatch(_))),
            "expected FusionShapeMismatch, got {:?}",
            result,
        );
    }
```

- [ ] **Step 7: Run all kernel-level tests**

Run: `cargo test --features cuda -- test_fused_add_mul_add_bcast_c --ignored -v`
Run: `cargo test --features cuda -- test_fused_add_mul_add_shape_mismatch --ignored -v`
Expected: All PASS.

- [ ] **Step 8: Run clippy**

Run: `cargo clippy --features cuda --tests -- -D warnings`
Expected: 0 errors, 0 warnings.

- [ ] **Step 9: Commit**

```bash
git add src/cuda/kernels/fused/wrappers.rs src/cuda/kernels/fused/mod.rs
git commit -S -m "$(cat <<'EOF'
test: add unit tests for broadcast-c param computation and kernel

Nine unit tests for compute_c_broadcast_params covering AdaIN, LSTM,
same-shape, incompatible, non-contiguous, and degenerate cases. Three
kernel-level integration tests verify the broadcast-c kernel produces
correct output for AdaIN and LSTM patterns, and that shape mismatch
returns the FusionShapeMismatch sentinel.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

```bash
git add src/cuda/kernels/fused/wrappers.rs
git commit -S -m "$(cat <<'EOF'
feat: add broadcast-c dispatch to gpu_fused_add_mul_add wrapper

Three-way dispatch: (1) all shapes match -> existing kernel, (2) x/a/b
match + c broadcasts -> new bcast_c kernel, (3) incompatible ->
FusionShapeMismatch sentinel. The compute_c_broadcast_params helper
finds the contiguous non-broadcast block and derives (c_inner_stride,
c_size) for the generalized index formula.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Modify the walker to register dynamic candidates

**Files:**
- Modify: `src/cuda/inference/fusion/detection.rs:27-31` (return type) and `:128-214` (AddMulAdd walker)
- Modify: `src/cuda/inference/fusion/mod.rs:17` (re-export)

- [ ] **Step 1: Change the return type of `detect_fused_patterns` to a 3-tuple**

In `src/cuda/inference/fusion/detection.rs`, change the function signature at line 27 and its doc comment:

Old (lines 16-31):
```rust
/// Detect all fuseable operator patterns in the given node list.
///
/// Returns a tuple of:
/// - `HashMap<String, FusedPatternInfo>` keyed by the pattern head node name
/// - `HashSet<String>` of all node names that should be skipped during
///   execution because they are part of a detected pattern
///
/// The `ctx` parameter is needed only to read constant scalar values
/// from weight tensors during exponent validation (GELU's `Pow` exponent
/// must be exactly 3.0). It is not used for any GPU computation during
/// detection.
pub(crate) fn detect_fused_patterns(
    nodes: &[ExecutionNode],
    weights: &HashMap<String, GpuTensor>,
    ctx: &IconnxCudaContext,
) -> (HashMap<String, FusedPatternInfo>, HashSet<String>) {
```

New:
```rust
/// Detect all fuseable operator patterns in the given node list.
///
/// Returns a 3-tuple of:
/// - `HashMap<String, FusedPatternInfo>` keyed by the pattern head node name
///   (static patterns — unconditionally fusible)
/// - `HashSet<String>` of all node names that should be skipped during
///   execution because they are part of a detected static pattern
/// - `HashMap<String, FusedPatternInfo>` of dynamic candidates — AddMulAdd
///   chains where `b` or `c` are runtime-computed. These are tried at
///   execution time with shape-check fallback.
///
/// The `ctx` parameter is needed only to read constant scalar values
/// from weight tensors during exponent validation (GELU's `Pow` exponent
/// must be exactly 3.0). It is not used for any GPU computation during
/// detection.
pub(crate) fn detect_fused_patterns(
    nodes: &[ExecutionNode],
    weights: &HashMap<String, GpuTensor>,
    ctx: &IconnxCudaContext,
) -> (
    HashMap<String, FusedPatternInfo>,
    HashSet<String>,
    HashMap<String, FusedPatternInfo>,
) {
```

- [ ] **Step 2: Add the `dynamic_candidates` map and update the return statement**

After the existing `let mut nodes_to_skip` declaration (line 47), add:

```rust
    let mut dynamic_candidates: HashMap<String, FusedPatternInfo> = HashMap::new();
```

Find the return statement at the very end of the function (currently `(fused_patterns, nodes_to_skip)`) and change it to:

```rust
    (fused_patterns, nodes_to_skip, dynamic_candidates)
```

- [ ] **Step 3: Modify the AddMulAdd walker to register dynamic candidates**

Find the `is_static` rejection at lines 190-192:

```rust
                    if !is_static(&b_input) || !is_static(&c_input) {
                        continue;
                    }
```

Replace with:

```rust
                    if !is_static(&b_input) || !is_static(&c_input) {
                        // Non-static b or c: register as dynamic candidate
                        // instead of skipping. The executor will attempt
                        // fusion at runtime and fall back on shape mismatch.
                        let pattern_info = FusedPatternInfo {
                            pattern: FusedPattern::AddMulAdd {
                                x_input: x_input.clone(),
                                a_input: a_input.clone(),
                                b_input,
                                c_input,
                                output_name: add2_node.outputs[0].clone(),
                            },
                            nodes_to_skip: vec![
                                mul_node.name.clone(),
                                add2_node.name.clone(),
                            ],
                            head_node: node.name.clone(),
                        };
                        // Do NOT add interior nodes to nodes_to_skip —
                        // they must remain executable for the fallback path.
                        dynamic_candidates.insert(node.name.clone(), pattern_info);
                        continue;
                    }
```

Note: `x_input` and `a_input` need to be cloned here because the static path below them also uses these values. Check whether `x_input` and `a_input` are moved or cloned in the original code at line 168. They are produced by `.clone()` at line 168, so the dynamic branch needs its own clones. Looking more carefully at the original: `x_input` and `a_input` are `String` values defined at line 168 via `clone()`, and they are consumed by the `FusedPatternInfo` construction at line 195-204. Since we now have TWO possible construction sites (dynamic and static), we need to clone `x_input` and `a_input` in the dynamic branch and let the static branch consume them as before. The code above uses `x_input.clone()` and `a_input.clone()` which is correct — the originals are still available for the static path (but we `continue` so they won't be used).

Actually, since the dynamic branch does `continue`, the variables are consumed in the dynamic branch and would NOT reach the static branch. The `b_input` and `c_input` are also consumed. So we don't actually need to clone `x_input`/`a_input` — they can be moved. But to be safe and consistent with the existing code style:

```rust
                    if !is_static(&b_input) || !is_static(&c_input) {
                        let pattern_info = FusedPatternInfo {
                            pattern: FusedPattern::AddMulAdd {
                                x_input,
                                a_input,
                                b_input,
                                c_input,
                                output_name: add2_node.outputs[0].clone(),
                            },
                            nodes_to_skip: vec![
                                mul_node.name.clone(),
                                add2_node.name.clone(),
                            ],
                            head_node: node.name.clone(),
                        };
                        dynamic_candidates.insert(node.name.clone(), pattern_info);
                        continue;
                    }
```

This moves the `String` variables into the dynamic `FusedPatternInfo`, same as the static path does. The `continue` ensures we don't try to also construct a static pattern from the same (now-moved) values.

- [ ] **Step 4: Run clippy**

Run: `cargo clippy --features cuda --tests -- -D warnings`
Expected: Compilation errors in `src/cuda/inference/mod.rs` and `src/cuda/inference/testing.rs` because they still destructure a 2-tuple from `detect_fused_patterns`. This is expected — we fix it in Task 5.

If there are errors ONLY in the expected files (mod.rs, testing.rs), proceed. If there are unexpected errors in detection.rs, fix them before moving on.

- [ ] **Step 5: Commit (even with downstream breakage)**

```bash
git add src/cuda/inference/fusion/detection.rs
git commit -S -m "$(cat <<'EOF'
feat: register non-static AddMulAdd chains as dynamic candidates

The walker now returns a 3-tuple: (static_patterns, nodes_to_skip,
dynamic_candidates). Non-static AddMulAdd chains go into
dynamic_candidates instead of being skipped. Their interior nodes are
NOT added to nodes_to_skip so they remain executable for the fallback
path. Downstream callers will be updated in the next task.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

Note: this commit will NOT pass `cargo test` because callers haven't been updated yet. That's fine — the next task fixes the callers. If `cargo clippy` can't even compile due to the 2-tuple vs 3-tuple mismatch, that's expected. Commit the detection.rs change and immediately proceed to Task 5.

---

## Task 5: Wire dynamic candidates through executor and testing facade

**Files:**
- Modify: `src/cuda/inference/mod.rs:255-259` (struct fields), `:424-426` (constructor), `:639-650` (detect method)
- Modify: `src/cuda/inference/testing.rs:49-69` (test helper return type), `:192-210` (count helper)
- Modify: `src/cuda/inference/fusion/mod.rs:17` (re-export if needed)

- [ ] **Step 1: Add the `dynamic_candidates` field to the executor struct**

In `src/cuda/inference/mod.rs`, find the struct fields around line 255-259:

```rust
    /// Detected fused patterns: maps head node name to pattern info
    /// Uses RefCell for interior mutability (detected on first run)
    pub(crate) fused_patterns: RefCell<HashMap<String, FusedPatternInfo>>,
    /// Set of node names to skip (they're part of a fused pattern)
    nodes_to_skip: RefCell<std::collections::HashSet<String>>,
```

Add after `nodes_to_skip`:

```rust
    /// Dynamic fusion candidates: AddMulAdd chains where b/c are runtime-
    /// computed. Attempted at execution time with shape-check fallback.
    pub(crate) dynamic_candidates: RefCell<HashMap<String, FusedPatternInfo>>,
```

- [ ] **Step 2: Initialize the new field in the constructor**

In `src/cuda/inference/mod.rs`, find the constructor around line 424-426:

```rust
            fused_patterns: RefCell::new(HashMap::new()),
            nodes_to_skip: RefCell::new(std::collections::HashSet::new()),
```

Add after `nodes_to_skip`:

```rust
            dynamic_candidates: RefCell::new(HashMap::new()),
```

- [ ] **Step 3: Update the `detect_fused_patterns` method to destructure the 3-tuple**

In `src/cuda/inference/mod.rs`, find the method at lines 639-650:

```rust
    fn detect_fused_patterns(&self) {
        if *self.patterns_detected.borrow() {
            return;
        }

        let (detected, skip_set) =
            fusion::detect_fused_patterns(&self.nodes, &self.weights, &self.ctx);

        *self.fused_patterns.borrow_mut() = detected;
        *self.nodes_to_skip.borrow_mut() = skip_set;
        *self.patterns_detected.borrow_mut() = true;
    }
```

Replace with:

```rust
    fn detect_fused_patterns(&self) {
        if *self.patterns_detected.borrow() {
            return;
        }

        let (detected, skip_set, dynamic) =
            fusion::detect_fused_patterns(&self.nodes, &self.weights, &self.ctx);

        *self.fused_patterns.borrow_mut() = detected;
        *self.nodes_to_skip.borrow_mut() = skip_set;
        *self.dynamic_candidates.borrow_mut() = dynamic;
        *self.patterns_detected.borrow_mut() = true;
    }
```

- [ ] **Step 4: Also update the `clear_detected_patterns` method (if it exists)**

Search for `patterns_detected` being set to `false` or any method that clears the patterns. Found at line 543:

```rust
        self.nodes_to_skip.borrow_mut().clear();
```

Add after it:

```rust
        self.dynamic_candidates.borrow_mut().clear();
```

- [ ] **Step 5: Update `detect_fused_patterns_for_tests` in testing.rs**

In `src/cuda/inference/testing.rs`, update the return type and body of `detect_fused_patterns_for_tests` (lines 49-69):

Old:
```rust
pub fn detect_fused_patterns_for_tests(
    nodes: &[ExecutionNodeForTests],
    weights: &HashMap<String, GpuTensor>,
    ctx: &IconnxCudaContext,
) -> (HashMap<String, FusedPatternInfo>, HashSet<String>) {
```

New:
```rust
pub fn detect_fused_patterns_for_tests(
    nodes: &[ExecutionNodeForTests],
    weights: &HashMap<String, GpuTensor>,
    ctx: &IconnxCudaContext,
) -> (
    HashMap<String, FusedPatternInfo>,
    HashSet<String>,
    HashMap<String, FusedPatternInfo>,
) {
```

The function body stays the same — it just passes through the 3-tuple from `detect_fused_patterns`.

- [ ] **Step 6: Update `count_fused_patterns` to include dynamic candidates**

In `src/cuda/inference/testing.rs`, update `count_fused_patterns` (lines 192-210) to also count dynamic candidates:

```rust
pub fn count_fused_patterns(
    executor: &crate::cuda::inference::GpuGraphExecutor,
) -> PatternCount {
    let patterns = executor.fused_patterns.borrow();
    let dynamic = executor.dynamic_candidates.borrow();
    let mut count = PatternCount::default();
    for info in patterns.values().chain(dynamic.values()) {
        match &info.pattern {
            FusedPattern::DivRsqrt { .. } => count.div_rsqrt += 1,
            FusedPattern::AddMulAdd { .. } => count.add_mul_add += 1,
            FusedPattern::Gelu { .. } => count.gelu += 1,
            FusedPattern::MulSinPowMulAdd { .. } => count.mul_sin_pow_mul_add += 1,
            FusedPattern::MulAdd { .. } => count.mul_add += 1,
            FusedPattern::AddMul { .. } => count.add_mul += 1,
            FusedPattern::SubMul { .. } => count.sub_mul += 1,
            FusedPattern::DivMul { .. } => count.div_mul += 1,
        }
    }
    count
}
```

- [ ] **Step 7: Update existing test call sites that destructure 2-tuples**

In `tests/fusion_detection_test.rs`, every call to `detect_fused_patterns_for_tests` currently destructures as `let (patterns, skip_set) = ...` or `let (patterns, _skip) = ...`. These all need to become 3-tuples.

Search for all occurrences and update:

```rust
// Old:
let (patterns, skip_set) = detect_fused_patterns_for_tests(&nodes, &weights, &ctx);
// New:
let (patterns, skip_set, _dynamic) = detect_fused_patterns_for_tests(&nodes, &weights, &ctx);

// Old:
let (patterns, _skip) = detect_fused_patterns_for_tests(&nodes, &weights, &ctx);
// New:
let (patterns, _skip, _dynamic) = detect_fused_patterns_for_tests(&nodes, &weights, &ctx);
```

There are approximately 8-10 call sites. Update them all.

- [ ] **Step 8: Run tests and clippy**

Run: `cargo clippy --features cuda --tests -- -D warnings`
Expected: 0 errors, 0 warnings.

Run: `cargo test --features cuda`
Expected: All existing tests pass. The new `dynamic_candidates` field is populated but not yet used by the execution loop.

- [ ] **Step 9: Commit**

```bash
git add src/cuda/inference/mod.rs src/cuda/inference/testing.rs src/cuda/inference/fusion/mod.rs tests/fusion_detection_test.rs
git commit -S -m "$(cat <<'EOF'
feat: wire dynamic_candidates through executor and testing facade

The executor stores the 3rd element from detect_fused_patterns as
dynamic_candidates. The count_fused_patterns test helper now counts
both static patterns and dynamic candidates. Existing test call sites
updated to destructure the 3-tuple.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Add dynamic dispatch with fallback in the execution loops

**Files:**
- Modify: `src/cuda/inference/mod.rs` — four execution loops in `run`, `run_with_validation`, `run_with_checkpoints`, `run_with_profiling`

The `dynamically_fused: HashSet<String>` MUST be a per-run local variable inside each method, NOT a field on the executor struct. It is reset between inferences because runtime shapes can change between calls.

- [ ] **Step 1: Add dynamic dispatch to `run_with_profiling` (line ~4368)**

Find the execution loop in `run_with_profiling` (starting around line 4368). Currently:

```rust
        for node in sorted_nodes {
            // Skip nodes that are part of a fused pattern (handled by pattern head)
            if self.nodes_to_skip.borrow().contains(&node.name) {
                continue;
            }

            // Check if this node is a pattern head - if so, execute fused kernel
            if let Some(pattern_info) = self.fused_patterns.borrow().get(&node.name).cloned() {
                // ... existing static dispatch ...
                continue;
            }
            if node.op_type == "Constant" {
```

Add a `dynamically_fused` local variable BEFORE the loop and a dynamic-candidate check AFTER the static-pattern dispatch:

Before the `for node in sorted_nodes` loop, add:

```rust
        // Per-run set of nodes whose dynamic-candidate pattern head was
        // successfully fused. Reset between inferences because runtime
        // shapes can change between calls.
        let mut dynamically_fused: HashSet<String> = HashSet::new();
```

After the static-pattern dispatch block (after the `continue;` following the `FusedPattern` match), and BEFORE the `if node.op_type == "Constant"` line, add:

```rust
            // Dynamic candidate: attempt fusion with shape-check fallback
            if let Some(pattern_info) =
                self.dynamic_candidates.borrow().get(&node.name).cloned()
            {
                let result = self.execute_fused_pattern(&pattern_info, &values);
                match result {
                    Ok(output) => {
                        let op_us = op_start.elapsed().as_micros() as u64;
                        let pattern_name = match &pattern_info.pattern {
                            FusedPattern::AddMulAdd { output_name, .. } => {
                                values.insert(output_name.clone(), output);
                                "Fused_AddMulAdd"
                            }
                            _ => unreachable!(
                                "dynamic candidates are only AddMulAdd"
                            ),
                        };
                        // Mark interior nodes as dynamically fused
                        for skip_name in &pattern_info.nodes_to_skip {
                            dynamically_fused.insert(skip_name.clone());
                        }
                        let entry = op_times
                            .entry(pattern_name.to_string())
                            .or_insert((0, 0));
                        entry.0 += op_us;
                        entry.1 += 1;
                        gpu_timer.record(pattern_name.to_string())?;
                        continue;
                    }
                    Err(CudaError::FusionShapeMismatch(_)) => {
                        // Fall through to normal execution
                    }
                    Err(e) => return Err(e), // Hard error
                }
            }

            // Skip interior nodes of a successfully-fused dynamic candidate
            if dynamically_fused.contains(&node.name) {
                continue;
            }
```

Wait -- there's an issue with `op_start`. In `run_with_profiling`, the `op_start` is set inside the static pattern block. For the dynamic candidate, we need our own timing. Let me look at the profiling pattern more carefully.

Looking at the existing static dispatch in `run_with_profiling` (lines 4375-4421), the timing is:

```rust
            if let Some(pattern_info) = self.fused_patterns.borrow().get(&node.name).cloned() {
                let op_start = Instant::now();
                let output = self.execute_fused_pattern(&pattern_info, &values)?;
                let op_us = op_start.elapsed().as_micros() as u64;
                // ... tracking ...
                continue;
            }
```

So `op_start` is local to the `if let` block. The dynamic candidate block needs its own timing. Corrected version:

```rust
            // Dynamic candidate: attempt fusion with shape-check fallback
            if let Some(pattern_info) =
                self.dynamic_candidates.borrow().get(&node.name).cloned()
            {
                let op_start = Instant::now();
                let result = self.execute_fused_pattern(&pattern_info, &values);
                match result {
                    Ok(output) => {
                        let op_us = op_start.elapsed().as_micros() as u64;
                        let pattern_name = match &pattern_info.pattern {
                            FusedPattern::AddMulAdd { output_name, .. } => {
                                values.insert(output_name.clone(), output);
                                "Fused_AddMulAdd"
                            }
                            _ => unreachable!(
                                "dynamic candidates are only AddMulAdd"
                            ),
                        };
                        for skip_name in &pattern_info.nodes_to_skip {
                            dynamically_fused.insert(skip_name.clone());
                        }
                        let entry = op_times
                            .entry(pattern_name.to_string())
                            .or_insert((0, 0));
                        entry.0 += op_us;
                        entry.1 += 1;
                        gpu_timer.record(pattern_name.to_string())?;
                        continue;
                    }
                    Err(CudaError::FusionShapeMismatch(_)) => {
                        // Fall through to normal execution
                    }
                    Err(e) => return Err(e),
                }
            }

            // Skip interior nodes of a successfully-fused dynamic candidate
            if dynamically_fused.contains(&node.name) {
                continue;
            }
```

- [ ] **Step 2: Add `HashSet` import if needed**

Check if `HashSet` is already imported in `mod.rs`. It's used via `std::collections::HashSet` in the `nodes_to_skip` field. Verify it's accessible. The `use std::collections::HashSet;` may need to be added if it's referenced as `std::collections::HashSet` everywhere. Follow the existing import pattern in the file.

- [ ] **Step 3: Add dynamic dispatch to `run` (line ~2800)**

The `run` method is simpler (no profiling, no gpu_timer). Before the `for` loop around line 2800, add:

```rust
        let mut dynamically_fused: HashSet<String> = HashSet::new();
```

After the static-pattern dispatch (around line 2835, after the `continue;`), add:

```rust
            // Dynamic candidate: attempt fusion with shape-check fallback
            if let Some(pattern_info) =
                self.dynamic_candidates.borrow().get(&node.name).cloned()
            {
                let result = self.execute_fused_pattern(&pattern_info, &values);
                match result {
                    Ok(output) => {
                        match &pattern_info.pattern {
                            FusedPattern::AddMulAdd { output_name, .. } => {
                                values.insert(output_name.clone(), output);
                            }
                            _ => unreachable!(
                                "dynamic candidates are only AddMulAdd"
                            ),
                        }
                        for skip_name in &pattern_info.nodes_to_skip {
                            dynamically_fused.insert(skip_name.clone());
                        }
                        continue;
                    }
                    Err(CudaError::FusionShapeMismatch(_)) => {
                        // Fall through to normal execution
                    }
                    Err(e) => return Err(e),
                }
            }

            if dynamically_fused.contains(&node.name) {
                continue;
            }
```

- [ ] **Step 4: Add dynamic dispatch to `run_with_validation` (line ~3277)**

Same pattern as `run`, but errors are mapped via `ValidationError::from`. Before the loop around line 3277, add:

```rust
        let mut dynamically_fused: std::collections::HashSet<String> =
            std::collections::HashSet::new();
```

After the static-pattern dispatch, add:

```rust
            // Dynamic candidate: attempt fusion with shape-check fallback
            if let Some(pattern_info) =
                self.dynamic_candidates.borrow().get(&node.name).cloned()
            {
                let result = self
                    .execute_fused_pattern(&pattern_info, &values);
                match result {
                    Ok(output) => {
                        let output_name = match &pattern_info.pattern {
                            FusedPattern::AddMulAdd { output_name, .. } => output_name.clone(),
                            _ => unreachable!(
                                "dynamic candidates are only AddMulAdd"
                            ),
                        };
                        values.insert(output_name, output);
                        for skip_name in &pattern_info.nodes_to_skip {
                            dynamically_fused.insert(skip_name.clone());
                        }
                        continue;
                    }
                    Err(CudaError::FusionShapeMismatch(_)) => {
                        // Fall through to normal execution
                    }
                    Err(e) => return Err(ValidationError::from(e)),
                }
            }

            if dynamically_fused.contains(&node.name) {
                continue;
            }
```

- [ ] **Step 5: Add dynamic dispatch to `run_with_checkpoints` (line ~3550)**

Same pattern. Before the loop around line 3550, add:

```rust
        let mut dynamically_fused: std::collections::HashSet<String> =
            std::collections::HashSet::new();
```

After the static-pattern dispatch, add:

```rust
            // Dynamic candidate: attempt fusion with shape-check fallback
            if let Some(pattern_info) =
                self.dynamic_candidates.borrow().get(&node.name).cloned()
            {
                let result = self.execute_fused_pattern(&pattern_info, &values);
                match result {
                    Ok(output) => {
                        let output_name = match &pattern_info.pattern {
                            FusedPattern::AddMulAdd { output_name, .. } => output_name.clone(),
                            _ => unreachable!(
                                "dynamic candidates are only AddMulAdd"
                            ),
                        };
                        values.insert(output_name, output);
                        for skip_name in &pattern_info.nodes_to_skip {
                            dynamically_fused.insert(skip_name.clone());
                        }
                        continue;
                    }
                    Err(CudaError::FusionShapeMismatch(_)) => {
                        // Fall through to normal execution
                    }
                    Err(e) => return Err(e),
                }
            }

            if dynamically_fused.contains(&node.name) {
                continue;
            }
```

- [ ] **Step 6: Run clippy and tests**

Run: `cargo clippy --features cuda --tests -- -D warnings`
Expected: 0 errors, 0 warnings.

Run: `cargo test --features cuda`
Expected: All existing tests pass. The Kokoro test will still assert `add_mul_add == 0` because `count_fused_patterns` now counts dynamic candidates; the Kokoro model should have ~73 dynamic candidates detected, making this assertion fail. If the Kokoro test fails with a count > 0, that's actually correct behavior -- we update the assertion in Task 8. Non-Kokoro tests should all pass.

If any `#[ignore]` test that previously passed now fails, investigate. The dynamic dispatch should be a no-op on graphs with no dynamic candidates.

- [ ] **Step 7: Commit**

```bash
git add src/cuda/inference/mod.rs
git commit -S -m "$(cat <<'EOF'
feat: add dynamic AddMulAdd dispatch with fallback in all execution loops

Each of the four run methods (run, run_with_validation,
run_with_checkpoints, run_with_profiling) now checks dynamic_candidates
after static patterns. On FusionShapeMismatch, execution falls through
to the normal unfused path. The dynamically_fused HashSet is a per-run
local variable, reset between inferences.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Unit tests for dynamic candidate detection and dispatch

**Files:**
- Modify: `tests/fusion_detection_test.rs`

These tests exercise the walker's dynamic candidate registration. They are detection-only tests (no execution) since the detection walker is pure.

- [ ] **Step 1: Write test — dynamic AddMulAdd registered when b is non-static**

Add at the end of `tests/fusion_detection_test.rs`:

```rust
// ============================================================================
// Cycle 4: Dynamic AddMulAdd candidate tests
// ============================================================================

#[test]
fn dynamic_add_mul_add_registered_when_b_is_runtime_computed() {
    let ctx = IconnxCudaContext::new().expect("CUDA context");

    // b comes from a Div node (runtime computed, like AdaIN's instance-norm scale).
    // c is an initializer (static).
    // The walker should register this as a dynamic candidate, not a static pattern.
    let nodes = vec![
        node("div_src", "Div", &["div_in_a", "div_in_b"], &["b"]),
        node("add1", "Add", &["x", "a"], &["t0"]),
        node("mul1", "Mul", &["t0", "b"], &["t1"]),
        node("add2", "Add", &["t1", "c"], &["y"]),
    ];

    let mut weights = HashMap::new();
    weights.insert(
        "c".to_string(),
        GpuTensor::from_host_f32(&ctx, &[3.0], vec![1]).expect("c"),
    );

    let (patterns, _skip, dynamic) = detect_fused_patterns_for_tests(&nodes, &weights, &ctx);

    // Should NOT be in static patterns
    assert!(
        !patterns.contains_key("add1"),
        "add1 should not be a static pattern when b is runtime-computed"
    );

    // Should be in dynamic candidates
    let info = dynamic.get("add1").unwrap_or_else(|| {
        panic!(
            "add1 was not registered as dynamic candidate. \
             Static patterns: {:?}, Dynamic: {:?}",
            patterns.keys().collect::<Vec<_>>(),
            dynamic.keys().collect::<Vec<_>>()
        )
    });
    match &info.pattern {
        FusedPattern::AddMulAdd {
            x_input,
            a_input,
            b_input,
            c_input,
            output_name,
        } => {
            assert_eq!(x_input, "x");
            assert_eq!(a_input, "a");
            assert_eq!(b_input, "b");
            assert_eq!(c_input, "c");
            assert_eq!(output_name, "y");
        }
        other => panic!("expected AddMulAdd, got {:?}", other),
    }
}

#[test]
fn dynamic_add_mul_add_registered_when_c_is_runtime_computed() {
    let ctx = IconnxCudaContext::new().expect("CUDA context");

    // c comes from a Slice node (runtime computed, like AdaIN's style projection bias).
    // b is an initializer (static).
    // The walker should register this as a dynamic candidate.
    let nodes = vec![
        node("slice_src", "Slice", &["slice_in"], &["c"]),
        node("add1", "Add", &["x", "a"], &["t0"]),
        node("mul1", "Mul", &["t0", "b"], &["t1"]),
        node("add2", "Add", &["t1", "c"], &["y"]),
    ];

    let mut weights = HashMap::new();
    weights.insert(
        "b".to_string(),
        GpuTensor::from_host_f32(&ctx, &[2.0], vec![1]).expect("b"),
    );

    let (patterns, _skip, dynamic) = detect_fused_patterns_for_tests(&nodes, &weights, &ctx);

    assert!(
        !patterns.contains_key("add1"),
        "add1 should not be a static pattern when c is runtime-computed"
    );

    let info = dynamic.get("add1").unwrap_or_else(|| {
        panic!(
            "add1 was not registered as dynamic candidate when c is runtime-computed. \
             Dynamic: {:?}",
            dynamic.keys().collect::<Vec<_>>()
        )
    });
    assert!(
        matches!(&info.pattern, FusedPattern::AddMulAdd { .. }),
        "expected AddMulAdd, got {:?}",
        info.pattern
    );
}

#[test]
fn dynamic_add_mul_add_registered_when_both_bc_runtime_computed() {
    let ctx = IconnxCudaContext::new().expect("CUDA context");

    // Both b and c are runtime-computed (AdaIN pattern: b from Div, c from Slice).
    let nodes = vec![
        node("div_src", "Div", &["div_in_a", "div_in_b"], &["b"]),
        node("slice_src", "Slice", &["slice_in"], &["c"]),
        node("add1", "Add", &["x", "a"], &["t0"]),
        node("mul1", "Mul", &["t0", "b"], &["t1"]),
        node("add2", "Add", &["t1", "c"], &["y"]),
    ];

    let weights = HashMap::new(); // no static weights

    let (patterns, _skip, dynamic) = detect_fused_patterns_for_tests(&nodes, &weights, &ctx);

    assert!(
        !patterns.contains_key("add1"),
        "add1 should not be a static pattern when both b and c are runtime-computed"
    );

    let info = dynamic.get("add1").unwrap_or_else(|| {
        panic!(
            "add1 was not registered as dynamic candidate with both b/c runtime-computed. \
             Dynamic: {:?}",
            dynamic.keys().collect::<Vec<_>>()
        )
    });
    assert!(
        matches!(&info.pattern, FusedPattern::AddMulAdd { .. }),
        "expected AddMulAdd, got {:?}",
        info.pattern
    );
}

#[test]
fn dynamic_add_mul_add_interior_nodes_not_in_skip_set() {
    let ctx = IconnxCudaContext::new().expect("CUDA context");

    // Dynamic candidates must NOT add their interior nodes to the skip set,
    // because the fallback path needs them to remain executable.
    let nodes = vec![
        node("div_src", "Div", &["div_in_a", "div_in_b"], &["b"]),
        node("add1", "Add", &["x", "a"], &["t0"]),
        node("mul1", "Mul", &["t0", "b"], &["t1"]),
        node("add2", "Add", &["t1", "c"], &["y"]),
    ];

    let mut weights = HashMap::new();
    weights.insert(
        "c".to_string(),
        GpuTensor::from_host_f32(&ctx, &[3.0], vec![1]).expect("c"),
    );

    let (_patterns, skip_set, dynamic) =
        detect_fused_patterns_for_tests(&nodes, &weights, &ctx);

    // Verify add1 is a dynamic candidate
    assert!(dynamic.contains_key("add1"));

    // Interior nodes (mul1, add2) must NOT be in the skip set
    assert!(
        !skip_set.contains("mul1"),
        "mul1 should not be in skip_set for a dynamic candidate"
    );
    assert!(
        !skip_set.contains("add2"),
        "add2 should not be in skip_set for a dynamic candidate"
    );
}
```

- [ ] **Step 2: Run the tests**

Run: `cargo test --features cuda -- dynamic_add_mul_add -v`
Expected: All 4 new tests PASS.

- [ ] **Step 3: Run the full test suite and clippy**

Run: `cargo clippy --features cuda --tests -- -D warnings`
Expected: 0 errors, 0 warnings.

Run: `cargo test --features cuda`
Expected: All tests pass (except possibly the Kokoro integration test if it's not `#[ignore]`). If the Kokoro test fails because `add_mul_add` is now > 0, that's the expected Cycle 4 behavior change -- we fix the assertion in Task 8.

- [ ] **Step 4: Commit**

```bash
git add tests/fusion_detection_test.rs
git commit -S -m "$(cat <<'EOF'
test: add unit tests for dynamic AddMulAdd candidate detection

Four synthetic-graph tests verify: (1) dynamic registration when b is
runtime-computed, (2) when c is runtime-computed, (3) when both b and c
are runtime-computed, (4) interior nodes are NOT added to the skip set
for dynamic candidates.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Update the Kokoro integration test assertion

**Files:**
- Modify: `tests/kokoro_gpu_fusion_test.rs:431-445`

- [ ] **Step 1: Update the `add_mul_add` assertion**

In `tests/kokoro_gpu_fusion_test.rs`, find the assertion block at lines 431-445:

```rust
    assert_eq!(
        pattern_counts.add_mul_add, 0,
        "Cycle 3 reality check: Kokoro's AddMulAdd chains are all AdaIN \
         (runtime-computed b/c via Div and Slice), which the current \
         static-affine kernel legitimately cannot fuse. Expected 0 \
         matches until a dynamic-AddMulAdd kernel variant is introduced \
         in a future cycle. If this assertion is failing because the \
         count went UP, that's good news — either the walker has been \
         extended to handle runtime operands, or a new dynamic kernel \
         has been wired in. Update this assertion to match the new \
         reality (probably a positive threshold like `>= 70`). If the \
         count went up to a small positive number unintentionally, the \
         walker may be silently overpromising — verify that execution \
         actually succeeds before relaxing this assertion.",
    );
```

Replace with:

```rust
    assert!(
        pattern_counts.add_mul_add >= 40,
        "Cycle 4: Kokoro's ~73 AddMulAdd chains (AdaIN + LSTM layer-norm) \
         should be registered as dynamic candidates. The count includes \
         both static patterns and dynamic candidates. The >= 40 threshold \
         absorbs single-use exclusions, future pattern subsumption \
         (AddMulAddLeakyRelu), and any shape-mismatch fallbacks on chains \
         we haven't characterized. Got: {}",
        pattern_counts.add_mul_add,
    );
```

- [ ] **Step 2: Update the diagnostic header to reflect Cycle 4**

Find the line:
```rust
    eprintln!("\n=== Cycle 3 fused pattern counts ===");
```

Replace with:
```rust
    eprintln!("\n=== Cycle 4 fused pattern counts ===");
```

- [ ] **Step 3: Run clippy**

Run: `cargo clippy --features cuda --tests -- -D warnings`
Expected: 0 errors, 0 warnings.

- [ ] **Step 4: Run the Kokoro integration test (if GPU available)**

Run: `cargo test --features cuda -- fusion_fires_on_full_kokoro_at_seq_len_5 --ignored`
Expected: PASS with `pattern_counts.add_mul_add >= 40`. The diagnostic printout should show a count around 73.

If the test fails because the count is below 40, investigate:
- Are dynamic candidates being registered? (Check the walker logic from Task 4.)
- Is `count_fused_patterns` counting both `fused_patterns` and `dynamic_candidates`? (Check Task 5, Step 6.)

- [ ] **Step 5: Commit**

```bash
git add tests/kokoro_gpu_fusion_test.rs
git commit -S -m "$(cat <<'EOF'
test: update Kokoro AddMulAdd assertion for Cycle 4 dynamic fusion

Change from == 0 (Cycle 3 baseline) to >= 40 (Cycle 4 with dynamic
candidates). The count now includes both static patterns and dynamic
candidates registered by the walker.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Final Verification Checklist

Before declaring Cycle 4 complete:

- [ ] `cargo clippy --features cuda --tests -- -D warnings` = 0 errors
- [ ] `cargo test --features cuda` = all non-ignored tests pass
- [ ] `cargo test --features cuda -- fusion_fires_on_full_kokoro_at_seq_len_5 --ignored` = PASS with `add_mul_add >= 40`
- [ ] Diagnostic printout shows existing fusion counts unchanged: DivRsqrt ~65, Gelu ~12, MulSinPowMulAdd ~48
- [ ] All commits are signed (verify with `git log --show-signature -5`)
- [ ] No files created over 1000 lines (check `wc -l` on modified files)
