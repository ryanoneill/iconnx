# MulSinPowMulAdd Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fused CUDA kernel for the 5-op chain `Mul → Sin → Pow → Mul → Add` (48 instances in Kokoro TTS, 192 launches eliminated) and extract fusion code from `src/cuda/inference/mod.rs` into a dedicated `src/cuda/inference/fusion/` module as a targeted refactor.

**Architecture:** Cycle 1 of the Leadline-driven optimization plan. The refactor moves `FusedPattern`, `FusedPatternInfo`, and `detect_fused_patterns` into a new `src/cuda/inference/fusion/` submodule without changing behavior. Then a new `MulSinPowMulAdd` variant is added end-to-end (kernel → detection → executor dispatch → Kokoro validation), all under TDD.

**Tech Stack:** Rust 2021, `cudarc` (CUDA 12), NVRTC for runtime-compiled kernels, `cublas`/`cudnn` feature, existing `FusedKernelCache` infrastructure. Tests use the in-repo test suite; the end-to-end numerical check uses `tests/kokoro_validation_test.rs`. External benchmarking uses `leadline-bench compare` from `~/workspace/ryanoneill/rust-ai-explorations`.

**Prerequisites for the implementing agent:**
- The spec at `docs/superpowers/specs/2026-04-10-mul-sin-pow-mul-add-fusion-design.md` should be read first for context on *why* this work is being done.
- Per `/home/ryano/.claude/CLAUDE.md`: TDD everywhere, commits after every substantive change, **all commits must be signed** (`git commit -S`), stop and ask if signing fails, warnings never ignored, no file over 1000 lines, no `--no-gpg-sign` ever.
- `cargo test --features cuda` is the baseline gate. Run it after each phase.
- `cargo clippy --features cuda -- -D warnings` must be clean before any commit.

---

## File Structure

### New files

- `src/cuda/inference/fusion/mod.rs` — Module declarations and re-exports for the fusion submodule (~60 lines).
- `src/cuda/inference/fusion/patterns.rs` — `FusedPattern` enum and `FusedPatternInfo` struct, moved out of `inference/mod.rs` (~220 lines).
- `src/cuda/inference/fusion/detection.rs` — Free function `detect_fused_patterns` and its helpers, moved out of `inference/mod.rs`. Later adds the `MulSinPowMulAdd` detection walker (~450 lines).
- `tests/operators/fused_mul_sin_pow_mul_add_test.rs` — Unit tests for the new fused kernel (~200 lines).
- `tests/fusion_detection_test.rs` — Integration tests for the new pattern's detection walker and dispatch (~180 lines).

### Modified files

- `src/cuda/inference/mod.rs` — Remove `FusedPattern` enum, `FusedPatternInfo` struct, and `detect_fused_patterns` method body (replaced with a thin wrapper that calls the free function). Add executor dispatch case for `MulSinPowMulAdd`. **Net delta: approximately −300 lines.**
- `src/cuda/kernels/fused.rs` — Add `fused_mul_sin_pow_mul_add_kernel` and `fused_mul_sin_pow_mul_add_broadcast_kernel` CUDA source, add `gpu_fused_mul_sin_pow_mul_add` host wrapper, register the two kernel names in `FUSED_KERNEL_NAMES`. **Net delta: approximately +150 lines (final size stays under 1000).**
- `tests/operators/mod.rs` — Add `mod fused_mul_sin_pow_mul_add_test;` declaration (1 line).

### Responsibility split

- `patterns.rs` owns only type definitions — no logic, no dependencies on the executor struct.
- `detection.rs` owns only pattern-matching logic against `&[ExecutionNode]` and `&HashMap<String, GpuTensor>` (weights). No dependency on executor struct fields. This keeps detection unit-testable without constructing a CUDA context.
- `inference/mod.rs` retains the executor struct, `execute_fused_pattern` (since it calls kernel launch functions that need `&self.ctx`), and profile tracking.
- `fused.rs` owns CUDA source strings and host wrappers — no change to its existing layered structure.

---

## Phase A: Refactor fusion code out of inference/mod.rs

Behavioral no-op. The existing test suite is the regression gate. No test code is added or modified in this phase.

### Task A1: Create empty fusion submodule

**Files:**
- Create: `src/cuda/inference/fusion/mod.rs`
- Modify: `src/cuda/inference/mod.rs`

- [ ] **Step 1: Create the empty submodule file**

Write `src/cuda/inference/fusion/mod.rs` with this content:

```rust
//! Fused kernel pattern detection and metadata.
//!
//! This module owns the `FusedPattern` enum (variant per fusible operator
//! chain), the `FusedPatternInfo` struct (pattern + nodes to skip + head
//! node), and the `detect_fused_patterns` graph walker that identifies
//! eligible chains before inference starts.
//!
//! Detection is intentionally separated from execution: the walker only
//! needs to read the node list and weights map, so it can be unit-tested
//! without a live CUDA context. Execution stays in `inference/mod.rs`
//! because it needs the executor's kernel caches and stream.

pub(super) mod patterns;
pub(super) mod detection;

pub(super) use patterns::{FusedPattern, FusedPatternInfo};
pub(super) use detection::detect_fused_patterns;
```

- [ ] **Step 2: Wire the submodule into `inference/mod.rs`**

Find the existing module declarations near the top of `src/cuda/inference/mod.rs` (they live just after the `use` statements, around line 44). Add this declaration:

```rust
mod fusion;
```

If there is no existing `mod` section in that file (all declarations are `use` imports), add the `mod fusion;` line on its own line immediately after the final `use` statement and before the first type definition.

- [ ] **Step 3: Create an empty `patterns.rs` and `detection.rs` stub**

Write `src/cuda/inference/fusion/patterns.rs`:

```rust
//! Type definitions for fused operator patterns.
//!
//! These types are populated by the `detection` module and consumed by
//! the executor's `execute_fused_pattern` method. They deliberately
//! carry only string names (not tensor references) so the detection
//! pass can run without touching any GPU state.
```

Write `src/cuda/inference/fusion/detection.rs`:

```rust
//! Graph walker that identifies fused operator patterns before inference.
//!
//! Detection is a free function (not a method on the executor) so it can
//! be tested with a fabricated node list and a dummy weights map —
//! no CUDA context required.
```

- [ ] **Step 4: Verify the project still builds**

Run: `cargo build --features cuda`

Expected: Compiles successfully. The `mod fusion;` declaration and its empty submodules must not introduce errors.

- [ ] **Step 5: Run the test suite as a regression gate**

Run: `cargo test --features cuda`

Expected: The same number of tests pass as before this task started. No new failures.

- [ ] **Step 6: Commit (signed)**

```bash
git add src/cuda/inference/fusion/mod.rs src/cuda/inference/fusion/patterns.rs src/cuda/inference/fusion/detection.rs src/cuda/inference/mod.rs
git commit -S -m "$(cat <<'EOF'
refactor(inference): add empty fusion submodule scaffold

Prepare for extraction of FusedPattern, FusedPatternInfo, and
detect_fused_patterns from inference/mod.rs. This commit only creates
the module structure; subsequent commits move the types and function.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

If `git commit -S` fails with a signing error, **stop and ask the user to resolve signing**. Do not use `--no-gpg-sign`.

---

### Task A2: Move `FusedPattern` and `FusedPatternInfo` into `patterns.rs`

**Files:**
- Modify: `src/cuda/inference/fusion/patterns.rs`
- Modify: `src/cuda/inference/mod.rs`

- [ ] **Step 1: Locate the types in `inference/mod.rs`**

Open `src/cuda/inference/mod.rs`. The types live near lines 253-322 (one large `enum FusedPattern` with `DivRsqrt`, `AddMulAdd`, `Gelu`, `MulAdd`, `AddMul`, `SubMul`, `DivMul` variants, followed by a small `struct FusedPatternInfo`). Select the entire block from `/// Types of fused kernel patterns we can detect and optimize` down to the closing `}` of the `FusedPatternInfo` struct.

- [ ] **Step 2: Paste the block into `patterns.rs`**

Append (after the module doc comment) to `src/cuda/inference/fusion/patterns.rs`:

```rust
/// Types of fused kernel patterns we can detect and optimize.
///
/// These types are `pub` so test-only re-exports through
/// `cuda::inference::testing` can return them. The containing
/// `fusion` module itself is `pub(crate)`, so external code can only
/// reach these types via the intended re-export path.
#[derive(Clone, Debug)]
pub enum FusedPattern {
    /// Add -> Sqrt -> Div: y / sqrt(x + eps)
    /// Stores: (numerator_input, variance_input, eps_input, output_name)
    DivRsqrt {
        numerator_input: String,
        variance_input: String,
        eps_input: String,
        output_name: String,
    },
    /// Add -> Mul -> Add: (x + a) * b + c
    /// Stores: (x, a, b, c, output_name)
    AddMulAdd {
        x_input: String,
        a_input: String,
        b_input: String,
        c_input: String,
        output_name: String,
    },
    /// Full GELU activation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    /// Replaces 8 ops: Pow -> Mul -> Add -> Mul -> Tanh -> Add -> Mul -> Mul
    Gelu {
        x_input: String,
        output_name: String,
    },
    /// Mul -> Add: y = a * b + c
    MulAdd {
        a_input: String,
        b_input: String,
        c_input: String,
        output_name: String,
    },
    /// Add -> Mul: y = (a + b) * c
    AddMul {
        a_input: String,
        b_input: String,
        c_input: String,
        output_name: String,
    },
    /// Sub -> Mul: y = (a - b) * c
    SubMul {
        a_input: String,
        b_input: String,
        c_input: String,
        output_name: String,
    },
    /// Div -> Mul: y = (a / b) * c
    DivMul {
        a_input: String,
        b_input: String,
        c_input: String,
        output_name: String,
    },
}

/// Information about a detected fused pattern.
#[derive(Clone, Debug)]
pub struct FusedPatternInfo {
    /// The pattern type and its inputs
    pub pattern: FusedPattern,
    /// Node names that are part of this pattern (to skip during execution)
    pub nodes_to_skip: Vec<String>,
    /// The first node name (head) that triggers execution of the fused kernel
    pub head_node: String,
}
```

Both types and their fields are `pub` so the testing re-export surface can refer to them. Access from outside the crate remains gated by the `pub(crate) mod fusion;` declaration (the module itself is not externally visible).

- [ ] **Step 3: Delete the original type definitions from `inference/mod.rs`**

Remove the same `FusedPattern` enum and `FusedPatternInfo` struct from `src/cuda/inference/mod.rs`. The file should now contain neither definition.

- [ ] **Step 4: Update imports in `inference/mod.rs`**

At the top of `src/cuda/inference/mod.rs`, the existing `mod fusion;` declaration is now sufficient because `patterns.rs` contains `pub(crate) enum FusedPattern` and `pub(crate) struct FusedPatternInfo`. Add these imports with the other `use` statements (alphabetize if the file alphabetizes them, otherwise append):

```rust
use self::fusion::{FusedPattern, FusedPatternInfo};
```

Note: `self::fusion` is needed because the submodule is declared within the same parent `cuda::inference` module. If the existing `use` section uses `super::` or `crate::cuda::inference::` prefixes, match that style.

- [ ] **Step 5: Build and run tests**

Run: `cargo build --features cuda`

Expected: Compiles. Any visibility or import errors are fixed in-place.

Run: `cargo test --features cuda`

Expected: Full test suite passes with the same count as before. No test code was changed, so any regression is a refactoring bug.

- [ ] **Step 6: Run clippy**

Run: `cargo clippy --features cuda -- -D warnings`

Expected: Zero warnings. Fix any that appear (they are most likely unused-import warnings from the moved code).

- [ ] **Step 7: Commit (signed)**

```bash
git add src/cuda/inference/fusion/patterns.rs src/cuda/inference/mod.rs
git commit -S -m "$(cat <<'EOF'
refactor(inference): move FusedPattern types into fusion submodule

Moves FusedPattern enum and FusedPatternInfo struct from
inference/mod.rs into src/cuda/inference/fusion/patterns.rs. No
behavior change — the executor imports the types from the submodule.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task A3: Move `detect_fused_patterns` into `detection.rs` as a free function

**Files:**
- Modify: `src/cuda/inference/fusion/detection.rs`
- Modify: `src/cuda/inference/mod.rs`

- [ ] **Step 1: Make `ExecutionNode` visible to the fusion submodule**

`ExecutionNode` is currently a private struct in `src/cuda/inference/mod.rs`. The detection free function needs to read it. Change its visibility to `pub(crate)` and ensure its fields (`name`, `op_type`, `inputs`, `outputs`, `attributes`, `precomputed_slice`) are also `pub(crate)`:

```rust
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub(crate) struct ExecutionNode {
    pub(crate) name: String,
    pub(crate) op_type: String,
    pub(crate) inputs: Vec<String>,
    pub(crate) outputs: Vec<String>,
    pub(crate) attributes: NodeAttributes,
    pub(crate) precomputed_slice: Option<PrecomputedSlice>,
}
```

Also make `PrecomputedSlice` `pub(crate)` with `pub(crate)` fields, since it is referenced in `ExecutionNode`.

- [ ] **Step 2: Paste the detection function into `detection.rs`**

Replace the doc-comment-only contents of `src/cuda/inference/fusion/detection.rs` with:

```rust
//! Graph walker that identifies fused operator patterns before inference.
//!
//! Detection is a free function (not a method on the executor) so it can
//! be tested with a fabricated node list and a dummy weights map —
//! no CUDA context required.

use std::collections::{HashMap, HashSet};

use super::patterns::{FusedPattern, FusedPatternInfo};
use crate::cuda::inference::ExecutionNode;
use crate::cuda::tensor::GpuTensor;
use crate::cuda::context::IconnxCudaContext;

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
    // Build lookup maps
    let output_to_node: HashMap<&str, &ExecutionNode> = nodes
        .iter()
        .flat_map(|n| n.outputs.iter().map(move |o| (o.as_str(), n)))
        .collect();

    // Count uses of each tensor output (for single-use check)
    let mut output_uses: HashMap<&str, usize> = HashMap::new();
    for node in nodes {
        for input in &node.inputs {
            *output_uses.entry(input.as_str()).or_insert(0) += 1;
        }
    }

    let mut fused_patterns: HashMap<String, FusedPatternInfo> = HashMap::new();
    let mut nodes_to_skip: HashSet<String> = HashSet::new();

    // === ALL EXISTING DETECTION BODIES GO HERE, UNCHANGED ===
    //
    // Copy the body of the existing GpuGraphExecutor::detect_fused_patterns
    // from src/cuda/inference/mod.rs starting at the line AFTER the
    // `let mut output_uses` block and ending BEFORE the line that sets
    // `*self.patterns_detected.borrow_mut() = true;`.
    //
    // Two mechanical substitutions must be made during the paste:
    //   1. `&self.nodes` becomes `nodes`
    //   2. `self.weights.get(...)` becomes `weights.get(...)`
    //   3. `self.weights.contains_key(...)` becomes `weights.contains_key(...)`
    //   4. `self.ctx` becomes `ctx`
    //   5. Any direct access to `self.fused_patterns.borrow_mut()` and
    //      `self.nodes_to_skip.borrow_mut()` becomes `&mut fused_patterns`
    //      and `&mut nodes_to_skip` (local variables).
    //
    // NO other logic changes. This is a pure mechanical move.

    (fused_patterns, nodes_to_skip)
}
```

- [ ] **Step 3: Perform the mechanical paste and substitutions**

Copy the body of `detect_fused_patterns` from `src/cuda/inference/mod.rs` (currently lines ~536 to ~1048, from the `let output_to_node` declaration through the last pattern detection block). Paste it into the placeholder in `detection.rs` between the two local-variable declarations and the return statement.

Apply these exact substitutions (search-replace in the pasted block only):

1. `&self.nodes` → `nodes`
2. `self.weights.get(` → `weights.get(`
3. `self.weights.contains_key(` → `weights.contains_key(`
4. `self.ctx` → `ctx`
5. `self.fused_patterns.borrow_mut()` → referenced via the local `fused_patterns` variable. Remove the `RefCell::borrow_mut()` call; the local is already a `&mut HashMap`.
6. `self.nodes_to_skip.borrow_mut()` → same, use the local `nodes_to_skip` variable directly.
7. Remove any `let mut fused_patterns = self.fused_patterns.borrow_mut();` shadow — the local variable is already in scope.
8. Remove any `let mut nodes_to_skip = self.nodes_to_skip.borrow_mut();` shadow — same reason.
9. Delete the final `*self.patterns_detected.borrow_mut() = true;` line — the caller will manage that flag.

- [ ] **Step 4: Replace `GpuGraphExecutor::detect_fused_patterns` with a thin wrapper**

In `src/cuda/inference/mod.rs`, replace the entire body of `fn detect_fused_patterns(&self)` (the method, not the free function) with this wrapper:

```rust
/// Detect fuseable operator patterns in the graph.
/// Called automatically before first execution.
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

The wrapper preserves the `patterns_detected` guard so repeated calls are still no-ops after the first.

- [ ] **Step 5: Build and fix any errors in one pass**

Run: `cargo build --features cuda`

Expected: Clean compile. Likely first-attempt errors and their fixes:
- *Missing `IconnxCudaContext` import in detection.rs* — add `use crate::cuda::context::IconnxCudaContext;` at the top.
- *Visibility errors on `ExecutionNode` fields* — revisit Step 1 to ensure all fields are `pub(crate)`.
- *Dead `use` statement in `inference/mod.rs`* — the old function body may have imported things only it used (`HashMap`, etc.). Leave any `use` statements that are still needed elsewhere in the file; remove truly unused ones.

- [ ] **Step 6: Run full test suite**

Run: `cargo test --features cuda`

Expected: All tests pass. The GELU, DivRsqrt, AddMulAdd, MulAdd, AddMul, SubMul, and DivMul fusion patterns must all still detect correctly — any regression means the mechanical paste was incomplete.

- [ ] **Step 7: Run clippy**

Run: `cargo clippy --features cuda -- -D warnings`

Expected: Zero warnings. If clippy complains about a `HashMap<&str, &ExecutionNode>` lifetime or similar, the detection function's internal lifetimes may need explicit annotation — keep the fix local to `detection.rs`.

- [ ] **Step 8: Commit (signed)**

```bash
git add src/cuda/inference/fusion/detection.rs src/cuda/inference/mod.rs
git commit -S -m "$(cat <<'EOF'
refactor(inference): extract detect_fused_patterns to free function

Moves the ~500-line detect_fused_patterns method body into
src/cuda/inference/fusion/detection.rs as a free function that takes
the node list, weights map, and CUDA context as parameters. The
executor method becomes a thin wrapper that calls the free function
and populates the RefCell-wrapped state.

Decoupling detection from executor state makes it unit-testable
without constructing a CUDA context. No behavior change — all
existing fusion tests pass.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task A4: Verify file sizes and check Phase A gate

**Files:**
- Read-only: `src/cuda/inference/mod.rs`, `src/cuda/inference/fusion/*.rs`

- [ ] **Step 1: Verify `inference/mod.rs` shrunk by ~300 lines**

Run: `wc -l src/cuda/inference/mod.rs`

Expected: approximately 4,685 lines (was 4,985, minus ~300 for the moved code). If it's larger than 4,700, the move was incomplete. If smaller than 4,600, something was over-deleted.

- [ ] **Step 2: Verify each new file is under 1,000 lines**

Run: `wc -l src/cuda/inference/fusion/mod.rs src/cuda/inference/fusion/patterns.rs src/cuda/inference/fusion/detection.rs`

Expected: `mod.rs` ~20, `patterns.rs` ~220, `detection.rs` ~500. All well under 1,000.

- [ ] **Step 3: Full test suite as phase gate**

Run: `cargo test --features cuda`

Expected: Green. If anything is red, fix it before proceeding to Phase B. Phase A must leave the suite in a clean state.

- [ ] **Step 4: Clippy as phase gate**

Run: `cargo clippy --features cuda -- -D warnings`

Expected: Zero warnings.

No commit for this task — it is verification only.

---

## Phase B: Add the new fused kernel (test-first)

### Task B1: Write failing tests for `gpu_fused_mul_sin_pow_mul_add` (same-shape case)

**Files:**
- Create: `tests/operators/fused_mul_sin_pow_mul_add_test.rs`
- Modify: `tests/operators/mod.rs`

- [ ] **Step 1: Register the test module**

Add this line to `tests/operators/mod.rs` in alphabetical order (after `fftw_stft_test` if present, before `gather_test`):

```rust
mod fused_mul_sin_pow_mul_add_test;
```

- [ ] **Step 2: Write the failing test file**

Create `tests/operators/fused_mul_sin_pow_mul_add_test.rs` with this exact content:

```rust
//! Tests for the fused Mul->Sin->Pow->Mul->Add kernel.
//!
//! Mathematical identity: y = sin(x * w0)^p * w1 + b
//!
//! The fused kernel must match the unfused chain of `gpu_mul`, `gpu_sin`,
//! `gpu_pow`, `gpu_mul`, `gpu_add` to within 1e-5 per element.

#![cfg(feature = "cuda")]

use iconnx::cuda::{
    gpu_add, gpu_fused_mul_sin_pow_mul_add, gpu_mul, gpu_pow, gpu_sin, FusedKernelCache,
    GpuMemoryPool, GpuTensor, IconnxCudaContext, KernelCache,
};

/// Build a GPU f32 tensor from a host Vec.
fn upload(ctx: &IconnxCudaContext, data: Vec<f32>, shape: Vec<usize>) -> GpuTensor {
    GpuTensor::from_host_f32(ctx, &data, shape).expect("upload failed")
}

/// Run the unfused reference chain: y = sin(x * w0)^p * w1 + b
fn unfused_reference(
    ctx: &IconnxCudaContext,
    elem_cache: &KernelCache,
    pool: &mut GpuMemoryPool,
    x: &GpuTensor,
    w0: &GpuTensor,
    w1: &GpuTensor,
    b: &GpuTensor,
    p: f32,
) -> GpuTensor {
    let t0 = gpu_mul(ctx, elem_cache, pool, x, w0).expect("gpu_mul t0");
    let t1 = gpu_sin(ctx, elem_cache, pool, &t0).expect("gpu_sin t1");
    let p_tensor = GpuTensor::from_host_f32(ctx, &[p], vec![1])
        .expect("p tensor");
    let t2 = gpu_pow(ctx, elem_cache, pool, &t1, &p_tensor).expect("gpu_pow t2");
    let t3 = gpu_mul(ctx, elem_cache, pool, &t2, w1).expect("gpu_mul t3");
    gpu_add(ctx, elem_cache, pool, &t3, b).expect("gpu_add y")
}

/// Compare two float tensors elementwise within a tolerance.
fn assert_close(ctx: &IconnxCudaContext, a: &GpuTensor, b: &GpuTensor, tol: f32) {
    let av = a.to_host_f32(ctx).expect("to_host_f32 a");
    let bv = b.to_host_f32(ctx).expect("to_host_f32 b");
    assert_eq!(av.len(), bv.len(), "length mismatch");
    for (i, (lhs, rhs)) in av.iter().zip(bv.iter()).enumerate() {
        let diff = (lhs - rhs).abs();
        assert!(
            diff <= tol,
            "mismatch at index {i}: fused={lhs}, unfused={rhs}, diff={diff}"
        );
    }
}

#[test]
fn naive_vs_fused_matches_same_shape() {
    let ctx = IconnxCudaContext::new().expect("CUDA context");
    let elem_cache = KernelCache::new(&ctx).expect("KernelCache");
    let fused_cache = FusedKernelCache::new(&ctx).expect("FusedKernelCache");
    let mut pool = GpuMemoryPool::new();

    // Deterministic synthetic inputs: shape [4, 8] so the test runs fast.
    let n = 4 * 8;
    let x_data: Vec<f32> = (0..n).map(|i| 0.01 * i as f32).collect();
    let w0_data: Vec<f32> = (0..n).map(|i| 0.5 + 0.03 * i as f32).collect();
    let w1_data: Vec<f32> = (0..n).map(|i| 1.0 - 0.02 * i as f32).collect();
    let b_data: Vec<f32> = (0..n).map(|i| 0.1 * i as f32).collect();

    let x = upload(&ctx, x_data, vec![4, 8]);
    let w0 = upload(&ctx, w0_data, vec![4, 8]);
    let w1 = upload(&ctx, w1_data, vec![4, 8]);
    let b = upload(&ctx, b_data, vec![4, 8]);
    let p: f32 = 2.0;

    let reference =
        unfused_reference(&ctx, &elem_cache, &mut pool, &x, &w0, &w1, &b, p);
    let fused = gpu_fused_mul_sin_pow_mul_add(
        &ctx, &fused_cache, &mut pool, &x, &w0, &w1, &b, p,
    )
    .expect("gpu_fused_mul_sin_pow_mul_add");

    assert_close(&ctx, &fused, &reference, 1e-5);
}
```

- [ ] **Step 3: Run the test to confirm it fails at compile time**

Run: `cargo test --features cuda --test operators -- fused_mul_sin_pow_mul_add`

Expected: Compilation error. The message should reference `gpu_fused_mul_sin_pow_mul_add` not being found in `iconnx::cuda::kernels::fused`. This is the red state.

- [ ] **Step 4: No commit yet — the test is in the red state and the next task will make it green**

Do not commit a failing test in isolation; the convention in this repo (per the existing operator tests) is test + implementation together per feature step.

---

### Task B2: Implement the fused kernel (same-shape case)

**Files:**
- Modify: `src/cuda/kernels/fused.rs`

- [ ] **Step 1: Add the kernel name to `FUSED_KERNEL_NAMES`**

In `src/cuda/kernels/fused.rs`, find `const FUSED_KERNEL_NAMES: &[&str]` (around line 57). Append two new entries:

```rust
const FUSED_KERNEL_NAMES: &[&str] = &[
    // ... existing entries ...
    // Mul-Sin-Pow-Mul-Add: y = sin(x * w0)^p * w1 + b
    // Vocoder periodic activation (48 occurrences in Kokoro)
    "fused_mul_sin_pow_mul_add_kernel",
    "fused_mul_sin_pow_mul_add_broadcast_kernel",
];
```

- [ ] **Step 2: Add the CUDA source for the same-shape kernel**

In the `const FUSED_KERNELS: &str = r#" ... "#;` raw string, append (before the closing `"#;`) a new kernel block. Place it after the existing `fused_div_mul_kernel` section:

```c
// ============================================================================
// Fused mul_sin_pow_mul_add: y = sin(x * w0)^p * w1 + b
// Vocoder periodic activation (48 occurrences in Kokoro)
// Replaces: Mul -> Sin -> Pow -> Mul -> Add
// ============================================================================

extern "C" __global__ void fused_mul_sin_pow_mul_add_kernel(
    float* out,
    const float* x,
    const float* w0,
    const float* w1,
    const float* b,
    float p,
    int p_is_int,
    size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float s = sinf(x[i] * w0[i]);
        float pow_val;
        if (p_is_int != 0) {
            // Integer exponent path: handles negative bases correctly
            int ip = (int) p;
            pow_val = 1.0f;
            float base = s;
            int exp = ip < 0 ? -ip : ip;
            while (exp > 0) {
                if (exp & 1) pow_val *= base;
                base *= base;
                exp >>= 1;
            }
            if (ip < 0) pow_val = 1.0f / pow_val;
        } else {
            // Float exponent path: caller guarantees base is non-negative
            pow_val = powf(s, p);
        }
        out[i] = pow_val * w1[i] + b[i];
    }
}

// Broadcast variant: w0, w1, and b may broadcast along the leading dimension
// (they have shape [1, C] while x has shape [N, C]).
// broadcast_stride is the number of elements per broadcast row
// (e.g., C when broadcasting [1, C] over [N, C]).
extern "C" __global__ void fused_mul_sin_pow_mul_add_broadcast_kernel(
    float* out,
    const float* x,
    const float* w0,
    const float* w1,
    const float* b,
    float p,
    int p_is_int,
    size_t n,
    size_t broadcast_stride
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        size_t bi = i % broadcast_stride;
        float s = sinf(x[i] * w0[bi]);
        float pow_val;
        if (p_is_int != 0) {
            int ip = (int) p;
            pow_val = 1.0f;
            float base = s;
            int exp = ip < 0 ? -ip : ip;
            while (exp > 0) {
                if (exp & 1) pow_val *= base;
                base *= base;
                exp >>= 1;
            }
            if (ip < 0) pow_val = 1.0f / pow_val;
        } else {
            pow_val = powf(s, p);
        }
        out[i] = pow_val * w1[bi] + b[bi];
    }
}
```

Integer exponentiation is written as manual exponentiation-by-squaring rather than calling `pown` because NVRTC's availability of `pown` varies across CUDA versions and the squaring loop is numerically identical for small integer exponents (the only exponent values the Kokoro graph uses).

- [ ] **Step 3: Add the host wrapper function**

At the bottom of `src/cuda/kernels/fused.rs`, after the existing `gpu_fused_*` functions, add:

```rust
/// GPU fused mul_sin_pow_mul_add: y = sin(x * w0)^p * w1 + b
///
/// Replaces the 5-op chain Mul -> Sin -> Pow -> Mul -> Add used in vocoder
/// periodic activations (48 occurrences in Kokoro).
///
/// All tensors must have the same shape (broadcast variant is a separate
/// function added in a later step).
///
/// `p` is passed as f32 and the caller signals whether it is an integer
/// via its fractional part: integer exponents use a squaring loop that
/// handles negative bases correctly; non-integer exponents use `powf`
/// (and require non-negative bases).
#[allow(clippy::too_many_arguments)]
pub fn gpu_fused_mul_sin_pow_mul_add(
    ctx: &IconnxCudaContext,
    cache: &FusedKernelCache,
    pool: &mut GpuMemoryPool,
    x: &GpuTensor,
    w0: &GpuTensor,
    w1: &GpuTensor,
    b: &GpuTensor,
    p: f32,
) -> Result<GpuTensor, CudaError> {
    // Same-shape fast path: all four tensors share one shape.
    let shape = x.shape();
    if w0.shape() == shape && w1.shape() == shape && b.shape() == shape {
        let n = x.len();
        let mut out = pool.get_tensor_f32(ctx, shape.to_vec())?;

        let func = cache
            .get("fused_mul_sin_pow_mul_add_kernel")
            .ok_or_else(|| {
                CudaError::Kernel("fused_mul_sin_pow_mul_add_kernel not found".into())
            })?;

        let p_is_int: i32 = if p.fract() == 0.0 { 1 } else { 0 };

        unsafe {
            ctx.stream()
                .launch_builder(func)
                .arg(out.data_f32_mut()?)
                .arg(x.data_f32()?)
                .arg(w0.data_f32()?)
                .arg(w1.data_f32()?)
                .arg(b.data_f32()?)
                .arg(&p)
                .arg(&p_is_int)
                .arg(&n)
                .launch(elementwise_config(n))
                .map_err(|e| {
                    CudaError::Kernel(format!(
                        "fused_mul_sin_pow_mul_add launch failed: {}",
                        e
                    ))
                })?;
        }

        return Ok(out);
    }

    // Broadcast cases are added in Task B3.
    Err(CudaError::Kernel(format!(
        "gpu_fused_mul_sin_pow_mul_add: broadcast variant not yet implemented \
         (x={:?}, w0={:?}, w1={:?}, b={:?})",
        x.shape(),
        w0.shape(),
        w1.shape(),
        b.shape()
    )))
}
```

- [ ] **Step 4: Re-export the new function and `FusedKernelCache` from `cuda/mod.rs`**

The test file imports `iconnx::cuda::{FusedKernelCache, gpu_fused_mul_sin_pow_mul_add, ...}`. Neither symbol is currently re-exported — `src/cuda/mod.rs` has `mod kernels;` (private) and the `pub use kernels::{...}` block does not list any fused kernels or `FusedKernelCache`.

Add a new re-export in `src/cuda/mod.rs` near the existing `pub use kernels::{...}` block:

```rust
#[allow(unused_imports)]
pub use kernels::fused::{
    gpu_fused_add_mul, gpu_fused_add_mul_add, gpu_fused_div_mul, gpu_fused_div_rsqrt,
    gpu_fused_gelu, gpu_fused_mul_add, gpu_fused_mul_sin_pow_mul_add, gpu_fused_sub_mul,
    FusedKernelCache,
};
```

Note that this re-exports all existing `gpu_fused_*` helpers too — they were previously used only inside `cuda::inference`, so they weren't re-exported. Re-exporting them is harmless (they're feature-gated behind `cuda`) and keeps the public surface for fused kernels consistent.

`src/cuda/kernels/mod.rs` already has `pub mod fused;`, so the `pub use kernels::fused::...` path resolves from `cuda/mod.rs` (which has access to the private `kernels` module as its direct child). No visibility change is needed for the `fused` submodule itself.

- [ ] **Step 5: Build**

Run: `cargo build --features cuda`

Expected: Clean compile. Likely first-attempt errors:
- *NVRTC compilation error inside the raw string* — any typo in the CUDA source surfaces as an NVRTC error at runtime (when tests run), not a Rust compile error. Review the CUDA source for typos before running tests.
- *`data_f32_mut` not found on `GpuTensor`* — the correct method is on `GpuTensor::Float32`; the `.data_f32_mut()` method on the enum should work. Compare to how `gpu_fused_div_rsqrt` uses it.

- [ ] **Step 6: Run the Task B1 test to verify it now passes**

Run: `cargo test --features cuda --test operators -- fused_mul_sin_pow_mul_add::naive_vs_fused_matches_same_shape`

Expected: PASS.

- [ ] **Step 7: Full test suite regression check**

Run: `cargo test --features cuda`

Expected: Full suite green, including the new test. No regression in existing tests.

- [ ] **Step 8: Clippy**

Run: `cargo clippy --features cuda -- -D warnings`

Expected: Zero warnings.

- [ ] **Step 9: Commit (signed)**

```bash
git add src/cuda/kernels/fused.rs src/cuda/kernels/mod.rs tests/operators/mod.rs tests/operators/fused_mul_sin_pow_mul_add_test.rs
git commit -S -m "$(cat <<'EOF'
feat(cuda): add fused mul_sin_pow_mul_add kernel (same-shape case)

Implements y = sin(x * w0)^p * w1 + b as a single CUDA kernel for
the same-shape case. Replaces the 5-op chain Mul->Sin->Pow->Mul->Add
which appears 48 times in Kokoro's vocoder.

Integer exponents use exponentiation-by-squaring so negative
intermediate values (sin can be negative) produce correct signs;
non-integer exponents fall back to powf.

Broadcast variants and detection walker arrive in subsequent commits.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task B3: Add broadcast-case test and implementation

**Files:**
- Modify: `tests/operators/fused_mul_sin_pow_mul_add_test.rs`
- Modify: `src/cuda/kernels/fused.rs`

- [ ] **Step 1: Add the failing broadcast test**

Append to `tests/operators/fused_mul_sin_pow_mul_add_test.rs`:

```rust
#[test]
fn broadcasting_w0_w1_b_per_channel() {
    let ctx = IconnxCudaContext::new().expect("CUDA context");
    let elem_cache = KernelCache::new(&ctx).expect("KernelCache");
    let fused_cache = FusedKernelCache::new(&ctx).expect("FusedKernelCache");
    let mut pool = GpuMemoryPool::new();

    // x is [N, C], w0/w1/b are [1, C] broadcast along the leading dim.
    let n_rows = 3usize;
    let channels = 8usize;
    let total = n_rows * channels;

    let x_data: Vec<f32> = (0..total).map(|i| 0.01 * i as f32 + 0.2).collect();
    let w0_row: Vec<f32> = (0..channels).map(|i| 0.5 + 0.05 * i as f32).collect();
    let w1_row: Vec<f32> = (0..channels).map(|i| 1.0 - 0.03 * i as f32).collect();
    let b_row: Vec<f32> = (0..channels).map(|i| 0.1 * i as f32).collect();

    let x = upload(&ctx, x_data, vec![n_rows, channels]);
    let w0 = upload(&ctx, w0_row.clone(), vec![1, channels]);
    let w1 = upload(&ctx, w1_row.clone(), vec![1, channels]);
    let b = upload(&ctx, b_row.clone(), vec![1, channels]);
    let p: f32 = 2.0;

    // Build the ground truth by tiling the broadcast rows to the full shape
    // and running the unfused reference chain.
    let mut w0_tiled = Vec::with_capacity(total);
    let mut w1_tiled = Vec::with_capacity(total);
    let mut b_tiled = Vec::with_capacity(total);
    for _ in 0..n_rows {
        w0_tiled.extend_from_slice(&w0_row);
        w1_tiled.extend_from_slice(&w1_row);
        b_tiled.extend_from_slice(&b_row);
    }
    let w0_full = upload(&ctx, w0_tiled, vec![n_rows, channels]);
    let w1_full = upload(&ctx, w1_tiled, vec![n_rows, channels]);
    let b_full = upload(&ctx, b_tiled, vec![n_rows, channels]);

    let reference = unfused_reference(
        &ctx, &elem_cache, &mut pool, &x, &w0_full, &w1_full, &b_full, p,
    );
    let fused = gpu_fused_mul_sin_pow_mul_add(
        &ctx, &fused_cache, &mut pool, &x, &w0, &w1, &b, p,
    )
    .expect("gpu_fused_mul_sin_pow_mul_add broadcast");

    assert_close(&ctx, &fused, &reference, 1e-5);
}
```

- [ ] **Step 2: Run the broadcast test to confirm it fails**

Run: `cargo test --features cuda --test operators -- fused_mul_sin_pow_mul_add::broadcasting_w0_w1_b_per_channel`

Expected: Test panics with the `"broadcast variant not yet implemented"` error from the stub. This is the red state.

- [ ] **Step 3: Implement the broadcast host wrapper branch**

In `src/cuda/kernels/fused.rs`, replace the `Err(CudaError::Kernel(format!("broadcast variant not yet implemented...`))` branch in `gpu_fused_mul_sin_pow_mul_add` with:

```rust
    // Broadcast case: w0, w1, b all share shape [1, C] or [..., 1, C] while
    // x has shape [..., N, C]. The broadcast stride is the number of
    // channels (last-axis length), computed as w0.len().
    let w0_shape = w0.shape();
    let w1_shape = w1.shape();
    let b_shape = b.shape();

    let broadcast_shape_matches =
        w0_shape == w1_shape && w1_shape == b_shape && w0.len() > 0;
    let broadcast_stride = w0.len();
    let x_len = x.len();

    if broadcast_shape_matches && x_len % broadcast_stride == 0 {
        let n = x_len;
        let mut out = pool.get_tensor_f32(ctx, shape.to_vec())?;

        let func = cache
            .get("fused_mul_sin_pow_mul_add_broadcast_kernel")
            .ok_or_else(|| {
                CudaError::Kernel(
                    "fused_mul_sin_pow_mul_add_broadcast_kernel not found".into(),
                )
            })?;

        let p_is_int: i32 = if p.fract() == 0.0 { 1 } else { 0 };

        unsafe {
            ctx.stream()
                .launch_builder(func)
                .arg(out.data_f32_mut()?)
                .arg(x.data_f32()?)
                .arg(w0.data_f32()?)
                .arg(w1.data_f32()?)
                .arg(b.data_f32()?)
                .arg(&p)
                .arg(&p_is_int)
                .arg(&n)
                .arg(&broadcast_stride)
                .launch(elementwise_config(n))
                .map_err(|e| {
                    CudaError::Kernel(format!(
                        "fused_mul_sin_pow_mul_add_broadcast launch failed: {}",
                        e
                    ))
                })?;
        }

        return Ok(out);
    }

    Err(CudaError::Kernel(format!(
        "gpu_fused_mul_sin_pow_mul_add: unsupported shape combination \
         (x={:?}, w0={:?}, w1={:?}, b={:?})",
        x.shape(),
        w0.shape(),
        w1.shape(),
        b.shape()
    )))
```

- [ ] **Step 4: Build and run the broadcast test**

Run: `cargo test --features cuda --test operators -- fused_mul_sin_pow_mul_add::broadcasting_w0_w1_b_per_channel`

Expected: PASS.

- [ ] **Step 5: Run the full test file to confirm no regression**

Run: `cargo test --features cuda --test operators -- fused_mul_sin_pow_mul_add`

Expected: Both tests pass (same-shape and broadcast).

- [ ] **Step 6: Clippy**

Run: `cargo clippy --features cuda -- -D warnings`

Expected: Zero warnings.

- [ ] **Step 7: Commit (signed)**

```bash
git add src/cuda/kernels/fused.rs tests/operators/fused_mul_sin_pow_mul_add_test.rs
git commit -S -m "$(cat <<'EOF'
feat(cuda): add broadcast variant to mul_sin_pow_mul_add fusion

Adds fused_mul_sin_pow_mul_add_broadcast_kernel for the common case
where w0, w1, and b share a broadcast shape [..., C] over an x of
shape [..., N, C]. The broadcast stride is derived from the weight
tensor length.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task B4: Add odd-exponent test (sign handling)

**Files:**
- Modify: `tests/operators/fused_mul_sin_pow_mul_add_test.rs`

- [ ] **Step 1: Add the test for p=3**

Append to `tests/operators/fused_mul_sin_pow_mul_add_test.rs`:

```rust
#[test]
fn odd_exponent_p3_preserves_sign() {
    let ctx = IconnxCudaContext::new().expect("CUDA context");
    let elem_cache = KernelCache::new(&ctx).expect("KernelCache");
    let fused_cache = FusedKernelCache::new(&ctx).expect("FusedKernelCache");
    let mut pool = GpuMemoryPool::new();

    // Deliberately choose inputs so that x * w0 lands in regions where
    // sin() is negative (e.g., values near 3pi/2). Then raising to an odd
    // exponent must preserve the negative sign — powf(neg, odd_int) in
    // CUDA returns NaN by default, which is the bug this test guards.
    let shape = vec![2, 6];
    let n = 2 * 6;
    let x_data: Vec<f32> = (0..n).map(|i| 4.0 + 0.1 * i as f32).collect();
    let w0_data: Vec<f32> = vec![1.0; n];
    let w1_data: Vec<f32> = (0..n).map(|i| 1.0 + 0.01 * i as f32).collect();
    let b_data: Vec<f32> = (0..n).map(|i| 0.01 * i as f32).collect();

    let x = upload(&ctx, x_data, shape.clone());
    let w0 = upload(&ctx, w0_data, shape.clone());
    let w1 = upload(&ctx, w1_data, shape.clone());
    let b = upload(&ctx, b_data, shape.clone());
    let p: f32 = 3.0;

    let reference =
        unfused_reference(&ctx, &elem_cache, &mut pool, &x, &w0, &w1, &b, p);
    let fused = gpu_fused_mul_sin_pow_mul_add(
        &ctx, &fused_cache, &mut pool, &x, &w0, &w1, &b, p,
    )
    .expect("gpu_fused_mul_sin_pow_mul_add p=3");

    // Sanity check: at least one reference output should be strictly negative,
    // otherwise the test isn't actually exercising the sign-handling path.
    let ref_host = reference.to_host_f32(&ctx).expect("to_host reference");
    assert!(
        ref_host.iter().any(|&v| v < -1e-3),
        "test inputs don't produce negative values — strengthen them"
    );

    assert_close(&ctx, &fused, &reference, 1e-5);
}
```

- [ ] **Step 2: Run the new test**

Run: `cargo test --features cuda --test operators -- fused_mul_sin_pow_mul_add::odd_exponent_p3_preserves_sign`

Expected: PASS. The integer-exponent path uses exponentiation-by-squaring, so negative bases are handled correctly. If the test fails at the `assert_close` step, the `p_is_int` detection is wrong. If it fails at the sanity assertion, the synthetic inputs don't actually produce negative intermediate values — widen the x range or use `w0_data = vec![2.0; n]` to push further into negative-sin territory.

- [ ] **Step 3: Full test file**

Run: `cargo test --features cuda --test operators -- fused_mul_sin_pow_mul_add`

Expected: All three tests pass.

- [ ] **Step 4: Clippy**

Run: `cargo clippy --features cuda -- -D warnings`

Expected: Zero warnings.

- [ ] **Step 5: Commit (signed)**

```bash
git add tests/operators/fused_mul_sin_pow_mul_add_test.rs
git commit -S -m "$(cat <<'EOF'
test(cuda): verify mul_sin_pow_mul_add handles odd exponents with negative sin

Guards the sign-handling path in the integer-exponent branch of the
fused kernel. CUDA powf(negative, odd_int) returns NaN by default; the
fused kernel uses exponentiation-by-squaring instead, which must
produce the correctly-signed result.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase C: Detect the new pattern

### Task C1: Write failing detection test

**Files:**
- Create: `tests/fusion_detection_test.rs`

- [ ] **Step 1: Write the new test file**

Create `tests/fusion_detection_test.rs` with this content. The test fabricates a minimal `ExecutionNode` list that mirrors the pattern and calls the detection free function directly.

```rust
//! Tests for the fused-pattern detection walker in
//! `src/cuda/inference/fusion/detection.rs`.
//!
//! These tests construct synthetic `ExecutionNode` lists — no real ONNX
//! model, no real CUDA graph. The detection function is pure (aside
//! from a CUDA context reference used to read constant scalar exponents
//! via `to_host_f32`), so we can unit-test it cheaply.

#![cfg(feature = "cuda")]

use std::collections::HashMap;

use iconnx::attributes::NodeAttributes;
use iconnx::cuda::inference::testing::{detect_fused_patterns_for_tests, ExecutionNodeForTests};
use iconnx::cuda::{GpuTensor, IconnxCudaContext};

/// Helper: build a node with the given name, op type, inputs, and outputs
/// (no attributes, no precomputed slice).
fn node(name: &str, op: &str, inputs: &[&str], outputs: &[&str]) -> ExecutionNodeForTests {
    ExecutionNodeForTests {
        name: name.to_string(),
        op_type: op.to_string(),
        inputs: inputs.iter().map(|s| s.to_string()).collect(),
        outputs: outputs.iter().map(|s| s.to_string()).collect(),
        attributes: NodeAttributes::default(),
    }
}

/// Build a dummy weights map containing a constant scalar `p` tensor under
/// the given name.
fn weights_with_scalar(
    ctx: &IconnxCudaContext,
    name: &str,
    value: f32,
) -> HashMap<String, GpuTensor> {
    let mut w = HashMap::new();
    let t = GpuTensor::from_host_f32(ctx, &[value], vec![1]).expect("scalar upload");
    w.insert(name.to_string(), t);
    w
}

#[test]
fn detects_mul_sin_pow_mul_add_chain() {
    let ctx = IconnxCudaContext::new().expect("CUDA context");

    // Build the 5-op chain:
    //   t0 = Mul(x, w0)
    //   t1 = Sin(t0)
    //   t2 = Pow(t1, p)
    //   t3 = Mul(t2, w1)
    //   y  = Add(t3, b)
    let nodes = vec![
        node("head_mul", "Mul", &["x", "w0"], &["t0"]),
        node("sin_node", "Sin", &["t0"], &["t1"]),
        node("pow_node", "Pow", &["t1", "p"], &["t2"]),
        node("tail_mul", "Mul", &["t2", "w1"], &["t3"]),
        node("bias_add", "Add", &["t3", "b"], &["y"]),
    ];

    // p must be a constant scalar weight for the pattern to be eligible.
    let weights = weights_with_scalar(&ctx, "p", 2.0);

    let (patterns, skip_set) = detect_fused_patterns_for_tests(&nodes, &weights, &ctx);

    assert!(
        patterns.contains_key("head_mul"),
        "head Mul was not registered as a pattern head. Registered: {:?}",
        patterns.keys().collect::<Vec<_>>()
    );
    assert!(skip_set.contains("sin_node"));
    assert!(skip_set.contains("pow_node"));
    assert!(skip_set.contains("tail_mul"));
    assert!(skip_set.contains("bias_add"));
}

#[test]
fn rejects_when_intermediate_has_multiple_consumers() {
    let ctx = IconnxCudaContext::new().expect("CUDA context");

    // Same chain but t2 is consumed by a second downstream node as well,
    // so the fusion is not safe.
    let nodes = vec![
        node("head_mul", "Mul", &["x", "w0"], &["t0"]),
        node("sin_node", "Sin", &["t0"], &["t1"]),
        node("pow_node", "Pow", &["t1", "p"], &["t2"]),
        node("tail_mul", "Mul", &["t2", "w1"], &["t3"]),
        node("bias_add", "Add", &["t3", "b"], &["y"]),
        // Second consumer of t2 — breaks single-use invariant
        node("other_mul", "Mul", &["t2", "q"], &["y2"]),
    ];

    let weights = weights_with_scalar(&ctx, "p", 2.0);

    let (patterns, _skip_set) = detect_fused_patterns_for_tests(&nodes, &weights, &ctx);

    // The MulSinPowMulAdd pattern must NOT be registered against head_mul.
    // (Other fusions may still match — that's fine, just not this one.)
    use iconnx::cuda::inference::testing::is_mul_sin_pow_mul_add;
    assert!(
        !patterns
            .get("head_mul")
            .map(is_mul_sin_pow_mul_add)
            .unwrap_or(false),
        "MulSinPowMulAdd was registered despite multi-use intermediate"
    );
}
```

- [ ] **Step 2: Introduce the test-support surface**

The test imports `iconnx::cuda::inference::testing::{detect_fused_patterns_for_tests, ExecutionNodeForTests, is_mul_sin_pow_mul_add}`. These do not exist yet; they form a thin public testing facade that exposes the private detection function without leaking the rest of the executor's internals.

Create `src/cuda/inference/testing.rs`:

```rust
//! Test-only bridge for exercising the fusion detection walker from
//! integration tests without exposing the private executor internals.
//!
//! This module is feature-gated behind `cuda` (since detection lives
//! beneath `cuda::inference`). Consumers should only call these
//! functions from `#[cfg(test)]` contexts.

use std::collections::{HashMap, HashSet};

use crate::attributes::NodeAttributes;
use crate::cuda::context::IconnxCudaContext;
use crate::cuda::inference::fusion::{
    detect_fused_patterns, FusedPattern, FusedPatternInfo,
};
use crate::cuda::inference::{ExecutionNode, PrecomputedSlice};
use crate::cuda::tensor::GpuTensor;

/// Testing mirror of the internal `ExecutionNode` shape. Constructing this
/// in tests avoids depending on the private struct's `precomputed_slice`
/// field (which has a non-trivial internal type).
pub struct ExecutionNodeForTests {
    pub name: String,
    pub op_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attributes: NodeAttributes,
}

impl From<ExecutionNodeForTests> for ExecutionNode {
    fn from(t: ExecutionNodeForTests) -> Self {
        ExecutionNode {
            name: t.name,
            op_type: t.op_type,
            inputs: t.inputs,
            outputs: t.outputs,
            attributes: t.attributes,
            precomputed_slice: None::<PrecomputedSlice>,
        }
    }
}

/// Run pattern detection on a test-fabricated node list.
pub fn detect_fused_patterns_for_tests(
    nodes: &[ExecutionNodeForTests],
    weights: &HashMap<String, GpuTensor>,
    ctx: &IconnxCudaContext,
) -> (HashMap<String, FusedPatternInfo>, HashSet<String>) {
    let converted: Vec<ExecutionNode> = nodes
        .iter()
        .map(|n| ExecutionNode {
            name: n.name.clone(),
            op_type: n.op_type.clone(),
            inputs: n.inputs.clone(),
            outputs: n.outputs.clone(),
            attributes: n.attributes.clone(),
            precomputed_slice: None,
        })
        .collect();
    detect_fused_patterns(&converted, weights, ctx)
}

/// Predicate: does this pattern info describe a MulSinPowMulAdd fusion?
pub fn is_mul_sin_pow_mul_add(info: &FusedPatternInfo) -> bool {
    matches!(info.pattern, FusedPattern::MulSinPowMulAdd { .. })
}
```

- [ ] **Step 3: Expose the testing module**

Three visibility edits are required to make the test imports resolve:

(a) In `src/cuda/inference/mod.rs`, add near the other module declarations:

```rust
pub mod testing;
```

(b) In `src/cuda/mod.rs`, change the existing `mod inference;` declaration to `pub mod inference;`. This exposes `iconnx::cuda::inference::testing` as a valid path. The existing `pub use inference::{GpuGraphExecutor, NodeExecutionLog, ProfileData, TensorLocation};` re-exports continue to work unchanged, so any code using the short paths is not affected.

(c) In `src/cuda/inference/fusion/mod.rs`, ensure the pattern types and detection function are at `pub(crate)` visibility so `testing.rs` (which is in the parent `inference` module) can import them:

```rust
pub(crate) mod patterns;
pub(crate) mod detection;

pub(crate) use patterns::{FusedPattern, FusedPatternInfo};
pub(crate) use detection::detect_fused_patterns;
```

Also, ensure `ExecutionNode` and `PrecomputedSlice` remain `pub(crate)` from Task A3 — the `testing` module lives in the same parent, so it can see them.

- [ ] **Step 4: Run the test to confirm it fails at compile time**

Run: `cargo test --features cuda --test fusion_detection_test`

Expected: Compile error. The critical missing symbol is `FusedPattern::MulSinPowMulAdd` (referenced by `is_mul_sin_pow_mul_add`) — the variant doesn't exist yet. This is the red state.

If any other compile errors appear that are unrelated to `MulSinPowMulAdd` (e.g., a missing `PrecomputedSlice` re-export, a visibility issue), fix them in place — only `MulSinPowMulAdd`-related errors should survive.

- [ ] **Step 5: No commit yet — red state**

The next task adds the variant and the detection walker.

---

### Task C2: Add `MulSinPowMulAdd` pattern variant and detection walker

**Files:**
- Modify: `src/cuda/inference/fusion/patterns.rs`
- Modify: `src/cuda/inference/fusion/detection.rs`

- [ ] **Step 1: Add the `MulSinPowMulAdd` enum variant**

In `src/cuda/inference/fusion/patterns.rs`, add a new variant to `FusedPattern` after the last existing variant:

```rust
    /// Mul -> Sin -> Pow -> Mul -> Add: y = sin(x * w0)^p * w1 + b
    /// Vocoder periodic activation (48 occurrences in Kokoro)
    MulSinPowMulAdd {
        x_input: String,
        w0_input: String,
        w1_input: String,
        b_input: String,
        /// Name of the weight tensor holding the scalar exponent.
        p_input: String,
        output_name: String,
    },
```

- [ ] **Step 2: Add the detection walker**

In `src/cuda/inference/fusion/detection.rs`, append a new detection block at the end of the function body, just before the final `(fused_patterns, nodes_to_skip)` return statement. Use the same style as the existing DivRsqrt and GELU walkers:

```rust
    // Detect Mul -> Sin -> Pow -> Mul -> Add pattern
    // (vocoder periodic activation: y = sin(x * w0)^p * w1 + b)
    for node in nodes {
        if node.op_type != "Mul" {
            continue;
        }
        // Skip if already claimed by another pattern
        if nodes_to_skip.contains(&node.name) {
            continue;
        }
        if node.inputs.len() < 2 {
            continue;
        }

        let head_mul_output = &node.outputs[0];
        if output_uses.get(head_mul_output.as_str()) != Some(&1) {
            continue;
        }

        // Step 1: Sin consuming head Mul's output
        let sin_node = output_to_node
            .iter()
            .find(|(_, n)| {
                n.op_type == "Sin"
                    && !nodes_to_skip.contains(&n.name)
                    && n.inputs.contains(head_mul_output)
            })
            .map(|(_, n)| *n);
        let Some(sin_node) = sin_node else { continue };
        let sin_output = &sin_node.outputs[0];
        if output_uses.get(sin_output.as_str()) != Some(&1) {
            continue;
        }

        // Step 2: Pow consuming Sin's output; exponent (second input) must be
        // a scalar constant in the weights map.
        let pow_node = output_to_node
            .iter()
            .find(|(_, n)| {
                n.op_type == "Pow"
                    && !nodes_to_skip.contains(&n.name)
                    && n.inputs.len() >= 2
                    && n.inputs[0] == *sin_output
            })
            .map(|(_, n)| *n);
        let Some(pow_node) = pow_node else { continue };
        let pow_output = &pow_node.outputs[0];
        if output_uses.get(pow_output.as_str()) != Some(&1) {
            continue;
        }
        let p_input = &pow_node.inputs[1];
        // Exponent must be a constant scalar in weights
        let p_is_scalar_weight = weights
            .get(p_input)
            .map(|t| t.len() == 1)
            .unwrap_or(false);
        if !p_is_scalar_weight {
            continue;
        }

        // Step 3: Tail Mul consuming Pow's output. Its other operand is w1
        // and should be a weight (or at least not a computed value produced
        // inside the chain).
        let tail_mul = output_to_node
            .iter()
            .find(|(_, n)| {
                n.op_type == "Mul"
                    && !nodes_to_skip.contains(&n.name)
                    && n.inputs.contains(pow_output)
            })
            .map(|(_, n)| *n);
        let Some(tail_mul_node) = tail_mul else { continue };
        let tail_mul_output = &tail_mul_node.outputs[0];
        if output_uses.get(tail_mul_output.as_str()) != Some(&1) {
            continue;
        }
        let w1_input = if tail_mul_node.inputs[0] == *pow_output {
            tail_mul_node.inputs[1].clone()
        } else {
            tail_mul_node.inputs[0].clone()
        };

        // Step 4: Add consuming tail Mul's output. The other operand is b.
        let bias_add = output_to_node
            .iter()
            .find(|(_, n)| {
                n.op_type == "Add"
                    && !nodes_to_skip.contains(&n.name)
                    && n.inputs.contains(tail_mul_output)
            })
            .map(|(_, n)| *n);
        let Some(bias_add_node) = bias_add else { continue };
        let b_input = if bias_add_node.inputs[0] == *tail_mul_output {
            bias_add_node.inputs[1].clone()
        } else {
            bias_add_node.inputs[0].clone()
        };

        // Identify x and w0 (the two inputs to the head Mul). Both orders
        // are accepted — the kernel is symmetric in x and w0 up to the
        // w0[i] broadcast path, and downstream execution will re-lookup
        // both by name.
        let (x_input, w0_input) = (node.inputs[0].clone(), node.inputs[1].clone());

        // Commit: register the pattern and mark all interior nodes as skip.
        let pattern_info = FusedPatternInfo {
            pattern: FusedPattern::MulSinPowMulAdd {
                x_input,
                w0_input,
                w1_input,
                b_input,
                p_input: p_input.clone(),
                output_name: bias_add_node.outputs[0].clone(),
            },
            nodes_to_skip: vec![
                sin_node.name.clone(),
                pow_node.name.clone(),
                tail_mul_node.name.clone(),
                bias_add_node.name.clone(),
            ],
            head_node: node.name.clone(),
        };

        for skip_name in &pattern_info.nodes_to_skip {
            nodes_to_skip.insert(skip_name.clone());
        }
        fused_patterns.insert(node.name.clone(), pattern_info);
    }

    // Silence unused-variable warnings in test builds where ctx is not read
    // inside this walker (ctx is used by the GELU walker above).
    let _ = ctx;
```

Ignore the final `let _ = ctx;` line if clippy does not complain — it is only there to suppress a warning if the GELU walker is the only other consumer of `ctx`.

- [ ] **Step 3: Run the Phase C tests**

Run: `cargo test --features cuda --test fusion_detection_test`

Expected: Both tests pass:
- `detects_mul_sin_pow_mul_add_chain` — the pattern is registered
- `rejects_when_intermediate_has_multiple_consumers` — the pattern is not registered

If `detects_*` fails because the head Mul is not in the pattern map, check whether the walker is running in the correct order (the new walker must come **after** any walker that could consume `Mul` nodes as another pattern's head, e.g. AddMul). If `rejects_*` fails, the `output_uses.get(...) != Some(&1)` checks are incorrect — the refactored walker must reject any intermediate with use count ≠ 1.

- [ ] **Step 4: Run the full test suite to guard against regressions in other patterns**

Run: `cargo test --features cuda`

Expected: Every pre-existing test still passes. If any fusion-related test regresses (especially GELU, DivRsqrt, AddMulAdd), the new walker is claiming nodes that should have belonged to another pattern — add more specific single-use/skip checks.

- [ ] **Step 5: Clippy**

Run: `cargo clippy --features cuda -- -D warnings`

Expected: Zero warnings. The `let _ = ctx;` line from Step 2 exists only if clippy warns about unused parameters; remove it if not needed.

- [ ] **Step 6: Commit (signed)**

```bash
git add src/cuda/inference/fusion/patterns.rs src/cuda/inference/fusion/detection.rs src/cuda/inference/testing.rs src/cuda/inference/mod.rs tests/fusion_detection_test.rs
git commit -S -m "$(cat <<'EOF'
feat(fusion): detect MulSinPowMulAdd pattern in graph

Adds the MulSinPowMulAdd variant to FusedPattern and a detection
walker that finds the 5-op chain Mul->Sin->Pow->Mul->Add. The walker
verifies:
  - Each intermediate has exactly one consumer (single-use invariant)
  - The Pow exponent is a scalar weight constant
  - No node in the chain is already claimed by another pattern

Also introduces src/cuda/inference/testing.rs as a narrow public
facade so fusion detection can be unit-tested without touching the
executor's internal state. Two synthetic-graph tests confirm the
walker fires on a minimal chain and correctly rejects chains whose
intermediates have multiple consumers.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase D: Dispatch the new pattern from the executor

### Task D1: Add failing end-to-end dispatch test

**Files:**
- Modify: `tests/fusion_detection_test.rs`

- [ ] **Step 1: Append the executor dispatch test**

Append to `tests/fusion_detection_test.rs`:

```rust
#[test]
fn executor_runs_mul_sin_pow_mul_add_and_records_profile_entry() {
    use iconnx::cuda::GpuGraphExecutor;
    use iconnx::tensor::Tensor;
    use iconnx::NodeAttributes;
    use std::collections::HashMap;

    let _ctx = IconnxCudaContext::new().expect("CUDA context");

    // Minimal graph: same 5-op chain, with inputs and a named output.
    // We register x, w0, w1, b, p as model initializers so the executor
    // can look them up. x flows as an input via the input map; the rest
    // are weights.
    let mut executor = GpuGraphExecutor::new().expect("executor");

    let shape = vec![2usize, 4usize];
    let n = 2 * 4;
    let w0_data: Vec<f32> = (0..n).map(|i| 0.5 + 0.05 * i as f32).collect();
    let w1_data: Vec<f32> = (0..n).map(|i| 1.0 - 0.03 * i as f32).collect();
    let b_data: Vec<f32> = (0..n).map(|i| 0.1 * i as f32).collect();

    executor
        .add_initializer(
            "w0".to_string(),
            &Tensor::from_vec_f32(w0_data.clone(), shape.clone()),
        )
        .expect("add w0");
    executor
        .add_initializer(
            "w1".to_string(),
            &Tensor::from_vec_f32(w1_data.clone(), shape.clone()),
        )
        .expect("add w1");
    executor
        .add_initializer(
            "b".to_string(),
            &Tensor::from_vec_f32(b_data.clone(), shape.clone()),
        )
        .expect("add b");
    executor
        .add_initializer(
            "p".to_string(),
            &Tensor::from_vec_f32(vec![2.0], vec![1]),
        )
        .expect("add p");

    executor.add_node(
        "head_mul",
        "Mul",
        vec!["x", "w0"],
        vec!["t0"],
        NodeAttributes::default(),
    );
    executor.add_node(
        "sin_node",
        "Sin",
        vec!["t0"],
        vec!["t1"],
        NodeAttributes::default(),
    );
    executor.add_node(
        "pow_node",
        "Pow",
        vec!["t1", "p"],
        vec!["t2"],
        NodeAttributes::default(),
    );
    executor.add_node(
        "tail_mul",
        "Mul",
        vec!["t2", "w1"],
        vec!["t3"],
        NodeAttributes::default(),
    );
    executor.add_node(
        "bias_add",
        "Add",
        vec!["t3", "b"],
        vec!["y"],
        NodeAttributes::default(),
    );

    let x_data: Vec<f32> = (0..n).map(|i| 0.01 * i as f32 + 0.2).collect();
    let mut inputs = HashMap::new();
    inputs.insert(
        "x".to_string(),
        Tensor::from_vec_f32(x_data.clone(), shape.clone()),
    );

    let (outputs, profile) = executor
        .run_with_profiling(inputs, vec!["y"])
        .expect("run");

    // The profile must contain the new fused pattern entry.
    let entry = profile
        .op_times
        .get("Fused_MulSinPowMulAdd")
        .expect("Fused_MulSinPowMulAdd missing from profile");
    assert_eq!(entry.1, 1, "expected exactly one fused-pattern execution");

    // And the intermediate ops (Sin, Pow, tail Mul, bias Add) must NOT
    // appear in the profile — they were skipped.
    assert!(
        !profile.op_times.contains_key("Sin"),
        "Sin should have been skipped by fusion"
    );
    assert!(
        !profile.op_times.contains_key("Pow"),
        "Pow should have been skipped by fusion"
    );
    assert!(
        !profile.op_times.contains_key("Add"),
        "Add should have been skipped by fusion"
    );
    // Note: Mul might still appear if the head Mul dispatches as Mul
    // rather than as the fused pattern head. In the current design the
    // head Mul *is* the pattern head, so Mul should also be absent.
    assert!(
        !profile.op_times.contains_key("Mul"),
        "Head Mul should have been consumed as pattern head"
    );

    // Numerical check: compute the same thing with plain ndarray and compare.
    let y_tensor = outputs
        .get("y")
        .expect("output y missing")
        .as_slice()
        .to_vec();
    let mut expected = vec![0.0f32; n];
    for i in 0..n {
        let s = (x_data[i] * w0_data[i]).sin();
        expected[i] = s * s * w1_data[i] + b_data[i];
    }
    for i in 0..n {
        let diff = (y_tensor[i] - expected[i]).abs();
        assert!(
            diff <= 1e-5,
            "y[{i}] = {} vs expected {}, diff {}",
            y_tensor[i],
            expected[i],
            diff
        );
    }
}
```

- [ ] **Step 2: Run the test**

Run: `cargo test --features cuda --test fusion_detection_test -- executor_runs_mul_sin_pow_mul_add_and_records_profile_entry`

Expected: Test fails. The failure mode depends on where dispatch breaks first — likely a panic from `match` not being exhaustive on `FusedPattern`, since the new variant has no arm in `execute_fused_pattern` yet.

- [ ] **Step 3: No commit yet**

---

### Task D2: Wire `MulSinPowMulAdd` into executor dispatch

**Files:**
- Modify: `src/cuda/inference/mod.rs`

- [ ] **Step 1: Add the dispatch arm in `execute_fused_pattern`**

In `src/cuda/inference/mod.rs`, find `fn execute_fused_pattern` (around line 1052). Its body is a `match &pattern_info.pattern { ... }` covering every existing variant. Add a new arm at the end, before the closing brace:

```rust
            FusedPattern::MulSinPowMulAdd {
                x_input,
                w0_input,
                w1_input,
                b_input,
                p_input,
                ..
            } => {
                let x = values
                    .get(x_input)
                    .or_else(|| self.weights.get(x_input))
                    .ok_or_else(|| {
                        CudaError::Kernel(format!(
                            "Fused MulSinPowMulAdd: x '{}' not found",
                            x_input
                        ))
                    })?;
                let w0 = values
                    .get(w0_input)
                    .or_else(|| self.weights.get(w0_input))
                    .ok_or_else(|| {
                        CudaError::Kernel(format!(
                            "Fused MulSinPowMulAdd: w0 '{}' not found",
                            w0_input
                        ))
                    })?;
                let w1 = values
                    .get(w1_input)
                    .or_else(|| self.weights.get(w1_input))
                    .ok_or_else(|| {
                        CudaError::Kernel(format!(
                            "Fused MulSinPowMulAdd: w1 '{}' not found",
                            w1_input
                        ))
                    })?;
                let b = values
                    .get(b_input)
                    .or_else(|| self.weights.get(b_input))
                    .ok_or_else(|| {
                        CudaError::Kernel(format!(
                            "Fused MulSinPowMulAdd: b '{}' not found",
                            b_input
                        ))
                    })?;
                let p_tensor = values
                    .get(p_input)
                    .or_else(|| self.weights.get(p_input))
                    .ok_or_else(|| {
                        CudaError::Kernel(format!(
                            "Fused MulSinPowMulAdd: p '{}' not found",
                            p_input
                        ))
                    })?;
                let p = p_tensor.to_host_f32(&self.ctx)?[0];

                gpu_fused_mul_sin_pow_mul_add(
                    &self.ctx,
                    &self.fused_kernels,
                    &mut pool,
                    x,
                    w0,
                    w1,
                    b,
                    p,
                )
            }
```

- [ ] **Step 2: Add the profiling entry**

In the same file, find the match in `run_with_profiling` that selects a profiling label (around line 4656). Add the new arm at the end:

```rust
                    FusedPattern::MulSinPowMulAdd { output_name, .. } => {
                        values.insert(output_name.clone(), output);
                        "Fused_MulSinPowMulAdd"
                    }
```

If there is a second match in `execute_node_fused_or_normal` or a similar function (Grep for `Fused_DivRsqrt` to find all sites — there are four), add the same arm everywhere `FusedPattern::DivRsqrt { output_name, .. } => { ... "Fused_DivRsqrt" }` already appears. Every match on `&FusedPattern` must now be exhaustive.

- [ ] **Step 3: Add the kernel import at the top of `inference/mod.rs`**

Find the existing import block for fused kernels (currently imports `gpu_fused_add_mul`, `gpu_fused_add_mul_add`, `gpu_fused_div_mul`, `gpu_fused_div_rsqrt`, `gpu_fused_gelu`, `gpu_fused_mul_add`, `gpu_fused_sub_mul`, `FusedKernelCache`). Add `gpu_fused_mul_sin_pow_mul_add` to the list in sorted order.

- [ ] **Step 4: Build and run the failing test**

Run: `cargo test --features cuda --test fusion_detection_test -- executor_runs_mul_sin_pow_mul_add_and_records_profile_entry`

Expected: PASS.

Likely first-attempt errors:
- *Non-exhaustive match* — add the arm wherever `match &pattern_info.pattern` or `match ... { FusedPattern::DivRsqrt` appears. There are four sites.
- *Panic in `to_host_f32` on non-Float32 tensor* — the `p` weight may be stored as Float32 in the test's initializer; this should succeed. If real-model initializers use a different type, extend the arm to handle it, but defer that to Phase E if the synthetic test passes.

- [ ] **Step 5: Run the full test suite**

Run: `cargo test --features cuda`

Expected: Full suite green. Every test that ran before Phase D must still pass, plus the two new dispatch tests from this task.

- [ ] **Step 6: Clippy**

Run: `cargo clippy --features cuda -- -D warnings`

Expected: Zero warnings.

- [ ] **Step 7: Commit (signed)**

```bash
git add src/cuda/inference/mod.rs tests/fusion_detection_test.rs
git commit -S -m "$(cat <<'EOF'
feat(inference): dispatch MulSinPowMulAdd fused kernel

Wires the newly-detected MulSinPowMulAdd pattern into the executor's
fused dispatch path. The executor looks up all five tensor inputs by
name, reads the scalar exponent via a single D2H, and calls the
fused kernel. Profiling records the pattern under the label
"Fused_MulSinPowMulAdd".

A synthetic-graph integration test verifies the dispatch path end
to end: it registers the five-node chain, runs inference, and
asserts that (a) Fused_MulSinPowMulAdd appears exactly once in the
profile, (b) Mul/Sin/Pow/Add are absent (skipped), and (c) the
output matches a CPU reference computation to within 1e-5.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase E: Kokoro validation

This phase is gated on an external tool (`leadline-bench`) and a real Kokoro model file. It does not add code; it runs existing tests plus the bench tool and captures results for the commit message. All acceptance criteria from the spec's Validation Gates are checked here.

### Task E1: Run the full test suite and confirm `tests/kokoro_validation_test.rs`

**Files:**
- Read-only

- [ ] **Step 1: Full test suite**

Run: `cargo test --features cuda`

Expected: 100% green.

- [ ] **Step 2: No-cuda build/test**

Run: `cargo test --no-default-features`

Expected: Compiles and tests pass. CPU fallback is unaffected. If the `pub mod testing;` line introduces a `cuda`-only type into a no-cuda build, gate the module declaration on `#[cfg(feature = "cuda")]`.

- [ ] **Step 3: Kokoro numerical validation**

Run: `cargo test --features cuda --test kokoro_validation_test`

Expected: PASS. This test uses `ort` as ground truth. If numerical divergence is detected, **stop** — the new fusion is producing incorrect outputs. Common causes:
- Detection is matching a pattern the original graph did not intend (for example, because operand order differs from the assumption)
- `w0` is actually a computed value rather than an input — the kernel's `w0[i]` index is wrong
- The exponent is not an integer in the real model and the float path was needed but not exercised

Investigation guide if this fails:
1. Add temporary eprintln in the dispatch arm recording `x.shape()`, `w0.shape()`, etc. on every fused call
2. Check whether `w0` has the same shape as `x` or is broadcast — if broadcast, verify the broadcast stride calculation
3. Check the value of `p` — if it is not an integer, the test's `p_is_int` flag is wrong

- [ ] **Step 4: Clippy**

Run: `cargo clippy --features cuda -- -D warnings`

Expected: Zero warnings.

No commit in this task.

---

### Task E2: Switch leadline-bench to a local path dep and run comparison at seq_len=5

**Files:**
- Modify: `~/workspace/ryanoneill/rust-ai-explorations/leadline-bench/Cargo.toml` (temporary)

- [ ] **Step 1: Switch `leadline-bench` to a local path dep**

Edit `~/workspace/ryanoneill/rust-ai-explorations/leadline-bench/Cargo.toml`. Find the `iconnx = { git = ..., branch = "main" }` line and replace it with:

```toml
iconnx = { path = "../../iconnx" }
```

Save the file. This allows benchmark iteration without pushing.

- [ ] **Step 2: Run the comparison at seq_len=5**

```bash
cd ~/workspace/ryanoneill/rust-ai-explorations
cargo run -p leadline-bench --bin compare --release -- \
  ~/workspace/ryanoneill/claudio/kokoro-v1.0.onnx \
  --seq-len 5 --iterations 10 --output-dir /tmp
```

Expected output (captured in a temporary note):
- `iconnx: avg ___ ms` (record it)
- `ORT: avg ___ ms` (record it)
- `Speedup: ___ x` (record it)

Then read `/tmp/iconnx.leadline` per the Leadline bench's instructions and extract the per-operator breakdown. Record:
- `Fused_MulSinPowMulAdd` total ms and call count
- `Mul`, `Sin`, `Pow`, `Add` call counts (expect each to drop by ~48 vs the pre-cycle baseline of 267/48/50/352 respectively)
- Variance

Acceptance criteria at seq_len=5:
1. `Fused_MulSinPowMulAdd` appears with 48 ± 5 calls
2. iconnx total latency dropped by ≥ 2 ms vs the 142.65 ms baseline
3. Variance is not worse than 46%

If any criterion fails, do not proceed to the next sequence length — investigate per the spec's Risks section.

- [ ] **Step 3: Capture memory pool stats at seq_len=5**

Leadline-bench records memory pool statistics if the executor is created with profiling. If the default `compare` binary does not already print `pool_stats()`, temporarily add a call to `executor.pool_stats()` before the executor is dropped and print the result. Record:
- Total allocations per inference (expected: ~192 fewer than baseline)
- Pool hit rate (must not regress)

Alternative: if modifying leadline-bench is too invasive, instead add a one-off print in a local iconnx test that loads Kokoro and runs a single inference, printing `pool_stats()` before and after the first few runs.

- [ ] **Step 4: Keep the leadline-bench `Cargo.toml` modified**

Do not revert the path-dep edit yet — Task E3 uses it again. It reverts in Task E5.

No commit in this task (no iconnx code changed).

---

### Task E3: Run comparison at seq_len=50

**Files:**
- Read-only

- [ ] **Step 1: Run at seq_len=50**

```bash
cd ~/workspace/ryanoneill/rust-ai-explorations
cargo run -p leadline-bench --bin compare --release -- \
  ~/workspace/ryanoneill/claudio/kokoro-v1.0.onnx \
  --seq-len 50 --iterations 10 --output-dir /tmp
```

Record the same data points as Task E2 step 2.

Acceptance criteria at seq_len=50:
1. `Fused_MulSinPowMulAdd` appears with 48 ± 5 calls
2. iconnx total latency dropped by ≥ 10 ms vs the 315.58 ms baseline (stretch: ≥ 25 ms)
3. The absolute improvement at seq_len=50 should be larger than at seq_len=5 — this is the spec's expectation (fusion savings scale with tensor size). If not, investigate.

- [ ] **Step 2: Capture memory pool stats at seq_len=50**

Same procedure as Task E2 step 3. Record allocations per inference and hit rate.

- [ ] **Step 3: Stretch goal — attempt seq_len=200**

```bash
cargo run -p leadline-bench --bin compare --release -- \
  ~/workspace/ryanoneill/claudio/kokoro-v1.0.onnx \
  --seq-len 200 --iterations 3 --output-dir /tmp
```

One of three outcomes is expected:
- **OOM again** (most likely after just one fusion pattern) — note this for the commit message. Continue.
- **No longer OOMs** — strong positive signal. Record the first-run latency.
- **Partial progress** (got further before OOM than the baseline did) — note the new failure point.

No criterion is required here; this is pure observation.

No commit in this task.

---

### Task E4: Re-validate detection if any acceptance criterion failed

**Files:**
- Modify (conditionally): `src/cuda/inference/fusion/detection.rs`

- [ ] **Step 1: Decide whether re-validation is needed**

If Tasks E2 and E3 both hit their acceptance criteria, skip this task.

If the `Fused_MulSinPowMulAdd` call count is noticeably lower than 48 (e.g., fewer than 40), the detection walker is not matching the real Kokoro pattern shape. Proceed with the following investigation.

- [ ] **Step 2: Log near-match candidates**

Add a one-off eprintln near the top of the `MulSinPowMulAdd` detection loop in `src/cuda/inference/fusion/detection.rs`:

```rust
if node.op_type == "Mul" && !nodes_to_skip.contains(&node.name) {
    if let Some(sin_candidate) = output_to_node.values().find(|n| {
        n.op_type == "Sin" && n.inputs.contains(&node.outputs[0])
    }) {
        eprintln!(
            "[detect] Mul->Sin candidate at {} -> {} (inputs {:?})",
            node.name, sin_candidate.name, node.inputs
        );
    }
}
```

Run Kokoro inference once (for example via `cargo test --features cuda --test kokoro_inference_test`) and observe the log. Count how many candidates are logged but then rejected further down the walker. The rejection reason (where the walker `continue`s) points at which assumption is wrong.

- [ ] **Step 3: Adjust the walker**

Common likely fixes:
- The `Pow` exponent is a **Constant node** output, not a weight. Fix: extend `p_is_scalar_weight` check to also match the static `Int64`/`Float32` constant cache or `values` map. Simplest approach: also accept `p_input` that corresponds to an output of a `Constant` node whose tensor has length 1.
- The tail Mul's operand order is swapped. Verify the `if tail_mul_node.inputs[0] == *pow_output` branch covers both orders.
- The Add's operand order is swapped. Same fix.
- There is an intermediate Cast or Reshape in the chain. If so, extend the walker to skip a single Cast/Reshape between Sin and Pow.

Make the minimal fix, re-run the bench at seq_len=5, and confirm the call count is now ~48.

- [ ] **Step 4: Remove the diagnostic eprintln**

Do not commit the temporary eprintln. Delete it before committing the walker fix.

- [ ] **Step 5: Commit the walker fix (only if changes were made)**

```bash
git add src/cuda/inference/fusion/detection.rs
git commit -S -m "$(cat <<'EOF'
fix(fusion): match real Kokoro MulSinPowMulAdd pattern shape

Adjusts the detection walker to accept <describe what was wrong>
in the real Kokoro graph. After this fix, leadline-bench at
seq_len=5 records <N> instances of Fused_MulSinPowMulAdd where N
was previously ~<M>.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

The placeholder `<describe what was wrong>` must be filled in with the actual finding before running this commit command; the engineer does not commit a message with `<...>` placeholders.

---

### Task E5: Revert the leadline-bench path-dep and commit cycle completion

**Files:**
- Modify: `~/workspace/ryanoneill/rust-ai-explorations/leadline-bench/Cargo.toml`
- No iconnx code changes

- [ ] **Step 1: Revert the leadline-bench path-dep to a git dep**

Edit `~/workspace/ryanoneill/rust-ai-explorations/leadline-bench/Cargo.toml`, replace:

```toml
iconnx = { path = "../../iconnx" }
```

with the original (adjust the git URL to match what was there before):

```toml
iconnx = { git = "https://github.com/ryanoneill/iconnx.git", branch = "main" }
```

Do not commit this change in the leadline-bench repo as part of iconnx's work — it belongs to the other repo and should only be committed there if it was actually changed. If the engineer originally made the path-dep change in a local unstaged edit, reverting cleanly restores the baseline.

- [ ] **Step 2: Write a completion summary for the Phase E commit message**

Compose a summary capturing, from the bench runs in tasks E2 and E3:

- seq_len=5: iconnx avg before / after, delta, Fused_MulSinPowMulAdd call count, Mul/Sin/Pow/Add deltas
- seq_len=50: same fields
- seq_len=200: OOM status (worse / same / fits now)
- Pool stats: allocations delta, hit rate before/after
- Commits landed in this cycle: list the hashes

- [ ] **Step 3: Commit the summary as a docs entry (optional but recommended)**

Write the summary to `docs/performance/2026-04-10-cycle1-mul-sin-pow-mul-add-results.md` using whatever format the existing `docs/performance/` file uses. Include the actual numbers from the benchmark — never the projected numbers from the spec.

```bash
git add docs/performance/2026-04-10-cycle1-mul-sin-pow-mul-add-results.md
git commit -S -m "$(cat <<'EOF'
perf(kokoro): Cycle 1 results — MulSinPowMulAdd fusion validated

See docs/performance/2026-04-10-cycle1-mul-sin-pow-mul-add-results.md
for full numbers. Headline:

- seq_len=5:  iconnx <BEFORE> -> <AFTER> (delta <D> ms)
- seq_len=50: iconnx <BEFORE> -> <AFTER> (delta <D> ms)
- Fused_MulSinPowMulAdd fires <N> times per inference
- Pool allocations reduced by <A> per inference
- seq_len=200: <status>

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

Fill in the placeholders with real values from the measurements. Do not commit with `<...>` placeholders.

---

## Final Validation Gates

Before declaring Cycle 1 complete, confirm every Validation Gate from the spec:

1. [ ] `cargo test --features cuda` passes
2. [ ] `cargo test` (no-default-features) passes
3. [ ] `cargo clippy --features cuda -- -D warnings` — zero warnings
4. [ ] `cargo fmt --check` — clean
5. [ ] `tests/kokoro_validation_test.rs` passes
6. [ ] `leadline-bench` at seq_len=5 and seq_len=50 shows:
   - `Fused_MulSinPowMulAdd` with ~48 calls at each length
   - `Mul`, `Sin`, `Pow`, `Add` counts each decreased by ~48
   - Latency improved ≥2 ms at seq_len=5 and ≥10 ms at seq_len=50
   - Variance not worse than baseline
7. [ ] Memory pool stats: allocations per inference dropped by ~192; hit rate not regressed; seq_len=200 status reported
8. [ ] `src/cuda/inference/mod.rs` is ~300 lines smaller than at cycle start
9. [ ] `src/cuda/inference/fusion/` exists with `patterns.rs`, `detection.rs`, `mod.rs`, each under 1000 lines
10. [ ] `src/cuda/kernels/fused.rs` remains under 1000 lines
11. [ ] Latency + memory numbers reported in the Phase E commit message
12. [ ] All commits signed (`git log --show-signature` shows `Good signature` for every new commit)

Run `wc -l src/cuda/inference/mod.rs src/cuda/inference/fusion/*.rs src/cuda/kernels/fused.rs` to confirm file sizes at the end of the cycle.

---

## Review Checklist (self-review at end)

When Cycle 1 is complete, walk through these questions before the next cycle starts:

- Did the fusion eliminate the expected launches at both seq_len=5 and seq_len=50?
- Did the latency improvement scale with tensor size (bigger win at seq_len=50 than at seq_len=5)? If not, the kernel may not actually be keeping intermediates in registers.
- Is the `fusion/` module's shape clean enough that Cycle 2 (`AddMulAddLeakyRelu`) and Cycle 3 (`AddMulAdd` detection fix) can land with minimal friction?
- Are there any lingering `#[allow(dead_code)]` or unused imports introduced during the refactor? Clean them up in a small follow-up commit if so.
