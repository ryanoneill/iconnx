# Clippy Baseline Cleanup — Implementation Plan

> **For agentic workers:** This plan is designed for inline execution by the controller (me). The steps are small and mechanical enough that dispatching a subagent per cluster adds more overhead than it saves. Use the checkbox (`- [ ]`) syntax to track progress.

**Goal:** Drive `cargo clippy --features cuda --tests -- -D warnings` to zero errors without changing runtime behavior.

**Architecture:** Five focused signed commits, each handling one logical error cluster. Full test gate after each commit. Strict monotonic decrease in clippy error count.

**Tech Stack:** Rust / cargo clippy / cargo test, CUDA-enabled build.

---

## Baseline snapshot (before any work)

- Baseline: 92 clippy errors (`cargo clippy --features cuda --tests -- -D warnings`)
- Test suite: 357 passed, 0 failed, 141 ignored
- Signed-commit git config active
- Branch: create new `cleanup/clippy-baseline-errors` in worktree `.worktrees/cleanup-clippy-baseline`

## Task 0: Worktree setup

**Files:** none

- [ ] **Step 1: Verify primary worktree is on main and clean**

```bash
git -C /home/ryano/workspace/ryanoneill/iconnx branch --show-current
git -C /home/ryano/workspace/ryanoneill/iconnx status
```

Expected: `main` and clean working tree.

- [ ] **Step 2: Create the worktree and branch**

```bash
git -C /home/ryano/workspace/ryanoneill/iconnx worktree add \
    .worktrees/cleanup-clippy-baseline \
    -b cleanup/clippy-baseline-errors
```

Expected: worktree created at `.worktrees/cleanup-clippy-baseline`, HEAD of new branch matches main.

- [ ] **Step 3: Build & baseline test gate**

```bash
cd /home/ryano/workspace/ryanoneill/iconnx/.worktrees/cleanup-clippy-baseline
cargo build --features cuda --tests
cargo test --features cuda 2>&1 | grep -E "^test result" | \
    awk '{total_pass+=$4; total_fail+=$6; total_ignored+=$8} \
         END {printf "Totals: %d passed, %d failed, %d ignored\n", total_pass, total_fail, total_ignored}'
```

Expected: `Totals: 357 passed, 0 failed, 141 ignored`.

- [ ] **Step 4: Baseline clippy count**

```bash
cargo clippy --features cuda --tests -- -D warnings 2>&1 | \
    grep -E "^error:" | grep -v "could not compile\|aborting" | wc -l
```

Expected: `92`.

---

## Task 1: Float precision cleanup (37 errors)

**Expected post-commit clippy count:** 55

**Files (to be confirmed by clippy output):**
- `src/cuda/conv/forward.rs`
- `src/cuda/conv/transpose.rs`
- `src/cuda/matmul.rs`
- `src/operators/conv.rs`
- `src/operators/conv_transpose.rs`
- `src/operators/resize.rs`
- Test files under `tests/operators/` that use float literals
- Any other locations clippy flags

**Approach:** For each `float has excessive precision` site, take clippy's truncation suggestion verbatim. The suggestion is guaranteed to be the same f32 bit pattern as the original over-precise literal. Verify the comment or surrounding context doesn't name the full-precision value (e.g., "Apply exact PI/2") — if it does, the site should move to Task 2 (math constants).

- [ ] **Step 1: Enumerate all "excessive precision" sites**

```bash
cargo clippy --features cuda --tests 2>&1 | \
    grep -B 3 "float has excessive precision" | \
    grep -E "^\s*-->" | sort -u
```

Record the full list.

- [ ] **Step 2: For each site, apply the truncation**

Read the file, locate the exact literal, replace with the truncation clippy suggested. Preserve any surrounding math-intent comments.

**DO NOT use `cargo clippy --fix`** — its blanket application doesn't leave a reviewable diff; manual per-site editing is required.

- [ ] **Step 3: Build**

```bash
cargo build --features cuda --tests
```

Expected: clean.

- [ ] **Step 4: Test**

```bash
cargo test --features cuda 2>&1 | grep -E "^test result" | \
    awk '{total_pass+=$4; total_fail+=$6; total_ignored+=$8} \
         END {printf "Totals: %d passed, %d failed, %d ignored\n", total_pass, total_fail, total_ignored}'
```

Expected: `Totals: 357 passed, 0 failed, 141 ignored`.

- [ ] **Step 5: Clippy progress check**

```bash
cargo clippy --features cuda --tests -- -D warnings 2>&1 | \
    grep -E "^error:" | grep -v "could not compile\|aborting" | wc -l
```

Expected: `55` (or fewer, if some float sites overlapped with other lints).

- [ ] **Step 6: Commit signed**

```bash
git add -u
git commit -S -m "$(cat <<'EOF'
cleanup(clippy): truncate over-precise f32 literals

Clippy's `excessive_precision` lint flags f32 literals that
specify more digits than f32 can actually represent. The
truncated values have identical bit patterns when coerced to
f32, so no semantic change.

Part 1/5 of the clippy baseline cleanup cycle. Brings clippy
--tests error count from 92 to 55.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Math constants (9 errors)

**Expected post-commit clippy count:** 46

**Approach:** Replace literal approximations of math constants with named constants from `std::f{32,64}::consts`. The lint only fires when the literal matches the full-precision constant closely enough; for each site, determine if the literal's f32/f64 variant is appropriate.

- [ ] **Step 1: Enumerate all math-constant sites**

```bash
cargo clippy --features cuda --tests 2>&1 | \
    grep -E "approximate value of" -B 3 | \
    grep -E "^\s*-->" | sort -u
```

- [ ] **Step 2: For each site, determine the target constant**

- `PI` → `std::f32::consts::PI` or `std::f64::consts::PI`
- `FRAC_PI_4` → `std::f32::consts::FRAC_PI_4` or `std::f64::consts::FRAC_PI_4`
- `E` → `std::f32::consts::E` or `std::f64::consts::E`

Determine `f32` vs. `f64` by the context's target type.

- [ ] **Step 3: Replace each literal**

Be careful with surrounding expressions:
- `2.0 * 3.14159265f32` → `2.0 * std::f32::consts::PI` (NOT `std::f32::consts::TAU` unless semantically correct)
- `PI / 4.0` style (already using a name) → skip; the lint won't fire
- If the original literal is slightly OFF from the constant (e.g., rounded to fewer digits), verify it's close enough that the target was truly the constant. If intentionally different, mark with `#[allow(clippy::approx_constant)]` + comment.

- [ ] **Step 4: Build**

```bash
cargo build --features cuda --tests
```

- [ ] **Step 5: Test**

```bash
cargo test --features cuda 2>&1 | grep -E "^test result" | \
    awk '{total_pass+=$4; total_fail+=$6; total_ignored+=$8} \
         END {printf "Totals: %d passed, %d failed, %d ignored\n", total_pass, total_fail, total_ignored}'
```

Expected: `Totals: 357 passed, 0 failed, 141 ignored`.

- [ ] **Step 6: Clippy progress**

```bash
cargo clippy --features cuda --tests -- -D warnings 2>&1 | \
    grep -E "^error:" | grep -v "could not compile\|aborting" | wc -l
```

Expected: `46`.

- [ ] **Step 7: Commit signed**

```bash
git add -u
git commit -S -m "$(cat <<'EOF'
cleanup(clippy): use std::f{32,64}::consts for PI/FRAC_PI_4/E

Replaces literal approximations (3.14159265f32, 0.78539816f32,
2.71828f32, etc.) with named constants from std::f32::consts
and std::f64::consts. No behavior change — the named constants
resolve to the same bit patterns.

Part 2/5 of the clippy baseline cleanup cycle. Brings clippy
--tests error count from 55 to 46.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Collection/comparison idioms (~30 errors)

**Expected post-commit clippy count:** ~16

**Sublints addressed:**
- `useless_vec` (22) — `vec![a, b, c]` → `[a, b, c]` where the slice works
- `len_zero` (4) — `x.len() == 0` → `x.is_empty()`
- `slice_from_ref` (2) — `[x.clone()]` → `std::slice::from_ref(&x)` where appropriate
- `collapsible_else_if` (1) — `else { if .. } { .. }` → `else if .. { .. }`
- `duplicated_attributes` (1) — remove duplicate `#[attr]`

**Approach:** Per-file manual review. These are truly mechanical but `cargo clippy --fix` can touch too much at once. Go file by file, apply the fix, verify the diff is minimal.

- [ ] **Step 1: Enumerate affected files**

```bash
cargo clippy --features cuda --tests 2>&1 | \
    grep -B 3 -E "useless use of |length comparison to zero|slice::from_ref|collapsible_else_if|duplicated attribute" | \
    grep -E "^\s*-->" | awk '{print $2}' | awk -F: '{print $1}' | sort -u
```

- [ ] **Step 2: For each file, apply the specific fixes clippy pointed to**

Read each file, locate each offending span, fix per clippy's suggestion. Do NOT use `clippy --fix` on the whole workspace.

- [ ] **Step 3: Build + Test + Clippy progress**

```bash
cargo build --features cuda --tests && \
cargo test --features cuda 2>&1 | grep -E "^test result" | \
    awk '{total_pass+=$4; total_fail+=$6; total_ignored+=$8} \
         END {printf "Totals: %d passed, %d failed, %d ignored\n", total_pass, total_fail, total_ignored}' && \
cargo clippy --features cuda --tests -- -D warnings 2>&1 | \
    grep -E "^error:" | grep -v "could not compile\|aborting" | wc -l
```

Expected: `357 passed, 0 failed, 141 ignored` and clippy count `~16`.

- [ ] **Step 4: Commit signed**

```bash
git add -u
git commit -S -m "$(cat <<'EOF'
cleanup(clippy): collection/comparison idiom fixes

Mechanical clippy fixes:
  - useless_vec: vec![...] -> [...] where a slice suffices
  - len_zero: x.len() == 0 -> x.is_empty()
  - slice_from_ref: [x.clone()] -> slice::from_ref(&x)
  - collapsible_else_if: nested -> flat
  - duplicated_attributes: remove duplicate attribute

No behavior change.

Part 3/5 of the clippy baseline cleanup cycle. Brings clippy
--tests error count from 46 to ~16.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Iterator conversions (7 errors)

**Expected post-commit clippy count:** ~9

**Sublint:** `needless_range_loop` — `for i in 0..n { x[i] = ... }` → iterator method.

**Approach:** Per-site review. Cannot blindly apply clippy's suggestion because some loops have side effects, bounds-checking intent, or early-exit conditions that the iterator form would silently alter. For each site, read the loop body and decide:

1. **Simple index-into-one-slice:** Use `.iter_mut().enumerate()`.
2. **Index-into-two-slices-parallel:** Use `.iter_mut().zip(other)`.
3. **Index with computed offset** (e.g., `x[i + stride]`): Preserve the index form with `#[allow(clippy::needless_range_loop)]` + a comment explaining why the iterator form would be less clear.
4. **Loop has `continue`/`break` with complex conditions:** Same — preserve with `#[allow]`.

- [ ] **Step 1: Enumerate sites**

```bash
cargo clippy --features cuda --tests 2>&1 | \
    grep -B 3 "the loop variable" | \
    grep -E "^\s*-->" | sort -u
```

- [ ] **Step 2: Per-site review and conversion**

For each site: read the surrounding context, pick case 1/2/3/4 above, apply the fix. If using `#[allow]`, include a brief comment.

- [ ] **Step 3: Build + Test + Clippy progress**

```bash
cargo build --features cuda --tests && \
cargo test --features cuda 2>&1 | grep -E "^test result" | \
    awk '{total_pass+=$4; total_fail+=$6; total_ignored+=$8} \
         END {printf "Totals: %d passed, %d failed, %d ignored\n", total_pass, total_fail, total_ignored}' && \
cargo clippy --features cuda --tests -- -D warnings 2>&1 | \
    grep -E "^error:" | grep -v "could not compile\|aborting" | wc -l
```

Expected: `357 passed, 0 failed, 141 ignored` and clippy count `~9`.

- [ ] **Step 4: Commit signed**

```bash
git add -u
git commit -S -m "$(cat <<'EOF'
cleanup(clippy): iterator conversions for needless_range_loop

Converts index-based loops to iterator methods where the
transformation preserves bounds semantics and error paths.
Sites where the index form is load-bearing (computed offsets,
complex early-exit conditions, etc.) are preserved with a
targeted #[allow(clippy::needless_range_loop)] + comment.

No behavior change.

Part 4/5 of the clippy baseline cleanup cycle. Brings clippy
--tests error count from ~16 to ~9.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: No-effect + dead code (7 errors)

**Expected post-commit clippy count:** 0

**Sublints addressed:**
- `no_effect` (4) — `x * 1.0`, `x + 0.0`, etc.
- `dead_code` (3) — unused test helper functions

**Approach (no-effect):** Read each site. If the no-op term is mathematically meaningful (conv formula preserving a conceptual parameter that happens to be zero), delete it AND preserve the intent in a comment. If it's a stale copy-paste, delete it.

**Approach (dead code):** For each unused function:
1. `grep -rn "<function_name>" src/ tests/` — check for any cfg-gated caller
2. If truly unused → delete
3. If cfg-gated caller exists → fix the cfg gate and leave the function
4. If uncertain → `#[allow(dead_code)]` + `// TODO(cleanup): check whether <function_name> has a valid caller` and call out in commit message

- [ ] **Step 1: Enumerate no-effect sites**

```bash
cargo clippy --features cuda --tests 2>&1 | \
    grep -B 3 "operation has no effect" | \
    grep -E "^\s*-->" | sort -u
```

- [ ] **Step 2: For each no-effect site, read and decide**

Delete if stale, preserve with comment if intentional.

- [ ] **Step 3: Handle dead code**

```bash
cargo clippy --features cuda --tests 2>&1 | \
    grep "is never used" | sort -u
```

For each: grep for callers, decide per the approach above.

- [ ] **Step 4: Build + Test**

```bash
cargo build --features cuda --tests && \
cargo test --features cuda 2>&1 | grep -E "^test result" | \
    awk '{total_pass+=$4; total_fail+=$6; total_ignored+=$8} \
         END {printf "Totals: %d passed, %d failed, %d ignored\n", total_pass, total_fail, total_ignored}'
```

Expected: `357 passed, 0 failed, 141 ignored`.

- [ ] **Step 5: Final clippy gate — zero errors**

```bash
cargo clippy --features cuda --tests -- -D warnings 2>&1 | \
    grep -E "^error:" | grep -v "could not compile\|aborting" | wc -l
```

Expected: `0`.

If any errors remain, investigate and resolve (or justify with `#[allow]` + comment) before committing.

- [ ] **Step 6: Commit signed**

```bash
git add -u
git commit -S -m "$(cat <<'EOF'
cleanup(clippy): remove no-effect ops and unused test helpers

Handles the final cluster of baseline clippy errors:
  - no_effect: delete stale x * 1.0 / x + 0.0 style ops;
    preserve any mathematically meaningful zero terms with
    an explanatory comment.
  - dead_code: delete test helper functions (test_inputs,
    load_fixture_inputs, setup_kokoro_executor) after grepping
    the codebase to confirm no cfg-gated callers.

Final part (5/5) of the clippy baseline cleanup cycle. Brings
clippy --tests error count from ~9 to 0.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Results doc

**Files:**
- Create: `docs/performance/2026-04-11-clippy-baseline-cleanup-results.md`

- [ ] **Step 1: Write the results doc**

Template:

```markdown
# Clippy Baseline Cleanup — Results

**Date:** 2026-04-11
**Spec:** [docs/superpowers/specs/2026-04-11-clippy-baseline-cleanup-design.md](../superpowers/specs/2026-04-11-clippy-baseline-cleanup-design.md)
**Plan:** [docs/superpowers/plans/2026-04-11-clippy-baseline-cleanup.md](../superpowers/plans/2026-04-11-clippy-baseline-cleanup.md)
**Branch:** `cleanup/clippy-baseline-errors` (5 signed commits + 1 docs commit)

## Headline

`cargo clippy --features cuda --tests -- -D warnings` now passes clean. 92
baseline errors resolved across ~20 files in 5 focused commits, no behavior
changes. Future cycles can use the stricter `--tests` clippy gate.

## What was done

<fill in per-commit summary with SHAs and error deltas>

## Validation

- `cargo test --features cuda`: <counts>
- `cargo clippy --features cuda --tests -- -D warnings`: 0 errors
- All 5 commits signed

## Follow-ups

- Update CI / local clippy gate to include `--tests` so future cycles catch
  baseline debt at the time of introduction
- Cycle 4 (dynamic AddMulAdd kernel) is now unblocked
```

- [ ] **Step 2: Commit signed**

```bash
git add docs/performance/2026-04-11-clippy-baseline-cleanup-results.md
git commit -S -m "..."
```

---

## Final validation gates

Before declaring complete:

- [ ] `cargo test --features cuda` — 357 passed, 0 failed, 141 ignored
- [ ] `cargo clippy --features cuda --tests -- -D warnings` — 0 errors
- [ ] `cargo clippy --features cuda -- -D warnings` (existing gate) — still clean
- [ ] All 6 commits signed (`git log --show-signature`)
- [ ] Worktree still on `cleanup/clippy-baseline-errors` branch (not detached)
- [ ] `git status` clean
- [ ] Results doc present

## Self-review

- Did any commit make a behavior change? (Should be zero.)
- Are all `#[allow]` suppressions accompanied by an explanatory comment?
- Does the results doc list real commit SHAs and real error deltas?
- Are the commit messages clear about what changed and why?
