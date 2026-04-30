# WS-5 Observability & Conformance — Design Spec

**Status:** Sections 1-6 reviewed and approved by leadline. Ready for plan-writing.
**Worktree:** `iconnx-ws5` on branch `ws5-observability` (off `main` at `231b47e`)
**Authors:** iconnx (Claude/Ryan) + leadline review

---

## Section 1 — Tracking Surface & Scope

**Spec-as-tracker.** WS-5 is tracked in this document. Brainstorming Q1-Q4 decisions are baked in; no design re-litigation in plan or implementation.

### In scope (5 milestones)

- **M5.1 — `validate_model()` public API.** Returns `Result<ModelCapabilities, ModelValidationFailure>`. Collect-all errors via `Vec<ModelIncompatibility>` + `Option<ModelCapabilities>` partial. Surface fields: `opset_imports`, `supported_ops` (used ∩ implemented), `used_dtypes`, `initializer_count`, `initializer_bytes`, `estimated_peak_bytes` (rough heuristic, not arena-precise; formula documented in `capabilities.rs` rustdoc), `expected_tolerance` (`ToleranceHint` with `Measured | Estimated | FallbackHeuristic` confidence + `SamePrecision | CrossPrecisionDepthScaled` basis), `warnings: Vec<ModelWarning>` (incl. `NonFiniteHalfConstants` from WS-3.5 hotfix). `#[non_exhaustive]` on all structs and enums; pre-1.0 churn-policy documented in rustdoc.

- **M5.2 — Opset range validation.** Static `MIN_OPSET_SUPPORTED` / `MAX_OPSET_SUPPORTED` constants derived empirically from the leadline-bench roster. Conservative: `MAX = roster_max`, NOT `roster_max + headroom`. Rustdoc encodes derivation policy: "Models with opset_imports outside this range may parse and run correctly, but iconnx does not claim validated semantics. To extend the range: add a roster model exercising the new opset and re-derive." Per-op support is checked separately by walking the graph against `supported_ops()` from the per-op metadata table.

- **M5.3 — ONNX conformance test subset.** Vendored `tests/conformance_data/onnx-1.<N>/` (version pinned in directory name; `node/` subtree only). Auto-skip via `OnnxParser` op_type extraction + `op_support::lookup(op, ctx)` helper. Single-file TOML allowlist (`tests/conformance_known_failures.toml`) with `dispatch: "cpu" | "gpu" | "both"` per entry (the `"cpu"` and `"both"` entries become inert under the WS-5 ParserOnlyCi reshape but remain as forward-compatible vocabulary for a future CPU executor workstream). Per-test the harness iterates **all** `test_data_set_N/` subdirectories (richer coverage than `_0`-only). Skipped-op summary at end-of-run for coverage prioritization. Per-dtype tolerances tighter than roster: FP32 `1e-4` abs, FP16 `1e-2` abs / `0.5%` rel, BF16 `5e-2` abs / `1%` rel, integer/Bool bit-exact. Separate `cargo test --test onnx_conformance` target. **CI runs `ParserOnlyCi` context (model-must-parse + op-must-be-in-`OP_SUPPORT_TABLE`); developer pre-merge runs `Gpu` context for full execution conformance (mirrors roster sweep discipline).**

- **M5.4 — Differential testing framework extension.** Generalize `cuda/differential.rs` to public `DifferentialRunner` API. `ReferenceSource::FixtureDir | InMemory` two-mode split. `output_names: &[&str]` (empty defaults to all graph outputs). `with_op_chain_context_depth(usize)` builder method (default 5). Shared `DifferentialTolerance` type living at top-level `src/tolerance.rs` (always-available); leadline-bench imports it directly or provides its own conversion. `DivergenceReport` includes node name, exceeded tolerance, diff magnitude, sample inputs (truncated), surrounding op chain context (parent op names, configurable depth). Explicit NaN/Inf forced-divergence locked as regression test, mirroring leadline-bench's `nan_in_iconnx_output_forces_infinite_diff` pattern. `debug-inference` feature gating preserved for the heavy machinery (CheckpointManager, runner); the tolerance type is always-available. **`InMemory` mode provides output-level divergence detection only**; per-node bisect requires `FixtureDir` mode (or a future leadline-bench addition that captures intermediate ORT activations — deferred).

- **M5.5 — Structured error taxonomy.** Subsumes WS-1 M1.6 — implement once, satisfy both. Top-level `IconnxError` enum at `src/errors.rs` wraps `ParseError`, `CudaError`, `ModelValidationFailure`, `DifferentialError`, `std::io::Error`. `thiserror::Error` consistent. Source chains preserved via `#[from]`. Public re-exports from crate root. `iconnx::Result<T>` becomes consumer-facing default. Soft cap at ~300 lines; promote to `errors/mod.rs` with submodules if exceeded.

### Out of scope (carry-forward bucket)

All previously-tracked cross-workstream Follow-Ups (kernel-file unification, garboard `gemm_strided_batched_*`, DistilBERT-INT8 long-seq noise investigation, Kokoro shape mismatch, line-count debt architecture-pass, WS-1 M1.5 shape-inference fall-through, WS-4.5 static quantization QLinear family) remain tracked in their original locations. **WS-5 neither adds nor closes any of these.** New WS-5-specific Follow-Ups (if any surface) get their own section at the bottom of this spec.

**Specifically excluded from M5.3:** No CPU executor lands in WS-5. iconnx is GPU-only via `cuda/`. Adding CPU op kernels (host-side execution path, broadcasting rules, dispatch infrastructure) is a future workstream sized for its own design. The conformance harness's CI gate is parser-only-recognition for WS-5.

### Success criteria

1. **`validate_model()` returns `Ok(ModelCapabilities)` for every roster profile that runs iconnx to completion (which is all 10).** When `validate_model()` returns `Err(ModelValidationFailure)`, `compare` would also fail to run iconnx — the predicate is consistency-of-prediction with `compare`'s actual `Ok/Err` outcome from `run_iconnx`, not with `compare`'s tolerance-gate result. (Roster-scope verification per `feedback_plan_oversight_roster_scope.md`.)
2. Opset range bounds match the roster's actual `opset_import` distribution; rustdoc states the derivation policy.
3. Conformance harness lands with empirical pass/skip/known-failure counts per dispatch context (ParserOnlyCi/Gpu); allowlist is non-empty but each entry is documented with reason + tracked issue/follow-up.
4. `DifferentialRunner` API is consumer-ready; leadline-bench `--bisect` integrates without iconnx-side adapters.
5. Project plan criteria 1-2 (formally provable model compatibility + structured failure modes) demonstrated end-to-end via the roster as conformance bed.

### Cross-repo handoff

- iconnx publishes the generalized API on the `ws5-observability` branch.
- leadline-bench Cargo.toml-pins to that branch (mirrors WS-3.5 worktree pin pattern).
- leadline-bench adds `--validate` (M5.1) AND `--bisect` (M5.4) flags **in a single commit** (sequencing observation from Q4: parallel-prep allows leadline-bench to start drafting against API stubs as soon as Y(1) and Y(4) commits land).
- Post-merge: leadline-bench reverts the pin to `../../iconnx` or main.

**leadline-bench `compare` exit-code convention** (recorded here as the agreed cross-repo contract):

| Exit code | Meaning |
|---|---|
| `0` | pass |
| `1` | gate fail (numerical divergence) |
| `2` | CLI arg / setup error (existing — `compare.rs` arg-parse failure) |
| `3` | validate-failure (model rejected by `validate_model`) |

iconnx's spec records this so neither side drifts. leadline-bench owns the implementation.

### Cadence

Compressed milestone cadence (mirror WS-3.5). 5-commit target, split if subsystems separate cleanly per Q2 nuance. All commits GPG-signed. Roster sweep + conformance harness both run pre-merge locally.

### Implementation-order dependency graph

```
M5.5 (errors.rs)              ← foundational; no deps
   ↓
M5.1 (validate/*)             ← uses IconnxError variants
   ↓                              uses op_support.rs (also needed by M5.3)
M5.2 (opset.rs)               ← uses M5.1's OpsetOutOfRange variant
M5.3 (onnx_conformance.rs)    ← uses M5.1's supported_ops() + op_support::lookup()
   
M5.4 (differential refactor)  ← orthogonal; depends only on existing parse/exec + IconnxError
```

Plausible plan ordering (refined in writing-plans phase):
- **Y(1):** M5.5 + M5.1 (foundational pair; M5.1 uses M5.5 errors heavily)
- **Y(2):** M5.2 (opset bounds + per-op support set)
- **Y(3):** M5.3 (conformance harness)
- **Y(4):** M5.4 (differential generalization)
- **Y(5):** Cross-repo integration sweep + verification (NOT integration start; leadline-bench parallel-prep against API stubs begins at Y(1)/Y(4) commits)

---

## Section 2 — Architecture

### Module structure

```
src/
├── lib.rs                      [TOUCH] public re-exports: validate_model,
│                                       DifferentialRunner, IconnxError, Result,
│                                       ModelCapabilities, ModelValidationFailure,
│                                       MIN_OPSET_SUPPORTED, MAX_OPSET_SUPPORTED,
│                                       DifferentialTolerance, ToleranceHint
├── errors.rs                   [NEW]   IconnxError taxonomy (M5.5);
│                                       #[from] adapters; ~80 lines
├── validate/
│   ├── mod.rs                  [NEW]   validate_model() entry point;
│   │                                   collect-all orchestration; ~250 lines
│   ├── capabilities.rs         [NEW]   ModelCapabilities, ModelWarning;
│   │                                   estimated_peak_bytes heuristic
│   │                                   (formula in rustdoc); ~180 lines
│   ├── incompat.rs             [NEW]   ModelValidationFailure,
│   │                                   ModelIncompatibility variants; ~150 lines
│   └── tolerance_hint.rs       [NEW]   ToleranceHint + ToleranceConfidence;
│                                       per-(dtype, depth-class) lookup table
│                                       seeded from WS-3.5 empirical data;
│                                       layer_count = distinct LayerNormalization
│                                       node count (rustdoc-documented); ~200 lines
├── op_support.rs               [NEW]   single-source-of-truth metadata table;
│                                       supported_ops() OnceLock-cached;
│                                       lookup(op, ctx) -> LookupResult helper; ~120 lines
├── opset.rs                    [NEW]   MIN/MAX_OPSET_SUPPORTED constants
│                                       w/ rustdoc-encoded derivation policy; ~50 lines
├── tolerance.rs                [NEW]   DifferentialTolerance (canonical type);
│                                       always-available (NO debug-inference gate); ~120 lines
├── onnx_parser.rs              [TOUCH] expose opset_imports() accessor;
│                                       no other changes
├── cuda/
│   ├── differential.rs         [REFAC] generalize: ReferenceSource enum,
│   │                                   builder API, output_names: &[&str],
│   │                                   with_op_chain_context_depth(usize);
│   │                                   target ~500 lines (was 368);
│   │                                   if approaches 1000, split into
│   │                                   differential/runner.rs + differential/report.rs +
│   │                                   differential/reference_source.rs
│   └── ...                     [unchanged]

tests/
├── conformance_data/onnx-1.<N>/        [NEW] vendored upstream test data
│   ├── node/                                  (only the node/ subtree;
│   │   ├── test_add/                          skip simple/, pytorch-converted/, etc)
│   │   ├── test_matmul_2d/
│   │   └── ...                          ~400 small dirs, each with model.onnx + test_data_set_*/
│   └── regenerate.sh                    [NEW] human-reference script for re-vendoring
│                                              new ONNX releases
├── onnx_conformance.rs                 [NEW] harness; ~400 lines
│                                              uses OnnxParser + op_support::lookup
├── conformance_known_failures.toml     [NEW] allowlist; per-entry dispatch field;
│                                              ~50-200 lines depending on initial fail set
├── validate_model_test.rs              [NEW] M5.1 surface tests + roster sweep
├── op_support_table_consistency_test.rs [NEW] drift-prevention: dispatch_op match arms
│                                              ⊆ OP_SUPPORT_TABLE entries (and vice versa)
└── opset_bounds_test.rs                [NEW] asserts MIN/MAX values match roster

CONTRIBUTING.md                          [TOUCH] document developer pre-merge discipline:
                                                  cargo test --features cuda --test onnx_conformance
                                                  before opening PR; mirror existing roster discipline
```

### Single-source-of-truth: `OP_SUPPORT_TABLE`

```rust
#[non_exhaustive]
pub struct OpSupport {
    pub op_type: &'static str,
    // Future fields (cpu: bool, opset_range, etc.) reserved via #[non_exhaustive].
    // Initial WS-5 lands op_type only — no CPU executor yet.
}

pub const OP_SUPPORT_TABLE: &[OpSupport] = &[
    OpSupport { op_type: "Add"     },
    OpSupport { op_type: "Conv"    },
    OpSupport { op_type: "Relu"    },
    OpSupport { op_type: "MatMul"  },
    // ~50 entries empirically populated from existing dispatch_op + dispatch_layout match arms
];

pub fn supported_ops() -> &'static HashSet<&'static str> {
    static CACHE: OnceLock<HashSet<&'static str>> = OnceLock::new();
    CACHE.get_or_init(|| {
        OP_SUPPORT_TABLE.iter().map(|e| e.op_type).collect()
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchContext {
    /// CI context: no GPU, no CPU executor today. Conformance harness verifies
    /// parse + op-recognition only; no execution comparison performed.
    ParserOnlyCi,
    /// Local developer / GPU runner: full GPU execution + reference comparison.
    Gpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LookupResult {
    /// Op is in OP_SUPPORT_TABLE and the context can run it fully.
    /// (Today: Gpu context with op present in table.)
    Runnable,
    /// Op is in OP_SUPPORT_TABLE; current context can verify parse + recognition
    /// but not execute. (Today: ParserOnlyCi context with op present in table.)
    ParserOnlyRecognized,
    /// Op not in OP_SUPPORT_TABLE. Test should skip.
    Unsupported,
}

pub fn lookup(op_type: &str, ctx: DispatchContext) -> LookupResult {
    if !supported_ops().contains(op_type) {
        return LookupResult::Unsupported;
    }
    match ctx {
        DispatchContext::Gpu => LookupResult::Runnable,
        DispatchContext::ParserOnlyCi => LookupResult::ParserOnlyRecognized,
    }
}
```

`supported_ops()` derives from the table; build-once via `OnceLock`, return reference forever. Adding a new op = one new table entry → automatic propagation. Drift impossible by construction. Per-op opset metadata deliberately absent from the table (Q2 deferral); `#[non_exhaustive]` reservation only.

### `IconnxError` taxonomy (M5.5)

```rust
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum IconnxError {
    #[error("parse error: {0}")]
    Parse(#[from] ParseError),

    #[error("CUDA error: {0}")]
    Cuda(#[from] CudaError),

    #[error("model validation failed: {0}")]
    Validation(#[from] ModelValidationFailure),

    #[error("differential test error: {0}")]
    Differential(#[from] DifferentialError),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, IconnxError>;
```

Subsumes WS-1 M1.6. Source chains preserved via `#[from]`. Re-exported from crate root so `iconnx::Result<T>` becomes the consumer-facing default. Existing `ParseError` and `CudaError` are unchanged — they remain typed leaf errors; `IconnxError` is the convenience wrapper for callers who don't want to enumerate.

### `ModelValidationFailure` and `ModelIncompatibility`

```rust
#[derive(Debug, thiserror::Error)]
#[error("model validation failed with {} incompatibilities", incompatibilities.len())]
#[non_exhaustive]
pub struct ModelValidationFailure {
    pub incompatibilities: Vec<ModelIncompatibility>,
    pub partial_capabilities: Option<ModelCapabilities>,
}

#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ModelIncompatibility {
    UnsupportedOp {
        op_type: String,
        count: usize,
        sample_node_names: Vec<String>,  // cap 3
    },
    UnsupportedDType {
        dtype: DType,
        reason: &'static str,
    },
    OpsetOutOfRange {
        domain: String,
        version: i64,
        supported: (i64, i64),
    },
    /// Model declares an opset import for a domain iconnx doesn't recognize
    /// (e.g., a custom domain). Non-fatal in the collect-all sense — added
    /// to incompatibilities but the rest of validation continues.
    OpsetDomainUnknown {
        domain: String,
        version: i64,
    },
    HardwareFloorUnmet {
        dtype: DType,
        required: &'static str,
        found: String,
    },
    Parse(ParseError),
}
```

### Differential runner architecture (M5.4)

```rust
// src/tolerance.rs (always-available, no debug-inference gate)
pub struct DifferentialTolerance {
    pub abs: Option<f32>,
    pub rel: Option<f32>,
    // Implicit mode: derive from abs.is_some() / rel.is_some()
    // (matches leadline-bench's existing ToleranceConfig shape)
}

// src/cuda/differential.rs (debug-inference gated; refactor of existing 368-line impl)
pub struct DifferentialRunner {
    executor: GpuGraphExecutor,
    reference: ReferenceSource,
    tolerance: DifferentialTolerance,
    output_names: Vec<String>,         // empty = all graph outputs (Q4.1)
    op_chain_context_depth: usize,     // default 5; configurable via builder
}

pub enum ReferenceSource {
    FixtureDir { path: PathBuf },                          // existing Kokoro flow; per-node refs
    InMemory   { tensors: HashMap<String, Tensor> },       // leadline-bench --bisect flow;
                                                           // output-level only by default
}

pub struct DivergenceReport {
    pub node_name: String,
    pub op_type: String,
    pub output_index: usize,
    pub exceeded: ExceededTolerance,        // Abs(value) | Rel(value) | NonFinite
    pub diff_magnitude: f32,
    pub sample_inputs: Vec<TensorPreview>,  // truncated <= 64 elements per tensor
    pub op_chain_context: Vec<String>,      // parent op names; cap by depth (default 5)
}

impl DifferentialRunner {
    pub fn new(model_path: &Path) -> Result<Self, DifferentialError>;
    pub fn with_tolerance(self, t: DifferentialTolerance) -> Self;
    pub fn with_reference(self, src: ReferenceSource) -> Self;
    pub fn with_outputs(self, names: &[&str]) -> Self;
    pub fn with_op_chain_context_depth(self, depth: usize) -> Self;
    pub fn run_until_divergence(&self, inputs: HashMap<String, Tensor>)
        -> Result<Option<DivergenceReport>, DifferentialError>;
}
```

NaN/Inf detection: explicit forced-divergence in the comparison routine. `compare(actual, expected)` checks `actual.iter().any(|x| !x.is_finite())` before any `f32::max` fold. Locked as regression test (folded into `cuda/differential.rs`'s `#[cfg(test)] mod nan_inf_tests`).

### Feature flag posture

- `debug-inference` (existing): continues to gate `CheckpointManager` + `DifferentialRunner` (heavy machinery; release builds stay lean). Unchanged posture.
- M5.1 (`validate_model`), M5.2 (opset bounds), M5.3 (conformance harness), M5.5 (`IconnxError`): **always-available** (public stable surface).
- `DifferentialTolerance` (M5.4 type): **always-available** at `src/tolerance.rs`. Type is config-side data; no GPU dependencies. leadline-bench imports it without enabling `debug-inference`.
- Conformance test target: always-available; only its execution dispatch context is gated on whether `cfg!(feature = "cuda")` AND a GPU is actually present.

### Cross-repo type-sharing direction

`From<iconnx::DifferentialTolerance> for leadline_bench::ToleranceConfig` (or vice versa) lives on **leadline-bench's side**. Dependency direction stays one-way: leadline-bench → iconnx. iconnx exposes its canonical types; leadline-bench imports them and provides any conversions to its own internal shapes. Avoids feature-flag bloat on iconnx's side and keeps iconnx unaware of downstream consumer shapes.

---

## Section 3 — Component Map

### New files (12 total, all under 1000-line ceiling)

| File | LOC est | Responsibility |
|---|---|---|
| `src/errors.rs` | ~80 | `IconnxError` enum (M5.5); `#[from]` adapters. Public `iconnx::Result<T>` alias. Soft cap ~300 lines; promote to `errors/mod.rs` if exceeded. |
| `src/validate/mod.rs` | ~250 | `validate_model()` orchestration. Walks parse → ops → opsets → initializers → tolerance → assemble. Collect-all error pattern. |
| `src/validate/capabilities.rs` | ~180 | `ModelCapabilities`, `ModelWarning` enum, `estimated_peak_bytes` heuristic. **Rustdoc above `estimated_peak_bytes` documents the formula explicitly** (e.g. `sum(initializer_bytes) + 2 * max_intermediate_tensor_bytes` — exact form decided at implementation). `#[non_exhaustive]`. |
| `src/validate/incompat.rs` | ~150 | `ModelValidationFailure`, `ModelIncompatibility` variants (incl. `OpsetDomainUnknown`). `#[non_exhaustive]`. |
| `src/validate/tolerance_hint.rs` | ~200 | `ToleranceHint`, `ToleranceConfidence`, `ToleranceBasis`. `expected_tolerance(used_dtypes, layer_count) -> ToleranceHint` lookup; seeded from WS-3.5 empirical table (Whisper-Tiny BF16 4-layer = 0.32/2.45%, BERT-base BF16 12-layer = 0.69/3.99%, whisper-fp16 = 0.039/0.30%). **`layer_count` heuristic = count of distinct `LayerNormalization` nodes in the graph** (rustdoc-documented; stable across BERT/Whisper/DistilBERT-style transformers). |
| `src/op_support.rs` | ~120 | `OP_SUPPORT_TABLE` const + `supported_ops()` + `DispatchContext` enum + `LookupResult` enum + `lookup(op, ctx)` helper. |
| `src/opset.rs` | ~50 | `MIN_OPSET_SUPPORTED` / `MAX_OPSET_SUPPORTED` w/ rustdoc-encoded derivation policy. |
| `src/tolerance.rs` | ~120 | `DifferentialTolerance` (canonical, always-available, no `debug-inference` gate). |
| `tests/onnx_conformance.rs` | ~400 | Conformance harness (M5.3). Iterates **all** `test_data_set_N/` per test directory. |
| `tests/conformance_known_failures.toml` | ~50–200 | Allowlist; `dispatch: "cpu" | "gpu" | "both"` per entry (cpu/both vocabulary forward-compat; only `"gpu"` entries gate today). |
| `tests/conformance_data/onnx-<N>/` | data | Vendored upstream `node/` test subtree. ~25-40 MB across ~400 small files. Pin specific ONNX release version in spec text + directory name. Direct commit (no LFS). |
| `tests/conformance_data/regenerate.sh` | ~30 | Human-reference script for re-vendoring new ONNX releases. Documents: clone ONNX repo, checkout tag, copy `node/` subtree to versioned directory, commit. Not invoked by tests. |
| `CONTRIBUTING.md` (or amend `README.md`) | +30 lines | Document developer pre-merge discipline: `cargo test --features cuda --test onnx_conformance` before opening PR. |

### Refactored files (1)

| File | Current | After | Change |
|---|---|---|---|
| `src/cuda/differential.rs` | 368 lines | ~500 lines | Generalize to public `DifferentialRunner` API. `ReferenceSource` enum, builder API (`with_tolerance`, `with_reference`, `with_outputs`, `with_op_chain_context_depth`), `output_names: &[&str]`. Replace fixture-dir-only with `FixtureDir | InMemory` two-mode. `DivergenceReport` gains `output_index`, `op_chain_context`, `sample_inputs`. NaN/Inf forced-divergence path + regression test (`#[cfg(test)] mod nan_inf_tests`). Remains `debug-inference`-gated. `DifferentialTolerance` definition moves OUT to `src/tolerance.rs`; this file imports it. |

### Touched files (3, minimal)

| File | Touch |
|---|---|
| `src/lib.rs` | Public re-exports: `validate_model`, `IconnxError`, `Result`, `ModelCapabilities`, `ModelValidationFailure`, `DifferentialTolerance`, `ToleranceHint`, `MIN_OPSET_SUPPORTED`, `MAX_OPSET_SUPPORTED`. `DifferentialRunner` re-exported only when `debug-inference`. |
| `src/onnx_parser.rs` | New public method `OnnxModel::opset_imports() -> Vec<(String, i64)>` (currently the data is internal). No other changes. |
| `CONTRIBUTING.md` (or `README.md`) | Add pre-merge discipline section (see above). |

### Test targets (3 new + extended existing)

| Test target | Purpose |
|---|---|
| `tests/validate_model_test.rs` | Unit + integration. Roster sweep: every leadline-bench profile path → `validate_model` returns `Ok(ModelCapabilities)`. Negative tests: synthetic models with unsupported op / opset out of range / unknown opset domain / non-finite half constants → `ModelValidationFailure` collects expected variants. |
| `tests/op_support_table_consistency_test.rs` | Drift-prevention: enumerates `OP_SUPPORT_TABLE` at compile time and asserts `dispatch_op` match arms ⊆ table (and the reverse). |
| `tests/opset_bounds_test.rs` | Asserts `MIN_OPSET_SUPPORTED` / `MAX_OPSET_SUPPORTED` values match the actual roster's opset_imports range. Locks the derivation policy. |
| Existing `cuda/differential.rs` tests | Extended with `ReferenceSource::InMemory` tests; existing `FixtureDir` tests preserved. NaN/Inf regression test folded in as `#[cfg(test)] mod nan_inf_tests`. |

### File-size discipline (CLAUDE.md 1000-line ceiling)

- All new files target ≤500 lines individually. Per-concern split (`validate/` subdirectory, `errors.rs` soft cap) keeps each module focused.
- `cuda/differential.rs` refactor may approach 500 lines; if it crosses, split into `differential/runner.rs` + `differential/report.rs` + `differential/reference_source.rs` per-concern. Decided at implementation time, not pre-emptively.
- No pre-existing over-limit files are touched by WS-5; carry-forward bucket from Section 1 covers structural debt.

---

## Section 4 — Data Flow

Three primary flows + the cross-repo handoff.

### Flow A — `validate_model(path)` invocation

```
caller
  │
  │  validate_model("path/to/model.onnx")
  ▼
┌──────────────────────────────────────────────────────────────┐
│  validate/mod.rs::validate_model                             │
│                                                              │
│  1. OnnxParser::parse_file(path)                             │
│     ├─ Err(ParseError) ─────────────► assemble Failure       │
│     │                                  with vec![Parse(_)],  │
│     │                                  partial: None         │
│     │                                  ──► return Err        │
│     └─ Ok(model)                                             │
│                                                              │
│     Note: model.computation_graph()? in step 2 also returns  │
│     ParseError on malformed graph topology — same family,    │
│     same short-circuit shape. ParseError is the only family  │
│     that short-circuits; everything below is collect-only.   │
│                                                              │
│  2. let graph = model.computation_graph()?  // ParseError    │
│     for node in graph.nodes():                               │
│         track op_type, dtypes, sample_node_names             │
│     unsupported = used_ops - supported_ops()                 │
│     for op in unsupported:                                   │
│         incompat.push(UnsupportedOp { op, count, samples })  │
│                                                              │
│  3. for (domain, version) in model.opset_imports():          │
│     match domain {                                           │
│         "" | "ai.onnx" =>                                    │
│             if version < MIN || version > MAX:               │
│                 incompat.push(OpsetOutOfRange { ... })       │
│         other =>                                             │
│             incompat.push(OpsetDomainUnknown { domain,       │
│                                                  version })  │
│     }                                                        │
│                                                              │
│  4. weights = model.extract_weights()?                       │
│     for (name, t) in weights:                                │
│         if t.is_bfloat16() && count_non_finite_bf16(...) > 0 │
│         || t.is_float16() && count_non_finite_f16(...) > 0:  │
│             warnings.push(NonFiniteHalfConstants {...})      │
│                                                              │
│  5. layer_count = count_layer_norm_nodes(&graph)             │
│     // distinct LayerNormalization op nodes — stable across  │
│     // BERT/Whisper/DistilBERT transformer architectures     │
│     hint = expected_tolerance(used_dtypes, layer_count)      │
│     // tolerance_hint.rs lookup:                             │
│     //   (BF16, 4-layer)  -> Measured 0.32/2.45%             │
│     //   (BF16, 12-layer) -> Measured 0.69/3.99%             │
│     //   (BF16, ?)        -> Estimated (depth interpolation) │
│     //   (UnseenDtype, _) -> FallbackHeuristic               │
│                                                              │
│  6. peak = sum(initializer_bytes) +                          │
│            2 * max_intermediate_tensor_bytes                  │
│     // exact formula in capabilities.rs rustdoc;             │
│     // documented as "rough heuristic, not arena-precise"    │
│                                                              │
│  7. if incompat.is_empty():                                  │
│         return Ok(ModelCapabilities { ... })                 │
│     else:                                                    │
│         return Err(ModelValidationFailure {                  │
│             incompatibilities: incompat,                     │
│             partial_capabilities: Some(...best-effort...),   │
│         })                                                   │
└──────────────────────────────────────────────────────────────┘
  │
  ▼ Result<ModelCapabilities, ModelValidationFailure>
```

**Important property:** Every step after parse is **collect-only**, never short-circuit. A model can hit 3 unsupported ops + opset-out-of-range + 1 unknown-domain + 1 non-finite warning in a single call. Only the `ParseError` family (`parse_file` and `computation_graph()`) short-circuits.

### Flow B — Conformance harness invocation (`cargo test --test onnx_conformance`)

```
runner
  │
  │  walk tests/conformance_data/onnx-<N>/node/test_*/
  ▼
┌──────────────────────────────────────────────────────────────┐
│  let context = if cfg!(feature = "cuda") && cuda_avail()     │
│                { DispatchContext::Gpu }                      │
│                else { DispatchContext::ParserOnlyCi };       │
│                                                              │
│  for each test_dir:                                          │
│                                                              │
│  1. let model = OnnxParser::parse_file(test_dir/model.onnx)  │
│     ├─ Err(_) -> outcome = UnexpectedFailure { reason:       │
│     │                       "parse error" }                  │
│     │             continue                                   │
│     └─ Ok(model)                                             │
│                                                              │
│  2. let op_type = model.computation_graph()?                 │
│                       .nodes()[0].op_type                    │
│     // (single-node graph by spec)                           │
│                                                              │
│  3. match op_support::lookup(&op_type, context) {            │
│       LookupResult::Unsupported -> {                         │
│         outcome = SkippedUnsupportedOp { op };               │
│         continue;                                            │
│       }                                                      │
│       LookupResult::ParserOnlyRecognized -> {                │
│         // CI gate: parse + recognition succeeded            │
│         outcome = ParserOnlyOk;                              │
│         continue;                                            │
│       }                                                      │
│       LookupResult::Runnable -> { /* run + compare below */ }│
│     }                                                        │
│                                                              │
│  4. // Only Runnable (Gpu context) reaches here.             │
│     for ds in test_dir/test_data_set_*/ {  // iterate ALL    │
│         inputs   = load_pb_inputs(ds);                       │
│         expected = load_pb_outputs(ds);                      │
│         actual   = GpuGraphExecutor::run(model, inputs);     │
│                                                              │
│         allowlist_entry = lookup_allowlist(test_name,        │
│                                            "gpu");           │
│         diff = compare(actual, expected, dtype_tolerance);   │
│         // diff explicitly checks NaN/Inf -> forced divergence│
│                                                              │
│         outcome[ds] = match (diff, allowlist_entry):         │
│             (Pass,    None)        => Pass                   │
│             (Pass,    Some(_))     => UnexpectedPass {entry} │
│             (Fail(_), None)        => UnexpectedFailure{...} │
│             (Fail(_), Some(entry)) => KnownFailure { entry } │
│     }                                                        │
│                                                              │
│  5. record outcomes; track op_type for skipped distribution  │
└──────────────────────────────────────────────────────────────┘
  │
  ▼  print result table + skipped-op distribution (top-N coverage gaps)
     CI gate (ParserOnlyCi):
       exit(0) iff zero UnexpectedFailure (parse + op recognition only)
     Local Gpu:
       exit(0) iff zero UnexpectedFailure + zero UnexpectedPass
```

**CI gate behavior:** `ParserOnlyCi` context. Parse failures or `OP_SUPPORT_TABLE` drift (op in dispatch but not table, or vice versa — caught by separate `op_support_table_consistency_test`) → CI fail. Execution-level failures only fire under `Gpu` context (developer pre-merge run).

### Flow C — `DifferentialRunner` bisect (M5.4)

```
caller (e.g. leadline-bench --bisect when a gate fails)
  │
  │  DifferentialRunner::new(model_path)
  │    .with_tolerance(tol)
  │    .with_reference(ReferenceSource::InMemory { ort_outputs })
  │    .with_outputs(&["logits"])    // or empty -> all graph outputs
  │    .with_op_chain_context_depth(5)   // optional; default 5
  │    .run_until_divergence(inputs)
  ▼
┌──────────────────────────────────────────────────────────────┐
│  cuda/differential.rs::DifferentialRunner                    │
│                                                              │
│  1. set up GpuGraphExecutor with model + initializers        │
│                                                              │
│  2. configure CheckpointManager from ReferenceSource:        │
│     - FixtureDir { path }   -> load_ort_references_from_dir  │
│                                  (per-node refs; full bisect)│
│     - InMemory { tensors }  -> seed mgr.ort_tensors directly │
│                                  (output-level only;          │
│                                   no per-node bisect unless  │
│                                   leadline-bench captures    │
│                                   intermediates — deferred)  │
│                                                              │
│  3. run_with_checkpoints(inputs, output_names, config)       │
│     -> per-node capture + comparison vs reference            │
│                                                              │
│  4. mgr.first_divergence() identifies first node where       │
│     |actual - expected| exceeds tolerance.abs OR             │
│     rel_diff exceeds tolerance.rel OR                        │
│     actual contains NaN/Inf (forced divergence)              │
│                                                              │
│  5. assemble DivergenceReport:                               │
│       node_name, op_type, output_index,                      │
│       exceeded: { Abs(...) | Rel(...) | NonFinite },         │
│       diff_magnitude,                                        │
│       sample_inputs: capture inputs to divergent node,       │
│         truncated to <= 64 elements per tensor,              │
│       op_chain_context: walk graph backward from divergent  │
│         node, collect <= op_chain_context_depth parent op    │
│         names                                                │
└──────────────────────────────────────────────────────────────┘
  │
  ▼  Result<Option<DivergenceReport>, DifferentialError>
     // None  = converged within tolerance for all checkpoints
     // Some(report) = first divergence
     // Err(_) = setup / execution failure (orthogonal to convergence)
```

NaN/Inf detection is **explicit and forced** — `compare(actual, expected)` checks `actual.iter().any(|x| !x.is_finite())` before any `f32::max` fold. Any non-finite in iconnx output → forced divergence with `exceeded: NonFinite`. Direct mirror of leadline-bench's `nan_in_iconnx_output_forces_infinite_diff` regression pattern.

### Flow D — Cross-repo handoff (leadline-bench `compare --validate --bisect`)

```
leadline-bench compare CLI
  │
  │  --validate <model_path>  --bisect (optional, applies on gate-fail)
  ▼
┌──────────────────────────────────────────────────────────────┐
│  compare main:                                               │
│                                                              │
│  1. iconnx::validate_model(model_path)        // M5.1        │
│     ├─ Err(ModelValidationFailure {                          │
│     │      incompatibilities, partial }) ->                  │
│     │    print "❌ pre-flight rejected:"                     │
│     │    for inc in incompatibilities: print(format(inc))    │
│     │    if partial: print partial summary                   │
│     │    exit 3  // validate-failure                         │
│     │            // (distinct from gate-fail's exit 1        │
│     │             and arg-parse's existing exit 2)           │
│     └─ Ok(caps) -> continue                                  │
│                                                              │
│  2. proceed with existing compare flow:                      │
│     - load model, run iconnx, run ORT reference              │
│     - keep ort_outputs in scope through gate-fail branch     │
│     - compute element-wise diff                              │
│     - check NaN/Inf (existing forced-INFINITY behavior)      │
│     - compare diff vs profile tolerance (existing gate)      │
│                                                              │
│  3. on gate-fail (exit 1 path) AND --bisect:                 │
│     iconnx::DifferentialRunner::new(model)    // M5.4        │
│         .with_tolerance(profile.tolerance.into())            │
│         .with_reference(ReferenceSource::InMemory {          │
│             tensors: ort_outputs_in_memory })                │
│         .with_outputs(&profile.output_names)                 │
│         .run_until_divergence(inputs)                        │
│     ├─ Ok(Some(report)) ->                                   │
│     │    print "first divergence at <node>: <report>"        │
│     │    // output-level only; per-node bisect requires      │
│     │    // FixtureDir mode (deferred)                       │
│     ├─ Ok(None) ->                                           │
│     │    print "no node-level divergence (output-level only)"│
│     └─ Err(_) -> propagate                                   │
└──────────────────────────────────────────────────────────────┘
```

leadline-bench Cargo.toml temporarily pins to `ws5-observability` branch during integration; reverts to main post-merge. Single PR on leadline-bench side bundles `--validate` + `--bisect` flags.

---

## Section 5 — Error Handling

Three layers: (1) typed leaf errors (existing + new), (2) `ModelValidationFailure` collect-all aggregator, (3) `IconnxError` top-level wrapper.

### Layer 1 — Typed leaf errors (existing, unchanged)

| Error type | Source | Used in |
|---|---|---|
| `ParseError` | `src/onnx_parser.rs` (existing) | All parse paths; short-circuits in `validate_model` Flow A. WS-1's typed taxonomy preserved. |
| `CudaError` | `src/cuda/context.rs` (existing) | All GPU dispatch paths. WS-3.5 added `UnsupportedHardware`, `UnsupportedDtype`, `ShapeMismatch`, `DtypeMismatch` variants. |
| `DifferentialError` | `src/cuda/differential.rs` (existing, refactored) | `DifferentialRunner` flows; covers `ModelLoad`, `GpuInit`, `FixtureLoad`, `Execution`. Generalization preserves existing variants. |
| `std::io::Error` | std | File I/O in parser, conformance harness, fixture loaders. |

### Layer 2 — Collect-all aggregator (NEW)

```rust
#[derive(Debug, thiserror::Error)]
#[error("model validation failed with {} incompatibilities", incompatibilities.len())]
#[non_exhaustive]
pub struct ModelValidationFailure {
    pub incompatibilities: Vec<ModelIncompatibility>,

    /// Best-effort capability information available even when validation fails.
    ///
    /// `Some(caps)` whenever parse succeeded, regardless of incompatibility count —
    /// `caps` reflects every field computable from what passed (e.g. `opset_imports`
    /// always populated, `supported_ops` = used_ops minus the unsupported set).
    /// Lets a UI render "this model uses ops [list], dtypes [list], opset [N],
    /// EXCEPT for these incompatibilities [list]" rather than just the failure list.
    ///
    /// `None` only on the parse short-circuit case (`vec![Parse(_)]` shape) — when
    /// the file failed to parse, no capability fields can be honestly reported.
    pub partial_capabilities: Option<ModelCapabilities>,
}
```

`ModelIncompatibility` variants (re-listed for completeness):
- `UnsupportedOp { op_type, count, sample_node_names: Vec<String> }` (sample names cap 3)
- `UnsupportedDType { dtype, reason: &'static str }`
- `OpsetOutOfRange { domain, version, supported: (i64, i64) }`
- `OpsetDomainUnknown { domain, version }` (added per Section 4 Q1 resolution)
- `HardwareFloorUnmet { dtype, required: &'static str, found: String }` (only producible by `validate_model_for_hardware`, see below)
- `Parse(ParseError)` (the short-circuit case; `partial_capabilities = None`)

`#[non_exhaustive]` on the struct AND the enum — additive variant evolution doesn't break consumers.

### Two `validate_model` entry points (resolves Section 5 Concern 1)

```rust
/// Static model validation — pure analysis with no CUDA dependency.
///
/// Cannot produce `ModelIncompatibility::HardwareFloorUnmet` (no hardware
/// context to compare against). Library consumers writing GPU-less tooling
/// (TUI inspectors, capability discovery for filtered roster sweeps, etc.)
/// call this entry point. leadline-bench `compare --validate` for hardware-
/// agnostic pre-flight also uses this; the hardware check is opt-in.
pub fn validate_model<P: AsRef<Path>>(path: P)
    -> Result<ModelCapabilities, ModelValidationFailure>;

/// Static model validation + hardware floor check.
///
/// Adds `HardwareFloorUnmet` checks for dtype/opset combinations that
/// require capabilities the supplied context lacks (e.g., BF16 on sm_75).
/// Used by callers that already have a `IconnxCudaContext` standing up
/// — typically inference clients about to run the model.
pub fn validate_model_for_hardware<P: AsRef<Path>>(
    path: P,
    ctx: &IconnxCudaContext,
) -> Result<ModelCapabilities, ModelValidationFailure>;
```

Both return identical types — consumers handle one return shape regardless of which entry point they used. The hardware-aware version is a strict superset of static analysis.

### Layer 3 — Top-level wrapper (NEW, M5.5)

```rust
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum IconnxError {
    #[error("parse error: {0}")]
    Parse(#[from] ParseError),

    #[error("CUDA error: {0}")]
    Cuda(#[from] CudaError),

    /// Convenience type-erasure for callers using `iconnx::Result<T>`.
    /// Note: this preserves the full `Vec<ModelIncompatibility>` and
    /// `partial_capabilities` payload from the inner failure. Callers
    /// who want to enumerate incompatibilities should use the structured
    /// `Result<_, ModelValidationFailure>` returned directly by
    /// `validate_model()` / `validate_model_for_hardware()` rather than
    /// destructuring through this variant. The convenience wrapper is
    /// intended for "log-and-bail" paths.
    #[error("model validation failed: {0}")]
    Validation(#[from] ModelValidationFailure),

    #[error("differential test error: {0}")]
    Differential(#[from] DifferentialError),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, IconnxError>;
```

Subsumes WS-1 M1.6. Source chains preserved via `#[from]`. Re-exported from crate root so `iconnx::Result<T>` becomes the consumer-facing default. Existing `ParseError` and `CudaError` are unchanged — they remain typed leaf errors; `IconnxError` is the convenience wrapper for callers who don't want to enumerate.

### Error-rendering policy (`Display` impls)

| Layer | `Display` form | Audience |
|---|---|---|
| `ParseError` | `"unsupported ONNX dtype 14 for tensor 'attention_mask'"` (existing) | Parser-side debugging; precise enum data preserved. |
| `CudaError` | `"unsupported dtype: op MatMul does not support dtype combination (BF16, Bool): reason"` (existing pattern from WS-3.5) | GPU-side debugging. |
| `ModelIncompatibility` | One variant per format — `"unsupported op 'CustomOp' (3 occurrences, e.g. /encoder/CustomOp_0, ...)"` for `UnsupportedOp`; symmetric for others. Helper `fn format(&self) -> String` lives on the enum (or via `Display` impl). | User-facing CLI output (`compare --validate`). |
| `ModelValidationFailure` | `"model validation failed with N incompatibilities"` (top-level summary). Iteration over `incompatibilities` + per-variant rendering is the caller's job — leadline-bench loops + prints. | User-facing CLI output. |
| `IconnxError` | Forwards to inner via `#[from]` source chain. `to_string()` returns the inner's display + outer prefix. | Convenience wrapper for callers who don't want to enumerate. |

### Source chain preservation

```rust
let err: IconnxError = some_io.into();
err.source()  // returns the inner std::io::Error
err.source().and_then(|e| e.source())  // chain walking
```

`#[from]` adapters preserve the chain. Standard `std::error::Error::source()` traversal works.

### Public re-export pattern

```rust
// src/lib.rs (additions)
pub use crate::errors::{IconnxError, Result};
pub use crate::onnx_parser::ParseError;          // top-level re-export
pub use crate::cuda::CudaError;                  // top-level re-export
pub use crate::validate::{
    validate_model, validate_model_for_hardware,
    ModelCapabilities, ModelValidationFailure, ModelIncompatibility, ModelWarning,
    ToleranceHint, ToleranceConfidence, ToleranceBasis,
};
pub use crate::op_support::{DispatchContext, LookupResult};
pub use crate::opset::{MIN_OPSET_SUPPORTED, MAX_OPSET_SUPPORTED};
pub use crate::tolerance::DifferentialTolerance;

#[cfg(feature = "debug-inference")]
pub use crate::cuda::differential::{
    DifferentialRunner, ReferenceSource, DivergenceReport, ExceededTolerance,
    DifferentialError,                            // gated re-export (mirrors DifferentialRunner)
};

// Existing module paths (`iconnx::onnx_parser::ParseError`,
// `iconnx::cuda::CudaError`) continue to resolve — re-exports are additive.
```

### Backwards compatibility

- `ParseError`, `CudaError` retain their public paths (`iconnx::onnx_parser::ParseError`, `iconnx::cuda::CudaError`). Existing consumers unbroken.
- `IconnxError` is **additive**. Existing callers that wrap iconnx errors in `anyhow::Error` (via `?` propagation through anyhow-using functions) continue to work — `IconnxError`, `ParseError`, and `CudaError` all implement `std::error::Error`, so anyhow-conversion remains free.
- Pre-WS-5 `cuda::differential::DifferentialError` becomes a leaf wrapped by `IconnxError::Differential` — existing consumers (just iconnx's own tests today) aren't broken.

### Failure-mode policy summary

| Failure | Where surfaces | Short-circuit? | Outcome |
|---|---|---|---|
| Parse error (file structure) | `validate_model` step 1 | Yes (`ParseError` family) | `ModelValidationFailure { vec![Parse(_)], partial: None }` |
| Malformed graph topology | `validate_model` step 2 | Yes (`ParseError` family) | Same as above |
| Unsupported op | `validate_model` step 2 | No | `incompatibilities.push(UnsupportedOp { ... })` |
| Opset version out of range | `validate_model` step 3 | No | `incompatibilities.push(OpsetOutOfRange { ... })` |
| Unknown opset domain | `validate_model` step 3 | No | `incompatibilities.push(OpsetDomainUnknown { ... })` |
| Non-finite half constant | `validate_model` step 4 | No (non-fatal) | `warnings.push(NonFiniteHalfConstants { ... })` |
| CUDA hardware floor unmet | `validate_model_for_hardware` (only) | No | `incompatibilities.push(HardwareFloorUnmet { ... })` |
| GPU execution failure | `DifferentialRunner::run_until_divergence` | Yes | `Err(DifferentialError::Execution(_))` (separate from convergence path) |
| NaN/Inf in iconnx output | `DifferentialRunner::compare` | No (forced divergence) | `Some(DivergenceReport { exceeded: NonFinite, ... })` |
| Conformance test parse fail | `tests/onnx_conformance.rs` Flow B step 1 | Per-test | `outcome = UnexpectedFailure { reason: "parse error" }` |

---

## Section 6 — Testing & Sign-Off

### Per-milestone test additions

**M5.5 (Y(1) leg 1) — `IconnxError` taxonomy:**
- Inline `#[cfg(test)] mod tests` in `src/errors.rs` (matches existing `cuda/differential.rs` convention; no standalone test target). Verifies:
  - Each `#[from]` adapter compiles and propagates source chain
  - `Display` round-trips through nested errors (e.g., `IconnxError::Parse(ParseError::ShapeMismatch(...))`)
  - `std::error::Error::source()` traversal walks the chain correctly
  - `iconnx::Result<T>` resolves to `Result<T, IconnxError>`

**M5.1 (Y(1) leg 2) — `validate_model()`:**
- `tests/validate_model_test.rs`:
  - **Synthetic negative tests** (no roster dependency; iconnx unaware of leadline-bench roster shape):
    - Unsupported op → `ModelValidationFailure { incompatibilities: [UnsupportedOp { op_type, count, sample_node_names: ... }] }`
    - Opset version out of range → `OpsetOutOfRange { domain, version, supported }`
    - Unknown opset domain → `OpsetDomainUnknown { domain, version }`
    - Non-finite half constant → `warnings: [NonFiniteHalfConstants { ... }]` + `incompatibilities: []` (warning is non-fatal)
    - Multi-incompatibility (3 unsupported ops + opset out of range simultaneously) → all four collected, `partial_capabilities = Some(...)`
    - Parse failure → `incompatibilities: [Parse(_)], partial_capabilities: None`
  - **`validate_model_for_hardware` tests:** mock context for sm_75 → BF16-using model → `HardwareFloorUnmet`. Same path on sm_80+ context → no `HardwareFloorUnmet`.
  - **Happy-path unit tests:** small synthetic models that should validate cleanly → `Ok(ModelCapabilities)`.
  - **Stability lock-in:** `#[non_exhaustive]` semantics relied on; documented in rustdoc rather than enforced via `trybuild` (well-known language feature, well-understood by consumers).
  - **Roster integration is NOT here.** Roster sweep of `validate_model` is exercised via leadline-bench's `compare --validate` (gate 6 territory at Y(5)). iconnx stays unaware of leadline-bench's roster directory shape.

**M5.2 (Y(2)) — Opset bounds + per-op support:**
- `tests/opset_bounds_test.rs`:
  - Asserts `MIN_OPSET_SUPPORTED` / `MAX_OPSET_SUPPORTED` match the actual roster's opset_imports range (drift-prevention against the rustdoc-encoded derivation policy).
  - Locks: extending the constants without updating roster fails this test.
- `tests/op_support_table_consistency_test.rs`:
  - Walks `dispatch_op` + `dispatch_layout` match arms (via compile-time `include_str!`-based parse, or runtime grep using `include_str!("../src/cuda/executor/mod.rs")`).
  - Asserts: every match-arm op_type → has `OP_SUPPORT_TABLE` entry; every table entry → has dispatch coverage.
  - Drift catches the WS-3.5 cross-tier dispatch lesson at WS-5 introduction time.

**M5.3 (Y(3)) — Conformance harness:**
- `tests/onnx_conformance.rs` (the harness itself):
  - **Walk pattern locked:** `tests/conformance_data/onnx-<N>/node/test_*/` (prevents stray data directories from being picked up).
  - **Dispatch context auto-detection:** if `IconnxCudaContext::new()` returns `Err(_)` for any reason (CUDA libs linked but no devices, `cudaGetDeviceCount = 0`, driver/lib skew, etc.), dispatch context falls back to `DispatchContext::ParserOnlyCi`. Any CUDA-init failure routes the harness to parse-only mode rather than raising a `CudaError` and aborting CI.
  - **CI gate (`ParserOnlyCi` context):** model parses + op recognized → `ParserOnlyOk`; OP_SUPPORT_TABLE drift → CI fail; conformance test parse failure → CI fail.
  - **Local Gpu context:** full execution + comparison; allowlist-driven outcome interpretation.
  - **Self-test fixtures live at `tests/conformance_data_self_test/`** (separate top-level tree). Hand-crafted dirs cover each outcome variant. Conformance walker does NOT iterate this tree.
- `tests/conformance_known_failures.toml` populated empirically from first GPU run; each entry includes `dispatch`, `reason`, optional `tracked_issue` field.

**M5.4 (Y(4)) — Differential framework extension:**
- Existing `cuda/differential.rs` `#[cfg(test)] mod tests` (existing 4 tests preserved):
  - `test_tolerance_defaults`, `test_tolerance_strict`, `test_extract_node_name`, `test_extract_op_type` — preserved
- New `#[cfg(test)] mod nan_inf_tests` (folded per Section 3 resolution):
  - `nan_in_iconnx_output_forces_infinite_diff` — mirrors leadline-bench regression pattern
  - `inf_in_iconnx_output_forces_infinite_diff` — same flavor for ±Inf
  - `nan_in_expected_reference_does_not_silently_pass` — covers the `f32::max(0.0, NaN)` trap from the other side
- New `#[cfg(test)] mod reference_source_tests`:
  - `in_memory_mode_runs_without_fixture_dir` — `ReferenceSource::InMemory` sanity check
  - `fixture_dir_mode_unchanged_from_kokoro_pattern` — backwards-compat for existing flow
  - `output_names_empty_resolves_to_all_graph_outputs` — Q4.1 default-resolution
  - `with_op_chain_context_depth_respects_cap` — Q3 configurability

### Sign-off gates

| Gate | Where it runs | What it enforces |
|---|---|---|
| 1. `cargo build --features cuda --tests` | CI + local | Compile cleanly; CUDA-feature build works |
| 2. `cargo clippy --features cuda --tests -- -D warnings` | CI + local | Zero clippy warnings (existing baseline + WS-5 additions) |
| 3. `cargo test --features cuda --lib` | CI + local | All lib unit tests pass — must include all M5.5 + M5.1 + M5.2 helper tests + inline differential tests |
| 4. `cargo test --features cuda --test onnx_conformance` (`ParserOnlyCi`) | CI | New M5.3 gate: parse + recognition; UnexpectedFailure / UnexpectedPass blocks merge |
| 5. `cargo test --features cuda --test onnx_conformance` (`Gpu`) | **Developer pre-merge, NOT CI** | Full execution conformance; mirrors roster-sweep pre-merge discipline |
| 6. Roster regression sweep + `compare --validate` integration (leadline-bench) | **Developer pre-merge, NOT CI** | All 10 profiles still PASS at WS-5 HEAD; zero regressions vs WS-3.5 baseline (`231b47e`); `--validate` exercises `validate_model` against the full roster (subsumes earlier gate 7 concept — owned by leadline-bench, not iconnx) |
| 7. `cargo test --features cuda --test op_support_table_consistency_test` | CI + local | Drift-prevention against `OP_SUPPORT_TABLE` |
| 8. `cargo test --features cuda --test opset_bounds_test` | CI + local | Drift-prevention against MIN/MAX_OPSET_SUPPORTED |

(Numbering compressed after dropping the iconnx-side roster gate per Section 6 Concern 1 resolution.)

**Excluded from sign-off** (WS-4 deferral inheritance, mirrors WS-3.5):
- `cargo build --no-default-features` — historically broken; not a WS-5 scope item

### CI pipeline composition

CI job structure (additive to existing `build + clippy + lib tests (CUDA libs, no GPU)`):

```yaml
# Existing
- name: build + clippy + lib tests (CUDA libs, no GPU)
  run: |
    cargo build --features cuda --tests
    cargo clippy --features cuda --tests -- -D warnings
    cargo test --features cuda --lib

# NEW for WS-5
- name: conformance + drift checks (parser-only CI context)
  run: |
    cargo test --features cuda --test onnx_conformance
    cargo test --features cuda --test op_support_table_consistency_test
    cargo test --features cuda --test opset_bounds_test
```

Both jobs gate PR merges. Conformance harness's `ParserOnlyCi` context auto-detects no-GPU and runs in parser-only mode (Concern 2 resolution). Implementation may parallelize the new step's three `cargo test` invocations as separate sub-steps for log isolation; not a spec concern.

### Per-milestone Definition of Done (DoD)

Each Y(N) commit is signed off only when ALL applicable gates pass:

| Y(N) | DoD = gates that must pass at this commit |
|---|---|
| Y(1) — M5.5 + M5.1 | 1, 2, 3 (lib tests now include errors inline tests + validate_model_test units) |
| Y(2) — M5.2 | 1, 2, 3, 7, 8 |
| Y(3) — M5.3 | 1, 2, 3, 4 (CI conformance lights up), 5 (developer pre-merge GPU conformance), 7, 8 |
| Y(4) — M5.4 | 1, 2, 3 (now includes differential nan_inf_tests + reference_source_tests), 4, 5, 7, 8 |
| Y(5) — Cross-repo verification | 1-8 all green; gate 6 fires (full roster sweep + leadline-bench `--validate` integration); leadline-bench `--validate` + `--bisect` integration runs against `ws5-observability` HEAD without iconnx-side adapters |

**Numerical-regression safety observation:** Y(1)-Y(4) add observability scaffolding without changing executor numerics. Y(4) (differential refactor) touches `cuda/differential.rs`, but the roster sweep doesn't exercise `DifferentialRunner` (leadline-bench's `compare` uses `analysis/diff.rs` for output-level comparison, not the differential framework). So no roster numerical regression is possible from Y(1)-Y(4) by construction. Gate 6 firing only at Y(5) is correct.

### Cross-repo verification at Y(5)

Y(5) is **not** "leadline-bench integration starts here" — leadline-bench has been parallel-prepping against API stubs since Y(1) and Y(4). Y(5) is the integration-sweep verification:

1. iconnx HEAD = Y(4) commit on `ws5-observability`
2. leadline-bench Cargo.toml temporarily pinned to `ws5-observability` branch
3. leadline-bench's `compare --validate <model>` runs against every roster profile and the synthetic-broken-model test matrix below
4. leadline-bench's `compare --bisect <model>` (when manually triggered against a synthetic gate-failing model) returns a `DivergenceReport`
5. Roster sweep at iconnx Y(5) HEAD: 10/10 PASS, zero regressions
6. Conformance harness `Gpu` context at iconnx Y(5) HEAD: zero UnexpectedFailure / UnexpectedPass; allowlist intact

#### Synthetic-broken-model test matrix for `--validate`

leadline-bench owns implementation; the matrix below is the iconnx-side specification of expected behavior:

| Synthetic case | Expected variant in report | Expected exit |
|---|---|---|
| Model with unsupported op (e.g., custom `FooOp`) | `UnsupportedOp { op_type: "FooOp", count, samples }` | `3` |
| Model with opset version above MAX | `OpsetOutOfRange { domain: "ai.onnx", version: MAX+1 }` | `3` |
| Model with unknown opset domain | `OpsetDomainUnknown { domain: "com.example.custom" }` | `3` |
| Model with non-finite BF16 constant | `warnings: [NonFiniteHalfConstants]`, `incompatibilities: []` | `0` |
| BF16 model + sm_75 mock context (`compare --validate-hw`) | `HardwareFloorUnmet { dtype: BF16, required: "sm_80+" }` | `3` |
| Multi-incompatibility (3 unsupported ops + opset out of range) | All four collected, `partial_capabilities: Some(_)` | `3` |
| File doesn't parse | `incompatibilities: [Parse(_)], partial_capabilities: None` | `3` |

#### Concrete `--bisect` verification case

Manually inject a NaN initializer into a roster Conv weight; run `compare --bisect <model>`; assert `Ok(Some(DivergenceReport { exceeded: NonFinite, op_type: "Conv", ... }))`. Validates the NaN-forced-divergence path end-to-end across the cross-repo seam.

### Merge & post-merge

- **All Y(N) commits GPG-signed** (key `0F0266D010EE6736`); never bypass signing without explicit permission per CLAUDE.md.
- **PR-via-GitHub merge at Y(5):** mirrors WS-3.5 protocol. Push to `ws5-observability` remote; open PR; CI green; surface PR URL to user as **shared-state action** for explicit consent (NOT autonomous).
- Merge strategy = `--merge` (preserves Y(N) milestone history; mirrors WS-4 PR #1 and WS-3.5 PR #2 precedent).
- **Post-merge:** leadline-bench reverts Cargo.toml pin from `ws5-observability` branch back to `../../iconnx` or main; lands the bundled `--validate` + `--bisect` flag commit on its own main.

### Tracked Follow-Ups (created by WS-5)

(none yet — populated during implementation as WS-5-introduced debt surfaces)

