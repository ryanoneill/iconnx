//! `OP_SUPPORT_TABLE` — single source of truth for iconnx's per-op support
//! discovery.
//!
//! Drift-prevention: `tests/op_support_table_consistency_test.rs` asserts
//! that the table entries match the dispatch_op + dispatch_layout match
//! arms in `src/cuda/executor/`. Adding a new op = one new table entry +
//! one new dispatch arm; the consistency test fails if either is missed.
//!
//! Per-op opset metadata (per-op min/max opset versions) is intentionally
//! absent — the model-level `MIN_OPSET_SUPPORTED` / `MAX_OPSET_SUPPORTED`
//! bounds in `src/opset.rs` are sufficient for WS-5. Future workstreams
//! may add per-op opset fields here without breaking consumers (struct is
//! `#[non_exhaustive]`).

use std::collections::HashSet;
use std::sync::OnceLock;

/// One row of the per-op support table.
#[non_exhaustive]
pub struct OpSupport {
    pub op_type: &'static str,
    // Future fields (cpu: bool, opset_range, ...) reserved.
}

/// The single-source-of-truth table.
///
/// Entries empirically derived from the dispatch match arms in
/// `src/cuda/executor/`. To add a new op:
/// 1. Add a dispatch arm in the appropriate executor file.
/// 2. Add an entry here.
/// 3. The consistency test in `tests/op_support_table_consistency_test.rs`
///    will pass once both are present.
pub const OP_SUPPORT_TABLE: &[OpSupport] = &[
    // Elementwise arith / unary math / activations / comparisons
    // (src/cuda/executor/mod.rs::dispatch_op):
    OpSupport { op_type: "Add" },
    OpSupport { op_type: "Sub" },
    OpSupport { op_type: "Mul" },
    OpSupport { op_type: "Div" },
    OpSupport { op_type: "Pow" },
    OpSupport { op_type: "Exp" },
    OpSupport { op_type: "Erf" },
    OpSupport { op_type: "Sqrt" },
    OpSupport { op_type: "Sin" },
    OpSupport { op_type: "Cos" },
    OpSupport { op_type: "Tanh" },
    OpSupport { op_type: "Sigmoid" },
    OpSupport { op_type: "Relu" },
    OpSupport { op_type: "LeakyRelu" },
    OpSupport { op_type: "Atan" },
    OpSupport { op_type: "Round" },
    OpSupport { op_type: "Floor" },
    OpSupport { op_type: "Clip" },
    OpSupport { op_type: "Softmax" },
    OpSupport { op_type: "Equal" },
    OpSupport { op_type: "Less" },
    OpSupport { op_type: "Greater" },
    OpSupport { op_type: "GreaterOrEqual" },
    OpSupport { op_type: "LessOrEqual" },
    OpSupport { op_type: "NotEqual" },
    // Conv (src/cuda/executor/mod.rs::dispatch_op):
    OpSupport { op_type: "Conv" },
    OpSupport { op_type: "ConvTranspose" },
    // Reductions (src/cuda/executor/mod.rs::dispatch_op):
    OpSupport { op_type: "ReduceMean" },
    OpSupport { op_type: "ReduceSum" },
    OpSupport { op_type: "LayerNormalization" },
    OpSupport { op_type: "GlobalAveragePool" },
    // MatMul (src/cuda/executor/mod.rs::dispatch_op + layout.rs for MatMulInteger):
    OpSupport { op_type: "Gemm" },
    OpSupport { op_type: "MatMul" },
    OpSupport { op_type: "MatMulInteger" },
    // Recurrent (src/cuda/executor/mod.rs::dispatch_op):
    OpSupport { op_type: "LSTM" },
    // STFT (src/cuda/executor/mod.rs::dispatch_op):
    OpSupport { op_type: "STFT" },
    // Logical (src/cuda/executor/mod.rs::dispatch_op):
    OpSupport { op_type: "And" },
    OpSupport { op_type: "Or" },
    OpSupport { op_type: "Not" },
    // Multi-output dispatch (src/cuda/executor/mod.rs::dispatch_op_multi):
    OpSupport { op_type: "Split" },
    OpSupport { op_type: "DynamicQuantizeLinear" },
    // Top-level Constant special-case (src/cuda/executor/mod.rs run loop +
    // src/cuda/executor/variants.rs four instrumented variants):
    OpSupport { op_type: "Constant" },
    // Layout / shape / metadata (src/cuda/executor/layout.rs::dispatch_layout):
    OpSupport { op_type: "Where" },
    OpSupport { op_type: "Shape" },
    OpSupport { op_type: "Cast" },
    OpSupport { op_type: "ConstantOfShape" },
    OpSupport { op_type: "Transpose" },
    OpSupport { op_type: "Gather" },
    OpSupport { op_type: "Slice" },
    OpSupport { op_type: "MaxPool" },
    OpSupport { op_type: "Flatten" },
    OpSupport { op_type: "Reshape" },
    OpSupport { op_type: "Squeeze" },
    OpSupport { op_type: "Unsqueeze" },
    OpSupport { op_type: "Identity" },
    OpSupport { op_type: "Concat" },
    OpSupport { op_type: "Expand" },
    OpSupport { op_type: "Pad" },
    OpSupport { op_type: "Resize" },
    OpSupport { op_type: "NonZero" },
    OpSupport { op_type: "ScatterND" },
    OpSupport { op_type: "Range" },
    OpSupport { op_type: "CumSum" },
    OpSupport { op_type: "DequantizeLinear" },
];

/// Returns the cached set of op_types iconnx supports. Build-once via
/// `OnceLock`; subsequent calls return the same `&'static HashSet`.
pub fn supported_ops() -> &'static HashSet<&'static str> {
    static CACHE: OnceLock<HashSet<&'static str>> = OnceLock::new();
    CACHE.get_or_init(|| OP_SUPPORT_TABLE.iter().map(|e| e.op_type).collect())
}

/// Conformance harness execution context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchContext {
    /// CI context: no GPU available. Conformance harness verifies parse +
    /// op-recognition only; no execution comparison performed. iconnx has
    /// no CPU executor today, so `ParserOnlyCi` is the sole non-GPU
    /// dispatch context.
    ParserOnlyCi,

    /// Local developer / GPU runner: full GPU execution + reference
    /// comparison.
    Gpu,
}

/// Result of looking up an op in `OP_SUPPORT_TABLE` for a given dispatch
/// context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LookupResult {
    /// Op is in the table and the context can execute it.
    Runnable,

    /// Op is in the table; current context can verify parse + recognition
    /// but cannot execute (e.g. ParserOnlyCi).
    ParserOnlyRecognized,

    /// Op is not in the table.
    Unsupported,
}

/// Look up an op_type's support status in the given dispatch context.
pub fn lookup(op_type: &str, ctx: DispatchContext) -> LookupResult {
    if !supported_ops().contains(op_type) {
        return LookupResult::Unsupported;
    }
    match ctx {
        DispatchContext::Gpu => LookupResult::Runnable,
        DispatchContext::ParserOnlyCi => LookupResult::ParserOnlyRecognized,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supported_ops_cached_returns_same_reference() {
        let s1 = supported_ops();
        let s2 = supported_ops();
        // Same OnceLock value -> same reference.
        assert!(std::ptr::eq(s1, s2));
    }

    #[test]
    fn supported_ops_includes_known_ops() {
        let ops = supported_ops();
        assert!(ops.contains("Add"));
        assert!(ops.contains("MatMul"));
        assert!(ops.contains("Conv"));
        assert!(ops.contains("Cast"));
        assert!(ops.contains("LayerNormalization"));
    }

    #[test]
    fn supported_ops_excludes_unknown() {
        let ops = supported_ops();
        assert!(!ops.contains("CustomOpFoo"));
    }

    #[test]
    fn lookup_runnable_for_supported_gpu() {
        assert_eq!(lookup("Add", DispatchContext::Gpu), LookupResult::Runnable);
    }

    #[test]
    fn lookup_parser_only_for_supported_ci() {
        assert_eq!(
            lookup("Add", DispatchContext::ParserOnlyCi),
            LookupResult::ParserOnlyRecognized
        );
    }

    #[test]
    fn lookup_unsupported_regardless_of_context() {
        assert_eq!(
            lookup("CustomOpFoo", DispatchContext::Gpu),
            LookupResult::Unsupported
        );
        assert_eq!(
            lookup("CustomOpFoo", DispatchContext::ParserOnlyCi),
            LookupResult::Unsupported
        );
    }
}
