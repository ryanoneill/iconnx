//! Drift-prevention: assert that every op_type with a dispatch arm in
//! `src/cuda/executor/` has an entry in `OP_SUPPORT_TABLE`, and vice versa.
//!
//! This is the single safety net catching the cross-tier dispatch lesson
//! from WS-3.5 (BERT BF16 Cast hotfix surfaced an executor-tier dispatch
//! site that a kernel-tier audit missed). Adding a new op now requires
//! both an executor match arm AND a table entry — drift fails this test.

#![cfg(feature = "cuda")]

use std::collections::HashSet;

use iconnx::op_support::OP_SUPPORT_TABLE;

/// Hand-curated extraction of op_type literals from the executor source.
/// Updated whenever a new dispatch arm is added.
///
/// Each entry corresponds to a `"OpName"` literal in a match arm under
/// `src/cuda/executor/`. The match-arm source comments where each lives
/// are below.
fn dispatch_arms() -> HashSet<&'static str> {
    [
        // From src/cuda/executor/mod.rs::dispatch_op (elementwise/conv/reduce/matmul/lstm/stft/misc):
        "Add", "Sub", "Mul", "Div", "Pow", "Exp", "Erf", "Sqrt", "Sin", "Cos",
        "Tanh", "Sigmoid", "Relu", "LeakyRelu", "Atan", "Round", "Floor", "Clip",
        "Softmax", "Equal", "Less", "Greater", "GreaterOrEqual", "LessOrEqual",
        "NotEqual", "Conv", "ConvTranspose", "ReduceMean", "ReduceSum",
        "LayerNormalization", "GlobalAveragePool", "Gemm", "MatMul",
        "MatMulInteger", "LSTM", "STFT", "And", "Or", "Not",
        // From src/cuda/executor/mod.rs::dispatch_op_multi:
        "Split", "DynamicQuantizeLinear",
        // From src/cuda/executor/mod.rs run loop + src/cuda/executor/variants.rs
        // (top-level Constant special-case, four instrumented variants):
        "Constant",
        // From src/cuda/executor/layout.rs::dispatch_layout:
        "Reshape", "Transpose", "Concat", "Slice", "Unsqueeze", "Squeeze",
        "Shape", "Gather", "Cast", "Pad", "Resize", "Range", "Where", "Expand",
        "ConstantOfShape", "ScatterND", "NonZero", "CumSum", "Identity",
        "Flatten", "MaxPool", "DequantizeLinear",
    ]
    .into_iter()
    .collect()
}

#[test]
fn every_dispatch_arm_has_table_entry() {
    let arms = dispatch_arms();
    let table: HashSet<_> = OP_SUPPORT_TABLE.iter().map(|e| e.op_type).collect();
    let missing: Vec<_> = arms.difference(&table).copied().collect();
    assert!(
        missing.is_empty(),
        "Dispatch arms missing from OP_SUPPORT_TABLE: {:?}\n\n\
         When adding an op, add a row to OP_SUPPORT_TABLE in src/op_support.rs.",
        missing
    );
}

#[test]
fn every_table_entry_has_dispatch_arm() {
    let arms = dispatch_arms();
    let table: HashSet<_> = OP_SUPPORT_TABLE.iter().map(|e| e.op_type).collect();
    let extra: Vec<_> = table.difference(&arms).copied().collect();
    assert!(
        extra.is_empty(),
        "OP_SUPPORT_TABLE entries with no dispatch arm: {:?}\n\n\
         Either add a dispatch arm in src/cuda/executor/, or remove the entry.",
        extra
    );
}

#[test]
fn dispatch_arms_set_matches_table_set() {
    let arms = dispatch_arms();
    let table: HashSet<_> = OP_SUPPORT_TABLE.iter().map(|e| e.op_type).collect();
    assert_eq!(arms, table, "drift between dispatch_arms() and OP_SUPPORT_TABLE");
}
