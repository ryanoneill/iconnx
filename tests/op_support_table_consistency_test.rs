//! Drift fence — source-parses dispatch arms via regex over executor source
//! files at compile time, compares to `OP_SUPPORT_TABLE`. Adding a new op
//! requires exactly two edits: dispatch arm + table entry. The test catches
//! single-side drift in either direction.
//!
//! This is the single safety net catching the cross-tier dispatch lesson
//! from WS-3.5 (BERT BF16 Cast hotfix surfaced an executor-tier dispatch
//! site that a kernel-tier audit missed). Encoded in the test mechanism
//! rather than the human reviewer's checklist: any future maintainer who
//! adds `"FooBar" =>` to `src/cuda/executor/*.rs` without an
//! `OP_SUPPORT_TABLE` entry (or vice versa) trips this fence.
//!
//! Mechanism:
//! - `include_str!` every `src/cuda/executor/*.rs` source file at compile
//!   time. New executor source files must be added to the list below;
//!   that's a one-time addition that's hard to forget (the test won't
//!   compile-cover the new file otherwise, but the existing body of arms
//!   is preserved).
//! - Walk each source with the regex `"([A-Z][a-zA-Z0-9]+)"\s*=>` to
//!   harvest op_type literals from match arms. The `=>` anchor naturally
//!   filters out string-concat / panic-message / format false positives.
//!   Alternation patterns like `"Add" | "Sub" =>` are handled because the
//!   regex matches each literal independently against the trailing `=>`
//!   (the alternatives without trailing `=>` are still picked up by the
//!   trailing `=>` of the final alternative once the match restarts).
//!   Note: alternation siblings without trailing `=>` are picked up
//!   because the regex re-runs at each position; `"Add"` followed by
//!   `| "Sub" =>` matches `"Sub" =>` directly, but `"Add"` is also
//!   harvested since the regex pattern fires when the cursor scans
//!   `"Sub" =>`. To be precise, alternation is handled by
//!   pre-processing: we strip `|` and re-run the regex per literal.
//! - Special-case: the `"Constant"` op is handled via
//!   `if op.node.op_type == "Constant"` in `mod.rs` and `variants.rs`,
//!   not via a match arm. The regex won't catch it; we add it manually
//!   below with a clear comment.

#![cfg(feature = "cuda")]

use std::collections::HashSet;

use iconnx::op_support::OP_SUPPORT_TABLE;
use regex::Regex;

/// Concatenated source of every executor file. New files added under
/// `src/cuda/executor/` must be appended here.
const EXECUTOR_SOURCES: &[(&str, &str)] = &[
    ("mod.rs", include_str!("../src/cuda/executor/mod.rs")),
    ("layout.rs", include_str!("../src/cuda/executor/layout.rs")),
    ("variants.rs", include_str!("../src/cuda/executor/variants.rs")),
    ("elementwise.rs", include_str!("../src/cuda/executor/elementwise.rs")),
    ("conv.rs", include_str!("../src/cuda/executor/conv.rs")),
    ("reduction.rs", include_str!("../src/cuda/executor/reduction.rs")),
    ("matmul.rs", include_str!("../src/cuda/executor/matmul.rs")),
    ("misc.rs", include_str!("../src/cuda/executor/misc.rs")),
    ("fused.rs", include_str!("../src/cuda/executor/fused.rs")),
    ("lstm.rs", include_str!("../src/cuda/executor/lstm.rs")),
    ("stft.rs", include_str!("../src/cuda/executor/stft.rs")),
];

/// Extract every op_type literal that drives a match arm in the executor
/// source. Returns the set across all source files.
fn extract_dispatch_arms_from_source() -> HashSet<&'static str> {
    // Match `"OpName" =>` patterns. The trailing `=>` filters out string
    // literals appearing in panic / format / error messages. Whitespace
    // between the literal and `=>` is permitted (formatter-friendly).
    let arm_re = Regex::new(r#""([A-Z][a-zA-Z0-9]+)"\s*=>"#).expect("regex compiles");

    // Some arms are written as alternation (`"Add" | "Sub" | "Mul" =>`).
    // The trailing `=>` only attaches to the final literal; siblings
    // need a separate pass. We handle this by also scanning for
    // `"OpName" |` patterns — every literal preceding a `|` followed by
    // (eventually) a `=>` belongs to the same arm.
    let alt_re = Regex::new(r#""([A-Z][a-zA-Z0-9]+)"\s*\|"#).expect("regex compiles");

    let mut extracted: HashSet<&'static str> = HashSet::new();

    for (_label, source) in EXECUTOR_SOURCES {
        // Skip lines containing panic!, eprintln!, format!, or error: —
        // belt-and-braces against false positives that would slip past
        // the `=>` anchor in unusual formatting.
        for line in source.lines() {
            let lower = line.to_ascii_lowercase();
            if lower.contains("panic!")
                || lower.contains("eprintln!")
                || lower.contains("format!")
                || lower.contains("error:")
            {
                continue;
            }
            for cap in arm_re.captures_iter(line) {
                // Leak the &'static str lifetime by using the source's
                // 'static lifetime: `include_str!` produces &'static str,
                // and `cap.get(1).unwrap().as_str()` borrows from that
                // 'static buffer.
                let lit: &'static str = cap.get(1).expect("group 1 present").as_str();
                extracted.insert(lit);
            }
            for cap in alt_re.captures_iter(line) {
                let lit: &'static str = cap.get(1).expect("group 1 present").as_str();
                extracted.insert(lit);
            }
        }
    }

    // The `"Constant"` op_type is handled via
    // `if op.node.op_type == "Constant"` (an if-let style guard) in
    // `mod.rs` and `variants.rs`, not a match arm. The regex won't catch
    // it; insert manually with a comment so future readers see why this
    // is here.
    extracted.insert("Constant"); // handled via if-let, not match

    extracted
}

#[test]
fn dispatch_arms_match_op_support_table() {
    let extracted = extract_dispatch_arms_from_source();
    let table: HashSet<&'static str> =
        OP_SUPPORT_TABLE.iter().map(|e| e.op_type).collect();

    let only_in_arms: Vec<&&str> = extracted.difference(&table).collect();
    let only_in_table: Vec<&&str> = table.difference(&extracted).collect();

    assert!(
        only_in_arms.is_empty() && only_in_table.is_empty(),
        "Drift between dispatch arms and OP_SUPPORT_TABLE:\n\
         Only in dispatch arms (add table entry): {:?}\n\
         Only in OP_SUPPORT_TABLE (add dispatch arm or remove entry): {:?}",
        only_in_arms,
        only_in_table,
    );
}

#[test]
fn extracted_dispatch_arms_set_nonempty_and_includes_known_arms() {
    // Sanity check that the regex/source-parse mechanism is wired up:
    // the extracted set should be non-empty and contain at least a few
    // canonical entries. Guards against e.g. a path typo silently
    // producing an empty include_str! body.
    let extracted = extract_dispatch_arms_from_source();
    assert!(
        !extracted.is_empty(),
        "source-parse extracted no dispatch arms — include_str! paths likely broken"
    );
    for known in &["Add", "MatMul", "Conv", "Cast", "Constant", "Split"] {
        assert!(
            extracted.contains(known),
            "expected source-parse to find {:?}; got {:?}",
            known,
            extracted
        );
    }
}
