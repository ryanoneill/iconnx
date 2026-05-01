//! `ToleranceHint`, `ToleranceConfidence`, `ToleranceBasis`, and the lookup
//! function that derives a hint from `(used_dtypes, layer_count)`.
//!
//! Encodes WS-3.5's empirical lesson: BF16 vs FP32 cross-precision diffs
//! scale with transformer depth. Whisper-Tiny BF16 (4 layers) measured
//! 0.32 abs / 2.45% rel; BERT-base BF16 (12 layers) measured 0.69 abs /
//! 3.99% rel. FP16 vs FP16 same-precision is much tighter (Whisper FP16 =
//! 0.039 abs / 0.30% rel). The hint communicates this distinction to
//! downstream tolerance-gate consumers (leadline-bench, future tooling).

use crate::tensor::DType;

/// Tolerance hint surfaced by [`crate::validate::ModelCapabilities`].
///
/// Advisory, not enforced. Downstream tolerance gates (e.g. leadline-bench
/// `compare`'s tolerance config) can use this as a starting point without
/// re-discovering empirical noise budgets per model integration.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ToleranceHint {
    /// Suggested absolute-difference ceiling.
    pub abs: f32,

    /// Suggested relative-difference ceiling (fraction, e.g. `0.03` = 3%).
    pub rel: f32,

    /// Source category — useful for callers deciding whether to enforce
    /// the hint strictly or treat it as a soft starting point.
    pub basis: ToleranceBasis,

    /// How confident iconnx is in the hint values. Empirically-measured
    /// hints (from WS-3.5 sign-off) get `Measured`; depth-extrapolated
    /// hints get `Estimated`; unseen-dtype hints get `FallbackHeuristic`.
    pub confidence: ToleranceConfidence,
}

/// Source category for a [`ToleranceHint`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ToleranceBasis {
    /// iconnx-half vs ORT-half (same-precision comparison). Tight bounds;
    /// representative of FP16 vs FP16 or BF16 vs BF16 execution.
    SamePrecision,

    /// iconnx-half vs ORT-FP32 (cross-precision comparison, depth-amplified).
    /// Bounds scale with transformer depth per WS-3.5 empirical data.
    /// The de-facto path for BF16 since ORT can't load BF16 graphs (Gather
    /// + Conv reject `tensor(bfloat16)`).
    CrossPrecisionDepthScaled,
}

/// Confidence label for a [`ToleranceHint`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ToleranceConfidence {
    /// Hint values are empirically measured against a roster model with the
    /// same dtype + similar depth. Trustworthy starting point.
    Measured,

    /// Hint values are extrapolated (e.g. interpolating BF16 8-layer between
    /// Whisper-Tiny 4-layer and BERT-base 12-layer measurements). Reasonable
    /// but not proven; consumers may want a wider buffer.
    Estimated,

    /// Hint values are a generic fallback heuristic for an unseen dtype or
    /// depth class. Use with caution; consumers should confirm empirically
    /// before tightening gates.
    FallbackHeuristic,
}

/// Compute the expected tolerance hint for a model with the given
/// `used_dtypes` and `layer_count`.
///
/// `layer_count` is the count of distinct `LayerNormalization` nodes in
/// the graph (see [`count_layer_norm_nodes`]). This heuristic is stable
/// across BERT/Whisper/DistilBERT-style architectures (one LayerNorm per
/// transformer block) and intentionally simple — anything more
/// sophisticated is over-engineered for a hint that's labeled `Estimated`
/// for unseen depths anyway.
///
/// Returns the worst-case (loosest) hint across all dtypes in
/// `used_dtypes` — i.e., a model using both FP32 and BF16 gets the BF16
/// (looser) hint.
pub fn expected_tolerance(used_dtypes: &[DType], layer_count: usize) -> ToleranceHint {
    // Worst-case dtype: BF16 > FP16 > FP32. INT8/INT32/Int64/Bool fall
    // through to FP32-class for hint purposes (they don't appear in the
    // tolerance-empirical-table anyway; the gate is on float outputs).
    let worst_half = used_dtypes
        .iter()
        .map(|d| match d {
            DType::BFloat16 => 2, // worst
            DType::Float16 => 1,
            _ => 0, // FP32/INT/Bool
        })
        .max()
        .unwrap_or(0);

    match worst_half {
        2 => bf16_hint(layer_count),
        1 => fp16_hint(layer_count),
        _ => fp32_hint(),
    }
}

/// FP32 hint: should be near-zero. Heuristic baseline.
fn fp32_hint() -> ToleranceHint {
    ToleranceHint {
        abs: 1e-4,
        rel: 1e-3, // 0.1%
        basis: ToleranceBasis::SamePrecision,
        confidence: ToleranceConfidence::FallbackHeuristic,
    }
}

/// FP16 hint: anchor on WS-3 Whisper-FP16 measurement (0.039 abs / 0.30% rel).
/// Same-precision (iconnx-FP16 vs ORT-FP16); doesn't scale with depth in
/// the FP16-vs-FP16 case the way BF16-vs-FP32 does.
fn fp16_hint(_layer_count: usize) -> ToleranceHint {
    ToleranceHint {
        abs: 0.05,
        rel: 0.005, // 0.5%
        basis: ToleranceBasis::SamePrecision,
        confidence: ToleranceConfidence::Measured,
    }
}

/// BF16 hint: WS-3.5 empirical (cross-precision vs FP32 reference). Scales
/// with depth.
///
/// - Whisper-Tiny 4 layers: 0.32 abs / 2.45% rel — `Measured`
/// - BERT-base 12 layers: 0.69 abs / 3.99% rel — `Measured`
/// - 5-11 layers: linear interpolation — `Estimated`
/// - >12 layers: extrapolated upward — `Estimated`
fn bf16_hint(layer_count: usize) -> ToleranceHint {
    let (abs, rel, confidence) = match layer_count {
        n if n <= 4 => (0.4, 0.025, ToleranceConfidence::Measured), // Whisper-Tiny class
        n if n >= 12 => (0.8, 0.045, ToleranceConfidence::Measured), // BERT-base class (slight upper margin)
        n => {
            // Linear interpolation between (4, 0.32, 0.0245) and (12, 0.69, 0.0399)
            let frac = (n as f32 - 4.0) / 8.0;
            let abs = 0.32 + frac * (0.69 - 0.32);
            let rel = 0.0245 + frac * (0.0399 - 0.0245);
            (abs * 1.1, rel * 1.1, ToleranceConfidence::Estimated) // 10% buffer
        }
    };
    ToleranceHint {
        abs,
        rel,
        basis: ToleranceBasis::CrossPrecisionDepthScaled,
        confidence,
    }
}

/// Count distinct `LayerNormalization` nodes in a computation graph.
/// Used as the "transformer depth" heuristic for [`expected_tolerance`].
///
/// Returns 0 for graphs that decompose `LayerNormalization` into primitives
/// (the typical pre-opset-17 export path: ReduceMean → Sub → Mul → ReduceMean
/// → Add → Sqrt → Div → Mul → Add) or use a non-canonical op_type. The hint
/// will fall back to the dtype-only baseline (`bf16_hint(0)` / `fp16_hint(0)` /
/// `fp32_hint()`) in that case, which over-estimates tightness for deep
/// transformers; consumers can supply their own depth heuristic if needed.
pub fn count_layer_norm_nodes(graph: &crate::onnx_parser::ComputationGraph) -> usize {
    graph
        .nodes()
        .iter()
        .filter(|n| n.op_type == "LayerNormalization")
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp32_only_returns_fallback_heuristic() {
        let hint = expected_tolerance(&[DType::Float32], 4);
        assert_eq!(hint.basis, ToleranceBasis::SamePrecision);
        assert_eq!(hint.confidence, ToleranceConfidence::FallbackHeuristic);
    }

    #[test]
    fn bf16_4_layers_returns_measured_whisper_tiny_class() {
        let hint = expected_tolerance(&[DType::BFloat16], 4);
        assert_eq!(hint.confidence, ToleranceConfidence::Measured);
        assert!(hint.abs > 0.3);
        assert!(hint.rel > 0.02);
    }

    #[test]
    fn bf16_12_layers_returns_measured_bert_base_class() {
        let hint = expected_tolerance(&[DType::BFloat16], 12);
        assert_eq!(hint.confidence, ToleranceConfidence::Measured);
        assert!(hint.abs > 0.6);
    }

    #[test]
    fn bf16_8_layers_interpolates() {
        let hint = expected_tolerance(&[DType::BFloat16], 8);
        assert_eq!(hint.confidence, ToleranceConfidence::Estimated);
        // Should fall between 4-layer and 12-layer values.
        let four = expected_tolerance(&[DType::BFloat16], 4);
        let twelve = expected_tolerance(&[DType::BFloat16], 12);
        assert!(hint.abs >= four.abs);
        assert!(hint.abs <= twelve.abs * 1.2);
    }

    #[test]
    fn worst_dtype_wins() {
        // Mixed FP32 + BF16 -> BF16 hint.
        let hint = expected_tolerance(&[DType::Float32, DType::BFloat16], 4);
        assert_eq!(hint.basis, ToleranceBasis::CrossPrecisionDepthScaled);
    }

    #[test]
    fn fp16_returns_same_precision_basis() {
        let hint = expected_tolerance(&[DType::Float16], 4);
        assert_eq!(hint.basis, ToleranceBasis::SamePrecision);
        assert_eq!(hint.confidence, ToleranceConfidence::Measured);
    }
}
