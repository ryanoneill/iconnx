//! `DifferentialTolerance` — canonical, always-available tolerance type
//! shared between iconnx and downstream consumers (leadline-bench).
//!
//! Always-available (NO `debug-inference` feature gate). The runtime side
//! that consumes this type (`DifferentialRunner`, `CheckpointManager`) is
//! still gated on `debug-inference`, but the type itself is pure config-
//! side data and lives at the crate root for cross-repo type re-use.
//!
//! Implicit mode: derive operating mode from `abs.is_some() / rel.is_some()`,
//! matching leadline-bench's existing `ToleranceConfig` shape (no separate
//! `ToleranceMode` field).

/// Tolerance configuration for differential comparison.
///
/// Both `abs` and `rel` are optional. Operating mode:
///
/// - `abs = Some(_), rel = None` → absolute-only gate
/// - `abs = None, rel = Some(_)` → relative-only gate
/// - `abs = Some(_), rel = Some(_)` → dual gate (either ceiling failing
///   triggers divergence)
/// - `abs = None, rel = None` → tolerance disabled (any nonzero diff
///   triggers divergence)
///
/// NaN/Inf in iconnx output is always forced-divergence regardless of
/// configuration; see the `DifferentialRunner::run_until_divergence`
/// implementation in `crate::cuda::differential` (gated on
/// `debug-inference`).
#[derive(Debug, Clone, Copy, Default)]
pub struct DifferentialTolerance {
    /// Maximum absolute difference. `None` = no absolute gate.
    pub abs: Option<f32>,

    /// Maximum relative difference (fraction; `0.05` = 5%). `None` = no
    /// relative gate.
    pub rel: Option<f32>,
}

impl DifferentialTolerance {
    /// Strict absolute gate (matches WS-3 Whisper FP16 sign-off ceiling).
    pub fn strict() -> Self {
        Self {
            abs: Some(1e-4),
            rel: None,
        }
    }

    /// Loose dual gate — for operations with known precision differences
    /// (e.g. BF16 cross-precision). Anchored on WS-3.5 BERT-base BF16
    /// measurement.
    pub fn loose() -> Self {
        Self {
            abs: Some(1.0),
            rel: Some(0.05),
        }
    }

    /// Returns true if either bound is set.
    pub fn is_active(&self) -> bool {
        self.abs.is_some() || self.rel.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_disabled() {
        let t = DifferentialTolerance::default();
        assert!(!t.is_active());
    }

    #[test]
    fn strict_uses_abs_only() {
        let t = DifferentialTolerance::strict();
        assert!(t.abs.is_some());
        assert!(t.rel.is_none());
    }

    #[test]
    fn loose_dual_gate() {
        let t = DifferentialTolerance::loose();
        assert!(t.abs.is_some());
        assert!(t.rel.is_some());
    }
}
