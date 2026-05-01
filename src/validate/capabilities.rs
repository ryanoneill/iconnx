//! `ModelCapabilities` + `ModelWarning` — the success payload returned by
//! `validate_model()` (and embedded as `partial_capabilities` inside
//! `ModelValidationFailure`).

use crate::tensor::DType;

use super::tolerance_hint::ToleranceHint;

/// Model capability summary returned by `validate_model()` /
/// `validate_model_for_hardware()`.
///
/// All fields are populated on success; on partial-failure, the
/// `partial_capabilities` field of `ModelValidationFailure` carries a
/// best-effort instance with whatever fields were computable from what
/// passed validation.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ModelCapabilities {
    /// Per-domain opset imports as `(domain, version)` pairs. Default ONNX
    /// domain renders as `""` or `"ai.onnx"`.
    pub opset_imports: Vec<(String, i64)>,

    /// Sorted list of op_types the model uses AND iconnx implements.
    /// `used_ops ∩ supported_ops()`.
    pub supported_ops: Vec<String>,

    /// Sorted list of dtypes flowing through the graph (initializers +
    /// declared input/output types).
    pub used_dtypes: Vec<DType>,

    /// Number of initializers in the model.
    pub initializer_count: usize,

    /// Total bytes of all initializers.
    pub initializer_bytes: usize,

    /// Rough estimate of peak GPU memory consumption during inference.
    ///
    /// **Rough Y(1) heuristic** computed as:
    ///
    /// ```text
    /// estimated_peak_bytes = sum(initializer_bytes) +
    ///                        2 * mean(initializer_bytes)
    /// ```
    ///
    /// — i.e. the total weight footprint plus a small slack term derived from
    /// the average initializer size as a stand-in for activation/intermediate
    /// memory. The proper graph-walk computing
    /// `max_intermediate_tensor_bytes` over non-initializer graph values
    /// (the originally-intended formula) is deferred to a future workstream.
    /// Consumers wanting precise memory plans should run the runtime arena
    /// planner (out of scope for capability discovery).
    pub estimated_peak_bytes: usize,

    /// Tolerance estimate consumers can use as a starting point for gating
    /// numerical comparisons. Encodes WS-3.5's empirical lesson that BF16
    /// vs FP32 cross-precision diffs scale with model depth.
    pub expected_tolerance: ToleranceHint,

    /// Non-fatal observations about the model (e.g. non-finite half-precision
    /// constants from a broken FP32 → BF16 export). Always populated; an
    /// empty Vec means "no warnings detected."
    pub warnings: Vec<ModelWarning>,
}

/// Non-fatal observation about a model. Surfaces as part of
/// `ModelCapabilities::warnings`. Distinct from `ModelIncompatibility` —
/// warnings don't fail validation.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ModelWarning {
    /// One or more half-precision (BF16/FP16) initializers contain non-finite
    /// values (NaN or ±Inf). Per WS-3.5 BERT BF16 root-cause: a naive FP32 →
    /// half-precision export can overflow ±FLT_MAX to ±Inf, triggering NaN
    /// cascades through downstream `Mul(0, ±Inf)` patterns. Non-fatal —
    /// the model may still run — but typically indicates a broken export.
    NonFiniteHalfConstants {
        tensor_name: String,
        count: usize,
        total: usize,
    },

    /// The model required iconnx's legacy shape-correction (an initializer's
    /// declared dims differ from its flat data length). Re-exporting with
    /// matching shapes is recommended.
    LegacyShapeRewriteRequired {
        tensor_name: String,
    },

    /// The model's transformer depth (LayerNormalization node count) exceeds
    /// the empirical tolerance-hint table's coverage. The returned
    /// `ToleranceHint` will be `Estimated` rather than `Measured`; callers
    /// may want to adjust expectations.
    DepthExceedsTolerancePrior {
        layers: usize,
        dtype: DType,
    },

    /// iconnx had to compress a source dtype that lacks a dedicated GPU
    /// [`DType`] tag, surfacing the source dtype under a different `DType`
    /// in [`ModelCapabilities::used_dtypes`]. Today this is FLOAT64(11) →
    /// `DType::Float32` (the GPU path doesn't accept f64 today, so for
    /// capability-discovery purposes f64 weights are reported under the
    /// Float32 bucket).
    ///
    /// Surfaced once per model regardless of how many initializers triggered
    /// the downcast — consumers should read it as "the model has at least
    /// one tensor whose source dtype was not preserved in the GPU dtype
    /// taxonomy," not as a per-tensor count.
    ///
    /// `from` is the source dtype label as returned by [`crate::tensor::Tensor::dtype`]
    /// (e.g. `"float64"`); `to` is the GPU [`DType`] iconnx reported in its place.
    DTypeDowncast {
        from: &'static str,
        to: DType,
    },
}
