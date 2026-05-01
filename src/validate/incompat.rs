//! `ModelValidationFailure` and `ModelIncompatibility` — the collect-all
//! aggregator returned by `validate_model()` and `validate_model_for_hardware()`.

use thiserror::Error;

use crate::onnx_parser::ParseError;
use crate::tensor::DType;

use super::capabilities::ModelCapabilities;

/// Aggregated validation failure with collect-all incompatibility list.
///
/// Returned by [`super::validate_model`] and
/// [`super::validate_model_for_hardware`] when one or more incompatibilities
/// are detected. Callers iterate over `incompatibilities` to enumerate every
/// surfaced issue rather than receiving a single short-circuited variant.
#[derive(Debug, Clone, Error)]
#[error("model validation failed with {} incompatibilities", incompatibilities.len())]
#[non_exhaustive]
pub struct ModelValidationFailure {
    /// All incompatibilities detected. Iteration order is the natural order
    /// the orchestration walks (parse → ops → opsets → initializers →
    /// hardware), but consumers should not rely on that order — sort or
    /// group by variant if rendering.
    pub incompatibilities: Vec<ModelIncompatibility>,

    /// Best-effort capability information available even when validation fails.
    ///
    /// `Some(caps)` whenever parse succeeded, regardless of incompatibility
    /// count — `caps` reflects every field computable from what passed (e.g.
    /// `opset_imports` always populated, `supported_ops` = used_ops minus the
    /// unsupported set). Lets a UI render "this model uses ops [list],
    /// dtypes [list], opset [N], EXCEPT for these incompatibilities [list]"
    /// rather than just the failure list.
    ///
    /// `None` only on the file-level parse short-circuit case (model proto
    /// decode); always `Some(_)` otherwise, even when the incompatibility
    /// list contains a `Parse(_)` from later-stage initializer decode.
    pub partial_capabilities: Option<ModelCapabilities>,
}

/// One incompatibility variant. WS-5 ships six variants; future workstreams
/// may add more (the enum is `#[non_exhaustive]`).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ModelIncompatibility {
    /// Op type used by the model but not in iconnx's `OP_SUPPORT_TABLE`.
    UnsupportedOp {
        op_type: String,
        count: usize,
        /// Up to 3 sample node names where this op_type appears, for
        /// downstream tooling locating the offenders in the graph.
        sample_node_names: Vec<String>,
    },

    /// Dtype used by the model that iconnx does not support.
    UnsupportedDType {
        dtype: DType,
        reason: &'static str,
    },

    /// Opset version is outside iconnx's roster-validated range.
    OpsetOutOfRange {
        domain: String,
        version: i64,
        supported: (i64, i64),
    },

    /// Opset import declares a domain iconnx does not recognize (e.g.
    /// `com.microsoft` custom domain). Non-fatal in collect-all sense.
    OpsetDomainUnknown {
        domain: String,
        version: i64,
    },

    /// Hardware floor (e.g. sm_80+ for BF16) not met by the supplied
    /// `IconnxCudaContext`. Only producible by `validate_model_for_hardware`;
    /// the static `validate_model` cannot produce this variant.
    HardwareFloorUnmet {
        dtype: DType,
        required: &'static str,
        found: String,
    },

    /// Underlying parse error.
    ///
    /// Appears in two cases:
    /// - **File-level parse failure** (model proto decode failed): the entire
    ///   validation short-circuits; `partial_capabilities` is `None`.
    /// - **Initializer decode failure** after a successful file-level parse:
    ///   composes with any prior incompatibilities; `partial_capabilities` is
    ///   `Some(_)` populated with fields computable from the parseable model
    ///   (opset_imports, supported_ops list, layer count, declared
    ///   input/output dtypes — initializer-derived fields zeroed).
    ///
    /// Inspect `partial_capabilities` to disambiguate.
    Parse(ParseError),
}
