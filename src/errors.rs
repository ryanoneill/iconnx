//! Top-level error type wrapping iconnx's typed leaf errors via `#[from]` adapters.
//!
//! Introduced in WS-5 M5.5 to subsume WS-1's M1.6 deferral. Provides
//! `iconnx::Result<T>` as the consumer-facing default for callers that don't
//! want to enumerate over `ParseError`, `CudaError`, etc. individually.
//!
//! Existing typed leaf errors (`ParseError`, `CudaError`, `DifferentialError`)
//! retain their public paths and are unchanged. `IconnxError` is purely
//! additive: callers who previously matched on `ParseError` directly continue
//! to compile; new code can opt into `IconnxError` for type-erasure.
//!
//! ## Stability
//!
//! Pre-1.0 prototype. The variant set is `#[non_exhaustive]` — additive
//! changes don't break consumers. Structural changes (rename a variant,
//! change a field's type) require a minor-version bump pre-1.0 and a major
//! bump post-1.0. Document any churn in CHANGELOG entries.

use thiserror::Error;

#[cfg(feature = "cuda")]
use crate::cuda::CudaError;
use crate::onnx_parser::ParseError;
#[cfg(feature = "cuda")]
use crate::validate::ModelValidationFailure;

#[cfg(feature = "debug-inference")]
use crate::cuda::differential::DifferentialError;

/// Top-level iconnx error type.
///
/// Wraps every leaf error iconnx can produce via `#[from]` adapters. Consumers
/// using `iconnx::Result<T>` for ergonomic propagation receive this enum and
/// should match on its variants (or destructure further into the inner leaf).
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum IconnxError {
    #[error("parse error: {0}")]
    Parse(#[from] ParseError),

    #[cfg(feature = "cuda")]
    #[error("CUDA error: {0}")]
    Cuda(#[from] CudaError),

    /// Convenience type-erasure for callers using `iconnx::Result<T>`.
    /// Note: this preserves the full `Vec<ModelIncompatibility>` and
    /// `partial_capabilities` payload from the inner failure. Callers who
    /// want to enumerate incompatibilities should use the structured
    /// `Result<_, ModelValidationFailure>` returned directly by
    /// `validate_model()` / `validate_model_for_hardware()` rather than
    /// destructuring through this variant. The convenience wrapper is
    /// intended for "log-and-bail" paths.
    ///
    /// Gated under `feature = "cuda"` because `ModelValidationFailure`
    /// transitively depends on `crate::tensor::DType` which is itself
    /// cuda-gated; mirroring the existing `Cuda` variant gating keeps
    /// the error surface internally consistent.
    #[cfg(feature = "cuda")]
    #[error("model validation failed: {0}")]
    Validation(#[from] ModelValidationFailure),

    #[cfg(feature = "debug-inference")]
    #[error("differential test error: {0}")]
    Differential(#[from] DifferentialError),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

/// Convenience alias for `Result<T, IconnxError>`.
pub type Result<T> = std::result::Result<T, IconnxError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_error_converts_via_from() {
        let pe = ParseError::UnsupportedDType {
            name: "x".into(),
            dtype_id: 99,
        };
        let ie: IconnxError = pe.into();
        assert!(matches!(ie, IconnxError::Parse(_)));
    }

    #[test]
    fn io_error_converts_via_from() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "missing");
        let ie: IconnxError = io_err.into();
        assert!(matches!(ie, IconnxError::Io(_)));
    }

    #[test]
    fn display_includes_inner_message() {
        let pe = ParseError::UnsupportedDType {
            name: "x".into(),
            dtype_id: 99,
        };
        let ie = IconnxError::from(pe);
        let s = format!("{}", ie);
        assert!(s.contains("parse error"));
        assert!(s.contains("99"));
    }

    #[test]
    fn source_chain_walks() {
        use std::error::Error;
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "missing");
        let ie = IconnxError::from(io_err);
        let inner = ie.source().expect("source present");
        // The inner is the std::io::Error.
        assert!(inner.to_string().contains("missing"));
    }

    #[test]
    fn result_alias_resolves() {
        // The `result_large_err` lint fires here because `IconnxError`
        // wraps the (intentionally-large) `ModelValidationFailure`
        // variant. Allowed for the same reason as the validate API:
        // boxing the error would break the public surface.
        #[allow(clippy::result_large_err)]
        fn returns_result() -> super::Result<()> {
            Ok(())
        }
        assert!(returns_result().is_ok());
    }

    /// Exercises actual `?`-propagation through the alias rather than just
    /// confirming the type compiles. A function returning `Result<_, ParseError>`
    /// composes via `?` into a function returning `iconnx::Result<_>` because
    /// of the `ParseError → IconnxError` `#[from]` impl.
    #[test]
    fn parse_error_propagates_via_question_mark() {
        fn inner() -> std::result::Result<i32, ParseError> {
            Err(ParseError::UnsupportedDType {
                name: "x".into(),
                dtype_id: 99,
            })
        }

        #[allow(clippy::result_large_err)]
        fn outer() -> super::Result<i32> {
            // `?` invokes the `From<ParseError> for IconnxError` impl
            // generated by `#[from]`, exercising the propagation path.
            let v = inner()?;
            Ok(v)
        }

        let err = outer().expect_err("outer should propagate inner error");
        assert!(
            matches!(err, IconnxError::Parse(ParseError::UnsupportedDType { .. })),
            "expected IconnxError::Parse(UnsupportedDType), got {err:?}"
        );
    }
}
