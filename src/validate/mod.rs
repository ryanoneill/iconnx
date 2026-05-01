//! `validate_model()` and `validate_model_for_hardware()` — public APIs for
//! pre-flight model compatibility discovery.
//!
//! See `docs/superpowers/specs/2026-04-29-ws5-observability-design.md`
//! Section 4 Flow A for the orchestration walk.

pub mod capabilities;
pub mod incompat;
pub mod tolerance_hint;

pub use capabilities::{ModelCapabilities, ModelWarning};
pub use incompat::{ModelIncompatibility, ModelValidationFailure};
pub use tolerance_hint::{ToleranceBasis, ToleranceConfidence, ToleranceHint};

use std::path::Path;

#[cfg(feature = "cuda")]
use crate::cuda::IconnxCudaContext;

/// Static model validation — pure analysis with no CUDA dependency.
///
/// Cannot produce `ModelIncompatibility::HardwareFloorUnmet` (no hardware
/// context to compare against). Library consumers writing GPU-less tooling
/// (TUI inspectors, capability discovery for filtered roster sweeps, etc.)
/// call this entry point.
///
/// # Note on `clippy::result_large_err`
///
/// The error type is intentionally not boxed. `ModelValidationFailure`
/// carries a `Vec<ModelIncompatibility>` plus an `Option<ModelCapabilities>`,
/// which clippy flags as a large `Err` variant; however, the public API
/// contract (locked at the WS-5 Y(1) signed commit) returns the failure
/// by-value so that downstream consumers (leadline-bench `--validate`)
/// can iterate `failure.incompatibilities` without an extra deref. Boxing
/// would change the surface and break that contract.
#[allow(clippy::result_large_err)]
pub fn validate_model<P: AsRef<Path>>(
    path: P,
) -> Result<ModelCapabilities, ModelValidationFailure> {
    use crate::onnx_parser::OnnxParser;

    // Step 1: Parse. ParseError short-circuits with partial: None.
    let model = match OnnxParser::parse_file(&path) {
        Ok(m) => m,
        Err(e) => {
            // OnnxParser::parse_file returns anyhow::Result; map to ParseError
            // when possible, otherwise wrap as InvalidTensorData.
            let pe = e
                .downcast::<crate::onnx_parser::ParseError>()
                .unwrap_or_else(|other| crate::onnx_parser::ParseError::InvalidTensorData {
                    name: String::new(),
                    reason: other.to_string(),
                });
            return Err(ModelValidationFailure {
                incompatibilities: vec![ModelIncompatibility::Parse(pe)],
                partial_capabilities: None,
            });
        }
    };

    let graph = match model.computation_graph() {
        Ok(g) => g,
        Err(e) => {
            return Err(ModelValidationFailure {
                incompatibilities: vec![ModelIncompatibility::Parse(e)],
                partial_capabilities: None,
            });
        }
    };

    // The op_support module isn't landed yet (Y(2)); for now use a
    // placeholder that returns "all ops supported." Y(2) replaces this
    // with the real lookup table.
    let mut incompatibilities: Vec<ModelIncompatibility> = Vec::new();
    let mut warnings: Vec<ModelWarning> = Vec::new();

    // Step 2: Walk ops.
    let mut op_counts: std::collections::HashMap<String, (usize, Vec<String>)> =
        std::collections::HashMap::new();
    for (idx, node) in graph.nodes().iter().enumerate() {
        let entry = op_counts
            .entry(node.op_type.clone())
            .or_insert_with(|| (0, Vec::with_capacity(3)));
        entry.0 += 1;
        if entry.1.len() < 3 {
            // Use first output as node name; fall back to a synthesized name.
            let name = node
                .outputs
                .first()
                .cloned()
                .unwrap_or_else(|| format!("node_{idx}"));
            entry.1.push(name);
        }
    }

    // PLACEHOLDER: Y(2) replaces with real op_support::supported_ops() check.
    // For now, no UnsupportedOp incompatibilities are surfaced — Y(2) will
    // wire this in.
    // (Tests in Task 1.5 use synthetic models with op_types that this
    // placeholder accepts; they don't exercise the unsupported-op path.
    // Y(2) re-runs the synthetic UnsupportedOp test against the real lookup.)

    // Step 3: Walk opset imports.
    let opset_imports = model.opset_imports();
    // PLACEHOLDER: Y(2) lands MIN_OPSET_SUPPORTED / MAX_OPSET_SUPPORTED;
    // for now use 1..=22 as a maximally permissive stub. Y(2) replaces.
    const PLACEHOLDER_MIN: i64 = 1;
    const PLACEHOLDER_MAX: i64 = 22;
    for (domain, version) in &opset_imports {
        match domain.as_str() {
            "" | "ai.onnx" => {
                if *version < PLACEHOLDER_MIN || *version > PLACEHOLDER_MAX {
                    incompatibilities.push(ModelIncompatibility::OpsetOutOfRange {
                        domain: domain.clone(),
                        version: *version,
                        supported: (PLACEHOLDER_MIN, PLACEHOLDER_MAX),
                    });
                }
            }
            _ => {
                incompatibilities.push(ModelIncompatibility::OpsetDomainUnknown {
                    domain: domain.clone(),
                    version: *version,
                });
            }
        }
    }

    // Step 4: Walk initializers for non-finite half-precision constants.
    let weights = match model.extract_weights() {
        Ok(w) => w,
        Err(e) => {
            return Err(ModelValidationFailure {
                incompatibilities: vec![ModelIncompatibility::Parse(e)],
                partial_capabilities: None,
            });
        }
    };

    let mut used_dtypes: std::collections::BTreeSet<crate::tensor::DType> =
        std::collections::BTreeSet::new();
    let mut total_init_bytes: usize = 0;
    let initializer_count = weights.len();

    for (name, tensor) in &weights {
        used_dtypes.insert(crate::tensor::dtype_for_tensor(tensor));
        total_init_bytes += crate::tensor::tensor_byte_size(tensor);

        // Non-finite half scan (re-uses helpers from WS-3.5 hotfix).
        if let crate::tensor::Tensor::BFloat16(arr) = tensor {
            let s: Vec<half::bf16> = arr.iter().copied().collect();
            let nf = s.iter().filter(|x| !x.is_finite()).count();
            if nf > 0 {
                warnings.push(ModelWarning::NonFiniteHalfConstants {
                    tensor_name: name.clone(),
                    count: nf,
                    total: s.len(),
                });
            }
        } else if let crate::tensor::Tensor::Float16(arr) = tensor {
            let s: Vec<half::f16> = arr.iter().copied().collect();
            let nf = s.iter().filter(|x| !x.is_finite()).count();
            if nf > 0 {
                warnings.push(ModelWarning::NonFiniteHalfConstants {
                    tensor_name: name.clone(),
                    count: nf,
                    total: s.len(),
                });
            }
        }
    }

    // Step 5: Compute layer count + tolerance hint.
    let layer_count = tolerance_hint::count_layer_norm_nodes(&graph);
    let used_dtypes_vec: Vec<_> = used_dtypes.iter().copied().collect();
    let expected_tolerance = tolerance_hint::expected_tolerance(&used_dtypes_vec, layer_count);

    // Step 6: Estimate peak bytes (rough heuristic per rustdoc).
    // Activation heuristic: 2 * largest-shape-product * 4 bytes (f32 default).
    // Replaced with proper graph-walk in Y(2)+ if needed; for now the
    // initializer_bytes dominates for transformer models.
    let max_intermediate_bytes = total_init_bytes / initializer_count.max(1);
    let estimated_peak_bytes = total_init_bytes + 2 * max_intermediate_bytes;

    // Step 7: supported_ops = used ops ∩ (placeholder supported set).
    // Y(2) replaces with real op_support::supported_ops() intersection.
    let mut supported_ops: Vec<String> = op_counts.keys().cloned().collect();
    supported_ops.sort();

    let caps = ModelCapabilities {
        opset_imports,
        supported_ops,
        used_dtypes: used_dtypes_vec,
        initializer_count,
        initializer_bytes: total_init_bytes,
        estimated_peak_bytes,
        expected_tolerance,
        warnings,
    };

    if incompatibilities.is_empty() {
        Ok(caps)
    } else {
        Err(ModelValidationFailure {
            incompatibilities,
            partial_capabilities: Some(caps),
        })
    }
}

/// Static model validation + hardware floor check.
///
/// Adds `HardwareFloorUnmet` checks for dtype/opset combinations that
/// require capabilities the supplied context lacks (e.g., BF16 on sm_75).
///
/// See `validate_model` rustdoc for the rationale on the
/// `clippy::result_large_err` allow.
#[cfg(feature = "cuda")]
#[allow(clippy::result_large_err)]
pub fn validate_model_for_hardware<P: AsRef<Path>>(
    path: P,
    ctx: &IconnxCudaContext,
) -> Result<ModelCapabilities, ModelValidationFailure> {
    // First run the static analysis.
    let static_result = validate_model(&path);

    let caps_for_hw_check: ModelCapabilities;
    let mut existing_incompat: Vec<ModelIncompatibility>;

    match static_result {
        Ok(c) => {
            caps_for_hw_check = c;
            existing_incompat = Vec::new();
        }
        Err(failure) => {
            // If static analysis bailed (parse error), surface that.
            if failure.partial_capabilities.is_none() {
                return Err(failure);
            }
            // Otherwise, continue with the partial capabilities and add
            // hardware-floor check on top.
            caps_for_hw_check = failure.partial_capabilities.unwrap();
            existing_incompat = failure.incompatibilities;
        }
    }

    // Hardware floor check: if any half-precision dtype is used but the
    // context is below sm_80, surface HardwareFloorUnmet.
    let (cc_major, cc_minor) = ctx.compute_capability();
    let device_str = format!("sm_{cc_major}{cc_minor}");

    for dtype in &caps_for_hw_check.used_dtypes {
        match dtype {
            crate::tensor::DType::BFloat16 if cc_major < 8 => {
                existing_incompat.push(ModelIncompatibility::HardwareFloorUnmet {
                    dtype: *dtype,
                    required: "sm_80+",
                    found: device_str.clone(),
                });
            }
            // FP16 has no hardware floor in iconnx today (works on sm_70+);
            // add explicit check if/when that changes.
            _ => {}
        }
    }

    if existing_incompat.is_empty() {
        Ok(caps_for_hw_check)
    } else {
        Err(ModelValidationFailure {
            incompatibilities: existing_incompat,
            partial_capabilities: Some(caps_for_hw_check),
        })
    }
}
