//! `validate_model()` and `validate_model_for_hardware()` — public APIs for
//! pre-flight model compatibility discovery.
//!
//! See `docs/superpowers/specs/2026-04-29-ws5-observability-design.md`
//! Section 4 Flow A for the orchestration walk.
//!
//! ## Relationship to the parser's `eprintln!` warnings
//!
//! `OnnxParser::extract_weights` emits an unconditional stderr warning when
//! it encounters non-finite half-precision (BF16/FP16) initializer values
//! (see `warn_non_finite_half` in `src/onnx_parser.rs`). The structured
//! [`ModelWarning::NonFiniteHalfConstants`] surfaced by `validate_model`
//! **subsumes** that stderr emission for callers using the validate API:
//! every initializer that triggered the parser eprintln also produces a
//! structured warning here. The parser's eprintln is retained because it
//! affects unrelated callers (anything that calls `extract_weights`
//! directly), but consumers of `validate_model` should rely on the
//! structured `warnings` vec rather than capturing stderr.

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

    // WS-5 Y(2) replacement points (this function):
    //   1. Unsupported-op check: replaces placeholder pass-through with op_support::lookup().
    //   2. Opset bounds: replaces placeholder (1..=22) with MIN/MAX_OPSET_SUPPORTED.
    // Inline `// PLACEHOLDER (Y(2)):` markers below identify each site.

    // Step 1: Parse. ParseError short-circuits with partial: None.
    let model = match OnnxParser::parse_file(&path) {
        Ok(m) => m,
        Err(e) => {
            // OnnxParser::parse_file returns anyhow::Result; map to ParseError
            // when possible, otherwise wrap in `ParseError::Other` (a generic
            // catchall — the previous `InvalidTensorData { name: "" }` shape
            // misled consumers into thinking a named tensor was at fault).
            let pe = e
                .downcast::<crate::onnx_parser::ParseError>()
                .unwrap_or_else(|other| crate::onnx_parser::ParseError::Other {
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

    // PLACEHOLDER (Y(2)): replaces with real op_support::supported_ops() check.
    // For now, no UnsupportedOp incompatibilities are surfaced — Y(2) will
    // wire this in.
    // (Tests in Task 1.5 use synthetic models with op_types that this
    // placeholder accepts; they don't exercise the unsupported-op path.
    // Y(2) re-runs the synthetic UnsupportedOp test against the real lookup.)

    // Step 3: Walk opset imports.
    let opset_imports = model.opset_imports();
    // PLACEHOLDER (Y(2)): lands MIN_OPSET_SUPPORTED / MAX_OPSET_SUPPORTED;
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

    // Pre-compute supported_ops (used ops ∩ placeholder supported set) so
    // we can include it in `partial_capabilities` even if `extract_weights`
    // bails below. Y(2) replaces with the real op_support intersection.
    let mut supported_ops: Vec<String> = op_counts.keys().cloned().collect();
    supported_ops.sort();

    // Pre-compute the layer count up here too — it's a pure graph walk.
    let layer_count = tolerance_hint::count_layer_norm_nodes(&graph);

    // Step 4: Walk initializers for non-finite half-precision constants.
    // If `extract_weights` fails we still surface a partial capability
    // payload — the contract on `ModelValidationFailure::partial_capabilities`
    // says `None` is reserved for the parse-short-circuit (file failed to
    // open / decode); a downstream initializer-decoding failure is not the
    // same case. Caller can still see opset_imports / supported_ops /
    // layer_count / fp32-fallback tolerance hint.
    let weights = match model.extract_weights() {
        Ok(w) => w,
        Err(e) => {
            // Even though initializer decode failed, the IO-walk over declared
            // input/output value-info dtypes is still computable from the
            // already-parsed model proto. Surface those dtypes in the failure
            // caps so callers see e.g. INT64 inputs for NLP models.
            let io_dtype_set = collect_io_dtypes(&model);
            let used_dtypes_vec: Vec<crate::tensor::DType> =
                io_dtype_set.into_iter().collect();
            let expected_tolerance = tolerance_hint::expected_tolerance(&used_dtypes_vec, 0);
            let caps = ModelCapabilities {
                opset_imports,
                supported_ops,
                used_dtypes: used_dtypes_vec,
                initializer_count: 0,
                initializer_bytes: 0,
                estimated_peak_bytes: 0,
                expected_tolerance,
                warnings: Vec::new(),
            };
            // Compose with any incompatibilities collected so far (opset
            // checks ran before `extract_weights`); add the new Parse one.
            let mut all_incompat = incompatibilities;
            all_incompat.push(ModelIncompatibility::Parse(e));
            return Err(ModelValidationFailure {
                incompatibilities: all_incompat,
                partial_capabilities: Some(caps),
            });
        }
    };

    let mut used_dtypes: std::collections::BTreeSet<crate::tensor::DType> =
        std::collections::BTreeSet::new();
    let mut total_init_bytes: usize = 0;
    let initializer_count = weights.len();

    // Per #2 review: surface ONE `DTypeDowncast` warning per model when at
    // least one initializer's source dtype was compressed (today only
    // `Tensor::Float64 → DType::Float32`). Flag-gated rather than
    // per-tensor; consumers should read it as a single global signal that
    // the dtype taxonomy lost information.
    let mut already_warned_fp64 = false;

    for (name, tensor) in &weights {
        used_dtypes.insert(crate::tensor::dtype_for_tensor(tensor));
        total_init_bytes += crate::tensor::tensor_byte_size(tensor);

        if let crate::tensor::Tensor::Float64(_) = tensor {
            if !already_warned_fp64 {
                warnings.push(ModelWarning::DTypeDowncast {
                    from: "float64",
                    to: crate::tensor::DType::Float32,
                });
                already_warned_fp64 = true;
            }
        }

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

    // Step 4b: Walk declared input/output value-info dtypes per the
    // `used_dtypes` rustdoc claim ("initializers + declared input/output
    // types"). NLP models in particular have INT64 input ids that don't
    // appear in any initializer.
    for id in model
        .input_dtypes()
        .into_iter()
        .chain(model.output_dtypes())
        .flatten()
    {
        if let Some(dt) = onnx_dtype_id_to_dtype(id) {
            used_dtypes.insert(dt);
        } else if id == 11 {
            // FLOAT64 declared on a graph IO; same downcast surface
            // applies (validate_model has no f64 GPU path).
            used_dtypes.insert(crate::tensor::DType::Float32);
            if !already_warned_fp64 {
                warnings.push(ModelWarning::DTypeDowncast {
                    from: "float64",
                    to: crate::tensor::DType::Float32,
                });
                already_warned_fp64 = true;
            }
        }
        // Other unknown ids (sequence/map/sparse) silently skipped —
        // they don't belong in `used_dtypes` and aren't validate-fatal.
    }

    // Step 5: Compute tolerance hint from the complete dtype set.
    let used_dtypes_vec: Vec<_> = used_dtypes.iter().copied().collect();
    let expected_tolerance = tolerance_hint::expected_tolerance(&used_dtypes_vec, layer_count);

    // Step 6: Estimate peak bytes (rough heuristic per rustdoc).
    // Y(1) implementation: total initializer footprint + 2 * mean
    // initializer size as a stand-in for activation/intermediate memory.
    // The proper non-initializer graph walk (originally-intended
    // `max_intermediate_tensor_bytes`) is deferred to a future workstream;
    // the runtime arena planner is the source of truth for precise memory
    // plans (out of scope for capability discovery).
    let mean_initializer_bytes = total_init_bytes / initializer_count.max(1);
    let estimated_peak_bytes = total_init_bytes + 2 * mean_initializer_bytes;

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

/// Walk declared input + output value-info dtypes into a sorted set.
///
/// Used by both the `extract_weights` failure path (where it's the only
/// available dtype source) and conceptually mirrors the IO-walk in the
/// success path. The success path inlines the walk because it also emits
/// a `DTypeDowncast` warning for FLOAT64 declared on a graph IO; that
/// warning side-effect doesn't apply on the failure path (caps.warnings is
/// empty), so the helper only handles the dtype extraction.
fn collect_io_dtypes(
    model: &crate::onnx_parser::OnnxModel,
) -> std::collections::BTreeSet<crate::tensor::DType> {
    let mut set = std::collections::BTreeSet::new();
    for id in model
        .input_dtypes()
        .into_iter()
        .chain(model.output_dtypes())
        .flatten()
    {
        if let Some(dt) = onnx_dtype_id_to_dtype(id) {
            set.insert(dt);
        } else if id == 11 {
            // FLOAT64 on a graph IO maps to FP32 in the GPU dtype taxonomy
            // (validate_model has no f64 GPU path); same downcast surface
            // as the success path, minus the warning emission.
            set.insert(crate::tensor::DType::Float32);
        }
        // Other unknown ids (sequence/map/sparse) silently skipped.
    }
    set
}

/// Map an ONNX `TensorProto.DataType` raw enum value to iconnx's GPU
/// [`crate::tensor::DType`] tag, or `None` if the id has no GPU representation.
///
/// FLOAT64(11) returns `None` — callers handle the downcast-and-warn path
/// explicitly so the surface is honest about the lossy mapping.
fn onnx_dtype_id_to_dtype(id: i32) -> Option<crate::tensor::DType> {
    use crate::tensor::DType;
    match id {
        1 => Some(DType::Float32),    // FLOAT
        7 => Some(DType::Int64),      // INT64
        6 => Some(DType::Int32),      // INT32
        9 => Some(DType::Bool),       // BOOL
        3 => Some(DType::Int8),       // INT8
        2 => Some(DType::UInt8),      // UINT8
        10 => Some(DType::Float16),   // FLOAT16
        16 => Some(DType::BFloat16),  // BFLOAT16
        // 11 = FLOAT64 — no DType tag, downcast surface (handled by caller).
        _ => None,
    }
}

/// Pure helper: produce `HardwareFloorUnmet` incompatibilities for any
/// dtype in `caps.used_dtypes` whose hardware floor is unmet by `cc`.
///
/// Today only BF16 has a hardware floor (sm_80+). FP16 works on sm_70+
/// without explicit gating; new dtype hardware floors get added as `match`
/// arms here.
///
/// Pure (no `IconnxCudaContext` parameter) so it's unit-testable without a
/// GPU. `validate_model_for_hardware` calls this with `ctx.compute_capability()`.
#[cfg(feature = "cuda")]
fn check_hardware_floor(
    caps: &ModelCapabilities,
    cc: (i32, i32),
) -> Vec<ModelIncompatibility> {
    let (cc_major, cc_minor) = cc;
    let device_str = format!("sm_{cc_major}{cc_minor}");
    let mut out = Vec::new();
    for dtype in &caps.used_dtypes {
        match dtype {
            crate::tensor::DType::BFloat16 if cc_major < 8 => {
                out.push(ModelIncompatibility::HardwareFloorUnmet {
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
    out
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

    // Hardware floor check delegated to a pure helper for unit-testability.
    let cc = ctx.compute_capability();
    existing_incompat.extend(check_hardware_floor(&caps_for_hw_check, cc));

    if existing_incompat.is_empty() {
        Ok(caps_for_hw_check)
    } else {
        Err(ModelValidationFailure {
            incompatibilities: existing_incompat,
            partial_capabilities: Some(caps_for_hw_check),
        })
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::tensor::DType;

    fn caps_with_dtypes(dtypes: Vec<DType>) -> ModelCapabilities {
        ModelCapabilities {
            opset_imports: vec![("".to_string(), 17)],
            supported_ops: Vec::new(),
            used_dtypes: dtypes,
            initializer_count: 0,
            initializer_bytes: 0,
            estimated_peak_bytes: 0,
            expected_tolerance: tolerance_hint::expected_tolerance(&[], 0),
            warnings: Vec::new(),
        }
    }

    #[test]
    fn check_hardware_floor_bf16_on_sm75_yields_unmet() {
        let caps = caps_with_dtypes(vec![DType::BFloat16]);
        let result = check_hardware_floor(&caps, (7, 5));
        assert_eq!(result.len(), 1);
        match &result[0] {
            ModelIncompatibility::HardwareFloorUnmet {
                dtype,
                required,
                found,
            } => {
                assert_eq!(*dtype, DType::BFloat16);
                assert_eq!(*required, "sm_80+");
                assert_eq!(found, "sm_75");
            }
            other => panic!("unexpected incompatibility: {:?}", other),
        }
    }

    #[test]
    fn check_hardware_floor_bf16_on_sm80_no_incompat() {
        let caps = caps_with_dtypes(vec![DType::BFloat16]);
        let result = check_hardware_floor(&caps, (8, 0));
        assert!(result.is_empty(), "BF16 on sm_80 should be supported");
    }

    #[test]
    fn check_hardware_floor_bf16_on_sm90_no_incompat() {
        let caps = caps_with_dtypes(vec![DType::BFloat16]);
        let result = check_hardware_floor(&caps, (9, 0));
        assert!(result.is_empty(), "BF16 on sm_90 (Hopper) should be supported");
    }

    #[test]
    fn check_hardware_floor_fp32_only_on_sm75_no_incompat() {
        let caps = caps_with_dtypes(vec![DType::Float32]);
        let result = check_hardware_floor(&caps, (7, 5));
        assert!(result.is_empty(), "FP32-only has no hardware floor");
    }

    #[test]
    fn check_hardware_floor_empty_used_dtypes_no_incompat() {
        let caps = caps_with_dtypes(Vec::new());
        let result = check_hardware_floor(&caps, (7, 5));
        assert!(
            result.is_empty(),
            "empty used_dtypes should produce no incompat"
        );
    }

    #[test]
    fn check_hardware_floor_mixed_bf16_fp32_on_sm75_only_bf16_unmet() {
        // BF16 + FP32 on sm_75: only BF16 surfaces an unmet floor.
        let caps = caps_with_dtypes(vec![DType::Float32, DType::BFloat16]);
        let result = check_hardware_floor(&caps, (7, 5));
        assert_eq!(result.len(), 1);
        assert!(matches!(
            result[0],
            ModelIncompatibility::HardwareFloorUnmet {
                dtype: DType::BFloat16,
                ..
            }
        ));
    }

    #[test]
    fn onnx_dtype_id_to_dtype_round_trip_supported_ids() {
        assert_eq!(onnx_dtype_id_to_dtype(1), Some(DType::Float32));
        assert_eq!(onnx_dtype_id_to_dtype(7), Some(DType::Int64));
        assert_eq!(onnx_dtype_id_to_dtype(6), Some(DType::Int32));
        assert_eq!(onnx_dtype_id_to_dtype(9), Some(DType::Bool));
        assert_eq!(onnx_dtype_id_to_dtype(3), Some(DType::Int8));
        assert_eq!(onnx_dtype_id_to_dtype(2), Some(DType::UInt8));
        assert_eq!(onnx_dtype_id_to_dtype(10), Some(DType::Float16));
        assert_eq!(onnx_dtype_id_to_dtype(16), Some(DType::BFloat16));
        // FLOAT64 (11) intentionally returns None; caller handles downcast.
        assert_eq!(onnx_dtype_id_to_dtype(11), None);
        // Unknown id (5 = INT16, not in iconnx) returns None.
        assert_eq!(onnx_dtype_id_to_dtype(5), None);
    }
}
