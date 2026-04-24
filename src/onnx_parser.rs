/// ONNX model parser for Iconnx inference framework
///
/// Parses ONNX protobuf files to extract:
/// - Model computation graph
/// - Weight tensors (initializers)
/// - Input/output specifications
/// - Operator attributes
///
/// Uses prost for protobuf decoding.
use anyhow::{Context, Result};
use prost::Message;
use std::path::Path;
use thiserror::Error;

/// Typed parser errors.
///
/// Introduced in WS-1 to replace silent-drop error handling
/// (`if let Ok(_) = …` and debug-only `eprintln!`). Every parse-time
/// failure surfaces through this enum so callers can decide whether
/// to fail fast, log, or (for legacy flows) request opt-in recovery.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ParseError {
    /// TensorProto or attribute used an ONNX dtype that the parser
    /// does not yet decode. The dtype_id is the raw ONNX enum value
    /// (see onnx.proto: 1=FLOAT, 7=INT64, 6=INT32, …).
    #[error("unsupported ONNX dtype {dtype_id} for tensor '{name}'")]
    UnsupportedDType { name: String, dtype_id: i32 },

    /// Initializer used an ONNX dtype that iconnx's Tensor enum
    /// does not currently represent (e.g. FP16, BF16, INT8). These
    /// are WS-3/WS-4 scope and must be surfaced rather than silently
    /// dropped.
    #[error("unsupported initializer dtype {dtype_id} for '{name}'")]
    UnsupportedInitializerDType { name: String, dtype_id: i32 },

    /// TensorProto payload was empty or could not be decoded into
    /// the expected element type (e.g. raw_data length not a
    /// multiple of sizeof(T), or neither typed-data nor raw_data
    /// was populated).
    #[error("invalid tensor data for '{name}': {reason}")]
    InvalidTensorData { name: String, reason: String },

    /// Declared tensor shape product disagrees with the flat data
    /// length. In strict mode (default) this is a hard error;
    /// callers can opt-in to the legacy silent-rewrite via
    /// `ParserOptions::allow_legacy_shape_correction`.
    #[error(
        "shape mismatch for '{name}': declared {declared:?} ({declared_len} elements), got {actual_len}"
    )]
    ShapeMismatch {
        name: String,
        declared: Vec<usize>,
        declared_len: usize,
        actual_len: usize,
    },
}

/// Generated ONNX protobuf types
mod onnx_proto {
    #![allow(clippy::all, clippy::pedantic)]
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

/// Opt-in knobs for legacy or lenient parser behaviours.
///
/// Introduced in WS-1 M1.3. The default (`Default::default()`) is
/// strict: every declared-vs-actual shape disagreement becomes a
/// [`ParseError::ShapeMismatch`]. Callers that must load older
/// models whose initializer dims drift from the flat data length
/// (e.g. Kokoro v1.0, which carries a few such initializers) can
/// set `allow_legacy_shape_correction = true` — the parser then
/// logs an unconditional warning to stderr and rewrites the shape
/// to `[actual_len]`, matching the pre-WS-1 behaviour.
#[derive(Clone, Debug, Default)]
pub struct ParserOptions {
    /// When `true`, declared-vs-actual shape mismatches are rewritten
    /// to `[actual_len]` with an unconditional warning instead of
    /// becoming a hard `ParseError::ShapeMismatch`. Defaults to
    /// `false` (strict).
    pub allow_legacy_shape_correction: bool,
}

/// ONNX model parser
pub struct OnnxParser;

impl OnnxParser {
    /// Parse an ONNX model file with default (strict) options.
    ///
    /// Equivalent to
    /// [`OnnxParser::parse_file_with_options`]`(path, ParserOptions::default())`.
    pub fn parse_file<P: AsRef<Path>>(path: P) -> Result<OnnxModel> {
        Self::parse_file_with_options(path, ParserOptions::default())
    }

    /// Parse an ONNX model file with explicit parser options.
    ///
    /// Use this entry when the caller needs to opt into legacy
    /// behaviours (see [`ParserOptions`]). Every model consumed in
    /// new code should keep the default strict options.
    pub fn parse_file_with_options<P: AsRef<Path>>(
        path: P,
        options: ParserOptions,
    ) -> Result<OnnxModel> {
        // 1. Read file bytes
        let bytes = std::fs::read(path.as_ref())
            .with_context(|| format!("Failed to read ONNX file: {:?}", path.as_ref()))?;

        // 2. Decode protobuf
        let proto =
            onnx_proto::ModelProto::decode(&bytes[..]).context("Failed to decode ONNX protobuf")?;

        // 3. Extract graph
        let _graph = proto
            .graph
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("ONNX model has no graph"))?;

        // 4. Return model
        Ok(OnnxModel { proto, options })
    }
}

/// Parsed ONNX model
///
/// Contains the parsed ONNX protobuf model and the parser options
/// chosen by the caller.
pub struct OnnxModel {
    #[allow(dead_code)]
    proto: onnx_proto::ModelProto,
    options: ParserOptions,
}

impl OnnxModel {
    /// Get model inputs
    ///
    /// Returns the names of all graph inputs
    pub fn inputs(&self) -> Vec<String> {
        self.proto
            .graph
            .as_ref()
            .map(|graph| {
                graph
                    .input
                    .iter()
                    .filter_map(|input| input.name.clone())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get declared shapes for all graph inputs.
    ///
    /// Returns a vector of `(name, shape)` pairs in the order the
    /// inputs appear in the ONNX graph. Each dim is `Some(n)` if the
    /// ONNX value-info records a numeric `dim_value`, or `None` when
    /// the dim is symbolic (a `dim_param` such as `seq_len` or
    /// unspecified). A missing `type_proto` or a non-tensor input
    /// yields an empty `Vec` — the codebase's "fully unknown"
    /// sentinel per `shape_inference::tensor_shapes_from_graph`.
    ///
    /// Callers use this to seed `OptimizableGraphBuilder::add_input`
    /// at loader time so the Phase 3 shape-inference pass has enough
    /// static rank to propagate into downstream ops (Gather, Transpose,
    /// Conv, LSTM …) and, ultimately, to pre-resolve Reshape
    /// `0`/`-1` placeholders.
    pub fn input_shapes(&self) -> Vec<(String, Vec<Option<usize>>)> {
        let Some(graph) = self.proto.graph.as_ref() else {
            return Vec::new();
        };
        let mut out = Vec::with_capacity(graph.input.len());
        for info in &graph.input {
            let name = info.name.clone().unwrap_or_default();
            let shape: Vec<Option<usize>> = info
                .r#type
                .as_ref()
                .and_then(|t| t.value.as_ref())
                .and_then(|v| match v {
                    onnx_proto::type_proto::Value::TensorType(t) => t.shape.as_ref(),
                    _ => None,
                })
                .map(|shape_proto| {
                    shape_proto
                        .dim
                        .iter()
                        .map(|d| match &d.value {
                            Some(onnx_proto::tensor_shape_proto::dimension::Value::DimValue(
                                v,
                            )) if *v >= 0 => Some(*v as usize),
                            // dim_param ("seq_len", "batch", …) or an
                            // unset/negative dim_value is dynamic — use
                            // `None`, which means "runtime decides."
                            _ => None,
                        })
                        .collect()
                })
                .unwrap_or_default();
            out.push((name, shape));
        }
        out
    }

    /// Get model outputs
    ///
    /// Returns the names of all graph outputs
    pub fn outputs(&self) -> Vec<String> {
        self.proto
            .graph
            .as_ref()
            .map(|graph| {
                graph
                    .output
                    .iter()
                    .filter_map(|output| output.name.clone())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// List names of all weight tensors (initializers)
    ///
    /// Returns the names of all weight tensors stored in graph.initializer
    pub fn list_weight_names(&self) -> Vec<String> {
        self.proto
            .graph
            .as_ref()
            .map(|graph| {
                graph
                    .initializer
                    .iter()
                    .filter_map(|tensor| tensor.name.clone())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Extract weight tensors from initializers.
    ///
    /// Returns a map of tensor name → Tensor. Supports ONNX dtypes
    /// FLOAT(1), INT64(7), INT32(6), FLOAT64(11), and BOOL(9) —
    /// the five dtypes currently representable by [`crate::tensor::Tensor`].
    ///
    /// Any other dtype (FP16=10, BF16=16, INT8=3, UINT8=2, INT16=5,
    /// UINT16=4, UINT32=12, UINT64=13, the FLOAT*E* variants, …) is
    /// surfaced as [`ParseError::UnsupportedInitializerDType`]
    /// rather than silently dropped: the pre-WS-1 `#[cfg(debug_assertions)]
    /// eprintln!` fallback produced release-mode-invisible weight
    /// drops that corrupt downstream inference.
    ///
    /// Payload-decoding failures (bad raw_data length, no data at
    /// all) are surfaced as [`ParseError::InvalidTensorData`].
    pub fn extract_weights(
        &self,
    ) -> Result<std::collections::HashMap<String, crate::tensor::Tensor>, ParseError> {
        use std::collections::HashMap;

        let mut weights = HashMap::new();

        let Some(graph) = self.proto.graph.as_ref() else {
            return Ok(weights);
        };

        for tensor_proto in &graph.initializer {
            let name = tensor_proto.name.clone().unwrap_or_default();

            // Declared shape + expected element count.
            let shape: Vec<usize> = tensor_proto.dims.iter().map(|&d| d as usize).collect();
            let expected_len: usize = if shape.is_empty() {
                1
            } else {
                shape.iter().product()
            };

            // ONNX dtype enum (see onnx.proto TensorProto.DataType).
            let data_type = tensor_proto.data_type.unwrap_or(1);

            // Zero-element initializer fast path — empty data is the
            // CORRECT representation, not corruption (same refinement as
            // parse_tensor_proto; see the fix-zero-element-tensors commit).
            if expected_len == 0 {
                let tensor = match data_type {
                    1 => crate::tensor::Tensor::from_vec_f32(Vec::new(), shape),
                    7 => crate::tensor::Tensor::from_vec_i64(Vec::new(), shape),
                    6 => crate::tensor::Tensor::from_vec_i32(Vec::new(), shape),
                    11 => crate::tensor::Tensor::from_vec_f64(Vec::new(), shape),
                    9 => crate::tensor::Tensor::from_vec_bool(Vec::new(), shape),
                    other => {
                        return Err(ParseError::UnsupportedInitializerDType {
                            name,
                            dtype_id: other,
                        });
                    }
                };
                weights.insert(name, tensor);
                continue;
            }

            let tensor = match data_type {
                1 => {
                    // FLOAT (f32)
                    let data = Self::decode_f32_payload(&name, tensor_proto)?;
                    let final_shape = self.correct_shape(&name, &shape, expected_len, data.len())?;
                    crate::tensor::Tensor::from_vec_f32(data, final_shape)
                }
                7 => {
                    // INT64 (i64)
                    let data = Self::decode_i64_payload(&name, tensor_proto)?;
                    let final_shape = self.correct_shape(&name, &shape, expected_len, data.len())?;
                    crate::tensor::Tensor::from_vec_i64(data, final_shape)
                }
                6 => {
                    // INT32 (i32)
                    let data = Self::decode_i32_payload(&name, tensor_proto)?;
                    let final_shape = self.correct_shape(&name, &shape, expected_len, data.len())?;
                    crate::tensor::Tensor::from_vec_i32(data, final_shape)
                }
                11 => {
                    // DOUBLE (f64) — WS-1 M1.2 expansion.
                    let data = Self::decode_f64_payload(&name, tensor_proto)?;
                    let final_shape = self.correct_shape(&name, &shape, expected_len, data.len())?;
                    crate::tensor::Tensor::from_vec_f64(data, final_shape)
                }
                9 => {
                    // BOOL — WS-1 M1.2 expansion. ONNX stores booleans
                    // in either int32_data (one i32 per bool) or
                    // raw_data (one byte per bool).
                    let data = Self::decode_bool_payload(&name, tensor_proto)?;
                    let final_shape = self.correct_shape(&name, &shape, expected_len, data.len())?;
                    crate::tensor::Tensor::from_vec_bool(data, final_shape)
                }
                other => {
                    return Err(ParseError::UnsupportedInitializerDType {
                        name,
                        dtype_id: other,
                    });
                }
            };

            weights.insert(name, tensor);
        }

        Ok(weights)
    }

    fn decode_f32_payload(
        name: &str,
        tensor_proto: &onnx_proto::TensorProto,
    ) -> Result<Vec<f32>, ParseError> {
        if !tensor_proto.float_data.is_empty() {
            return Ok(tensor_proto.float_data.clone());
        }
        let raw = tensor_proto.raw_data.as_ref().ok_or_else(|| ParseError::InvalidTensorData {
            name: name.to_string(),
            reason: "no float_data and no raw_data".to_string(),
        })?;
        if raw.is_empty() {
            return Err(ParseError::InvalidTensorData {
                name: name.to_string(),
                reason: "empty raw_data for FLOAT tensor".to_string(),
            });
        }
        if raw.len() % 4 != 0 {
            return Err(ParseError::InvalidTensorData {
                name: name.to_string(),
                reason: format!("raw_data length {} not a multiple of 4 for FLOAT", raw.len()),
            });
        }
        Ok(raw.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    fn decode_i64_payload(
        name: &str,
        tensor_proto: &onnx_proto::TensorProto,
    ) -> Result<Vec<i64>, ParseError> {
        if !tensor_proto.int64_data.is_empty() {
            return Ok(tensor_proto.int64_data.clone());
        }
        let raw = tensor_proto.raw_data.as_ref().ok_or_else(|| ParseError::InvalidTensorData {
            name: name.to_string(),
            reason: "no int64_data and no raw_data".to_string(),
        })?;
        if raw.is_empty() {
            return Err(ParseError::InvalidTensorData {
                name: name.to_string(),
                reason: "empty raw_data for INT64 tensor".to_string(),
            });
        }
        if raw.len() % 8 != 0 {
            return Err(ParseError::InvalidTensorData {
                name: name.to_string(),
                reason: format!("raw_data length {} not a multiple of 8 for INT64", raw.len()),
            });
        }
        Ok(raw.chunks_exact(8)
            .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect())
    }

    fn decode_i32_payload(
        name: &str,
        tensor_proto: &onnx_proto::TensorProto,
    ) -> Result<Vec<i32>, ParseError> {
        if !tensor_proto.int32_data.is_empty() {
            return Ok(tensor_proto.int32_data.clone());
        }
        let raw = tensor_proto.raw_data.as_ref().ok_or_else(|| ParseError::InvalidTensorData {
            name: name.to_string(),
            reason: "no int32_data and no raw_data".to_string(),
        })?;
        if raw.is_empty() {
            return Err(ParseError::InvalidTensorData {
                name: name.to_string(),
                reason: "empty raw_data for INT32 tensor".to_string(),
            });
        }
        if raw.len() % 4 != 0 {
            return Err(ParseError::InvalidTensorData {
                name: name.to_string(),
                reason: format!("raw_data length {} not a multiple of 4 for INT32", raw.len()),
            });
        }
        Ok(raw.chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    fn decode_f64_payload(
        name: &str,
        tensor_proto: &onnx_proto::TensorProto,
    ) -> Result<Vec<f64>, ParseError> {
        if !tensor_proto.double_data.is_empty() {
            return Ok(tensor_proto.double_data.clone());
        }
        let raw = tensor_proto.raw_data.as_ref().ok_or_else(|| ParseError::InvalidTensorData {
            name: name.to_string(),
            reason: "no double_data and no raw_data".to_string(),
        })?;
        if raw.is_empty() {
            return Err(ParseError::InvalidTensorData {
                name: name.to_string(),
                reason: "empty raw_data for DOUBLE tensor".to_string(),
            });
        }
        if raw.len() % 8 != 0 {
            return Err(ParseError::InvalidTensorData {
                name: name.to_string(),
                reason: format!("raw_data length {} not a multiple of 8 for DOUBLE", raw.len()),
            });
        }
        Ok(raw.chunks_exact(8)
            .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect())
    }

    fn decode_bool_payload(
        name: &str,
        tensor_proto: &onnx_proto::TensorProto,
    ) -> Result<Vec<bool>, ParseError> {
        // ONNX carries BOOL in int32_data (one i32 per bool) when
        // the type-specific path is used; each nonzero value means
        // `true`. raw_data uses one byte per bool (again, nonzero
        // = true).
        if !tensor_proto.int32_data.is_empty() {
            return Ok(tensor_proto.int32_data.iter().map(|&v| v != 0).collect());
        }
        let raw = tensor_proto.raw_data.as_ref().ok_or_else(|| ParseError::InvalidTensorData {
            name: name.to_string(),
            reason: "no int32_data and no raw_data".to_string(),
        })?;
        if raw.is_empty() {
            return Err(ParseError::InvalidTensorData {
                name: name.to_string(),
                reason: "empty raw_data for BOOL tensor".to_string(),
            });
        }
        Ok(raw.iter().map(|&b| b != 0).collect())
    }

    /// Reconcile declared shape with actual data length.
    ///
    /// * Default / strict: any mismatch becomes
    ///   [`ParseError::ShapeMismatch`].
    /// * Legacy (opt-in via
    ///   [`ParserOptions::allow_legacy_shape_correction`]): emits an
    ///   unconditional warning to stderr and returns
    ///   `[actual_len]`, matching the pre-WS-1 behaviour. The
    ///   warning is intentionally not gated by
    ///   `#[cfg(debug_assertions)]` — silent release-mode rewrites
    ///   were the exact correctness hazard WS-1 removes.
    fn correct_shape(
        &self,
        name: &str,
        shape: &[usize],
        expected_len: usize,
        actual_len: usize,
    ) -> Result<Vec<usize>, ParseError> {
        if actual_len == expected_len {
            return Ok(shape.to_vec());
        }
        if self.options.allow_legacy_shape_correction {
            eprintln!(
                "warn: iconnx parser rewriting shape for '{}': {} elements, ONNX says {:?} (expects {}), using [{}] (ParserOptions::allow_legacy_shape_correction=true)",
                name, actual_len, shape, expected_len, actual_len
            );
            return Ok(vec![actual_len]);
        }
        Err(ParseError::ShapeMismatch {
            name: name.to_string(),
            declared: shape.to_vec(),
            declared_len: expected_len,
            actual_len,
        })
    }

    /// List all unique operators used
    ///
    /// Returns a sorted list of unique operator types used in the computation graph
    pub fn list_unique_operators(&self) -> Vec<String> {
        use std::collections::HashSet;

        let mut operators = HashSet::new();

        if let Some(graph) = self.proto.graph.as_ref() {
            for node in &graph.node {
                if let Some(op_type) = &node.op_type {
                    operators.insert(op_type.clone());
                }
            }
        }

        let mut result: Vec<String> = operators.into_iter().collect();
        result.sort();
        result
    }

    /// Get computation graph.
    ///
    /// Returns the computation graph with all nodes and their
    /// attributes. Any attribute that references an unsupported
    /// ONNX dtype surfaces as a [`ParseError`] rather than being
    /// silently dropped.
    pub fn computation_graph(&self) -> Result<ComputationGraph, ParseError> {
        let mut nodes: Vec<GraphNode> = Vec::new();
        if let Some(graph) = self.proto.graph.as_ref() {
            nodes.reserve(graph.node.len());
            for node in &graph.node {
                let attributes = Self::parse_attributes(&node.attribute)?;
                nodes.push(GraphNode {
                    op_type: node.op_type.clone().unwrap_or_default(),
                    inputs: node.input.clone(),
                    outputs: node.output.clone(),
                    attributes,
                });
            }
        }
        Ok(ComputationGraph { nodes })
    }

    /// Parse attributes from ONNX AttributeProto.
    ///
    /// Tensor-valued attributes are decoded via
    /// [`Self::parse_tensor_proto`] and any failure is propagated
    /// to the caller instead of being silently dropped (pre-WS-1
    /// this used `if let Ok(_) = …`, which turned unsupported
    /// dtypes into invisible correctness bugs).
    ///
    /// Non-tensor attributes that fail (e.g. non-UTF-8 string
    /// bytes) are still skipped today; they are out of scope for
    /// WS-1 M1.1.
    fn parse_attributes(
        attrs: &[onnx_proto::AttributeProto],
    ) -> Result<crate::attributes::NodeAttributes, ParseError> {
        use crate::attributes::NodeAttributes;

        let mut node_attrs = NodeAttributes::new();

        for attr in attrs {
            let name = attr.name.clone().unwrap_or_default();

            // Integer attribute
            if let Some(i) = attr.i {
                node_attrs.add_int(name.clone(), i);
            }

            // Integer list
            if !attr.ints.is_empty() {
                node_attrs.add_ints(name.clone(), attr.ints.clone());
            }

            // Float attribute
            if let Some(f) = attr.f {
                node_attrs.add_float(name.clone(), f);
            }

            // Float list
            if !attr.floats.is_empty() {
                node_attrs.add_floats(name.clone(), attr.floats.clone());
            }

            // String attribute
            if let Some(s) = &attr.s {
                if let Ok(string) = String::from_utf8(s.clone()) {
                    node_attrs.add_string(name.clone(), string);
                }
            }

            // Tensor attribute — propagate ParseError instead of silent drop.
            if let Some(tensor_proto) = &attr.t {
                let tensor = Self::parse_tensor_proto(tensor_proto)?;
                node_attrs.add_tensor(name.clone(), tensor);
            }
        }

        Ok(node_attrs)
    }

    /// Parse a TensorProto into a Tensor.
    ///
    /// Supports ONNX dtypes FLOAT(1), INT64(7), INT32(6). Every
    /// failure path — unsupported dtype, empty/misaligned payload,
    /// shape/length mismatch — returns a typed [`ParseError`] so
    /// callers can decide what to do instead of silently dropping
    /// the attribute (the pre-WS-1 behaviour).
    fn parse_tensor_proto(
        tensor_proto: &onnx_proto::TensorProto,
    ) -> Result<crate::tensor::Tensor, ParseError> {
        let name = tensor_proto.name.clone().unwrap_or_default();
        let shape: Vec<usize> = tensor_proto.dims.iter().map(|&d| d as usize).collect();
        let expected_len: usize = if shape.is_empty() {
            1
        } else {
            shape.iter().product()
        };
        let data_type = tensor_proto.data_type.unwrap_or(1); // default to FLOAT

        // Zero-element tensor fast path (e.g. dims = [0] or [0, 3]). Valid
        // per ONNX spec: empty float_data / int_data / raw_data is the
        // CORRECT representation, not corruption. Catches this case before
        // the typed-data decoders reject `"empty raw_data for FLOAT tensor"`
        // — the WS-1 M1.2 strict-mode refinement surfaced by YOLOS-tiny's
        // `/vit/embeddings/interpolation/Constant_{2,3}` nodes.
        if expected_len == 0 {
            return match data_type {
                1 => Ok(crate::tensor::Tensor::from_vec_f32(Vec::new(), shape)),
                7 => Ok(crate::tensor::Tensor::from_vec_i64(Vec::new(), shape)),
                6 => Ok(crate::tensor::Tensor::from_vec_i32(Vec::new(), shape)),
                other => Err(ParseError::UnsupportedDType {
                    name,
                    dtype_id: other,
                }),
            };
        }

        match data_type {
            1 => {
                // FLOAT (f32)
                let data: Vec<f32> = if !tensor_proto.float_data.is_empty() {
                    tensor_proto.float_data.clone()
                } else if let Some(raw) = &tensor_proto.raw_data {
                    if raw.is_empty() {
                        return Err(ParseError::InvalidTensorData {
                            name,
                            reason: "empty raw_data for FLOAT tensor".to_string(),
                        });
                    }
                    if raw.len() % 4 != 0 {
                        return Err(ParseError::InvalidTensorData {
                            name,
                            reason: format!(
                                "raw_data length {} not a multiple of 4 for FLOAT",
                                raw.len()
                            ),
                        });
                    }
                    raw.chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect()
                } else {
                    return Err(ParseError::InvalidTensorData {
                        name,
                        reason: "no float_data and no raw_data".to_string(),
                    });
                };

                if data.len() != expected_len {
                    return Err(ParseError::ShapeMismatch {
                        name,
                        declared: shape,
                        declared_len: expected_len,
                        actual_len: data.len(),
                    });
                }
                Ok(crate::tensor::Tensor::from_vec_f32(data, shape))
            }
            7 => {
                // INT64 (i64)
                let data: Vec<i64> = if !tensor_proto.int64_data.is_empty() {
                    tensor_proto.int64_data.clone()
                } else if let Some(raw) = &tensor_proto.raw_data {
                    if raw.is_empty() {
                        return Err(ParseError::InvalidTensorData {
                            name,
                            reason: "empty raw_data for INT64 tensor".to_string(),
                        });
                    }
                    if raw.len() % 8 != 0 {
                        return Err(ParseError::InvalidTensorData {
                            name,
                            reason: format!(
                                "raw_data length {} not a multiple of 8 for INT64",
                                raw.len()
                            ),
                        });
                    }
                    raw.chunks_exact(8)
                        .map(|c| {
                            i64::from_le_bytes([
                                c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7],
                            ])
                        })
                        .collect()
                } else {
                    return Err(ParseError::InvalidTensorData {
                        name,
                        reason: "no int64_data and no raw_data".to_string(),
                    });
                };

                if data.len() != expected_len {
                    return Err(ParseError::ShapeMismatch {
                        name,
                        declared: shape,
                        declared_len: expected_len,
                        actual_len: data.len(),
                    });
                }
                Ok(crate::tensor::Tensor::from_vec_i64(data, shape))
            }
            6 => {
                // INT32 (i32)
                let data: Vec<i32> = if !tensor_proto.int32_data.is_empty() {
                    tensor_proto.int32_data.clone()
                } else if let Some(raw) = &tensor_proto.raw_data {
                    if raw.is_empty() {
                        return Err(ParseError::InvalidTensorData {
                            name,
                            reason: "empty raw_data for INT32 tensor".to_string(),
                        });
                    }
                    if raw.len() % 4 != 0 {
                        return Err(ParseError::InvalidTensorData {
                            name,
                            reason: format!(
                                "raw_data length {} not a multiple of 4 for INT32",
                                raw.len()
                            ),
                        });
                    }
                    raw.chunks_exact(4)
                        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect()
                } else {
                    return Err(ParseError::InvalidTensorData {
                        name,
                        reason: "no int32_data and no raw_data".to_string(),
                    });
                };

                if data.len() != expected_len {
                    return Err(ParseError::ShapeMismatch {
                        name,
                        declared: shape,
                        declared_len: expected_len,
                        actual_len: data.len(),
                    });
                }
                Ok(crate::tensor::Tensor::from_vec_i32(data, shape))
            }
            other => Err(ParseError::UnsupportedDType {
                name,
                dtype_id: other,
            }),
        }
    }
}

/// Computation graph structure
///
/// Contains all computation nodes in execution order
pub struct ComputationGraph {
    nodes: Vec<GraphNode>,
}

impl ComputationGraph {
    /// Get all nodes in execution order
    pub fn nodes(&self) -> &[GraphNode] {
        &self.nodes
    }

    /// Get number of nodes in graph
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

/// A node in the computation graph
///
/// Represents a single operation in the ONNX graph
#[derive(Clone)]
pub struct GraphNode {
    pub op_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attributes: crate::attributes::NodeAttributes,
}

#[cfg(test)]
mod tests {
    //! Unit tests for WS-1 M1.1: typed dtype handling in
    //! `parse_tensor_proto` and caller error surfacing.
    //!
    //! These tests drive the private `parse_tensor_proto` via a
    //! synthetic `TensorProto` built from raw_data / int*_data,
    //! exercising the four cases called out in the WS-1 acceptance
    //! criteria: INT64 scalar, INT64 vector, INT32 scalar, and
    //! unsupported dtype.
    use super::*;
    use crate::tensor::Tensor;

    fn proto_i64_int64_data(name: &str, values: Vec<i64>, dims: Vec<i64>) -> onnx_proto::TensorProto {
        onnx_proto::TensorProto {
            name: Some(name.to_string()),
            data_type: Some(7), // INT64
            dims,
            int64_data: values,
            ..Default::default()
        }
    }

    fn proto_i64_raw_data(name: &str, values: &[i64], dims: Vec<i64>) -> onnx_proto::TensorProto {
        let mut raw = Vec::with_capacity(values.len() * 8);
        for v in values {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        onnx_proto::TensorProto {
            name: Some(name.to_string()),
            data_type: Some(7),
            dims,
            raw_data: Some(raw),
            ..Default::default()
        }
    }

    fn proto_i32_int32_data(name: &str, values: Vec<i32>, dims: Vec<i64>) -> onnx_proto::TensorProto {
        onnx_proto::TensorProto {
            name: Some(name.to_string()),
            data_type: Some(6), // INT32
            dims,
            int32_data: values,
            ..Default::default()
        }
    }

    #[test]
    fn parse_tensor_proto_int64_scalar_via_int64_data() {
        // 0-d INT64 scalar (dims = []) carrying a single value,
        // analogous to Whisper-Tiny's Constant_output_0.
        let proto = proto_i64_int64_data("const", vec![42], vec![]);
        let tensor = OnnxModel::parse_tensor_proto(&proto).expect("INT64 scalar should parse");
        match tensor {
            Tensor::Int64(arr) => {
                assert!(arr.shape().is_empty(), "expected 0-d shape, got {:?}", arr.shape());
                assert_eq!(arr.iter().copied().collect::<Vec<_>>(), vec![42]);
            }
            other => panic!("expected Tensor::Int64, got {:?}", other),
        }
    }

    #[test]
    fn parse_tensor_proto_int64_vector_via_raw_data() {
        // 1-d INT64 vector carried in raw_data (little-endian),
        // covering the chunks_exact(8) path.
        let values = vec![-1_i64, 0, 1, 2, 3];
        let proto = proto_i64_raw_data("shape", &values, vec![values.len() as i64]);
        let tensor = OnnxModel::parse_tensor_proto(&proto).expect("INT64 vector should parse");
        match tensor {
            Tensor::Int64(arr) => {
                assert_eq!(arr.shape(), &[values.len()]);
                assert_eq!(arr.iter().copied().collect::<Vec<_>>(), values);
            }
            other => panic!("expected Tensor::Int64, got {:?}", other),
        }
    }

    #[test]
    fn parse_tensor_proto_int32_scalar_via_int32_data() {
        // 0-d INT32 scalar (dims = []).
        let proto = proto_i32_int32_data("axis", vec![-7], vec![]);
        let tensor = OnnxModel::parse_tensor_proto(&proto).expect("INT32 scalar should parse");
        match tensor {
            Tensor::Int32(arr) => {
                assert!(arr.shape().is_empty());
                assert_eq!(arr.iter().copied().collect::<Vec<_>>(), vec![-7]);
            }
            other => panic!("expected Tensor::Int32, got {:?}", other),
        }
    }

    fn init_proto(
        name: &str,
        data_type: i32,
        dims: Vec<i64>,
    ) -> onnx_proto::TensorProto {
        onnx_proto::TensorProto {
            name: Some(name.to_string()),
            data_type: Some(data_type),
            dims,
            ..Default::default()
        }
    }

    /// Helper: build an `OnnxModel` carrying a single initializer
    /// so we can exercise `extract_weights` without a real file.
    /// Defaults to strict parser options; individual tests override
    /// via [`model_with_initializer_opts`] when they need the
    /// legacy shape-correction path.
    fn model_with_initializer(init: onnx_proto::TensorProto) -> OnnxModel {
        model_with_initializer_opts(init, ParserOptions::default())
    }

    fn model_with_initializer_opts(
        init: onnx_proto::TensorProto,
        options: ParserOptions,
    ) -> OnnxModel {
        OnnxModel {
            proto: onnx_proto::ModelProto {
                graph: Some(onnx_proto::GraphProto {
                    initializer: vec![init],
                    ..Default::default()
                }),
                ..Default::default()
            },
            options,
        }
    }

    #[test]
    fn extract_weights_decodes_float64_via_double_data() {
        // FLOAT64 via double_data — exercises M1.2 dtype 11.
        let mut init = init_proto("bias_f64", 11, vec![3]);
        init.double_data = vec![1.0_f64, 2.5, -0.25];
        let model = model_with_initializer(init);

        let weights = model.extract_weights().expect("f64 initializer should load");
        let tensor = weights.get("bias_f64").expect("tensor present");
        match tensor {
            Tensor::Float64(arr) => {
                assert_eq!(arr.shape(), &[3]);
                assert_eq!(arr.iter().copied().collect::<Vec<_>>(), vec![1.0, 2.5, -0.25]);
            }
            other => panic!("expected Tensor::Float64, got {:?}", other),
        }
    }

    #[test]
    fn extract_weights_decodes_float64_via_raw_data() {
        let values: Vec<f64> = vec![0.0, 1.0, -1.0, 42.5];
        let mut raw = Vec::with_capacity(values.len() * 8);
        for v in &values {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        let mut init = init_proto("weights_f64", 11, vec![values.len() as i64]);
        init.raw_data = Some(raw);
        let model = model_with_initializer(init);

        let weights = model.extract_weights().expect("f64 raw_data should decode");
        match weights.get("weights_f64").unwrap() {
            Tensor::Float64(arr) => {
                assert_eq!(arr.iter().copied().collect::<Vec<_>>(), values);
            }
            other => panic!("expected Tensor::Float64, got {:?}", other),
        }
    }

    #[test]
    fn extract_weights_decodes_bool_via_int32_data() {
        // BOOL via int32_data — one i32 per bool, nonzero = true.
        let mut init = init_proto("mask", 9, vec![4]);
        init.int32_data = vec![0, 1, 0, 2];
        let model = model_with_initializer(init);

        let weights = model.extract_weights().expect("bool initializer should load");
        match weights.get("mask").unwrap() {
            Tensor::Bool(arr) => {
                assert_eq!(arr.shape(), &[4]);
                assert_eq!(
                    arr.iter().copied().collect::<Vec<_>>(),
                    vec![false, true, false, true]
                );
            }
            other => panic!("expected Tensor::Bool, got {:?}", other),
        }
    }

    #[test]
    fn extract_weights_decodes_bool_via_raw_data() {
        // BOOL via raw_data — one byte per bool.
        let mut init = init_proto("flags", 9, vec![3]);
        init.raw_data = Some(vec![0u8, 1u8, 0xFFu8]);
        let model = model_with_initializer(init);

        let weights = model.extract_weights().expect("bool raw_data should decode");
        match weights.get("flags").unwrap() {
            Tensor::Bool(arr) => {
                assert_eq!(
                    arr.iter().copied().collect::<Vec<_>>(),
                    vec![false, true, true]
                );
            }
            other => panic!("expected Tensor::Bool, got {:?}", other),
        }
    }

    #[test]
    fn extract_weights_strict_mode_returns_shape_mismatch_err() {
        // Declared shape says 4 elements, but the payload carries 3.
        // In strict mode (default) this must surface as a typed
        // ShapeMismatch error rather than being silently rewritten.
        let mut init = init_proto("misshapen_weights", 1, vec![4]);
        init.float_data = vec![0.0_f32, 1.0, 2.0]; // 3 elements, not 4
        let model = model_with_initializer(init);

        match model.extract_weights() {
            Err(ParseError::ShapeMismatch {
                name,
                declared,
                declared_len,
                actual_len,
            }) => {
                assert_eq!(name, "misshapen_weights");
                assert_eq!(declared, vec![4]);
                assert_eq!(declared_len, 4);
                assert_eq!(actual_len, 3);
            }
            other => panic!("expected ShapeMismatch, got {:?}", other),
        }
    }

    #[test]
    fn extract_weights_legacy_flag_rewrites_shape_to_actual_len() {
        // Same mismatch, but opt-in legacy flag: shape should be
        // rewritten to [actual_len] (matching pre-WS-1 behaviour)
        // and the initializer should load successfully.
        let mut init = init_proto("misshapen_weights", 1, vec![4]);
        init.float_data = vec![0.0_f32, 1.0, 2.0];
        let model = model_with_initializer_opts(
            init,
            ParserOptions {
                allow_legacy_shape_correction: true,
            },
        );

        let weights = model
            .extract_weights()
            .expect("legacy shape correction should accept mismatch");
        match weights.get("misshapen_weights").unwrap() {
            Tensor::Float32(arr) => {
                assert_eq!(arr.shape(), &[3]);
                assert_eq!(arr.iter().copied().collect::<Vec<_>>(), vec![0.0, 1.0, 2.0]);
            }
            other => panic!("expected Tensor::Float32, got {:?}", other),
        }
    }

    #[test]
    fn extract_weights_unsupported_dtype_returns_error() {
        // dtype 10 = FP16 — not representable in today's Tensor enum;
        // must be an Err per WS-1, not silently dropped.
        let mut init = init_proto("fp16_weights", 10, vec![4]);
        init.raw_data = Some(vec![0u8; 8]);
        let model = model_with_initializer(init);

        match model.extract_weights() {
            Err(ParseError::UnsupportedInitializerDType { name, dtype_id }) => {
                assert_eq!(name, "fp16_weights");
                assert_eq!(dtype_id, 10);
            }
            other => panic!("expected UnsupportedInitializerDType, got {:?}", other),
        }
    }

    #[test]
    fn parse_tensor_proto_unsupported_dtype_returns_error() {
        // Use a dtype the parser does not yet decode. Pick 10
        // (FP16) — a concrete WS-3 dtype — so we also validate the
        // error carries a recognisable dtype_id.
        let proto = onnx_proto::TensorProto {
            name: Some("fp16_weights".to_string()),
            data_type: Some(10),
            dims: vec![4],
            raw_data: Some(vec![0u8; 8]),
            ..Default::default()
        };
        match OnnxModel::parse_tensor_proto(&proto) {
            Err(ParseError::UnsupportedDType { name, dtype_id }) => {
                assert_eq!(name, "fp16_weights");
                assert_eq!(dtype_id, 10);
            }
            other => panic!("expected UnsupportedDType, got {:?}", other),
        }
    }

    // ---- Zero-element tensor strict-mode refinement -----------------
    //
    // Surfaced by YOLOS-tiny's
    // `/vit/embeddings/interpolation/Constant_{2,3}` nodes, which carry
    // FLOAT tensors with `dims = [0]` and empty float_data / raw_data /
    // int_data. Per the ONNX spec this is a zero-element placeholder —
    // the correct payload representation is "nothing to decode." WS-1
    // M1.2's strict-mode check rejected this case as
    // `InvalidTensorData { reason: "empty raw_data for FLOAT tensor" }`.
    // The refinement distinguishes `expected_len == 0` (valid empty
    // tensor) from `expected_len > 0 + no data` (genuine corruption).

    #[test]
    fn parse_tensor_proto_zero_element_float_succeeds() {
        // Mirrors YOLOS-tiny's Constant_{2,3} attribute tensors.
        let proto = onnx_proto::TensorProto {
            name: Some("empty_float".to_string()),
            data_type: Some(1), // FLOAT
            dims: vec![0],
            ..Default::default()
        };
        let tensor = OnnxModel::parse_tensor_proto(&proto).expect("zero-element FLOAT should parse");
        match tensor {
            Tensor::Float32(arr) => {
                assert_eq!(arr.shape(), &[0]);
                assert_eq!(arr.len(), 0);
            }
            other => panic!("expected Tensor::Float32, got {:?}", other),
        }
    }

    #[test]
    fn parse_tensor_proto_zero_element_int64_succeeds() {
        let proto = onnx_proto::TensorProto {
            name: Some("empty_i64".to_string()),
            data_type: Some(7),
            dims: vec![0],
            ..Default::default()
        };
        let tensor = OnnxModel::parse_tensor_proto(&proto).expect("zero-element INT64 should parse");
        match tensor {
            Tensor::Int64(arr) => {
                assert_eq!(arr.shape(), &[0]);
                assert_eq!(arr.len(), 0);
            }
            other => panic!("expected Tensor::Int64, got {:?}", other),
        }
    }

    #[test]
    fn parse_tensor_proto_zero_element_int32_succeeds() {
        let proto = onnx_proto::TensorProto {
            name: Some("empty_i32".to_string()),
            data_type: Some(6),
            dims: vec![0],
            ..Default::default()
        };
        let tensor = OnnxModel::parse_tensor_proto(&proto).expect("zero-element INT32 should parse");
        match tensor {
            Tensor::Int32(arr) => {
                assert_eq!(arr.shape(), &[0]);
                assert_eq!(arr.len(), 0);
            }
            other => panic!("expected Tensor::Int32, got {:?}", other),
        }
    }

    #[test]
    fn parse_tensor_proto_zero_element_multi_dim_succeeds() {
        // dims = [0, 3] — rank-2 tensor with zero rows; still
        // zero elements, empty data is correct.
        let proto = onnx_proto::TensorProto {
            name: Some("empty_matrix".to_string()),
            data_type: Some(1),
            dims: vec![0, 3],
            ..Default::default()
        };
        let tensor = OnnxModel::parse_tensor_proto(&proto).expect("zero-element matrix should parse");
        match tensor {
            Tensor::Float32(arr) => {
                assert_eq!(arr.shape(), &[0, 3]);
                assert_eq!(arr.len(), 0);
            }
            other => panic!("expected Tensor::Float32, got {:?}", other),
        }
    }

    #[test]
    fn parse_tensor_proto_nonempty_dims_with_no_data_still_errors() {
        // Genuine corruption case: declared 3 elements but no payload.
        // Must remain an error — the zero-element refinement only
        // applies when expected_len == 0.
        let proto = onnx_proto::TensorProto {
            name: Some("corrupted".to_string()),
            data_type: Some(1),
            dims: vec![3],
            ..Default::default()
        };
        match OnnxModel::parse_tensor_proto(&proto) {
            Err(ParseError::InvalidTensorData { name, .. }) => {
                assert_eq!(name, "corrupted");
            }
            other => panic!("expected InvalidTensorData, got {:?}", other),
        }
    }

    #[test]
    fn parse_tensor_proto_zero_element_unsupported_dtype_still_errors() {
        // Zero-element refinement must not mask unsupported dtypes —
        // an FP16 tensor with dims=[0] should still surface as
        // UnsupportedDType rather than silently succeeding.
        let proto = onnx_proto::TensorProto {
            name: Some("empty_fp16".to_string()),
            data_type: Some(10), // FP16 — WS-3 scope
            dims: vec![0],
            ..Default::default()
        };
        match OnnxModel::parse_tensor_proto(&proto) {
            Err(ParseError::UnsupportedDType { name, dtype_id }) => {
                assert_eq!(name, "empty_fp16");
                assert_eq!(dtype_id, 10);
            }
            other => panic!("expected UnsupportedDType, got {:?}", other),
        }
    }

    #[test]
    fn extract_weights_zero_element_initializer_succeeds() {
        // Symmetric coverage for extract_weights — zero-element
        // initializers are uncommon but valid per ONNX.
        let init = init_proto("empty_init", 1, vec![0]);
        let model = model_with_initializer(init);
        let weights = model
            .extract_weights()
            .expect("zero-element initializer should load");
        match weights.get("empty_init").unwrap() {
            Tensor::Float32(arr) => {
                assert_eq!(arr.shape(), &[0]);
                assert_eq!(arr.len(), 0);
            }
            other => panic!("expected Tensor::Float32, got {:?}", other),
        }
    }
}
