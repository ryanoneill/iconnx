//! Differential Testing Framework for GPU Inference
//!
//! Compares GPU execution against an external reference (per-node ORT
//! fixture directory or in-memory caller-supplied tensors), stopping at
//! the first divergence with detailed diagnostics.
//!
//! Two reference modes:
//! - [`ReferenceSource::FixtureDir`] — per-node ORT references on disk;
//!   supports full per-node bisect via the checkpoint system.
//! - [`ReferenceSource::InMemory`] — caller-supplied output-level
//!   reference tensors (e.g., leadline-bench's just-computed ORT
//!   outputs); compares only the requested graph outputs, no per-node
//!   bisect.
//!
//! Builder usage:
//! ```ignore
//! let runner = DifferentialRunner::new(model_path)?
//!     .with_reference(ReferenceSource::FixtureDir { path: fixture_path })
//!     .with_tolerance(DifferentialTolerance::strict())
//!     .with_op_chain_context_depth(5);
//! let report = runner.run_until_divergence(inputs)?;
//! ```
//!
//! `DifferentialTolerance` itself lives at `crate::tolerance` and is
//! always-available (NOT gated on `debug-inference`). Only this runtime
//! side is `debug-inference` gated.

use super::checkpoints::CheckpointConfig;
use super::inference::GpuGraphExecutor;
use crate::onnx_parser::OnnxParser;
use crate::tensor::Tensor;
use crate::tolerance::DifferentialTolerance;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Backward-compatible alias for `DifferentialTolerance`.
///
/// Older WS-3.x callers use `DiffTolerance` with the legacy
/// `correlation_threshold`/`max_abs_diff`/`allow_shape_mismatch` shape;
/// the canonical type has moved to `crate::tolerance::DifferentialTolerance`
/// (just `abs` + `rel` optionals). The alias preserves the import path
/// during the migration window.
pub type DiffTolerance = DifferentialTolerance;

/// Where iconnx finds reference outputs to compare against.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ReferenceSource {
    /// Existing fixture-directory format: per-node ORT references in
    /// `<path>/checkpoints/` with `metadata.json`. Supports full per-node
    /// bisect.
    FixtureDir { path: PathBuf },

    /// In-memory references provided by the caller (e.g. leadline-bench
    /// passes its just-computed ORT outputs). **Output-level only** —
    /// per-node bisect requires `FixtureDir` mode (or a future leadline-
    /// bench addition that captures intermediate ORT activations).
    InMemory {
        tensors: HashMap<String, Tensor>,
    },
}

/// Variant indicating which tolerance bound was exceeded.
#[derive(Debug, Clone)]
pub enum ExceededTolerance {
    /// Absolute difference exceeded the configured ceiling. Carries the
    /// observed max-abs-diff value.
    Abs(f32),
    /// Relative difference exceeded the configured ceiling. Carries the
    /// observed max-rel-diff fraction.
    Rel(f32),
    /// Forced divergence due to non-finite values (NaN/Inf) in the actual
    /// output. Mirrors leadline-bench's
    /// `nan_in_iconnx_output_forces_infinite_diff` regression pattern.
    NonFinite,
}

/// Truncated preview of a tensor for inclusion in divergence reports.
#[derive(Debug, Clone)]
pub struct TensorPreview {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    /// Up to 64 sample values (f32-coerced for half-precision dtypes).
    pub samples: Vec<f32>,
}

/// Report of a divergence found during differential testing.
#[derive(Debug, Clone)]
pub struct DivergenceReport {
    /// Name of the first divergent node (or graph output, in InMemory mode).
    pub node_name: String,
    /// Operator type of the divergent node (or `"GraphOutput"` for an
    /// InMemory-mode top-level divergence).
    pub op_type: String,
    /// Index of the divergent output among the node's outputs (0 for
    /// single-output ops, or position in the `output_names` list for
    /// graph-output mode).
    pub output_index: usize,
    /// Which tolerance bound triggered the divergence.
    pub exceeded: ExceededTolerance,
    /// The observed magnitude (raw max-abs-diff for `Abs`, raw fraction
    /// for `Rel`, `f32::INFINITY` for `NonFinite`).
    pub diff_magnitude: f32,
    /// Inputs to the divergent node (truncated to <= 64 elements per
    /// tensor). Best-effort: populated where the executor exposes the
    /// input tensors at divergence time, empty otherwise.
    pub sample_inputs: Vec<TensorPreview>,
    /// Parent op names leading to the divergent node, walked backwards
    /// up to `op_chain_context_depth` (default 5).
    pub op_chain_context: Vec<String>,
    /// Number of nodes executed before divergence (FixtureDir mode only;
    /// 0 for InMemory mode).
    pub nodes_before_divergence: usize,
}

impl DivergenceReport {
    /// Format as a human-readable string.
    pub fn to_string_detailed(&self) -> String {
        let mut s = String::new();
        s.push_str("\n=== Divergence Report ===\n");
        s.push_str(&format!("Node: {} ({})\n", self.node_name, self.op_type));
        s.push_str(&format!("Output index: {}\n", self.output_index));
        s.push_str(&format!(
            "Nodes executed before divergence: {}\n",
            self.nodes_before_divergence
        ));
        match &self.exceeded {
            ExceededTolerance::Abs(v) => {
                s.push_str(&format!("Exceeded: Abs(max_abs_diff = {:.6e})\n", v));
            }
            ExceededTolerance::Rel(v) => {
                s.push_str(&format!("Exceeded: Rel(max_rel_diff = {:.6})\n", v));
            }
            ExceededTolerance::NonFinite => {
                s.push_str("Exceeded: NonFinite (NaN/Inf in iconnx output)\n");
            }
        }
        s.push_str(&format!("Diff magnitude: {:.6e}\n", self.diff_magnitude));

        if !self.op_chain_context.is_empty() {
            s.push_str("\nOp chain context (most recent first):\n");
            for (i, name) in self.op_chain_context.iter().enumerate() {
                s.push_str(&format!("  [-{}] {}\n", i + 1, name));
            }
        }

        if !self.sample_inputs.is_empty() {
            s.push_str("\nSample inputs:\n");
            for tp in &self.sample_inputs {
                s.push_str(&format!(
                    "  {} ({:?}, {}): {} samples\n",
                    tp.name,
                    tp.shape,
                    tp.dtype,
                    tp.samples.len()
                ));
            }
        }
        s
    }
}

/// Per-node metadata captured at runner construction so divergence reports
/// can reconstruct an op-chain context without re-parsing the model.
#[derive(Clone, Debug)]
struct NodeMeta {
    name: String,
    op_type: String,
    inputs: Vec<String>,
    outputs: Vec<String>,
}

/// Differential testing runner.
///
/// Construct via [`DifferentialRunner::new`], then chain builder methods
/// to configure the reference source, tolerance, output names, and op-
/// chain context depth. Finally call [`DifferentialRunner::run_until_divergence`]
/// with model inputs.
pub struct DifferentialRunner {
    executor: GpuGraphExecutor,
    reference: ReferenceSource,
    tolerance: DifferentialTolerance,
    /// Empty = all graph outputs (Q4.1 default).
    output_names: Vec<String>,
    /// Default 5 (Q3 configurability).
    op_chain_context_depth: usize,
    /// Captured at construction so we don't re-parse the model on report
    /// assembly.
    graph_nodes: Vec<NodeMeta>,
    /// Default outputs declared in the ONNX graph (used when
    /// `output_names` is empty).
    model_outputs: Vec<String>,
}

impl DifferentialRunner {
    /// Construct a new runner from an ONNX model path. Defaults: tolerance
    /// disabled, in-memory empty reference (must be replaced via
    /// [`Self::with_reference`] before [`Self::run_until_divergence`]),
    /// all graph outputs requested, op_chain_context_depth = 5.
    pub fn new(model_path: &Path) -> Result<Self, DifferentialError> {
        let model = OnnxParser::parse_file(model_path).map_err(|e| {
            DifferentialError::ModelLoad(format!("Failed to parse ONNX: {:?}", e))
        })?;

        let weights = model.extract_weights().map_err(|e| {
            DifferentialError::ModelLoad(format!("Failed to extract weights: {}", e))
        })?;
        let graph = model.computation_graph().map_err(|e| {
            DifferentialError::ModelLoad(format!("Failed to build computation graph: {}", e))
        })?;
        let nodes = graph.nodes();

        let mut executor = GpuGraphExecutor::new()
            .map_err(|e| DifferentialError::GpuInit(format!("GPU init failed: {}", e)))?;

        for (name, tensor) in &weights {
            executor.add_initializer(name.clone(), tensor).map_err(|e| {
                DifferentialError::GpuInit(format!("Weight upload failed: {}", e))
            })?;
        }

        let mut graph_nodes = Vec::with_capacity(nodes.len());
        for (idx, node) in nodes.iter().enumerate() {
            let fallback_name = format!("node_{}", idx);
            let node_name = node
                .outputs
                .first()
                .map(|s| s.as_str())
                .unwrap_or(&fallback_name)
                .to_string();
            let inputs: Vec<&str> = node.inputs.iter().map(|s| s.as_str()).collect();
            let outputs: Vec<&str> = node.outputs.iter().map(|s| s.as_str()).collect();
            executor.add_node(
                &node_name,
                &node.op_type,
                inputs,
                outputs,
                node.attributes.clone(),
            );

            graph_nodes.push(NodeMeta {
                name: node_name,
                op_type: node.op_type.clone(),
                inputs: node.inputs.clone(),
                outputs: node.outputs.clone(),
            });
        }

        let model_outputs = model.outputs();

        Ok(Self {
            executor,
            reference: ReferenceSource::InMemory {
                tensors: HashMap::new(),
            },
            tolerance: DifferentialTolerance::default(),
            output_names: Vec::new(),
            op_chain_context_depth: 5,
            graph_nodes,
            model_outputs,
        })
    }

    /// Override the tolerance gate (default: disabled).
    pub fn with_tolerance(mut self, t: DifferentialTolerance) -> Self {
        self.tolerance = t;
        self
    }

    /// Set the reference source.
    pub fn with_reference(mut self, src: ReferenceSource) -> Self {
        self.reference = src;
        self
    }

    /// Restrict the output names compared at the graph-output level.
    /// Empty = all model outputs.
    pub fn with_outputs(mut self, names: &[&str]) -> Self {
        self.output_names = names.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Override the op-chain context depth (default: 5). When a
    /// divergence is detected, the report includes up to this many
    /// preceding op names walked backward through the producer chain.
    pub fn with_op_chain_context_depth(mut self, depth: usize) -> Self {
        self.op_chain_context_depth = depth;
        self
    }

    /// Get the underlying executor.
    pub fn executor(&self) -> &GpuGraphExecutor {
        &self.executor
    }

    /// Test-only accessor: configured op-chain context depth.
    #[cfg(test)]
    fn op_chain_context_depth_for_test(&self) -> usize {
        self.op_chain_context_depth
    }

    /// Test-only accessor: configured reference source variant tag.
    #[cfg(test)]
    fn reference_variant_for_test(&self) -> &'static str {
        match &self.reference {
            ReferenceSource::FixtureDir { .. } => "FixtureDir",
            ReferenceSource::InMemory { .. } => "InMemory",
        }
    }

    /// Test-only accessor: configured output names (empty = all).
    #[cfg(test)]
    fn output_names_for_test(&self) -> &[String] {
        &self.output_names
    }

    /// Execute the model and detect first divergence vs the configured
    /// reference. Returns `Ok(None)` if all checkpoints converged within
    /// tolerance; `Ok(Some(report))` for the first divergent node;
    /// `Err(_)` for setup or execution failure (orthogonal to convergence).
    pub fn run_until_divergence(
        &self,
        inputs: HashMap<String, Tensor>,
    ) -> Result<Option<DivergenceReport>, DifferentialError> {
        let resolved_outputs: Vec<String> = if self.output_names.is_empty() {
            self.model_outputs.clone()
        } else {
            self.output_names.clone()
        };

        match &self.reference {
            ReferenceSource::FixtureDir { path } => {
                self.run_until_divergence_fixture_dir(inputs, &resolved_outputs, path)
            }
            ReferenceSource::InMemory { tensors } => {
                self.run_until_divergence_in_memory(inputs, &resolved_outputs, tensors)
            }
        }
    }

    fn run_until_divergence_fixture_dir(
        &self,
        inputs: HashMap<String, Tensor>,
        resolved_outputs: &[String],
        fixture_path: &Path,
    ) -> Result<Option<DivergenceReport>, DifferentialError> {
        // Translate the new tolerance shape into the legacy CheckpointConfig
        // gate values. The checkpoint system uses correlation_threshold
        // (f64) + max_diff_threshold (f32). For the abs gate we forward
        // tolerance.abs; correlation_threshold is set permissively (since
        // the new model doesn't track correlation explicitly — divergence
        // is gated on abs alone here, matching the WS-3 sign-off ceiling
        // semantics).
        let max_diff_threshold = self.tolerance.abs.unwrap_or(f32::MAX);
        let config = CheckpointConfig {
            base_dir: fixture_path.to_path_buf(),
            compare_to_ort: true,
            correlation_threshold: 0.0, // permissive: don't gate on correlation
            max_diff_threshold,
            ..CheckpointConfig::default()
        };

        let output_refs: Vec<&str> = resolved_outputs.iter().map(|s| s.as_str()).collect();

        match self.executor.run_with_checkpoints(inputs, output_refs, config) {
            Ok((_, mgr)) => {
                let Some(first_div) = mgr.first_divergence() else {
                    return Ok(None);
                };
                let Some(entry) = mgr.entries().iter().find(|e| e.name == first_div) else {
                    return Ok(None);
                };
                let nodes_before = mgr
                    .entries()
                    .iter()
                    .position(|e| e.name == first_div)
                    .unwrap_or(0);

                let raw_diff = entry.max_diff_from_ort.unwrap_or(f32::INFINITY);
                let stats_non_finite = entry.stats.has_nan || entry.stats.has_inf;

                let exceeded = if stats_non_finite || !raw_diff.is_finite() {
                    ExceededTolerance::NonFinite
                } else if let Some(abs_ceiling) = self.tolerance.abs {
                    if raw_diff > abs_ceiling {
                        ExceededTolerance::Abs(raw_diff)
                    } else {
                        // Should not reach here because mgr only flagged
                        // divergence when raw_diff > max_diff_threshold;
                        // surface the raw diff under Abs as a fallback.
                        ExceededTolerance::Abs(raw_diff)
                    }
                } else {
                    ExceededTolerance::Abs(raw_diff)
                };

                let op_chain_context = self.op_chain_context_for(&entry.name);

                Ok(Some(DivergenceReport {
                    node_name: entry.name.clone(),
                    op_type: entry.op_type.clone(),
                    output_index: 0,
                    exceeded,
                    diff_magnitude: raw_diff,
                    sample_inputs: Vec::new(),
                    op_chain_context,
                    nodes_before_divergence: nodes_before,
                }))
            }
            Err(e) => {
                let err_msg = e.to_string();
                if err_msg.contains("not broadcastable") || err_msg.contains("shapes") {
                    if let Some(node_name) = extract_node_name_from_error(&err_msg) {
                        let op_type = extract_op_type_from_error(&err_msg)
                            .unwrap_or("Unknown")
                            .to_string();
                        let op_chain_context = self.op_chain_context_for(node_name);
                        return Ok(Some(DivergenceReport {
                            node_name: node_name.to_string(),
                            op_type,
                            output_index: 0,
                            exceeded: ExceededTolerance::Abs(f32::INFINITY),
                            diff_magnitude: f32::INFINITY,
                            sample_inputs: Vec::new(),
                            op_chain_context,
                            nodes_before_divergence: 0,
                        }));
                    }
                }
                Err(DifferentialError::Execution(err_msg))
            }
        }
    }

    fn run_until_divergence_in_memory(
        &self,
        inputs: HashMap<String, Tensor>,
        resolved_outputs: &[String],
        reference: &HashMap<String, Tensor>,
    ) -> Result<Option<DivergenceReport>, DifferentialError> {
        let output_refs: Vec<&str> = resolved_outputs.iter().map(|s| s.as_str()).collect();
        let actual = self
            .executor
            .run(inputs, output_refs)
            .map_err(|e| DifferentialError::Execution(e.to_string()))?;

        for (idx, name) in resolved_outputs.iter().enumerate() {
            let actual_tensor = actual.get(name).ok_or_else(|| {
                DifferentialError::Execution(format!(
                    "executor did not produce requested output {:?}",
                    name
                ))
            })?;
            let expected_tensor = match reference.get(name) {
                Some(t) => t,
                None => continue, // No reference for this output — skip.
            };

            let actual_f32 = tensor_to_f32_vec(actual_tensor);
            let expected_f32 = tensor_to_f32_vec(expected_tensor);

            if let Some(exceeded) =
                compare_with_tolerance(&actual_f32, &expected_f32, self.tolerance)
            {
                let diff_magnitude = match &exceeded {
                    ExceededTolerance::Abs(v) | ExceededTolerance::Rel(v) => *v,
                    ExceededTolerance::NonFinite => f32::INFINITY,
                };
                let op_chain_context = self.op_chain_context_for(name);
                return Ok(Some(DivergenceReport {
                    node_name: name.clone(),
                    op_type: "GraphOutput".to_string(),
                    output_index: idx,
                    exceeded,
                    diff_magnitude,
                    sample_inputs: Vec::new(),
                    op_chain_context,
                    nodes_before_divergence: 0,
                }));
            }
        }
        Ok(None)
    }

    /// Walk the producer chain backwards from `node_name`, collecting up
    /// to `op_chain_context_depth` parent op names (most-recent-first).
    /// Returns an empty vec if the node isn't found in the graph or the
    /// configured depth is zero.
    fn op_chain_context_for(&self, node_name: &str) -> Vec<String> {
        if self.op_chain_context_depth == 0 {
            return Vec::new();
        }
        // Find the node by output name.
        let mut current_idx = match self
            .graph_nodes
            .iter()
            .position(|n| n.outputs.iter().any(|o| o == node_name))
        {
            Some(i) => i,
            None => return Vec::new(),
        };

        let mut chain = Vec::new();
        let mut visited = vec![false; self.graph_nodes.len()];
        visited[current_idx] = true;

        // Walk back through the first input's producer until depth exhausted.
        for _ in 0..self.op_chain_context_depth {
            let node = &self.graph_nodes[current_idx];
            // Find the first input that has a producer.
            let mut next_idx: Option<usize> = None;
            for input in &node.inputs {
                if let Some(producer_idx) = self
                    .graph_nodes
                    .iter()
                    .position(|n| n.outputs.iter().any(|o| o == input))
                {
                    if !visited[producer_idx] {
                        next_idx = Some(producer_idx);
                        break;
                    }
                }
            }
            match next_idx {
                Some(idx) => {
                    let parent = &self.graph_nodes[idx];
                    chain.push(format!("{} ({})", parent.name, parent.op_type));
                    visited[idx] = true;
                    current_idx = idx;
                }
                None => break,
            }
        }
        chain
    }
}

/// Compare actual vs expected; returns `None` if within tolerance, else
/// `Some(ExceededTolerance)` describing how the gate was exceeded.
///
/// **NaN/Inf in `actual` is always forced-divergence**, regardless of
/// tolerance configuration. Mirrors leadline-bench's
/// `nan_in_iconnx_output_forces_infinite_diff` regression pattern: we
/// must NOT silently `f32::max(0.0, NaN) = 0.0` and pass; the output is
/// garbage if it contains NaN/Inf and the gate must fail loudly.
pub(super) fn compare_with_tolerance(
    actual: &[f32],
    expected: &[f32],
    tolerance: DifferentialTolerance,
) -> Option<ExceededTolerance> {
    // Step 1: explicit NaN/Inf check on actual.
    if actual.iter().any(|x| !x.is_finite()) {
        return Some(ExceededTolerance::NonFinite);
    }

    // Step 2: shape mismatch is also a forced divergence.
    if actual.len() != expected.len() {
        return Some(ExceededTolerance::Abs(f32::INFINITY));
    }

    // Step 3: standard abs/rel gate. Compute max-abs-diff treating any
    // NaN diff (e.g. expected[i] is NaN) as +inf so it cannot silently
    // pass an abs ceiling.
    let max_abs = actual
        .iter()
        .zip(expected.iter())
        .map(|(a, e)| {
            let d = (a - e).abs();
            if d.is_nan() {
                f32::INFINITY
            } else {
                d
            }
        })
        .fold(0.0_f32, f32::max);

    if let Some(abs_ceiling) = tolerance.abs {
        if max_abs > abs_ceiling {
            return Some(ExceededTolerance::Abs(max_abs));
        }
    }

    if let Some(rel_ceiling) = tolerance.rel {
        let max_val = actual
            .iter()
            .map(|x| x.abs())
            .chain(expected.iter().filter(|x| x.is_finite()).map(|x| x.abs()))
            .fold(0.0_f32, f32::max);
        if max_val > 0.0 {
            let rel = max_abs / max_val;
            if rel > rel_ceiling {
                return Some(ExceededTolerance::Rel(rel));
            }
        }
    }

    // No tolerance configured: any nonzero diff triggers divergence.
    if !tolerance.is_active() && max_abs > 0.0 {
        return Some(ExceededTolerance::Abs(max_abs));
    }

    None
}

/// Coerce a `Tensor` to a flat `Vec<f32>` for comparison purposes.
/// Half-precision dtypes are upcast; integer dtypes are cast lossily to
/// f32 (acceptable for divergence comparison since the tolerance is the
/// gate of record).
fn tensor_to_f32_vec(t: &Tensor) -> Vec<f32> {
    match t {
        Tensor::Float32(arr) => arr.iter().copied().collect(),
        Tensor::Float64(arr) => arr.iter().map(|x| *x as f32).collect(),
        Tensor::Float16(arr) => arr.iter().map(|x| x.to_f32()).collect(),
        Tensor::BFloat16(arr) => arr.iter().map(|x| x.to_f32()).collect(),
        Tensor::Int32(arr) => arr.iter().map(|x| *x as f32).collect(),
        Tensor::Int64(arr) => arr.iter().map(|x| *x as f32).collect(),
        Tensor::Int8(arr) => arr.iter().map(|x| *x as f32).collect(),
        Tensor::UInt8(arr) => arr.iter().map(|x| *x as f32).collect(),
        Tensor::Bool(arr) => arr.iter().map(|x| if *x { 1.0 } else { 0.0 }).collect(),
    }
}

/// Extract node name from error message.
fn extract_node_name_from_error(err: &str) -> Option<&str> {
    if let Some(start) = err.find("Node '") {
        let rest = &err[start + 6..];
        if let Some(end) = rest.find('\'') {
            return Some(&rest[..end]);
        }
    }
    None
}

/// Extract op type from error message.
fn extract_op_type_from_error(err: &str) -> Option<&str> {
    if let Some(start) = err.find("' (") {
        let rest = &err[start + 3..];
        if let Some(end) = rest.find(')') {
            return Some(&rest[..end]);
        }
    }
    None
}

/// Differential testing errors.
#[derive(Debug, Clone)]
pub enum DifferentialError {
    /// Model loading error.
    ModelLoad(String),
    /// GPU initialization error.
    GpuInit(String),
    /// Fixture loading error.
    FixtureLoad(String),
    /// Execution error.
    Execution(String),
}

impl std::fmt::Display for DifferentialError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DifferentialError::ModelLoad(msg) => write!(f, "Model load error: {}", msg),
            DifferentialError::GpuInit(msg) => write!(f, "GPU init error: {}", msg),
            DifferentialError::FixtureLoad(msg) => write!(f, "Fixture load error: {}", msg),
            DifferentialError::Execution(msg) => write!(f, "Execution error: {}", msg),
        }
    }
}

impl std::error::Error for DifferentialError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tolerance_defaults() {
        let default = DifferentialTolerance::default();
        assert!(default.abs.is_none());
        assert!(default.rel.is_none());
        assert!(!default.is_active());
    }

    #[test]
    fn test_tolerance_strict() {
        let strict = DifferentialTolerance::strict();
        assert!(strict.abs.is_some());
        assert!(strict.rel.is_none());
        assert!(strict.is_active());
    }

    #[test]
    fn test_extract_node_name() {
        let err = "CUDA kernel failed: Node '/decoder/Add_2_output_0' (Add): shapes mismatch";
        assert_eq!(
            extract_node_name_from_error(err),
            Some("/decoder/Add_2_output_0")
        );
    }

    #[test]
    fn test_extract_op_type() {
        let err = "CUDA kernel failed: Node '/decoder/Add_2_output_0' (Add): shapes mismatch";
        assert_eq!(extract_op_type_from_error(err), Some("Add"));
    }
}

#[cfg(test)]
mod nan_inf_tests {
    use super::*;

    #[test]
    fn nan_in_iconnx_output_forces_infinite_diff() {
        let actual = vec![1.0_f32, f32::NAN, 3.0];
        let expected = vec![1.0_f32, 2.0, 3.0];
        let tol = DifferentialTolerance::loose();
        let result = compare_with_tolerance(&actual, &expected, tol);
        assert!(matches!(result, Some(ExceededTolerance::NonFinite)));
    }

    #[test]
    fn inf_in_iconnx_output_forces_infinite_diff() {
        let actual = vec![1.0_f32, f32::INFINITY, 3.0];
        let expected = vec![1.0_f32, 2.0, 3.0];
        let tol = DifferentialTolerance::loose();
        let result = compare_with_tolerance(&actual, &expected, tol);
        assert!(matches!(result, Some(ExceededTolerance::NonFinite)));
    }

    #[test]
    fn nan_in_expected_reference_does_not_silently_pass() {
        // The expected may legitimately contain NaN (e.g., if the reference
        // model itself diverged); we still want to surface the difference
        // rather than treat NaN-vs-finite as "matching."
        let actual = vec![1.0_f32, 2.0, 3.0];
        let expected = vec![1.0_f32, f32::NAN, 3.0];
        let tol = DifferentialTolerance::strict();
        let result = compare_with_tolerance(&actual, &expected, tol);
        assert!(result.is_some());
    }

    #[test]
    fn matching_outputs_within_tolerance_pass() {
        let actual = vec![1.001_f32, 2.0, 3.0];
        let expected = vec![1.0_f32, 2.0, 3.0];
        let tol = DifferentialTolerance::strict();
        let result = compare_with_tolerance(&actual, &expected, tol);
        assert!(result.is_some()); // strict is 1e-4; 0.001 exceeds it.

        let tol_loose = DifferentialTolerance::loose();
        let result_loose = compare_with_tolerance(&actual, &expected, tol_loose);
        assert!(result_loose.is_none()); // loose accepts 0.001.
    }
}

#[cfg(test)]
mod reference_source_tests {
    use super::*;

    #[test]
    fn output_names_empty_resolves_to_all_graph_outputs() {
        let runner_path =
            std::path::PathBuf::from("tests/fixtures/ws5_synthetic/minimal_add.onnx");
        if !runner_path.exists() {
            eprintln!("skipping: synthetic fixture not present");
            return;
        }
        let runner = match DifferentialRunner::new(&runner_path) {
            Ok(r) => r,
            Err(_) => return,
        };
        // No with_outputs call: builder default is empty.
        assert!(runner.output_names_for_test().is_empty());
    }

    #[test]
    fn with_op_chain_context_depth_respects_cap() {
        let runner_path =
            std::path::PathBuf::from("tests/fixtures/ws5_synthetic/minimal_add.onnx");
        if !runner_path.exists() {
            return;
        }
        let runner = match DifferentialRunner::new(&runner_path) {
            Ok(r) => r,
            Err(_) => return,
        };
        let runner = runner.with_op_chain_context_depth(10);
        assert_eq!(runner.op_chain_context_depth_for_test(), 10);
    }

    #[test]
    fn in_memory_mode_sets_reference_source() {
        let runner_path =
            std::path::PathBuf::from("tests/fixtures/ws5_synthetic/minimal_add.onnx");
        if !runner_path.exists() {
            return;
        }
        let runner = match DifferentialRunner::new(&runner_path) {
            Ok(r) => r,
            Err(_) => return,
        };
        let runner = runner.with_reference(ReferenceSource::InMemory {
            tensors: HashMap::new(),
        });
        assert_eq!(runner.reference_variant_for_test(), "InMemory");
    }

    #[test]
    fn fixture_dir_mode_sets_reference_source() {
        let runner_path =
            std::path::PathBuf::from("tests/fixtures/ws5_synthetic/minimal_add.onnx");
        if !runner_path.exists() {
            return;
        }
        let runner = match DifferentialRunner::new(&runner_path) {
            Ok(r) => r,
            Err(_) => return,
        };
        let runner = runner.with_reference(ReferenceSource::FixtureDir {
            path: std::path::PathBuf::from("/tmp/fake/dir"),
        });
        assert_eq!(runner.reference_variant_for_test(), "FixtureDir");
    }
}
