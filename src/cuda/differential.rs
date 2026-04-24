//! Differential Testing Framework for GPU Inference
//!
//! Compares GPU execution against ORT reference values node-by-node,
//! stopping at the first divergence with detailed diagnostics.
//!
//! Usage:
//! ```ignore
//! let runner = DifferentialRunner::new(model_path, fixture_path)?;
//! let result = runner.run_until_divergence(inputs)?;
//! if let Some(divergence) = result {
//!     println!("First divergence at: {}", divergence.node_name);
//! }
//! ```
//!
//! This module is compile-gated behind the `debug-inference` feature.

use super::checkpoints::CheckpointConfig;
use super::inference::GpuGraphExecutor;
use crate::onnx_parser::OnnxParser;
use crate::tensor::Tensor;
use std::collections::HashMap;
use std::path::Path;

/// Tolerance settings for differential comparison
#[derive(Clone, Debug)]
pub struct DiffTolerance {
    /// Correlation threshold below which we consider divergence
    pub correlation_threshold: f64,
    /// Max absolute difference threshold
    pub max_abs_diff: f32,
    /// Whether to allow shape broadcasting mismatches
    pub allow_shape_mismatch: bool,
}

impl Default for DiffTolerance {
    fn default() -> Self {
        Self {
            correlation_threshold: 0.9999,
            max_abs_diff: 1e-4,
            allow_shape_mismatch: false,
        }
    }
}

impl DiffTolerance {
    /// Strict tolerance (high correlation, low diff)
    pub fn strict() -> Self {
        Self {
            correlation_threshold: 0.99999,
            max_abs_diff: 1e-5,
            allow_shape_mismatch: false,
        }
    }

    /// Loose tolerance (for operations with known precision differences)
    pub fn loose() -> Self {
        Self {
            correlation_threshold: 0.99,
            max_abs_diff: 1e-2,
            allow_shape_mismatch: false,
        }
    }
}

/// Report of a divergence found during differential testing
#[derive(Debug, Clone)]
pub struct DivergenceReport {
    /// Name of the first divergent node
    pub node_name: String,
    /// Operator type of the divergent node
    pub op_type: String,
    /// Expected shape (from ORT)
    pub expected_shape: Vec<usize>,
    /// Actual shape (from GPU)
    pub actual_shape: Vec<usize>,
    /// Correlation with ORT reference
    pub correlation: f64,
    /// Maximum absolute difference
    pub max_diff: f32,
    /// Whether this is a shape mismatch
    pub is_shape_mismatch: bool,
    /// Sample values at indices with largest differences
    pub sample_diffs: Vec<DiffSample>,
    /// Number of nodes executed before divergence
    pub nodes_before_divergence: usize,
}

/// Sample of a value difference at a specific index
#[derive(Debug, Clone)]
pub struct DiffSample {
    /// Flat index in the tensor
    pub index: usize,
    /// Expected value (ORT)
    pub expected: f32,
    /// Actual value (GPU)
    pub actual: f32,
    /// Absolute difference
    pub diff: f32,
}

impl DivergenceReport {
    /// Format as a human-readable string
    pub fn to_string_detailed(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("\n=== Divergence Report ===\n"));
        s.push_str(&format!("Node: {} ({})\n", self.node_name, self.op_type));
        s.push_str(&format!(
            "Nodes executed before divergence: {}\n",
            self.nodes_before_divergence
        ));

        if self.is_shape_mismatch {
            s.push_str(&format!(
                "Shape mismatch: expected {:?}, got {:?}\n",
                self.expected_shape, self.actual_shape
            ));
        }

        s.push_str(&format!(
            "Correlation: {:.6} (threshold: <)\n",
            self.correlation
        ));
        s.push_str(&format!("Max diff: {:.6e}\n", self.max_diff));

        if !self.sample_diffs.is_empty() {
            s.push_str("\nLargest differences:\n");
            for sample in &self.sample_diffs {
                s.push_str(&format!(
                    "  [{}]: expected={:.6e}, actual={:.6e}, diff={:.6e}\n",
                    sample.index, sample.expected, sample.actual, sample.diff
                ));
            }
        }

        s
    }
}

/// Differential testing runner
pub struct DifferentialRunner {
    executor: GpuGraphExecutor,
    fixture_path: std::path::PathBuf,
    tolerance: DiffTolerance,
}

impl DifferentialRunner {
    /// Create a new differential runner
    ///
    /// # Arguments
    /// * `model_path` - Path to the ONNX model file
    /// * `fixture_path` - Path to the ORT fixture directory
    /// * `tolerance` - Tolerance settings for comparison
    pub fn new(
        model_path: &Path,
        fixture_path: &Path,
        tolerance: DiffTolerance,
    ) -> Result<Self, DifferentialError> {
        // Parse model
        let model = OnnxParser::parse_file(model_path).map_err(|e| {
            DifferentialError::ModelLoad(format!("Failed to parse ONNX: {:?}", e))
        })?;

        let weights = model.extract_weights().expect("extract weights");
        let graph = model.computation_graph().map_err(|e| {
            DifferentialError::ModelLoad(format!("Failed to build computation graph: {}", e))
        })?;
        let nodes = graph.nodes();

        // Create GPU executor
        let mut executor = GpuGraphExecutor::new()
            .map_err(|e| DifferentialError::GpuInit(format!("GPU init failed: {}", e)))?;

        // Load weights
        for (name, tensor) in &weights {
            executor.add_initializer(name.clone(), tensor).map_err(|e| {
                DifferentialError::GpuInit(format!("Weight upload failed: {}", e))
            })?;
        }

        // Add nodes
        for (idx, node) in nodes.iter().enumerate() {
            let fallback_name = format!("node_{}", idx);
            let node_name = node
                .outputs
                .first()
                .map(|s| s.as_str())
                .unwrap_or(&fallback_name);
            let inputs: Vec<&str> = node.inputs.iter().map(|s| s.as_str()).collect();
            let outputs: Vec<&str> = node.outputs.iter().map(|s| s.as_str()).collect();
            executor.add_node(
                node_name,
                &node.op_type,
                inputs,
                outputs,
                node.attributes.clone(),
            );
        }

        Ok(Self {
            executor,
            fixture_path: fixture_path.to_path_buf(),
            tolerance,
        })
    }

    /// Run inference until first divergence is found
    ///
    /// Returns None if no divergence is found (GPU matches ORT).
    pub fn run_until_divergence(
        &self,
        inputs: HashMap<String, Tensor>,
        output_name: &str,
    ) -> Result<Option<DivergenceReport>, DifferentialError> {
        // Configure checkpoint system
        let mut config = CheckpointConfig::default();
        config.base_dir = self.fixture_path.clone();
        config.compare_to_ort = true;
        config.correlation_threshold = self.tolerance.correlation_threshold;
        config.max_diff_threshold = self.tolerance.max_abs_diff;

        // Run with checkpoints
        match self
            .executor
            .run_with_checkpoints(inputs, vec![output_name], config)
        {
            Ok((_, mgr)) => {
                // Check for divergence in captured checkpoints
                if let Some(first_div) = mgr.first_divergence() {
                    // Find the entry for this divergence
                    if let Some(entry) = mgr.entries().iter().find(|e| e.name == first_div) {
                        let nodes_before = mgr
                            .entries()
                            .iter()
                            .position(|e| e.name == first_div)
                            .unwrap_or(0);

                        return Ok(Some(DivergenceReport {
                            node_name: entry.name.clone(),
                            op_type: entry.op_type.clone(),
                            expected_shape: vec![], // Would need ORT shapes
                            actual_shape: entry.shape.clone(),
                            correlation: entry.correlation_with_ort.unwrap_or(0.0),
                            max_diff: entry.max_diff_from_ort.unwrap_or(f32::MAX),
                            is_shape_mismatch: false,
                            sample_diffs: vec![], // Would need to save sample values
                            nodes_before_divergence: nodes_before,
                        }));
                    }
                }
                Ok(None) // No divergence found
            }
            Err(e) => {
                // Check if this is a shape mismatch error
                let err_msg = e.to_string();
                if err_msg.contains("not broadcastable") || err_msg.contains("shapes") {
                    // Extract node name from error
                    if let Some(node_name) = extract_node_name_from_error(&err_msg) {
                        return Ok(Some(DivergenceReport {
                            node_name: node_name.to_string(),
                            op_type: extract_op_type_from_error(&err_msg)
                                .unwrap_or("Unknown")
                                .to_string(),
                            expected_shape: vec![],
                            actual_shape: vec![],
                            correlation: 0.0,
                            max_diff: f32::MAX,
                            is_shape_mismatch: true,
                            sample_diffs: vec![],
                            nodes_before_divergence: 0,
                        }));
                    }
                }
                Err(DifferentialError::Execution(e.to_string()))
            }
        }
    }

    /// Get the underlying executor
    pub fn executor(&self) -> &GpuGraphExecutor {
        &self.executor
    }
}

/// Extract node name from error message
fn extract_node_name_from_error(err: &str) -> Option<&str> {
    // Pattern: "Node 'xxx' (yyy):"
    if let Some(start) = err.find("Node '") {
        let rest = &err[start + 6..];
        if let Some(end) = rest.find('\'') {
            return Some(&rest[..end]);
        }
    }
    None
}

/// Extract op type from error message
fn extract_op_type_from_error(err: &str) -> Option<&str> {
    // Pattern: "Node 'xxx' (yyy):"
    if let Some(start) = err.find("' (") {
        let rest = &err[start + 3..];
        if let Some(end) = rest.find(')') {
            return Some(&rest[..end]);
        }
    }
    None
}

/// Differential testing errors
#[derive(Debug, Clone)]
pub enum DifferentialError {
    /// Model loading error
    ModelLoad(String),
    /// GPU initialization error
    GpuInit(String),
    /// Fixture loading error
    FixtureLoad(String),
    /// Execution error
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
        let default = DiffTolerance::default();
        assert!((default.correlation_threshold - 0.9999).abs() < 1e-6);
        assert!((default.max_abs_diff - 1e-4).abs() < 1e-8);
    }

    #[test]
    fn test_tolerance_strict() {
        let strict = DiffTolerance::strict();
        assert!(strict.correlation_threshold > 0.9999);
        assert!(strict.max_abs_diff < 1e-4);
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
