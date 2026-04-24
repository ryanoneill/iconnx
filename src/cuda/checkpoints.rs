//! GPU Tensor Checkpoint System
//!
//! Provides infrastructure for saving GPU tensor snapshots and comparing
//! them against ORT reference values. Features:
//!
//! - Save GPU tensors to disk at any node
//! - Load ORT reference checkpoints from the fixture system
//! - Compare GPU vs ORT with correlation and max_diff metrics
//! - Find first divergent node automatically
//!
//! This module is compile-gated behind the `debug-inference` feature.

use super::context::IconnxCudaContext;
use super::tensor::GpuTensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Configuration for checkpoint operations
#[derive(Clone, Debug)]
pub struct CheckpointConfig {
    /// Base directory for checkpoint files
    pub base_dir: PathBuf,
    /// Specific nodes to capture (empty = capture all)
    pub capture_nodes: Vec<String>,
    /// Only capture every Nth node (0 = all nodes)
    pub capture_every_nth: usize,
    /// Whether to save GPU tensors to disk
    pub save_gpu: bool,
    /// Whether to compare against ORT references
    pub compare_to_ort: bool,
    /// Correlation threshold below which we consider divergence
    pub correlation_threshold: f64,
    /// Max absolute diff threshold
    pub max_diff_threshold: f32,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            base_dir: PathBuf::from("tests/fixtures/ort_reference"),
            capture_nodes: Vec::new(),
            capture_every_nth: 0,
            save_gpu: false,
            compare_to_ort: true,
            correlation_threshold: 0.9999,
            max_diff_threshold: 1e-4,
        }
    }
}

impl CheckpointConfig {
    /// Create config for comparing against ORT fixtures
    pub fn ort_comparison() -> Self {
        Self::default()
    }

    /// Create config for saving GPU snapshots
    pub fn save_snapshots(output_dir: PathBuf) -> Self {
        Self {
            base_dir: output_dir,
            save_gpu: true,
            compare_to_ort: false,
            ..Default::default()
        }
    }
}

/// Statistics for a single tensor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorStats {
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub has_nan: bool,
    pub has_inf: bool,
}

/// Single checkpoint entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointEntry {
    /// Tensor name (node output)
    pub name: String,
    /// Operator type
    pub op_type: String,
    /// Tensor shape
    pub shape: Vec<usize>,
    /// Data type name
    pub dtype: String,
    /// Tensor statistics
    pub stats: TensorStats,
    /// Correlation with ORT reference (if compared)
    pub correlation_with_ort: Option<f64>,
    /// Max absolute difference from ORT (if compared)
    pub max_diff_from_ort: Option<f32>,
    /// Whether this is a divergent node
    pub is_divergent: bool,
}

impl CheckpointEntry {
    /// Format as a single line for logging
    pub fn to_line(&self) -> String {
        let status = if self.is_divergent {
            "DIVERGE"
        } else if self.correlation_with_ort.is_some() {
            "PASS"
        } else {
            "---"
        };

        let ort_info = match (self.correlation_with_ort, self.max_diff_from_ort) {
            (Some(corr), Some(diff)) => format!(" corr={:.6} diff={:.2e}", corr, diff),
            _ => String::new(),
        };

        format!(
            "{} {} ({}) {:?}{}",
            status, self.name, self.op_type, self.shape, ort_info
        )
    }
}

/// ORT reference metadata (from Python fixture generator)
#[derive(Debug, Deserialize)]
struct OrtMetadata {
    #[allow(dead_code)]
    seed: u64,
    #[allow(dead_code)]
    inputs: serde_json::Value,
    checkpoints: Vec<OrtCheckpointInfo>,
}

#[derive(Debug, Deserialize)]
struct OrtCheckpointInfo {
    name: String,
    path: String,
    shape: Vec<usize>,
    #[allow(dead_code)]
    dtype: String,
    #[allow(dead_code)]
    checksum: String,
    #[allow(dead_code)]
    min: f32,
    #[allow(dead_code)]
    max: f32,
    #[allow(dead_code)]
    has_nan: bool,
    #[allow(dead_code)]
    has_inf: bool,
}

/// Checkpoint manager for saving and comparing tensors
pub struct CheckpointManager {
    /// Configuration for checkpoint operations
    pub config: CheckpointConfig,
    /// Captured checkpoint entries
    entries: Vec<CheckpointEntry>,
    /// ORT reference tensors (name -> (data, shape))
    ort_tensors: HashMap<String, (Vec<f32>, Vec<usize>)>,
    /// First divergent node found
    first_divergence: Option<String>,
    /// Node counter for capture_every_nth
    node_count: usize,
}

impl CheckpointManager {
    /// Create a new checkpoint manager
    pub fn new(config: CheckpointConfig) -> Self {
        Self {
            config,
            entries: Vec::new(),
            ort_tensors: HashMap::new(),
            first_divergence: None,
            node_count: 0,
        }
    }

    /// Load ORT reference tensors from fixture directory
    pub fn load_ort_references(&mut self) -> Result<usize, CheckpointError> {
        let metadata_path = self.config.base_dir.join("metadata.json");
        if !metadata_path.exists() {
            return Err(CheckpointError::MetadataNotFound(
                metadata_path.display().to_string(),
            ));
        }

        let content = fs::read_to_string(&metadata_path)
            .map_err(|e| CheckpointError::IoError(e.to_string()))?;

        let metadata: OrtMetadata = serde_json::from_str(&content)
            .map_err(|e| CheckpointError::ParseError(e.to_string()))?;

        let checkpoints_dir = self.config.base_dir.join("checkpoints");
        let mut loaded = 0;

        for checkpoint in &metadata.checkpoints {
            let tensor_path = checkpoints_dir.join(&checkpoint.path);
            if tensor_path.exists() {
                let data = load_binary_f32(&tensor_path)?;
                self.ort_tensors
                    .insert(checkpoint.name.clone(), (data, checkpoint.shape.clone()));
                loaded += 1;
            }
        }

        Ok(loaded)
    }

    /// Check if a node should be captured
    fn should_capture(&self, name: &str) -> bool {
        // If specific nodes are configured, only capture those
        if !self.config.capture_nodes.is_empty() {
            return self.config.capture_nodes.iter().any(|n| n == name);
        }

        // Otherwise, use capture_every_nth
        if self.config.capture_every_nth > 0 {
            return self.node_count.is_multiple_of(self.config.capture_every_nth);
        }

        true
    }

    /// Capture a GPU tensor checkpoint
    pub fn capture(
        &mut self,
        name: &str,
        op_type: &str,
        tensor: &GpuTensor,
        ctx: &IconnxCudaContext,
    ) -> Result<Option<CheckpointEntry>, CheckpointError> {
        self.node_count += 1;

        if !self.should_capture(name) {
            return Ok(None);
        }

        // Only process Float32 tensors
        if !tensor.is_f32() {
            return Ok(None);
        }

        // Download tensor data
        let gpu_data = tensor
            .to_host_f32(ctx)
            .map_err(|e| CheckpointError::CudaError(e.to_string()))?;

        // Compute stats
        let stats = compute_stats(&gpu_data);

        // Compare with ORT if available
        let (correlation, max_diff, is_divergent) =
            if self.config.compare_to_ort && self.ort_tensors.contains_key(name) {
                let (ort_data, ort_shape) = self.ort_tensors.get(name).unwrap();

                // Check shape compatibility
                if tensor.shape() != ort_shape.as_slice() {
                    // Shape mismatch - still compare what we can
                    let min_len = gpu_data.len().min(ort_data.len());
                    let corr = correlation_f32(&gpu_data[..min_len], &ort_data[..min_len]);
                    let diff = max_abs_diff_f32(&gpu_data[..min_len], &ort_data[..min_len]);
                    let divergent = corr < self.config.correlation_threshold
                        || diff > self.config.max_diff_threshold;
                    (Some(corr), Some(diff), divergent)
                } else {
                    let corr = correlation_f32(&gpu_data, ort_data);
                    let diff = max_abs_diff_f32(&gpu_data, ort_data);
                    let divergent = corr < self.config.correlation_threshold
                        || diff > self.config.max_diff_threshold;
                    (Some(corr), Some(diff), divergent)
                }
            } else {
                (None, None, false)
            };

        // Track first divergence
        if is_divergent && self.first_divergence.is_none() {
            self.first_divergence = Some(name.to_string());
        }

        // Save to disk if configured
        if self.config.save_gpu {
            let gpu_dir = self.config.base_dir.join("gpu");
            fs::create_dir_all(&gpu_dir)
                .map_err(|e| CheckpointError::IoError(e.to_string()))?;

            let safe_name = name.replace(['/', '\\'], "_");
            let tensor_path = gpu_dir.join(format!("{}.bin", safe_name));
            save_binary_f32(&tensor_path, &gpu_data)?;
        }

        let entry = CheckpointEntry {
            name: name.to_string(),
            op_type: op_type.to_string(),
            shape: tensor.shape().to_vec(),
            dtype: "float32".to_string(),
            stats,
            correlation_with_ort: correlation,
            max_diff_from_ort: max_diff,
            is_divergent,
        };

        self.entries.push(entry.clone());
        Ok(Some(entry))
    }

    /// Get all checkpoint entries
    pub fn entries(&self) -> &[CheckpointEntry] {
        &self.entries
    }

    /// Get the first divergent node
    pub fn first_divergence(&self) -> Option<&str> {
        self.first_divergence.as_deref()
    }

    /// Get divergent entries
    pub fn divergent_entries(&self) -> Vec<&CheckpointEntry> {
        self.entries.iter().filter(|e| e.is_divergent).collect()
    }

    /// Check if any divergence was found
    pub fn has_divergence(&self) -> bool {
        self.first_divergence.is_some()
    }

    /// Get number of ORT references loaded
    pub fn ort_count(&self) -> usize {
        self.ort_tensors.len()
    }

    /// Reset for a new inference run
    pub fn reset(&mut self) {
        self.entries.clear();
        self.first_divergence = None;
        self.node_count = 0;
    }

    /// Print summary to stderr
    pub fn print_summary(&self) {
        eprintln!("\n=== Checkpoint Summary ===");
        eprintln!("  Nodes captured: {}", self.entries.len());
        eprintln!("  ORT references: {}", self.ort_tensors.len());
        eprintln!("  Divergent nodes: {}", self.divergent_entries().len());

        if let Some(first) = &self.first_divergence {
            eprintln!("  First divergence: {}", first);
        }

        if !self.divergent_entries().is_empty() {
            eprintln!("\n=== Divergent Nodes ===");
            for entry in self.divergent_entries() {
                eprintln!("  {}", entry.to_line());
            }
        }
    }
}

/// Checkpoint errors
#[derive(Debug, Clone)]
pub enum CheckpointError {
    /// Metadata file not found
    MetadataNotFound(String),
    /// IO error
    IoError(String),
    /// JSON parse error
    ParseError(String),
    /// CUDA error
    CudaError(String),
}

impl std::fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CheckpointError::MetadataNotFound(path) => {
                write!(f, "Metadata not found: {}", path)
            }
            CheckpointError::IoError(msg) => write!(f, "IO error: {}", msg),
            CheckpointError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            CheckpointError::CudaError(msg) => write!(f, "CUDA error: {}", msg),
        }
    }
}

impl std::error::Error for CheckpointError {}

// ==================== Helper functions ====================

fn load_binary_f32(path: &Path) -> Result<Vec<f32>, CheckpointError> {
    let bytes = fs::read(path).map_err(|e| CheckpointError::IoError(e.to_string()))?;
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn save_binary_f32(path: &Path, data: &[f32]) -> Result<(), CheckpointError> {
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    fs::write(path, bytes).map_err(|e| CheckpointError::IoError(e.to_string()))
}

fn compute_stats(data: &[f32]) -> TensorStats {
    let mut min = f32::MAX;
    let mut max = f32::MIN;
    let mut sum = 0.0f64;
    let mut has_nan = false;
    let mut has_inf = false;
    let mut valid_count = 0usize;

    for &val in data {
        if val.is_nan() {
            has_nan = true;
        } else if val.is_infinite() {
            has_inf = true;
        } else {
            if val < min {
                min = val;
            }
            if val > max {
                max = val;
            }
            sum += val as f64;
            valid_count += 1;
        }
    }

    let mean = if valid_count > 0 {
        (sum / valid_count as f64) as f32
    } else {
        0.0
    };

    TensorStats {
        min,
        max,
        mean,
        has_nan,
        has_inf,
    }
}

fn correlation_f32(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let n = a.len() as f64;
    let sum_a: f64 = a.iter().map(|&x| x as f64).sum();
    let sum_b: f64 = b.iter().map(|&x| x as f64).sum();
    let mean_a = sum_a / n;
    let mean_b = sum_b / n;

    let mut cov = 0.0f64;
    let mut var_a = 0.0f64;
    let mut var_b = 0.0f64;

    for i in 0..a.len() {
        let da = a[i] as f64 - mean_a;
        let db = b[i] as f64 - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    if var_a < 1e-10 || var_b < 1e-10 {
        if (var_a - var_b).abs() < 1e-10 {
            1.0
        } else {
            0.0
        }
    } else {
        cov / (var_a.sqrt() * var_b.sqrt())
    }
}

fn max_abs_diff_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_correlation_identical() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let corr = correlation_f32(&a, &b);
        assert!((corr - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_correlation_opposite() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let corr = correlation_f32(&a, &b);
        assert!((corr - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_max_abs_diff() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.1, 2.0, 3.5, 4.0, 5.0];
        let diff = max_abs_diff_f32(&a, &b);
        assert!((diff - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_compute_stats() {
        let data = vec![1.0, 2.0, f32::NAN, 4.0, 5.0, f32::INFINITY];
        let stats = compute_stats(&data);
        assert!((stats.min - 1.0).abs() < 1e-6);
        assert!((stats.max - 5.0).abs() < 1e-6);
        assert!((stats.mean - 3.0).abs() < 1e-6); // (1+2+4+5)/4
        assert!(stats.has_nan);
        assert!(stats.has_inf);
    }
}
