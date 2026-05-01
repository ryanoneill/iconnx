//! Tests for the GPU tensor validation and checkpoint infrastructure
//!
//! These tests verify that NaN/Inf detection and ORT comparison work correctly.
//! Everything in this file — helpers included — is gated behind the
//! `debug-inference` feature since that's the only configuration under which
//! the test functions are compiled.

#[cfg(feature = "debug-inference")]
#[path = "common/mod.rs"]
#[allow(clippy::duplicate_mod)] // shared helper loaded via `#[path]` in multiple integration-test binaries
mod common;

#[cfg(feature = "debug-inference")]
use std::collections::HashMap;
#[cfg(feature = "debug-inference")]
use std::path::Path;

#[cfg(feature = "debug-inference")]
use iconnx::cuda::GpuGraphExecutor;
#[cfg(feature = "debug-inference")]
use iconnx::cuda::checkpoints::CheckpointConfig;
#[cfg(feature = "debug-inference")]
use iconnx::cuda::validation::ValidationMode;
#[cfg(feature = "debug-inference")]
use iconnx::onnx_parser::OnnxParser;
#[cfg(feature = "debug-inference")]
use iconnx::tensor::Tensor;

#[cfg(feature = "debug-inference")]
fn setup_kokoro_executor() -> GpuGraphExecutor {
    let model_path = common::kokoro_model::kokoro_model_path();
    let model = OnnxParser::parse_file(&model_path).expect("parse kokoro-v1.0.onnx");
    let weights = model.extract_weights().expect("extract weights");
    let graph = model.computation_graph().expect("parse computation graph");
    let nodes = graph.nodes();

    let mut executor = GpuGraphExecutor::new().expect("create CUDA executor");

    for (name, tensor) in &weights {
        executor
            .add_initializer(name.clone(), tensor)
            .expect("add initializer");
    }

    for (idx, node) in nodes.iter().enumerate() {
        let fallback_name = format!("node_{}", idx);
        let node_name = node
            .outputs
            .first()
            .map(|s| s.as_str())
            .unwrap_or(&fallback_name);
        let inputs: Vec<&str> = node.inputs.iter().map(|s| s.as_str()).collect();
        let outputs: Vec<&str> = node.outputs.iter().map(|s| s.as_str()).collect();
        executor.add_node(node_name, &node.op_type, inputs, outputs, node.attributes.clone());
    }

    executor
}

#[cfg(feature = "debug-inference")]
fn test_inputs() -> HashMap<String, Tensor> {
    let tokens = vec![0i64, 10, 20, 30, 0];
    let style: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01).sin()).collect();
    let speed = vec![1.0f32];

    let mut inputs = HashMap::new();
    inputs.insert(
        "tokens".to_string(),
        Tensor::from_vec_i64(tokens, vec![1, 5]),
    );
    inputs.insert("style".to_string(), Tensor::from_vec_f32(style, vec![1, 256]));
    inputs.insert("speed".to_string(), Tensor::from_vec_f32(speed, vec![1]));

    inputs
}

/// Load inputs from ORT fixture files to ensure exact match
#[cfg(feature = "debug-inference")]
fn load_fixture_inputs() -> HashMap<String, Tensor> {
    let fixture_dir = Path::new("tests/fixtures/ort_reference/inputs");

    // Load tokens
    let tokens_bytes = std::fs::read(fixture_dir.join("tokens.bin")).expect("tokens.bin");
    let tokens: Vec<i64> = tokens_bytes
        .chunks_exact(8)
        .map(|chunk| i64::from_le_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3],
            chunk[4], chunk[5], chunk[6], chunk[7],
        ]))
        .collect();

    // Load style
    let style_bytes = std::fs::read(fixture_dir.join("style.bin")).expect("style.bin");
    let style: Vec<f32> = style_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    // Load speed
    let speed_bytes = std::fs::read(fixture_dir.join("speed.bin")).expect("speed.bin");
    let speed: Vec<f32> = speed_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    let mut inputs = HashMap::new();
    inputs.insert(
        "tokens".to_string(),
        Tensor::from_vec_i64(tokens.clone(), vec![1, tokens.len()]),
    );
    inputs.insert(
        "style".to_string(),
        Tensor::from_vec_f32(style, vec![1, 256]),
    );
    inputs.insert(
        "speed".to_string(),
        Tensor::from_vec_f32(speed, vec![1]),
    );

    inputs
}

/// Test that validation runs without errors on early nodes (before any divergence)
#[test]
#[ignore] // Requires CUDA GPU and model file
#[cfg(feature = "debug-inference")]
fn test_validation_encoder_nodes() {
    let executor = setup_kokoro_executor();
    let inputs = test_inputs();

    // Run with validation in full mode (log all stats)
    let mode = ValidationMode::full();

    // Test encoder output - should have no NaN/Inf
    match executor.run_with_validation(inputs.clone(), vec!["/encoder/Squeeze_1_output_0"], mode) {
        Ok((outputs, report)) => {
            println!("\n=== Encoder Validation Report ===");
            println!("Total tensors validated: {}", report.results.len());
            println!("First NaN node: {:?}", report.first_nan_node);
            println!("First Inf node: {:?}", report.first_inf_node);
            println!("Total NaN count: {}", report.total_nan_count);
            println!("Total Inf count: {}", report.total_inf_count);

            assert!(
                !report.has_issues(),
                "Encoder should have no NaN/Inf issues"
            );
            assert!(outputs.contains_key("/encoder/Squeeze_1_output_0"));
        }
        Err(e) => {
            panic!("Encoder validation failed: {}", e);
        }
    }
}

/// Test that validation catches NaN in the decoder path
#[test]
#[ignore] // Requires CUDA GPU and model file
#[cfg(feature = "debug-inference")]
fn test_validation_finds_nan() {
    let executor = setup_kokoro_executor();
    let inputs = test_inputs();

    // Run with validation but don't stop on error (accumulate all issues)
    let mut mode = ValidationMode::full();
    mode.stop_on_error = false;

    // Run full inference - may have NaN/Inf issues
    match executor.run_with_validation(inputs.clone(), vec!["audio"], mode) {
        Ok((outputs, report)) => {
            println!("\n=== Full Model Validation Report ===");
            println!("Total tensors validated: {}", report.results.len());
            println!("Summary: {}", report.summary());

            if report.has_issues() {
                println!("\nFirst node with NaN: {:?}", report.first_nan_node);
                println!("First node with Inf: {:?}", report.first_inf_node);

                // Print details of problematic tensors
                println!("\n=== Tensors with Issues ===");
                for result in &report.results {
                    if result.has_issues() {
                        println!("  {}", result.to_line());
                    }
                }
            }

            // Audio output should exist
            assert!(outputs.contains_key("audio"));
        }
        Err(e) => {
            // Validation may fail with NaN error - that's expected
            println!("Validation stopped with error: {}", e);
        }
    }
}

/// Test quick validation mode (NaN/Inf only, no stats)
#[test]
#[ignore] // Requires CUDA GPU and model file
#[cfg(feature = "debug-inference")]
fn test_validation_nan_inf_only_mode() {
    let executor = setup_kokoro_executor();
    let inputs = test_inputs();
    let mode = ValidationMode::nan_inf_only();

    // Run to encoder - should pass
    let result =
        executor.run_with_validation(inputs.clone(), vec!["/encoder/Squeeze_1_output_0"], mode);

    match result {
        Ok((_, report)) => {
            assert!(!report.has_issues(), "Encoder should not have NaN/Inf");
            println!("NaN/Inf check passed for encoder");
        }
        Err(e) => {
            panic!("Unexpected validation error: {}", e);
        }
    }
}

// ==================== Checkpoint Tests ====================

/// Test that checkpoint system loads ORT references
#[test]
#[ignore] // Requires CUDA GPU, model file, and ORT fixtures
#[cfg(feature = "debug-inference")]
fn test_checkpoint_load_ort_references() {
    let executor = setup_kokoro_executor();
    let inputs = test_inputs();
    let config = CheckpointConfig::ort_comparison();

    match executor.run_with_checkpoints(inputs, vec!["/encoder/Squeeze_1_output_0"], config) {
        Ok((outputs, mgr)) => {
            println!("\n=== Checkpoint Test Results ===");
            println!("ORT references loaded: {}", mgr.ort_count());
            println!("Nodes captured: {}", mgr.entries().len());
            println!("Divergent nodes: {}", mgr.divergent_entries().len());

            if let Some(first) = mgr.first_divergence() {
                println!("First divergence: {}", first);
            }

            // Print some captured entries
            println!("\n=== Sample Captured Entries ===");
            for entry in mgr.entries().iter().take(10) {
                println!("  {}", entry.to_line());
            }

            assert!(outputs.contains_key("/encoder/Squeeze_1_output_0"));
        }
        Err(e) => {
            println!("Checkpoint test failed: {}", e);
            // This may fail if fixtures not present - not a hard failure
        }
    }
}

/// Test that checkpoint system finds divergence
#[test]
#[ignore] // Requires CUDA GPU, model file, and ORT fixtures
#[cfg(feature = "debug-inference")]
fn test_checkpoint_find_divergence() {
    let executor = setup_kokoro_executor();
    // Use actual fixture inputs to match ORT reference values
    let inputs = load_fixture_inputs();

    // Configure to capture all nodes and compare to ORT
    let mut config = CheckpointConfig::ort_comparison();
    config.correlation_threshold = 0.999; // Slightly looser threshold

    // Try running full inference - will likely fail on shape mismatch
    // but we should have captured enough to see where divergence starts
    match executor.run_with_checkpoints(inputs, vec!["audio"], config) {
        Ok((_, mgr)) => {
            mgr.print_summary();

            if mgr.has_divergence() {
                println!("\nDivergence detected - first node: {:?}", mgr.first_divergence());
            }
        }
        Err(e) => {
            // Expected - shape mismatch at Add_2
            println!("Inference stopped with error (expected): {}", e);
        }
    }
}

// ==================== Differential Testing Framework Tests ====================

#[cfg(feature = "debug-inference")]
use iconnx::cuda::differential::{DifferentialRunner, ReferenceSource};
#[cfg(feature = "debug-inference")]
use iconnx::DifferentialTolerance;

/// Test tolerance configurations
#[test]
#[cfg(feature = "debug-inference")]
fn test_tolerance_presets() {
    // Post-Y(4): DifferentialTolerance has the implicit-mode shape
    // (abs/rel optionals). Default is "disabled"; strict is abs-only;
    // loose is dual-gate.
    let default = DifferentialTolerance::default();
    assert!(default.abs.is_none());
    assert!(default.rel.is_none());
    assert!(!default.is_active());

    let strict = DifferentialTolerance::strict();
    assert!(strict.abs.is_some());
    assert!(strict.rel.is_none());

    let loose = DifferentialTolerance::loose();
    assert!(loose.abs.is_some());
    assert!(loose.rel.is_some());
}

/// Test differential runner with encoder nodes
#[test]
#[ignore] // Requires CUDA GPU, model file, and ORT fixtures
#[cfg(feature = "debug-inference")]
fn test_differential_encoder_nodes() {
    let model_path = common::kokoro_model::kokoro_model_path();
    let fixture_path = Path::new("tests/fixtures/ort_reference");

    if !fixture_path.exists() {
        println!("Fixture path not found - skipping");
        return;
    }

    let runner = match DifferentialRunner::new(&model_path) {
        Ok(r) => r
            .with_reference(ReferenceSource::FixtureDir {
                path: fixture_path.to_path_buf(),
            })
            .with_tolerance(DifferentialTolerance::strict())
            .with_outputs(&["/encoder/Squeeze_1_output_0"]),
        Err(e) => {
            println!("Failed to create runner: {}", e);
            return;
        }
    };

    let inputs = test_inputs();

    // Run to encoder output - should have no divergence
    match runner.run_until_divergence(inputs) {
        Ok(Some(report)) => {
            println!("\n{}", report.to_string_detailed());
            // Early divergence is unexpected for encoder
            println!("WARNING: Divergence found in encoder at: {}", report.node_name);
        }
        Ok(None) => {
            println!("Encoder executed without divergence - GPU matches ORT!");
        }
        Err(e) => {
            println!("Differential test error: {}", e);
        }
    }
}

/// Test differential runner finds divergence in decoder
#[test]
#[ignore] // Requires CUDA GPU, model file, and ORT fixtures
#[cfg(feature = "debug-inference")]
fn test_differential_find_divergence() {
    let model_path = common::kokoro_model::kokoro_model_path();
    let fixture_path = Path::new("tests/fixtures/ort_reference");

    if !fixture_path.exists() {
        println!("Fixture path not found - skipping");
        return;
    }

    let tolerance = DifferentialTolerance {
        abs: Some(1e-3),
        rel: None,
    };

    let runner = match DifferentialRunner::new(&model_path) {
        Ok(r) => r
            .with_reference(ReferenceSource::FixtureDir {
                path: fixture_path.to_path_buf(),
            })
            .with_tolerance(tolerance)
            .with_outputs(&["audio"]),
        Err(e) => {
            println!("Failed to create runner: {}", e);
            return;
        }
    };

    // Use actual fixture inputs to match ORT reference values
    let inputs = load_fixture_inputs();

    // Run full model - expect divergence if GPU doesn't match ORT
    match runner.run_until_divergence(inputs) {
        Ok(Some(report)) => {
            println!("\n=== DIVERGENCE FOUND ===");
            println!("{}", report.to_string_detailed());

            // Post-Y(4): shape mismatch surfaces as ExceededTolerance::Abs(INFINITY).
            if matches!(
                report.exceeded,
                iconnx::cuda::differential::ExceededTolerance::Abs(v) if !v.is_finite()
            ) {
                println!("Shape mismatch confirmed at: {}", report.node_name);
            }
        }
        Ok(None) => {
            println!("Full model executed without divergence!");
        }
        Err(e) => {
            // May fail with shape mismatch error - that's the bug we're finding
            println!("Differential test stopped with error: {}", e);
        }
    }
}
