//! Minimal GPU inference example for iconnx.
//!
//! Usage:
//!   cargo run --example gpu_infer --features cuda -- path/to/model.onnx
//!
//! Note: actually *running* inference requires a CUDA GPU and a valid ONNX
//! model file.  This example compiles and performs all setup steps; the
//! `executor.run(...)` call is guarded so the binary can be exercised
//! (argument-parse + model-load) without a GPU present.

use std::collections::HashMap;

use anyhow::Context;
use iconnx::{GpuGraphExecutor, OnnxParser, Tensor};

fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .context("Usage: gpu_infer <path/to/model.onnx>")?;

    // Parse the ONNX model from disk.
    let model = OnnxParser::parse_file(&path)
        .with_context(|| format!("failed to parse ONNX model at '{path}'"))?;

    // Build the GPU executor from the parsed model.
    // GpuGraphExecutor is !Sync — use one executor per thread.
    let executor = GpuGraphExecutor::from_model(&model)
        .context("failed to build GpuGraphExecutor")?;

    // Construct demo inputs: replace every dynamic (None) dimension with 1.
    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    for (name, shape) in model.input_shapes() {
        let dims: Vec<usize> = shape.iter().map(|d| d.unwrap_or(1)).collect();
        let numel: usize = dims.iter().product();
        let tensor = Tensor::from_vec_f32(vec![0.0_f32; numel], dims);
        inputs.insert(name, tensor);
    }

    if inputs.is_empty() {
        println!("Model has no declared inputs — nothing to run.");
        return Ok(());
    }

    // Collect output names from the executor's plan.
    // We use the input keys as a proxy for now; real usage derives output
    // names from model.graph().output or from prior knowledge of the model.
    // For a concrete call the caller must supply the actual output node names.
    let output_names: Vec<&str> = vec![];

    // Guard: skip the actual GPU dispatch when no output names are supplied, so
    // this binary can be compiled and exercised (parse + build) without a GPU.
    // Populate `output_names` above with the model's real output node names to
    // enable live inference.
    if !output_names.is_empty() {
        let outputs = executor
            .run(inputs, output_names)
            .context("inference failed")?;

        for (name, tensor) in &outputs {
            println!("output '{}': shape {:?}", name, tensor.shape());
        }
    } else {
        println!(
            "Executor built successfully with {} input(s). \
             Provide output_names to run inference.",
            inputs.len()
        );
    }

    Ok(())
}
