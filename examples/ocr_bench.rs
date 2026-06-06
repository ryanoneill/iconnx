//! WS-6 R5 — OCR inference performance characterisation.
//!
//! ## Purpose
//!
//! Measures wall-clock latency and GPU-memory behaviour for the two
//! PP-OCRv3 models used by iconnx's GPU-OCR pipeline:
//!
//! * **Recognition** (`en_number_mobile_rec.onnx`) — single-crop inference
//!   at shape `[1, 3, 32, 320]` (a representative crop width).
//! * **Detection**  (`en_PP-OCRv3_det.onnx`)       — full-page inference at
//!   `[1, 3, 960, 960]` and `[1, 3, 1500, 2000]`.
//!
//! ## Batch-size confirmation
//!
//! The recognition model contains an LSTM (`dispatch_lstm`) which is
//! hardcoded to `batch_size=1` in `src/cuda/executor/lstm.rs` (a cuDNN
//! `prepare_lstm_plan` constraint). Running recognition with `N=2` will
//! produce an error message that confirms the limitation.
//!
//! ## Model-presence gate
//!
//! All live measurements are guarded behind a file-existence check. If the
//! OCR model symlinks are absent the binary prints an actionable message
//! and exits 0, so `cargo build --example ocr_bench --features cuda`
//! remains the acceptance gate. The symlinks expected are:
//!
//! ```text
//! ./en_number_mobile_rec.onnx   (recognition model)
//! ./en_PP-OCRv3_det.onnx        (detection model)
//! ```
//!
//! They are the same symlinks used by the `ocr_parity_test` suite (WS-6
//! M6.5), placed in the repository root (or wherever you run the binary
//! from).
//!
//! ## Usage
//!
//! ```sh
//! cargo run --release --example ocr_bench --features cuda
//! ```
//!
//! With the OCR model symlinks in `$PWD` and a CUDA GPU present.

#![cfg(feature = "cuda")]

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use iconnx::{GpuGraphExecutor, OnnxParser, Tensor};

/// Path to the recognition model symlink (same convention as M6.5 tests).
const REC_MODEL: &str = "./en_number_mobile_rec.onnx";
/// Path to the detection model symlink (same convention as M6.5 tests).
const DET_MODEL: &str = "./en_PP-OCRv3_det.onnx";

/// Number of warm-up runs before timing.
const WARMUP_RUNS: usize = 3;
/// Number of timed runs for the average.
const TIMED_RUNS: usize = 10;

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    let rec_present = Path::new(REC_MODEL).exists();
    let det_present = Path::new(DET_MODEL).exists();

    if !rec_present || !det_present {
        println!("OCR models not present — run with the M6.5 symlinks for live figures.");
        println!();
        println!("Expected files (symlinks) in $PWD:");
        println!("  {REC_MODEL}  (recognition model)");
        println!("  {DET_MODEL}  (detection model)");
        println!();
        println!(
            "Place the rapidAI exports at those paths, then re-run:\n  \
             cargo run --release --example ocr_bench --features cuda"
        );
        std::process::exit(0);
    }

    println!("=============================================================");
    println!("  iconnx WS-6 R5 — OCR Inference Performance Characterisation");
    println!("=============================================================");
    println!();

    // --- Recognition benchmarks ------------------------------------------
    bench_rec_single_crop(REC_MODEL, 320);

    // Confirm batch>1 is rejected by dispatch_lstm (batch=1 hardcoded).
    confirm_lstm_batch1_requirement(REC_MODEL, 320);

    // --- Detection benchmarks --------------------------------------------
    bench_det(DET_MODEL, 960, 960);
    bench_det(DET_MODEL, 1500, 2000);

    println!();
    println!("Done.");
}

// ---------------------------------------------------------------------------
// Recognition — single crop wall-clock
// ---------------------------------------------------------------------------

/// Measure single-crop recognition latency at `[1, 3, 32, width]`.
fn bench_rec_single_crop(model_path: &str, width: usize) {
    let label = format!("rec [1, 3, 32, {width}]");
    println!("--- {label} ---");

    let model = OnnxParser::parse_file(model_path)
        .unwrap_or_else(|e| panic!("Failed to parse {model_path}: {e}"));

    let exec = GpuGraphExecutor::from_model(&model)
        .unwrap_or_else(|e| panic!("GpuGraphExecutor::from_model failed: {e}"));

    // Build a representative single-crop input.
    // Shape: [1 (batch), 3 (channels), 32 (height), width].
    let shape = vec![1usize, 3, 32, width];
    let numel: usize = shape.iter().product();
    let data = simple_seeded_f32(numel);

    let input_names = model.inputs();
    let input_name = input_names
        .first()
        .cloned()
        .unwrap_or_else(|| panic!("{model_path} has no inputs"));
    let output_names_owned = model.outputs();
    let output_names: Vec<&str> = output_names_owned.iter().map(|s| s.as_str()).collect();

    // Warmup.
    for _ in 0..WARMUP_RUNS {
        let mut inputs = HashMap::new();
        inputs.insert(
            input_name.clone(),
            Tensor::from_vec_f32(data.clone(), shape.clone()),
        );
        exec.run(inputs, output_names.clone())
            .expect("rec warmup run failed");
    }

    // Timed runs.
    let mut elapsed_ms = Vec::with_capacity(TIMED_RUNS);
    for _ in 0..TIMED_RUNS {
        let mut inputs = HashMap::new();
        inputs.insert(
            input_name.clone(),
            Tensor::from_vec_f32(data.clone(), shape.clone()),
        );
        let t0 = Instant::now();
        exec.run(inputs, output_names.clone())
            .expect("rec timed run failed");
        elapsed_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
    }

    let avg_ms = elapsed_ms.iter().sum::<f64>() / elapsed_ms.len() as f64;
    let min_ms = elapsed_ms.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_ms = elapsed_ms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!("  Wall-clock (avg/min/max over {TIMED_RUNS} runs):");
    println!("    avg = {avg_ms:.2} ms   min = {min_ms:.2} ms   max = {max_ms:.2} ms");

    // GPU memory: weights bytes after first run (plan compiled).
    let weight_bytes = exec.weights_memory_bytes();
    println!(
        "  GPU weight memory : {:.2} MiB  ({weight_bytes} bytes)",
        weight_bytes as f64 / (1024.0 * 1024.0)
    );

    // Memory pool statistics.
    let (hits, misses, hit_rate) = exec.pool_stats();
    println!("  Pool stats: hits={hits}  misses={misses}  hit_rate={hit_rate:.1}%");

    println!();
}

// ---------------------------------------------------------------------------
// Batch > 1 guard — confirm dispatch_lstm errors with batch=2
// ---------------------------------------------------------------------------

/// Run recognition with `N=2` (batch=2) and report the resulting error.
///
/// `dispatch_lstm` in `src/cuda/executor/lstm.rs` hardcodes `batch_size=1`
/// via `prepare_lstm_plan(..., 1, seq_length)`. The cuDNN plan is keyed on
/// `(W_ptr, R_ptr, seq_length)` and the cuDNN workspace is sized for a
/// single batch. Dispatching `N=2` produces an error when the runtime
/// batch dimension disagrees with the prepared plan.
///
/// This function intentionally runs the batch=2 inference, catches the
/// resulting `Err`, and prints it so the limitation is explicitly confirmed
/// in benchmark output.
fn confirm_lstm_batch1_requirement(model_path: &str, width: usize) {
    println!("--- batch>1 LSTM constraint (rec N=2 [2, 3, 32, {width}]) ---");

    let model = OnnxParser::parse_file(model_path)
        .unwrap_or_else(|e| panic!("Failed to parse {model_path}: {e}"));

    let exec = GpuGraphExecutor::from_model(&model)
        .unwrap_or_else(|e| panic!("GpuGraphExecutor::from_model failed: {e}"));

    // Shape: [2 (batch), 3 (channels), 32 (height), width].
    let shape = vec![2usize, 3, 32, width];
    let numel: usize = shape.iter().product();
    let data = simple_seeded_f32(numel);

    let input_names = model.inputs();
    let input_name = input_names.first().cloned().expect("model has no inputs");
    let output_names_owned = model.outputs();
    let output_names: Vec<&str> = output_names_owned.iter().map(|s| s.as_str()).collect();

    let mut inputs = HashMap::new();
    inputs.insert(
        input_name,
        Tensor::from_vec_f32(data, shape),
    );

    match exec.run(inputs, output_names) {
        Ok(_) => {
            // If somehow it succeeded, note it — the LSTM code may have
            // been updated to support batch>1 since this bench was written.
            println!("  Result: UNEXPECTEDLY succeeded with batch=2.");
            println!(
                "  NOTE: if dispatch_lstm now supports batch>1, update the \
                 follow-up plan (ws6-followups-batching-perf.md) accordingly."
            );
        }
        Err(e) => {
            println!("  Result: Correctly returned error with batch=2:");
            println!("    {e}");
            println!("  => batch=1 is required; batching deferred (see ws6-followups-batching-perf.md).");
        }
    }

    println!();
}

// ---------------------------------------------------------------------------
// Detection — full-page wall-clock + GPU memory
// ---------------------------------------------------------------------------

/// Measure detection model latency and GPU memory at `[1, 3, H, W]`.
fn bench_det(model_path: &str, h: usize, w: usize) {
    let label = format!("det [1, 3, {h}, {w}]");
    println!("--- {label} ---");

    let model = OnnxParser::parse_file(model_path)
        .unwrap_or_else(|e| panic!("Failed to parse {model_path}: {e}"));

    let exec = GpuGraphExecutor::from_model(&model)
        .unwrap_or_else(|e| panic!("GpuGraphExecutor::from_model failed: {e}"));

    let shape = vec![1usize, 3, h, w];
    let numel: usize = shape.iter().product();
    let data = simple_seeded_f32(numel);

    let input_names = model.inputs();
    let input_name = input_names.first().cloned().expect("det model has no inputs");
    let output_names_owned = model.outputs();
    let output_names: Vec<&str> = output_names_owned.iter().map(|s| s.as_str()).collect();

    // Warmup.
    for _ in 0..WARMUP_RUNS {
        let mut inputs = HashMap::new();
        inputs.insert(
            input_name.clone(),
            Tensor::from_vec_f32(data.clone(), shape.clone()),
        );
        exec.run(inputs, output_names.clone())
            .expect("det warmup failed");
    }

    // Timed runs.
    let mut elapsed_ms = Vec::with_capacity(TIMED_RUNS);
    for _ in 0..TIMED_RUNS {
        let mut inputs = HashMap::new();
        inputs.insert(
            input_name.clone(),
            Tensor::from_vec_f32(data.clone(), shape.clone()),
        );
        let t0 = Instant::now();
        exec.run(inputs, output_names.clone())
            .expect("det timed run failed");
        elapsed_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
    }

    let avg_ms = elapsed_ms.iter().sum::<f64>() / elapsed_ms.len() as f64;
    let min_ms = elapsed_ms.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_ms = elapsed_ms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!("  Wall-clock (avg/min/max over {TIMED_RUNS} runs):");
    println!("    avg = {avg_ms:.2} ms   min = {min_ms:.2} ms   max = {max_ms:.2} ms");

    // GPU memory.
    let weight_bytes = exec.weights_memory_bytes();
    let activation_shape_note = format!("[1, 3, {h}, {w}]");
    println!(
        "  GPU weight memory : {:.2} MiB  ({weight_bytes} bytes)",
        weight_bytes as f64 / (1024.0 * 1024.0)
    );
    println!(
        "  Activation input  : {activation_shape_note}  \
         ({:.2} MiB raw fp32)",
        (numel * 4) as f64 / (1024.0 * 1024.0)
    );

    // Memory pool stats.
    let (hits, misses, hit_rate) = exec.pool_stats();
    println!("  Pool stats: hits={hits}  misses={misses}  hit_rate={hit_rate:.1}%");

    // Top allocation sizes.
    let top_allocs = exec.pool_allocation_analysis(5);
    if !top_allocs.is_empty() {
        println!("  Pool top-5 sizes (bytes | allocs | returns | reuse_rate):");
        for (size, alloc_count, return_count, reuse_rate) in &top_allocs {
            println!(
                "    {size:>12} B   allocs={alloc_count:>4}   returns={return_count:>4}   \
                 reuse={reuse_rate:.1}%"
            );
        }
    }

    println!();
}

// ---------------------------------------------------------------------------
// Helper: simple seeded data (no external deps, no unsafe)
// ---------------------------------------------------------------------------

/// Generate `n` f32 values in a simple non-zero pattern suitable for model
/// inputs.
///
/// Values cycle through `[-0.5, -0.25, 0.0, 0.25, 0.5]` which avoids the
/// pathological all-zeros fixture (which suppresses attention signals in
/// transformers and is non-representative for convolutional pipelines).
fn simple_seeded_f32(n: usize) -> Vec<f32> {
    let pattern = [-0.5f32, -0.25, 0.0, 0.25, 0.5];
    (0..n).map(|i| pattern[i % pattern.len()]).collect()
}
