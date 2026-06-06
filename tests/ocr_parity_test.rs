//! WS-6 M6.5 — whole-model OCR det + rec parity scaffolding.
//!
//! ## Live-run status: BLOCKED on pdfocr fixtures
//!
//! The model tests (`rec_whole_model_matches_ort`, `det_whole_model_matches_ort`)
//! require two things that pdfocr has NOT delivered yet:
//!
//! 1. Two `_rapidai` model exports symlinked as:
//!    * `./en_PP-OCRv3_det.onnx`
//!    * `./en_number_mobile_rec.onnx`
//! 2. Two preprocessed `.npy` input fixtures:
//!    * `./ocr_det_input.npy`   (shape [1, 3, H, W])
//!    * `./ocr_rec_input.npy`   (shape [1, 3, 32, W])
//!
//! Until those files exist on disk, the model tests **skip with a printed
//! reason** and the suite stays green. DO NOT fabricate fixtures.
//!
//! ## What IS provable now (and tested here without `--ignored`):
//!
//! * `ctc_greedy` — argmax per timestep → collapse dups → drop blank (index 0).
//! * `mask_iou` — IoU of probability maps thresholded at `thresh`.
//!
//! These helper-unit tests run in the default lane, no GPU or fixtures needed.

#![cfg(feature = "cuda")]

// Live-ORT parity helpers.  Declared ONCE at file scope so
// `clippy::duplicate_mod` is satisfied (per-function #[path] includes would
// load the same file as multiple distinct modules).
#[path = "common/ort_parity.rs"]
mod ort_parity;

// Npy-load, SHA-256, and skip helpers (WS-6 M6.5 T15).
#[path = "common/npy.rs"]
mod npy;

use std::collections::HashMap;

use iconnx::{GpuGraphExecutor, OnnxParser, Tensor};

use ort_parity::{assert_close, OrtSession};

// ---------------------------------------------------------------------------
// Fixture constants
// ---------------------------------------------------------------------------

/// Symlink path for the rapidAI recognition model.
const REC_MODEL: &str = "./en_number_mobile_rec.onnx";
/// Symlink path for the rapidAI detection model.
const DET_MODEL: &str = "./en_PP-OCRv3_det.onnx";
/// Preprocessed recognition input fixture (shape [1, 3, 32, W]).
const REC_INPUT: &str = "./ocr_rec_input.npy";
/// Preprocessed detection input fixture (shape [1, 3, H, W]).
const DET_INPUT: &str = "./ocr_det_input.npy";

/// Pinned SHA-256 of the recognition model (rapidAI export).
const REC_MODEL_SHA256: &str =
    "e679ba625c544444be78292a50d9e1af9caa1569239a88bb8b864cb688b11c01";
/// Pinned SHA-256 of the detection model (rapidAI export).
const DET_MODEL_SHA256: &str =
    "f139598bc2af4e4b6fe98dec11574e30edfdd91fc94ac1425c18ace3bd5a866b";

// ---------------------------------------------------------------------------
// Helper: CTC greedy decoding
// ---------------------------------------------------------------------------

/// CTC greedy decode of `logits` with shape `[T, C]` (T timesteps, C classes).
///
/// Algorithm:
/// 1. For each timestep, take argmax over C → raw sequence of class indices.
/// 2. Collapse consecutive duplicate indices.
/// 3. Remove the blank index (index 0) from the collapsed sequence.
///
/// Returns the decoded token index sequence (blank-free, duplicate-collapsed).
///
/// # Parameters
/// * `logits` — flat row-major slice of length `t * c`.
/// * `t` — number of timesteps.
/// * `c` — number of classes (including blank at index 0).
fn ctc_greedy(logits: &[f32], t: usize, c: usize) -> Vec<usize> {
    assert_eq!(logits.len(), t * c, "ctc_greedy: logits length mismatch");

    // Step 1: argmax per timestep.
    let raw: Vec<usize> = (0..t)
        .map(|step| {
            let row = &logits[step * c..(step + 1) * c];
            row.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .expect("ctc_greedy: empty row")
        })
        .collect();

    // Step 2: collapse consecutive duplicates.
    let mut collapsed: Vec<usize> = vec![];
    for &idx in &raw {
        if collapsed.last() != Some(&idx) {
            collapsed.push(idx);
        }
    }

    // Step 3: drop blank (index 0).
    collapsed.into_iter().filter(|&i| i != 0).collect()
}

// ---------------------------------------------------------------------------
// Helper: mask IoU
// ---------------------------------------------------------------------------

/// Compute the IoU of the foreground masks produced by thresholding `a` and
/// `b` at `thresh`.
///
/// Returns `f64::NAN` when the union is empty (both maps are all-background
/// at this threshold); callers guard for non-NaN before asserting.
///
/// # Parameters
/// * `a`, `b` — flat f32 probability maps (same length).
/// * `thresh` — binarisation threshold (e.g. 0.3 for PP-OCRv3 det).
fn mask_iou(a: &[f32], b: &[f32], thresh: f32) -> f64 {
    assert_eq!(a.len(), b.len(), "mask_iou: length mismatch");

    let mut intersection: u64 = 0;
    let mut union: u64 = 0;
    for (&av, &bv) in a.iter().zip(b.iter()) {
        let fa = av > thresh;
        let fb = bv > thresh;
        if fa && fb {
            intersection += 1;
        }
        if fa || fb {
            union += 1;
        }
    }

    if union == 0 {
        f64::NAN
    } else {
        intersection as f64 / union as f64
    }
}

// ---------------------------------------------------------------------------
// Unit tests for the helper logic (default lane — no GPU, no fixtures)
// ---------------------------------------------------------------------------

/// `ctc_greedy` correctly collapses duplicates and strips blank.
///
/// Hand-built logit array for T=8, C=6 where the argmax sequence is
/// [0, 3, 3, 0, 5, 5, 5, 2]:
/// * Collapse duplicates → [0, 3, 0, 5, 2]
/// * Drop blank (0) → [3, 5, 2]
#[test]
fn ctc_greedy_collapses_and_strips_blank() {
    // T=8, C=6.  We set one logit per row to be clearly maximal (10.0)
    // and the rest to 0.0, giving a forced argmax sequence.
    //
    // Argmax per timestep: [0, 3, 3, 0, 5, 5, 5, 2]
    // After collapsing duplicates:            [0, 3, 0, 5, 2]
    // After dropping blank (0):              [3, 5, 2]
    let t = 8;
    let c = 6;
    let mut logits = vec![0.0f32; t * c];
    let argmaxes = [0usize, 3, 3, 0, 5, 5, 5, 2];
    for (step, &class) in argmaxes.iter().enumerate() {
        logits[step * c + class] = 10.0;
    }

    let decoded = ctc_greedy(&logits, t, c);
    assert_eq!(decoded, vec![3usize, 5, 2], "ctc_greedy decoded mismatch");
}

/// `ctc_greedy` on an all-blank sequence decodes to empty.
#[test]
fn ctc_greedy_all_blank_decodes_empty() {
    // T=4, C=3.  All argmaxes are 0 (blank).
    let t = 4;
    let c = 3;
    let mut logits = vec![0.0f32; t * c];
    for step in 0..t {
        logits[step * c] = 10.0; // argmax = 0 (blank) each step
    }

    let decoded = ctc_greedy(&logits, t, c);
    assert!(decoded.is_empty(), "all-blank should decode to empty");
}

/// `mask_iou` returns the correct IoU for two synthetic probability maps
/// with a known overlap.
///
/// Map layout (4×4 = 16 pixels, thresh 0.5):
///   a foreground pixels:  indices 0..8   (top half)
///   b foreground pixels:  indices 4..12  (middle half)
///   Intersection:         indices 4..8   → 4 pixels
///   Union:                indices 0..12  → 12 pixels
///   IoU:                  4 / 12 ≈ 0.333
#[test]
fn mask_iou_known_overlap() {
    let mut a = vec![0.0f32; 16];
    let mut b = vec![0.0f32; 16];
    // a is hot in [0..8)
    for v in a[0..8].iter_mut() {
        *v = 1.0;
    }
    // b is hot in [4..12)
    for v in b[4..12].iter_mut() {
        *v = 1.0;
    }

    let iou = mask_iou(&a, &b, 0.5);
    let expected = 4.0f64 / 12.0;
    assert!(
        (iou - expected).abs() < 1e-9,
        "mask_iou expected {expected:.6}, got {iou:.6}"
    );
}

/// `mask_iou` returns NaN when both masks are all-background.
#[test]
fn mask_iou_empty_union_is_nan() {
    let a = vec![0.0f32; 16];
    let b = vec![0.0f32; 16];
    let iou = mask_iou(&a, &b, 0.5);
    assert!(
        iou.is_nan(),
        "mask_iou with empty union should be NaN, got {iou}"
    );
}

/// `mask_iou` returns 1.0 for identical all-foreground maps.
#[test]
fn mask_iou_perfect_overlap() {
    let a = vec![1.0f32; 16];
    let b = vec![1.0f32; 16];
    let iou = mask_iou(&a, &b, 0.5);
    assert!(
        (iou - 1.0).abs() < 1e-9,
        "identical foreground maps should have IoU=1.0, got {iou}"
    );
}

// ---------------------------------------------------------------------------
// Whole-model parity: recognition (rec)
//
// BLOCKED: requires `./en_number_mobile_rec.onnx` + `./ocr_rec_input.npy`
// from pdfocr.  Skips cleanly when absent.
// ---------------------------------------------------------------------------

/// Whole-model iconnx-GPU vs ORT parity for the PP-OCRv3 **recognition** model.
///
/// Checks:
/// (a) Tensor allclose: atol=rtol=1e-3 on the raw CTC logit output.
/// (b) CTC-greedy decode of iconnx output == CTC-greedy decode of ORT output
///     (exact sequence match).
/// (c) Non-degeneracy guard: the ORT-decoded sequence must have ≥ 1 non-blank
///     token; if the fixture is all-blank the test fails loudly (degenerate
///     fixture, not a real character crop).
///
/// A second fixture width `W2` can be added by extending the slice passed to
/// `assert_rec_parity` — the structure is easy to extend; one width is
/// sufficient until pdfocr delivers multi-crop fixtures.
#[test]
#[ignore = "requires CUDA GPU + pdfocr model/fixture files"]
fn rec_whole_model_matches_ort() {
    if npy::skip_if_missing(&[REC_MODEL, REC_INPUT]) {
        return;
    }

    // Verify fixture integrity before attempting inference.
    let model_path = std::path::Path::new(REC_MODEL);
    assert!(
        npy::verify_sha256(model_path, REC_MODEL_SHA256)
            .expect("sha256_hex failed for rec model"),
        "SHA-256 mismatch for {REC_MODEL} — wrong fixture file on disk"
    );

    // Load the preprocessed input fixture (shape [1, 3, 32, W]).
    let (input_data, input_shape) =
        npy::load_npy_f32(std::path::Path::new(REC_INPUT)).expect("load rec input npy");
    assert_eq!(
        input_shape.len(),
        4,
        "rec input must be 4-D [1, 3, 32, W], got shape {:?}",
        input_shape
    );
    assert_eq!(
        input_shape[0], 1,
        "rec input batch must be 1, got {:?}",
        input_shape
    );
    assert_eq!(
        input_shape[1], 3,
        "rec input must be 3-channel, got {:?}",
        input_shape
    );
    assert_eq!(
        input_shape[2], 32,
        "rec input height must be 32, got {:?}",
        input_shape
    );

    // Run iconnx GPU path.
    let model = OnnxParser::parse_file(model_path).expect("parse rec model");
    let exec = GpuGraphExecutor::from_model(&model).expect("GpuGraphExecutor::from_model");

    // Use the first model input/output name from the parsed spec so the test
    // is robust to whatever name the rapidAI export uses (e.g. "x", "input",
    // "images", …).
    let rec_input_names = model.inputs();
    let rec_input_name = rec_input_names
        .first()
        .cloned()
        .expect("rec model has no inputs");
    let rec_outputs = model.outputs();
    let rec_output_name = rec_outputs
        .first()
        .map(|s| s.as_str())
        .expect("rec model has no outputs");

    let mut inputs = HashMap::new();
    inputs.insert(
        rec_input_name.clone(),
        Tensor::from_vec_f32(input_data.clone(), input_shape.clone()),
    );
    let iconnx_out = exec
        .run(inputs, vec![rec_output_name])
        .expect("iconnx rec run");
    let iconnx_logits = iconnx_out[rec_output_name].as_slice();

    // Run ORT CPU EP oracle.
    let mut ort = OrtSession::open(model_path).expect("OrtSession::open rec model");
    let (ort_logits, ort_shape) = ort
        .run_f32(
            &[(&rec_input_name, &input_data, input_shape.clone())],
            rec_output_name,
        )
        .expect("ort rec run");

    // (a) Tensor closeness.
    assert_close(&iconnx_logits, &ort_logits, 1e-3, 1e-3);

    // (b) CTC greedy decoded sequence must match exactly.
    // Output shape: [1, T, C] — T timesteps, C classes.
    assert_eq!(
        ort_shape.len(),
        3,
        "rec output must be 3-D [1, T, C], got {:?}",
        ort_shape
    );
    let (t, c) = (ort_shape[1], ort_shape[2]);

    let iconnx_seq = ctc_greedy(&iconnx_logits, t, c);
    let ort_seq = ctc_greedy(&ort_logits, t, c);
    assert_eq!(
        iconnx_seq, ort_seq,
        "CTC-greedy decoded sequences differ: iconnx={iconnx_seq:?} vs ort={ort_seq:?}"
    );

    // (c) Non-degeneracy guard.
    assert!(
        !ort_seq.is_empty(),
        "ORT decoded sequence is empty (all blank) — the fixture is degenerate \
         and is not a real character crop; replace with a fixture that produces \
         at least one non-blank token"
    );
}

// ---------------------------------------------------------------------------
// Whole-model parity: detection (det)
//
// BLOCKED: requires `./en_PP-OCRv3_det.onnx` + `./ocr_det_input.npy`
// from pdfocr.  Also covers R6 Resize parity.  Skips cleanly when absent.
// ---------------------------------------------------------------------------

/// Whole-model iconnx-GPU vs ORT parity for the PP-OCRv3 **detection** model.
///
/// Checks:
/// (a) Probability-map allclose: atol=rtol=1e-3 on the det output tensor.
/// (b) Binarise both maps at threshold 0.3 → `mask_iou` ≥ 0.99 (near-identical
///     binary foreground regions).
/// (c) Non-empty-mask guard: the ORT binary mask must contain at least one
///     foreground pixel; if the input fixture has no text the test fails loudly.
///
/// This also exercises R6 Resize parity end-to-end: PP-OCRv3 det's DBNet head
/// uses nearest/asymmetric/floor Resize (verified in the model graph), which is
/// exactly the GPU `ResizeCoordMode` support — a mode mismatch would shift the
/// probability map and fail this parity check.
#[test]
#[ignore = "requires CUDA GPU + pdfocr model/fixture files"]
fn det_whole_model_matches_ort() {
    if npy::skip_if_missing(&[DET_MODEL, DET_INPUT]) {
        return;
    }

    // Verify fixture integrity before attempting inference.
    let model_path = std::path::Path::new(DET_MODEL);
    assert!(
        npy::verify_sha256(model_path, DET_MODEL_SHA256)
            .expect("sha256_hex failed for det model"),
        "SHA-256 mismatch for {DET_MODEL} — wrong fixture file on disk"
    );

    // Load the preprocessed detection input (shape [1, 3, H, W]).
    let (input_data, input_shape) =
        npy::load_npy_f32(std::path::Path::new(DET_INPUT)).expect("load det input npy");
    assert_eq!(
        input_shape.len(),
        4,
        "det input must be 4-D [1, 3, H, W], got shape {:?}",
        input_shape
    );
    assert_eq!(
        input_shape[0], 1,
        "det input batch must be 1, got {:?}",
        input_shape
    );
    assert_eq!(
        input_shape[1], 3,
        "det input must be 3-channel, got {:?}",
        input_shape
    );

    // Run iconnx GPU path.
    let model = OnnxParser::parse_file(model_path).expect("parse det model");
    let exec = GpuGraphExecutor::from_model(&model).expect("GpuGraphExecutor::from_model");

    // Use the first model input/output name from the parsed spec so the test
    // is robust to whatever name the rapidAI export uses.
    let det_input_names = model.inputs();
    let det_input_name = det_input_names
        .first()
        .cloned()
        .expect("det model has no inputs");
    let det_outputs = model.outputs();
    let det_output_name = det_outputs
        .first()
        .map(|s| s.as_str())
        .expect("det model has no outputs");

    let mut inputs = HashMap::new();
    inputs.insert(
        det_input_name.clone(),
        Tensor::from_vec_f32(input_data.clone(), input_shape.clone()),
    );
    let iconnx_out = exec
        .run(inputs, vec![det_output_name])
        .expect("iconnx det run");
    let iconnx_prob = iconnx_out[det_output_name].as_slice();

    // Run ORT CPU EP oracle.
    let mut ort = OrtSession::open(model_path).expect("OrtSession::open det model");
    let (ort_prob, _ort_shape) = ort
        .run_f32(
            &[(&det_input_name, &input_data, input_shape.clone())],
            det_output_name,
        )
        .expect("ort det run");

    // (a) Probability-map closeness.
    assert_close(&iconnx_prob, &ort_prob, 1e-3, 1e-3);

    // (b) Binary mask IoU ≥ 0.99 at threshold 0.3.
    let iou = mask_iou(&iconnx_prob, &ort_prob, 0.3);
    assert!(
        !iou.is_nan(),
        "mask_iou is NaN — both iconnx and ORT produced all-background maps \
         at threshold 0.3; check that the fixture contains text"
    );
    assert!(
        iou >= 0.99,
        "mask_iou {iou:.4} < 0.99 — binary foreground regions differ more than \
         expected at threshold 0.3"
    );

    // (c) Non-empty-mask guard.
    let ort_has_foreground = ort_prob.iter().any(|&v| v > 0.3);
    assert!(
        ort_has_foreground,
        "ORT probability map is all-background at threshold 0.3 — the det \
         input fixture does not contain detectable text; replace with a fixture \
         that has visible text regions"
    );
}
