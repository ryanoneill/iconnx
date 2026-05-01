//! Model-level opset version range supported by iconnx.
//!
//! `MIN_OPSET_SUPPORTED` and `MAX_OPSET_SUPPORTED` correspond to the lowest
//! and highest opset versions among the roster of validated models in
//! leadline-bench. Models with `opset_imports` outside this range may parse
//! and run correctly, but iconnx does not claim validated semantics beyond
//! the bounds. To extend the range, add a roster model exercising the new
//! opset and re-derive.
//!
//! Per-op opset metadata is deliberately absent from `OP_SUPPORT_TABLE`
//! (Q2 deferral). The model-level (min, max) bound is sufficient for WS-5.
//!
//! ## Empirical derivation (2026-04-29)
//!
//! Observed `opset_import` for the default ai.onnx domain across the 10
//! ModelProfile gates in leadline-bench:
//!
//! | Roster model                                | opset |
//! |---------------------------------------------|------:|
//! | whisper-tiny.en-export/onnx/encoder_model   |    11 |
//! | whisper-tiny.en-export/onnx/decoder_model   |    11 |
//! | whisper-tiny.en-bf16/encoder_model          |    11 |
//! | bert-base-uncased-onnx/onnx/model           |    11 |
//! | bert-base-uncased-onnx-bf16/model           |    11 |
//! | distilbert-base-uncased-onnx/onnx/model_int8 |   11 |
//! | resnet-50-onnx/onnx/model                   |    11 |
//! | yolos-tiny-onnx/onnx/model                  |    12 |
//! | gpt2-onnx/onnx/decoder_model                |    13 |
//! | kokoro-v1.0.onnx                            |    20 |
//!
//! MIN contributors: BERT, BERT-BF16, DistilBERT-INT8, ResNet-50, Whisper
//! encoder/decoder (FP32 + BF16). MAX contributor: Kokoro v1.0.

/// Lowest opset version iconnx claims to support.
///
/// Derived from the lowest `opset_import` version among leadline-bench's
/// 10 ModelProfile gates (as of 2026-04-29). MIN = 11, contributed by 7 of
/// the 10 roster models (BERT/DistilBERT/ResNet-50/Whisper × FP32+BF16).
pub const MIN_OPSET_SUPPORTED: i64 = 11;

/// Highest opset version iconnx claims to support.
///
/// Derived from the highest `opset_import` version among leadline-bench's
/// 10 ModelProfile gates (as of 2026-04-29). MAX = 20, contributed by
/// kokoro-v1.0.onnx (the only roster model declaring opset >= 14).
/// Conservative MAX = roster_max policy — claims only what's been
/// validated. Models declaring opset_import above this value receive a
/// `ModelIncompatibility::OpsetOutOfRange` from `validate_model`. To
/// extend: add a roster model exercising the new opset and update.
pub const MAX_OPSET_SUPPORTED: i64 = 20;
