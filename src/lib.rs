//! iconnx - High-performance ONNX inference engine with CUDA GPU acceleration
//!
//! iconnx is a pure Rust ONNX inference engine designed for production workloads
//! requiring high-throughput GPU inference with minimal latency.
//!
//! ## Features
//!
//! - **ONNX Opset 17+ support** - Modern operators for transformer models
//! - **CUDA GPU acceleration** - Native cuBLAS/cuDNN integration via garboard
//! - **Concurrent inference** - Multiple streams share weights for optimal GPU utilization
//! - **CPU fallback** - Full CPU implementation for testing and portability
//! - **Pure Rust** - No C++ dependencies, just Rust + CUDA
//!
//! ## Architecture
//!
//! - **Shared weights:** Model parameters loaded once, shared via Arc
//! - **Per-inference contexts:** Each inference has own buffers + CUDA stream
//! - **No Mutex serialization:** Multiple threads call run(&self) concurrently
//!
//! ## Quick Start
//!
//! ```text
//! use iconnx::{OnnxParser, GpuGraphExecutor, Tensor};
//!
//! // Load ONNX model
//! let model = OnnxParser::parse_file("model.onnx")?;
//!
//! // Create GPU executor
//! let mut executor = GpuGraphExecutor::new()?;
//!
//! // Add model weights
//! for (name, tensor) in model.extract_weights()? {
//!     executor.add_initializer(name, &tensor)?;
//! }
//!
//! // Run inference
//! let outputs = executor.run(inputs, &["output"])?;
//! ```
//!
//! ## Feature Flags
//!
//! - `cuda` (default) - Enable CUDA GPU acceleration
//! - `debug-inference` - Enable tensor validation, checkpoints, differential testing
//!
//! ## Design Principles
//!
//! 1. **Test-Driven:** Every operator validated against ONNX Runtime
//! 2. **Validated:** All outputs compared against ground truth
//! 3. **Incremental:** Build piece by piece, each independently useful
//! 4. **Pure Rust:** No C++ dependencies, just Rust + CUDA
//! 5. **Educational:** Deep understanding of ML inference internals

// Allow dead code - many components are reference implementations,
// fallbacks, or used only in specific test scenarios. The GPU paths are primary,
// but CPU implementations are kept for testing/validation.
#![allow(dead_code)]

pub mod attributes;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod engine;
pub mod errors;
pub mod graph;
pub mod graph_executor;
pub mod ir;
pub mod onnx_parser;
// `op_support` and `opset` are gated under `cuda` to mirror `validate`
// (the only consumer); neither has an intrinsic cuda dependency, but
// ungating them while `validate` remains gated would leave dead code
// paths visible on no-default-features builds (e.g. `OP_SUPPORT_TABLE`
// would compile against a `validate` module that doesn't exist).
// Ungate together if/when validate's cuda coupling is removed.
#[cfg(feature = "cuda")]
pub mod op_support;
pub mod operators;
#[cfg(feature = "cuda")]
pub mod opset;
pub mod tensor;
// `validate` depends on `crate::tensor::DType`, which is itself gated under
// `feature = "cuda"` (see `src/tensor.rs`'s `pub use crate::cuda::DType`).
// Mirror that gating here so a future no-default-features build doesn't
// surface link errors from a partially-defined module surface. Public
// re-exports below match the gating exactly.
#[cfg(feature = "cuda")]
pub mod validate;

// Re-export commonly used types at crate root
pub use attributes::NodeAttributes;
pub use errors::{IconnxError, Result};
pub use graph_executor::GraphExecutor;
pub use onnx_parser::{OnnxModel, OnnxParser, ParseError};
pub use tensor::Tensor;

#[cfg(feature = "cuda")]
pub use validate::{
    validate_model, ModelCapabilities, ModelIncompatibility, ModelValidationFailure, ModelWarning,
    ToleranceBasis, ToleranceConfidence, ToleranceHint,
};

#[cfg(feature = "cuda")]
pub use op_support::{DispatchContext, LookupResult};

#[cfg(feature = "cuda")]
pub use opset::{MAX_OPSET_SUPPORTED, MIN_OPSET_SUPPORTED};

#[cfg(feature = "cuda")]
pub use cuda::{GpuGraphExecutor, GpuMemoryPool, GpuTensor, IconnxCudaContext};

#[cfg(feature = "cuda")]
pub use validate::validate_model_for_hardware;
