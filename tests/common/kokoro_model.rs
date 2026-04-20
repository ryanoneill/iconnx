//! Locates the Kokoro v1.0 ONNX model for integration tests.
//!
//! Panics loudly when the model is missing. Silent-skip masked the
//! Phase 3 Commit 3 Gather shape-inference correctness bug until the
//! model was explicitly symlinked; future silent defects (memory
//! planner, new passes) would be hidden the same way. We make the
//! failure mode explicit.

use std::path::PathBuf;

/// Resolve the Kokoro model path from the test working directory.
/// Panics with a single consistent diagnostic if the model is missing.
pub fn kokoro_model_path() -> PathBuf {
    let path = PathBuf::from("kokoro-v1.0.onnx");
    assert!(
        path.exists(),
        "kokoro-v1.0.onnx must be reachable from the working directory \
         (symlink or copy it in). A silent skip here would hide real \
         correctness regressions — see Phase 3 Commit 3's Gather bug \
         for an example of what silent-skipping can mask."
    );
    path
}
