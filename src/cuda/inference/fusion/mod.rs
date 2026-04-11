//! Fused kernel pattern detection and metadata.
//!
//! This module owns the `FusedPattern` enum (variant per fusible operator
//! chain), the `FusedPatternInfo` struct (pattern + nodes to skip + head
//! node), and the `detect_fused_patterns` graph walker that identifies
//! eligible chains before inference starts.
//!
//! Detection is intentionally separated from execution: the walker only
//! needs to read the node list and weights map, so it can be unit-tested
//! without a live CUDA context. Execution stays in `inference/mod.rs`
//! because it needs the executor's kernel caches and stream.

pub(super) mod patterns;
pub(super) mod detection;

pub(super) use patterns::{FusedPattern, FusedPatternInfo};
// detect_fused_patterns re-export added in Task A3 when the function moves
