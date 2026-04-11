//! Type definitions for fused operator patterns.
//!
//! These types are populated by the `detection` module and consumed by
//! the executor's `execute_fused_pattern` method. They deliberately
//! carry only string names (not tensor references) so the detection
//! pass can run without touching any GPU state.
