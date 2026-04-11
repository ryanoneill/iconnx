//! Graph walker that identifies fused operator patterns before inference.
//!
//! Detection is a free function (not a method on the executor) so it can
//! be tested with a fabricated node list and a dummy weights map —
//! no CUDA context required.
