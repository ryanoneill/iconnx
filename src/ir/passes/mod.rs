//! Graph passes — pure functions over `OptimizableGraph`.

pub mod fusion;
pub mod precompute;
pub mod topo_sort;

pub use fusion::{FusionAnnotations, detect_fusion};
pub use precompute::precompute_params;
pub use topo_sort::topo_sort;
