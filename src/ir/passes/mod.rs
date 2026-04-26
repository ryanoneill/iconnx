//! Graph passes — pure functions over `OptimizableGraph`.

pub mod constant_folding;
pub mod dce;
pub mod elementwise_fusion;
pub mod elementwise_fusion_codegen;
pub mod folding_core;
pub mod fusion;
pub mod memory_planner;
pub mod precompute;
pub mod shape_extraction_folding;
pub mod shape_inference;
pub mod topo_sort;

pub use constant_folding::constant_folding as constant_folding_pass;
pub use dce::dead_code_elimination;
pub use elementwise_fusion::{
    elementwise_fusion as elementwise_fusion_pass, fusion_annotations_from_graph,
};
pub use fusion::{FusionAnnotations, detect_fusion, gate_fusion_to_float32};
pub use memory_planner::memory_planner;
pub use precompute::precompute_params;
pub use shape_extraction_folding::{shape_extraction_folding, ShapeExtractionFoldingCounts};
pub use shape_inference::{shape_inference, tensor_shapes_from_graph};
pub use topo_sort::topo_sort;
