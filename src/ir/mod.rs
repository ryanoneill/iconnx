//! Graph intermediate representation for iconnx.
//!
//! Splits "what to compute" (`OptimizableGraph`) from the frozen compiled
//! form (`ExecutionPlan`) from the GPU runtime (`cuda::executor::Executor`).
//! See docs/superpowers/specs/2026-04-17-phase2-graph-ir-design.md.

pub mod graph;
pub mod lowering;
pub mod passes;
pub mod plan;

pub use graph::{GraphInput, GraphNode, OptimizableGraph, OptimizableGraphBuilder};
pub use lowering::lower;
pub use plan::{
    ExecutionPlan, LstmPlanCache, LstmWeightCache, NodePrecomputed, OpKind, PlannedOp,
    PrecomputedSlice,
};
