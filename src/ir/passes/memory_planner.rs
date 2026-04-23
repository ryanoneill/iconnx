//! Static memory planner — assigns arena offsets to eligible tensors.
//!
//! Runs after fusion annotation; consumes tensor_shapes + tensor_last_use
//! + tensor_first_use + fusion annotations + graph I/O names. Produces a
//! [`MemoryPlan`] that stamps `arena_offset: Option<usize>` onto each
//! [`PlannedOp`], plus a summary of allocation decisions exposed via
//! [`MemoryPlan::classification_counts`].
//!
//! Philosophy: coexist with `GpuMemoryPool`. Arena-eligible tensors
//! (those passing [`is_planner_eligible`] AND not filtered as
//! fusion-skipped) get a slot. Everything else (dynamic shapes, small
//! tensors, graph I/O, fusion-skipped outputs) takes the pool path at
//! runtime — same behavior as pre-Phase-4.
//!
//! # Phase 4 A1 scope
//!
//! This module is the pure-data planner. It produces a [`MemoryPlan`]
//! whose offsets are stamped onto [`PlannedOp::arena_offset`] at plan
//! time. The runtime arena path (Executor allocating an arena buffer
//! and ops consuming slices from it) is deferred to Phase 5 A2 — the
//! dispatch-model refactor (`GpuTensor` accepting borrowed / trait-object
//! dispatch) required to bridge iconnx's Arc-based ownership model to
//! arena-allocated sub-slices is out of scope for this phase. Under A1,
//! the stamped `arena_offset` is written but not consumed; the runtime
//! continues to take the pool path for every tensor. Having the
//! plan-interface contract in place now lets A2 wire the runtime without
//! a second plumbing pass through `lower()`.

use std::collections::{BTreeMap, HashMap};

use crate::ir::graph::OptimizableGraph;
use crate::ir::passes::fusion::FusionAnnotations;
use crate::ir::plan::{MemoryPlan, PlannerClassification};

/// Semantic eligibility predicate.
///
/// Returns `true` iff the tensor's shape is fully known, its total byte
/// size is ≥ 4 KiB (matching `GpuMemoryPool`'s size-class threshold),
/// and it is not a graph input or graph output.
///
/// * **Graph inputs** are written by the input-transfer path, not by
///   any op's output; arena-backing them would have no clean
///   reconciliation with the input buffer at `run()` time.
/// * **Graph outputs** are excluded so ownership can be handed back to
///   the caller past the arena's lifetime.
/// * **Shapes with a `None` dim** cannot be sized at plan time.
/// * **Tensors < 4 KiB** are better served by the pool's small-object
///   path; the arena overhead per slot is not worth it.
pub(crate) fn is_planner_eligible(
    tensor_name: &str,
    tensor_shapes: &HashMap<String, Vec<Option<usize>>>,
    graph_inputs: &[String],
    graph_outputs: &[String],
    dtype_bytes: usize,
    element_count: usize,
) -> bool {
    if graph_inputs.iter().any(|i| i == tensor_name) {
        return false;
    }
    if graph_outputs.iter().any(|o| o == tensor_name) {
        return false;
    }
    let Some(shape) = tensor_shapes.get(tensor_name) else {
        return false;
    };
    if shape.is_empty() || shape.iter().any(|d| d.is_none()) {
        return false;
    }
    if dtype_bytes.saturating_mul(element_count) < 4096 {
        return false;
    }
    true
}

/// Structural filter separate from the semantic predicate.
///
/// Returns `true` iff `tensor_name` is produced by an op that belongs
/// to a fused chain as a skipped sub-node (i.e., the fused head writes
/// directly into the head's slot, and this op's output is never
/// materialized at runtime). Arena-assigning such a tensor would waste
/// a slot.
///
/// The fusion annotations track **node names** in their `skipped` set,
/// not tensor names — so we look up the tensor's producer in the graph
/// and check the producer's node name against the skipped set.
pub(crate) fn is_fusion_skipped(
    tensor_name: &str,
    graph: &OptimizableGraph,
    fusion: &FusionAnnotations,
) -> bool {
    if fusion.skipped.is_empty() {
        return false;
    }
    for node in graph.nodes.iter() {
        if node.outputs.iter().any(|o| o == tensor_name) {
            return fusion.skipped.contains(&node.name);
        }
    }
    false
}

/// Size-class bucket matching `GpuMemoryPool`'s convention:
///
/// * exact match for element counts `< 4096`
/// * power-of-2 rounding for counts `≥ 4096`
#[inline]
fn size_class_elements(len: usize) -> usize {
    if len < 4096 {
        len
    } else {
        len.next_power_of_two()
    }
}

/// Per-tensor planning record. All fields known at enumeration time.
#[derive(Clone, Debug)]
struct EligibleTensor {
    name: String,
    dtype_bytes: usize,
    size_class_elements: usize,
    slot_bytes: usize,
    first_use_op_idx: usize,
    last_use_op_idx: usize,
}

/// One slot in the arena. May be reused by disjoint-lifetime tensors
/// with matching `dtype_bytes` and `size_class_elements`.
#[derive(Clone, Debug)]
struct ArenaSlot {
    dtype_bytes: usize,
    size_class_elements: usize,
    byte_offset: usize,
    /// Index of the op where the slot's current tenant is last used.
    /// A new tenant can take the slot only when its `first_use_op_idx`
    /// is strictly greater than this value.
    current_tenant_last_use: usize,
}

/// Conservative dtype inference for op outputs.
///
/// Returns the byte size of an op's primary output dtype, or `0` for
/// ops whose output dtype is not reliably known. A return of `0`
/// propagates to "total bytes < 4 KiB" in [`is_planner_eligible`] and
/// the tensor falls through to the pool path — safe, matches pre-Phase-4
/// behavior.
///
/// We deliberately do **not** attempt to derive the dtype from
/// `tensor_shapes` or by walking the graph: mis-guessing would undersize
/// a slot and cause silent corruption at runtime. The conservative f32
/// set below covers Kokoro's activation path and all supported
/// binary / unary / reduce / matmul-family ops.
fn infer_output_dtype_bytes(op_type: &str) -> usize {
    match op_type {
        // Unary / binary elementwise f32 ops.
        "Add" | "Sub" | "Mul" | "Div" | "Pow" | "Sqrt" | "Rsqrt" | "Exp" | "Log"
        | "Sin" | "Cos" | "Tan" | "Tanh" | "Sigmoid" | "Atan" | "Erf" | "Relu"
        | "LeakyRelu" | "Gelu" | "Elu" | "Clip" | "Floor" | "Round" | "Ceil"
        | "Abs" | "Neg" | "Reciprocal" | "Softmax" | "LogSoftmax" | "Softplus"
        | "HardSigmoid" | "HardSwish" | "Swish" | "Mish" => 4,
        // Reduce / norm (output f32).
        "ReduceMean" | "ReduceSum" | "ReduceMax" | "ReduceMin" | "ReduceProd"
        | "InstanceNormalization" | "LayerNormalization" | "BatchNormalization"
        | "GroupNormalization" | "RMSNormalization" => 4,
        // MatMul / Conv (output f32 in our supported paths).
        "MatMul" | "Gemm" | "Conv" | "ConvTranspose" => 4,
        // Other f32-producing ops used in Kokoro.
        "Where" | "Expand" | "Tile" | "Pad" | "ConstantOfShape" => 4,
        // RNN family — output f32.
        "LSTM" | "GRU" | "RNN" => 4,
        // Shape-family ops (Shape, Range, Size) produce i64. Reshape /
        // Unsqueeze / Squeeze / Transpose / Slice / Concat / Split /
        // Gather / Cast outputs depend on input dtype (for f32 paths they
        // are f32 but often function as views — better handled by the
        // pool). Return 0 to defer; refine on demand if profiling shows
        // coverage loss.
        _ => 0,
    }
}

/// Plan arena offsets for eligible tensors.
///
/// Deterministic: given the same inputs, produces byte-identical
/// [`MemoryPlan`] output. Determinism falls out of:
///
/// 1. The eligibility enumeration walking `graph.nodes` in stored
///    order (topo-sorted by this point in the pipeline).
/// 2. A stable sort on `(first_use_op_idx, tensor_name)` — both
///    inputs are deterministic.
/// 3. A linear scan of the `slots` vector picking the first matching
///    slot, deterministic given the same enumeration.
///
/// The snapshot assertion in the test suite guards against any
/// future change accidentally breaking determinism.
pub fn memory_planner(
    graph: &OptimizableGraph,
    tensor_shapes: &HashMap<String, Vec<Option<usize>>>,
    tensor_last_use: &HashMap<String, usize>,
    tensor_first_use: &HashMap<String, usize>,
    fusion: &FusionAnnotations,
) -> MemoryPlan {
    let graph_inputs: Vec<String> = graph
        .inputs
        .iter()
        .map(|input| input.name.clone())
        .collect();
    let graph_outputs: Vec<String> = graph.outputs.clone();

    // Enumerate tensors produced by nodes. Apply eligibility predicate
    // and fusion-skipped filter. Tensors that don't produce a valid
    // (dtype, element_count) pair for arena assignment fall through to
    // the pool path and contribute to `pool_classified`.
    let mut eligible: Vec<EligibleTensor> = Vec::new();
    let mut arena_eligible_count = 0usize;
    let mut pool_classified_count = 0usize;

    for node in graph.nodes.iter() {
        for out_name in node.outputs.iter() {
            if out_name.is_empty() {
                continue;
            }

            // Compute dtype + element count.
            let dtype_bytes = infer_output_dtype_bytes(&node.op_type);
            let element_count = match tensor_shapes.get(out_name) {
                Some(shape)
                    if !shape.is_empty() && shape.iter().all(|d| d.is_some()) =>
                {
                    shape.iter().map(|d| d.expect("checked")).product::<usize>()
                }
                _ => 0, // Will be rejected by is_planner_eligible below.
            };

            let semantic_eligible = is_planner_eligible(
                out_name,
                tensor_shapes,
                &graph_inputs,
                &graph_outputs,
                dtype_bytes,
                element_count,
            );
            if !semantic_eligible {
                pool_classified_count += 1;
                continue;
            }

            if is_fusion_skipped(out_name, graph, fusion) {
                // Fusion-skipped tensors are counted as pool_classified
                // (they're not arena-eligible — a future change to the
                // dispatch model might eliminate them entirely, but for
                // now they still flow through the pool at runtime as
                // virtual outputs of the skipped op).
                pool_classified_count += 1;
                continue;
            }

            // Lifetimes are required for arena assignment. If either is
            // missing (e.g., a tensor consumed only as a graph output and
            // therefore stored under last_use but not first_use — which
            // cannot happen under our compute_tensor_first_use mirror,
            // but we guard defensively) we re-classify as pool.
            let Some(&last_use) = tensor_last_use.get(out_name) else {
                pool_classified_count += 1;
                continue;
            };
            let Some(&first_use) = tensor_first_use.get(out_name) else {
                pool_classified_count += 1;
                continue;
            };

            arena_eligible_count += 1;
            let size_class = size_class_elements(element_count);
            let slot_bytes = dtype_bytes.saturating_mul(size_class);

            eligible.push(EligibleTensor {
                name: out_name.clone(),
                dtype_bytes,
                size_class_elements: size_class,
                slot_bytes,
                first_use_op_idx: first_use,
                last_use_op_idx: last_use,
            });
        }
    }

    // Deterministic sort: by first_use ASC, tiebreak by name ASC.
    eligible.sort_by(|a, b| {
        a.first_use_op_idx
            .cmp(&b.first_use_op_idx)
            .then_with(|| a.name.cmp(&b.name))
    });

    // Greedy linear scan allocator. For each tensor, pick the first
    // existing compatible slot whose current tenant is no longer live;
    // otherwise allocate a new slot appended at the current end-of-arena.
    let mut slots: Vec<ArenaSlot> = Vec::new();
    // BTreeMap chosen over HashMap so `Debug`-format output is
    // deterministic across independent `memory_planner` calls — the
    // determinism-snapshot test relies on that.
    let mut slot_assignments: BTreeMap<String, usize> = BTreeMap::new();
    let mut next_byte_offset: usize = 0;

    for tensor in eligible.iter() {
        let mut chosen: Option<usize> = None;
        for (idx, slot) in slots.iter().enumerate() {
            if slot.dtype_bytes == tensor.dtype_bytes
                && slot.size_class_elements == tensor.size_class_elements
                && slot.current_tenant_last_use < tensor.first_use_op_idx
            {
                chosen = Some(idx);
                break;
            }
        }

        let slot_idx = match chosen {
            Some(idx) => {
                slots[idx].current_tenant_last_use = tensor.last_use_op_idx;
                idx
            }
            None => {
                let idx = slots.len();
                slots.push(ArenaSlot {
                    dtype_bytes: tensor.dtype_bytes,
                    size_class_elements: tensor.size_class_elements,
                    byte_offset: next_byte_offset,
                    current_tenant_last_use: tensor.last_use_op_idx,
                });
                next_byte_offset += tensor.slot_bytes;
                idx
            }
        };

        slot_assignments.insert(tensor.name.clone(), slot_idx);
    }

    // Invariant: under the current greedy algorithm every eligible tensor
    // gets a slot. If this ever changes (e.g., a fragmentation cap is
    // added), the classification summary will surface the gap.
    debug_assert_eq!(
        slot_assignments.len(),
        arena_eligible_count,
        "greedy allocator must assign every eligible tensor",
    );

    let arena_bytes = next_byte_offset;
    let slot_count = slots.len();
    let slot_offsets_bytes: Vec<usize> = slots.iter().map(|s| s.byte_offset).collect();

    let classification = PlannerClassification {
        arena_eligible: arena_eligible_count,
        arena_assigned: slot_assignments.len(),
        pool_classified: pool_classified_count,
    };

    MemoryPlan::new(
        arena_bytes,
        slot_count,
        slot_assignments,
        slot_offsets_bytes,
        classification,
    )
}

#[cfg(test)]
mod predicate_tests {
    use super::*;

    fn shapes_map(entries: &[(&str, Vec<Option<usize>>)]) -> HashMap<String, Vec<Option<usize>>> {
        entries
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    #[test]
    fn is_planner_eligible_all_dims_known_yields_true() {
        let shapes = shapes_map(&[("t", vec![Some(16), Some(64)])]);
        // 4 bytes (f32) × 1024 elements = 4096 bytes — at threshold.
        assert!(is_planner_eligible(
            "t", &shapes, &[], &[], 4, 16 * 64,
        ));
    }

    #[test]
    fn is_planner_eligible_any_none_dim_yields_false() {
        let shapes = shapes_map(&[("t", vec![None, Some(64)])]);
        assert!(!is_planner_eligible("t", &shapes, &[], &[], 4, 256));
    }

    #[test]
    fn is_planner_eligible_graph_input_yields_false() {
        let shapes = shapes_map(&[("t", vec![Some(4), Some(64)])]);
        assert!(!is_planner_eligible(
            "t", &shapes, &["t".to_string()], &[], 4, 256,
        ));
    }

    #[test]
    fn is_planner_eligible_graph_output_yields_false() {
        let shapes = shapes_map(&[("t", vec![Some(4), Some(64)])]);
        assert!(!is_planner_eligible(
            "t", &shapes, &[], &["t".to_string()], 4, 256,
        ));
    }

    #[test]
    fn is_planner_eligible_below_4kb_yields_false() {
        let shapes = shapes_map(&[("t", vec![Some(4), Some(64)])]);
        // 4 bytes × 1000 = 4000 bytes < 4096 threshold.
        assert!(!is_planner_eligible("t", &shapes, &[], &[], 4, 1000));
    }

    #[test]
    fn is_planner_eligible_unknown_tensor_yields_false() {
        // Tensor not in tensor_shapes map — predicate must be tolerant.
        let shapes = HashMap::new();
        assert!(!is_planner_eligible("ghost", &shapes, &[], &[], 4, 10_000));
    }
}

#[cfg(test)]
mod allocator_tests {
    use super::*;
    use crate::attributes::NodeAttributes;
    use crate::ir::graph::OptimizableGraphBuilder;
    use crate::tensor::Tensor;

    /// Build a two-Add chain on a single f32 activation `a`, then a
    /// final Add producing `b` (graph output). Useful baseline for
    /// overlapping-lifetime / fusion-skipping scenarios.
    fn build_two_add_graph() -> crate::ir::graph::OptimizableGraph {
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("x".into(), vec![Some(64), Some(128)]);
        b.add_initializer(
            "w1".into(),
            Tensor::from_vec_f32(vec![0.0; 64 * 128], vec![64, 128]),
        );
        b.add_initializer(
            "w2".into(),
            Tensor::from_vec_f32(vec![0.0; 64 * 128], vec![64, 128]),
        );
        b.add_node(
            "add1",
            "Add",
            vec!["x".into(), "w1".into()],
            vec!["a".into()],
            NodeAttributes::new(),
        );
        b.add_node(
            "add2",
            "Add",
            vec!["a".into(), "w2".into()],
            vec!["b".into()],
            NodeAttributes::new(),
        );
        b.add_output("b".into());
        b.build()
    }

    fn base_shapes_for_two_add() -> HashMap<String, Vec<Option<usize>>> {
        let mut s = HashMap::new();
        s.insert("x".into(), vec![Some(64), Some(128)]);
        s.insert("a".into(), vec![Some(64), Some(128)]);
        s.insert("b".into(), vec![Some(64), Some(128)]);
        s
    }

    /// Two f32 tensors, same size, disjoint lifetimes — expect a single
    /// shared slot.
    #[test]
    fn slot_assignment_reuses_slot_for_disjoint_lifetimes() {
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("x".into(), vec![Some(64), Some(128)]);
        b.add_initializer(
            "w".into(),
            Tensor::from_vec_f32(vec![0.0; 64 * 128], vec![64, 128]),
        );
        b.add_node(
            "first",
            "Add",
            vec!["x".into(), "w".into()],
            vec!["a".into()],
            NodeAttributes::new(),
        );
        b.add_node(
            "second",
            "Add",
            vec!["a".into(), "w".into()],
            vec!["a2".into()],
            NodeAttributes::new(),
        );
        b.add_node(
            "third",
            "Add",
            vec!["a2".into(), "w".into()],
            vec!["out".into()],
            NodeAttributes::new(),
        );
        b.add_output("out".into());
        let graph = b.build();

        let mut shapes = HashMap::new();
        shapes.insert("x".into(), vec![Some(64), Some(128)]);
        shapes.insert("a".into(), vec![Some(64), Some(128)]);
        shapes.insert("a2".into(), vec![Some(64), Some(128)]);
        shapes.insert("out".into(), vec![Some(64), Some(128)]);

        let mut first = HashMap::new();
        first.insert("a".into(), 0);
        first.insert("a2".into(), 1);
        first.insert("out".into(), 2);

        // Disjoint lifetimes: `a` lives [0..=0] (produced at op 0, not
        // consumed past op 0); `a2` lives [1..=1]. Slot-reuse predicate
        // `current_tenant_last_use < first_use` is 0 < 1 → true, so
        // `a2` can take `a`'s slot. Expect 1 slot total, 2 assignments.
        // (`out` is a graph output — excluded by is_planner_eligible.)
        let mut last = HashMap::new();
        last.insert("a".into(), 0);
        last.insert("a2".into(), 1);

        let plan = memory_planner(
            &graph,
            &shapes,
            &last,
            &first,
            &FusionAnnotations::default(),
        );
        assert_eq!(
            plan.slot_count, 1,
            "two disjoint-lifetime f32 tensors of same size must share a single slot"
        );
        assert_eq!(plan.classification_counts().arena_assigned, 2);
    }

    /// Two tensors with overlapping lifetimes — expect two slots.
    #[test]
    fn slot_assignment_creates_new_slot_for_overlapping_lifetimes() {
        // Chain: add1 -> a, add2 -> b, add3 -> c. We force overlap by
        // making op-idx 2 (producing c) consume both a and b.
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("x".into(), vec![Some(64), Some(128)]);
        b.add_initializer(
            "w".into(),
            Tensor::from_vec_f32(vec![0.0; 64 * 128], vec![64, 128]),
        );
        b.add_node(
            "add1",
            "Add",
            vec!["x".into(), "w".into()],
            vec!["a".into()],
            NodeAttributes::new(),
        );
        b.add_node(
            "add2",
            "Add",
            vec!["x".into(), "w".into()],
            vec!["b".into()],
            NodeAttributes::new(),
        );
        b.add_node(
            "add3",
            "Add",
            vec!["a".into(), "b".into()],
            vec!["c".into()],
            NodeAttributes::new(),
        );
        b.add_output("c".into());
        let graph = b.build();

        let mut shapes = HashMap::new();
        shapes.insert("x".into(), vec![Some(64), Some(128)]);
        shapes.insert("a".into(), vec![Some(64), Some(128)]);
        shapes.insert("b".into(), vec![Some(64), Some(128)]);
        shapes.insert("c".into(), vec![Some(64), Some(128)]);

        let mut first = HashMap::new();
        first.insert("a".into(), 0);
        first.insert("b".into(), 1);
        first.insert("c".into(), 2);

        let mut last = HashMap::new();
        // Both a and b are consumed by op-idx 2 — simultaneously live.
        last.insert("a".into(), 2);
        last.insert("b".into(), 2);

        let plan = memory_planner(
            &graph,
            &shapes,
            &last,
            &first,
            &FusionAnnotations::default(),
        );
        // `c` is a graph output — excluded. `a` and `b` are both
        // eligible and overlap — expect two slots.
        assert_eq!(plan.slot_count, 2, "overlapping lifetimes require two slots");
        assert_eq!(plan.classification_counts().arena_assigned, 2);
    }

    /// Two tensors with matching element count but different dtypes
    /// cannot share. Ops whose output dtype is i64 (Shape, Range)
    /// return 0 from `infer_output_dtype_bytes`, which causes them to
    /// fall through to pool — so we can't directly construct this case
    /// with mismatched-dtype arena tensors today. Instead we pin the
    /// invariant by constructing two size-class-compatible f32 tensors
    /// and asserting they DO share (negative space for "different
    /// dtypes CAN'T share"). The determinism of slot reuse is exercised
    /// elsewhere; the size-class / dtype separation is a structural
    /// invariant.
    #[test]
    fn slot_assignment_preserves_size_class_separation() {
        // Three f32 tensors, three distinct size classes. Expect three
        // slots regardless of lifetime disjointness — size classes
        // cannot share.
        let mut b = OptimizableGraphBuilder::new();
        b.add_input("x".into(), vec![Some(64), Some(128)]);
        b.add_initializer(
            "w".into(),
            Tensor::from_vec_f32(vec![0.0; 64 * 128], vec![64, 128]),
        );
        b.add_node(
            "small",
            "Add",
            vec!["x".into(), "w".into()],
            vec!["s".into()],
            NodeAttributes::new(),
        );
        b.add_node(
            "medium",
            "Add",
            vec!["s".into(), "w".into()],
            vec!["m".into()],
            NodeAttributes::new(),
        );
        b.add_node(
            "large",
            "Add",
            vec!["m".into(), "w".into()],
            vec!["l".into()],
            NodeAttributes::new(),
        );
        b.add_output("l".into());
        let graph = b.build();

        // Different element counts -> different power-of-2 size classes.
        let mut shapes = HashMap::new();
        shapes.insert("x".into(), vec![Some(64), Some(128)]);
        shapes.insert("s".into(), vec![Some(2048)]); // 2048 elements → 2048
        shapes.insert("m".into(), vec![Some(8192)]); // 8192 → 8192 (power of 2)
        shapes.insert("l".into(), vec![Some(16_384)]); // graph output, excluded
        // Adjust so all three are eligible (l is output → excluded),
        // so we only care about s and m having different size classes.
        let mut first = HashMap::new();
        first.insert("s".into(), 0);
        first.insert("m".into(), 1);
        first.insert("l".into(), 2);
        let mut last = HashMap::new();
        last.insert("s".into(), 0); // disjoint from m
        last.insert("m".into(), 1);

        let plan = memory_planner(
            &graph,
            &shapes,
            &last,
            &first,
            &FusionAnnotations::default(),
        );

        // s (size_class 2048) and m (size_class 8192) — two slots
        // despite disjoint lifetimes, because the size classes differ.
        assert_eq!(plan.slot_count, 2, "different size classes cannot share a slot");
        assert_eq!(plan.classification_counts().arena_assigned, 2);
    }

    /// Fusion-skipped tensor is filtered out even when otherwise
    /// eligible.
    #[test]
    fn fusion_skipped_tensor_filtered_out_even_when_otherwise_eligible() {
        let graph = build_two_add_graph();
        let shapes = base_shapes_for_two_add();
        let mut first = HashMap::new();
        first.insert("a".into(), 0);
        first.insert("b".into(), 1);
        let mut last = HashMap::new();
        last.insert("a".into(), 1);

        // Build a FusionAnnotations whose `skipped` set contains the
        // node `add1` — the op that produces tensor `a`. Even though
        // `a` passes is_planner_eligible (fully-known shape, not graph
        // I/O, ≥ 4 KiB), it must be filtered out.
        let mut fusion = FusionAnnotations::default();
        fusion.skipped.insert("add1".to_string());

        let plan = memory_planner(&graph, &shapes, &last, &first, &fusion);
        let c = plan.classification_counts();

        assert_eq!(
            c.arena_eligible, 0,
            "fusion-skipped tensor must not count as arena_eligible"
        );
        assert_eq!(c.arena_assigned, 0);
        // `a` (fusion-skipped) + `b` (graph output) = 2 pool-classified.
        assert_eq!(c.pool_classified, 2);
    }

    /// Byte-identical MemoryPlan across back-to-back calls with the
    /// same inputs — Phase 4 determinism contract.
    #[test]
    fn determinism_snapshot() {
        let graph = build_two_add_graph();
        let shapes = base_shapes_for_two_add();
        let mut first = HashMap::new();
        first.insert("a".into(), 0);
        first.insert("b".into(), 1);
        let mut last = HashMap::new();
        last.insert("a".into(), 1);

        let plan_1 = memory_planner(
            &graph,
            &shapes,
            &last,
            &first,
            &FusionAnnotations::default(),
        );
        let plan_2 = memory_planner(
            &graph,
            &shapes,
            &last,
            &first,
            &FusionAnnotations::default(),
        );

        assert_eq!(plan_1.arena_bytes, plan_2.arena_bytes);
        assert_eq!(plan_1.slot_count, plan_2.slot_count);
        assert_eq!(plan_1.slot_offsets_bytes, plan_2.slot_offsets_bytes);
        // Debug-format snapshot equivalence — guards anything else in
        // the struct (including classification + slot_assignments).
        assert_eq!(format!("{:?}", plan_1), format!("{:?}", plan_2));
    }
}
