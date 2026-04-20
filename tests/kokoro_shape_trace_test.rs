//! Regression test for the Phase 3 Commit 3 shape-inference bug on
//! Kokoro.
//!
//! Loads the real Kokoro v1.0 ONNX, builds an `OptimizableGraph` exactly
//! like `kokoro_gpu_fusion_test::setup_kokoro_gpu_executor` does, runs
//! the CPU-side pipeline passes, and asserts that the inferred shape at
//! `/encoder/text_encoder/Transpose_output_0` matches the runtime-truth
//! shape `[Some(1), Some(512), None]` (batch=1, channels=512, seq_len
//! symbolic).
//!
//! Previously this inference produced `[None, Some(512), Some(2),
//! Some(256)]` (rank 4, batch=512) because `infer_gather` treated the
//! empty `Shape::new()` of an unseeded graph-input `tokens` as a known
//! rank-0 scalar and fabricated an output rank of `data.rank - 1`.
//! Downstream passes (Transpose, Conv, LSTM, final Transpose_1)
//! inherited that bogus dim.
//!
//! The test also checks `/encoder/text_encoder/lstm/Transpose_1_output_0`
//! — the Y-shape `[seq_len, 2, batch, hidden]` permuted by
//! `perm=[0,2,1,3]` — which in the original bug report surfaced as
//! `[None, Some(512), Some(2), Some(256)]` and must now be
//! `[None, Some(1), Some(2), Some(256)]` (batch=1).
//!
//! This test is `#[ignore]`d because it requires the full Kokoro ONNX
//! (~325 MB) in the workspace. Run it manually with:
//!   cargo test --features cuda --release --test kokoro_shape_trace_test \
//!       -- --ignored

#![cfg(feature = "cuda")]

use std::path::Path;

use iconnx::ir::passes::{
    constant_folding::constant_folding, dce::dead_code_elimination,
    shape_inference::tensor_shapes_from_graph, topo_sort::topo_sort,
};
use iconnx::ir::OptimizableGraphBuilder;
use iconnx::onnx_parser::OnnxParser;

fn build_kokoro_graph_same_as_executor() -> iconnx::ir::OptimizableGraph {
    let model_path = Path::new("kokoro-v1.0.onnx");
    assert!(
        model_path.exists(),
        "kokoro-v1.0.onnx must be reachable from the working directory \
         (symlink or copy it in). A silent skip here would hide real \
         shape-inference regressions — see Phase 3 Commit 3's Gather bug."
    );
    let model = OnnxParser::parse_file(model_path).expect("parse kokoro-v1.0.onnx");
    let weights = model.extract_weights();
    let graph = model.computation_graph();
    let nodes = graph.nodes();

    let mut b = OptimizableGraphBuilder::new();
    for (name, tensor) in &weights {
        b.add_initializer(name.clone(), tensor.clone());
    }
    for (idx, node) in nodes.iter().enumerate() {
        let fallback = format!("node_{}", idx);
        let node_name = node
            .outputs
            .first()
            .map(|s| s.as_str())
            .unwrap_or(&fallback)
            .to_string();
        b.add_node(
            node_name,
            node.op_type.clone(),
            node.inputs.clone(),
            node.outputs.clone(),
            node.attributes.clone(),
        );
    }
    // Seed graph inputs exactly as the updated production loader does.
    for (name, shape) in model.input_shapes() {
        b.add_input(name, shape);
    }
    b.add_output("audio".into());
    b.build()
}

#[test]
#[ignore]
fn text_encoder_transpose_shape_matches_runtime_truth() {
    let graph = build_kokoro_graph_same_as_executor();
    let graph = topo_sort(graph).expect("topo");
    let graph = constant_folding(graph);
    let graph = dead_code_elimination(graph);
    let shapes = tensor_shapes_from_graph(&graph);

    // Node 1 of the text_encoder: Transpose of embedding Gather output,
    // perm=[0,2,1]. Runtime tensor is `[1, 512, seq_len]`. Post-fix
    // inferred shape must have `Some(512)` at dim 1 (channels axis),
    // not at any other axis.
    let transpose_0 = shapes
        .get("/encoder/text_encoder/Transpose_output_0")
        .cloned()
        .expect("Transpose_output_0 must be in shape map");
    assert_eq!(
        transpose_0,
        vec![Some(1), Some(512), None],
        "text_encoder Transpose_output_0 shape regressed — this is the \
         canary for the Gather-unknown-indices bug. Expected \
         [Some(1), Some(512), None] (batch=1, channels=512, seq_len=?). \
         If you see a rank other than 3 here, the Gather inferencer is \
         once again fabricating a rank from an empty indices shape. If \
         `Some(512)` moved to dim 0 or dim 2, perm handling is broken."
    );

    // Downstream: the LSTM Y-output transpose with perm=[0,2,1,3].
    // Y has shape `[seq_len, num_directions, batch, hidden] =
    // [None, 2, 1, 256]`, so perm=[0,2,1,3] gives
    // `[None, 1, 2, 256]`. The original bug report surfaced this as
    // `[None, Some(512), Some(2), Some(256)]`.
    let lstm_transpose = shapes
        .get("/encoder/text_encoder/lstm/Transpose_1_output_0")
        .cloned()
        .expect("lstm Transpose_1_output_0 must be in shape map");
    assert_eq!(
        lstm_transpose,
        vec![None, Some(1), Some(2), Some(256)],
        "lstm Transpose_1_output_0 shape regressed. Expected \
         [None, Some(1), Some(2), Some(256)] (seq_len=?, batch=1, \
         num_directions=2, hidden=256). The original Phase 3 Commit 3 \
         bug had this as [None, Some(512), Some(2), Some(256)] — a \
         non-None but wrong batch dim baked in from the token-embedding \
         Gather's fabricated rank."
    );

    // Canary for the originating Gather — must be rank 3, not rank 1.
    let gather = shapes
        .get("/encoder/text_encoder/embedding/Gather_output_0")
        .cloned()
        .expect("embedding Gather output must be in shape map");
    assert_eq!(
        gather.len(),
        3,
        "embedding Gather output rank regressed. With data [178, 512] \
         and indices [1, ?], ONNX Gather axis=0 yields rank \
         indices.rank + data.rank - 1 = 2 + 2 - 1 = 3. Got: {:?}",
        gather
    );
}
