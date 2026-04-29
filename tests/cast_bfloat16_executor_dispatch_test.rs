//! WS-3.5 hotfix — executor-level Cast type-tag dispatch must accept
//! ONNX TensorProto.DataType = 16 (BFLOAT16).
//!
//! Leadline-bench gate 4 surfaced this gap: kernel-level Cast dispatch
//! arms in `cuda/ops/cast.rs` cover all WS-3.5 BF16 dtype-pairs, but
//! the executor-level type-tag → DType mapping at
//! `cuda/executor/layout.rs:152` rejected target_type=16 before any
//! kernel call. Both BERT (int64 attention_mask → BF16) and Whisper
//! (rewritten `graph_input_cast_0`) hit this. R3 wildcard-removal
//! against the FP16 surface didn't cover this dispatch tier — same
//! flavor as M3.7b's Bool-on-Expand miss, just at a different layer.
//!
//! This regression test exercises a Cast(to=16) node end-to-end
//! through `GpuGraphExecutor`, asserting both that dispatch succeeds
//! and that the output tensor is `Tensor::BFloat16`. Skipped on
//! pre-sm_80 hardware (BF16 ctor floor).

#![cfg(feature = "cuda")]

use std::collections::HashMap;

use half::bf16;
use iconnx::attributes::NodeAttributes;
use iconnx::cuda::inference::GpuGraphExecutor;
use iconnx::cuda::IconnxCudaContext;
use iconnx::tensor::Tensor;

#[test]
#[ignore = "requires CUDA GPU"]
fn executor_cast_to_bfloat16_dispatches_via_layout_rs_152() {
    // sm_80+ floor: ctx ctor would fail on BF16 upload anyway.
    let ctx = IconnxCudaContext::new().expect("ctx");
    if ctx.compute_capability().0 < 8 {
        return;
    }
    drop(ctx);

    let mut executor = GpuGraphExecutor::new().expect("executor");

    // Constant Float32 source → Cast(to=16, BFLOAT16) → output.
    // 1.5 and -2.0 are exactly representable in BF16 so the
    // round-trip via to_f32() is lossless.
    let src = Tensor::from_vec(vec![1.5_f32, -2.0, 0.0], vec![3]);

    let mut const_attrs = NodeAttributes::new();
    const_attrs.add_tensor("value".into(), src);
    executor.add_node(
        "const_src",
        "Constant",
        Vec::<&str>::new(),
        vec!["src"],
        const_attrs,
    );

    let mut cast_attrs = NodeAttributes::new();
    cast_attrs.add_int("to".to_string(), 16); // ONNX TensorProto.DataType.BFLOAT16
    executor.add_node("cast_to_bf16", "Cast", vec!["src"], vec!["out"], cast_attrs);

    executor.compile().expect("compile graph with Cast(to=16)");

    let outputs = executor.run(HashMap::new(), vec!["out"]).expect(
        "Cast(to=16) must dispatch — pre-fix this errors with 'unsupported target type 16'",
    );

    let out = outputs.get("out").expect("out present");
    assert!(
        out.is_bfloat16(),
        "Cast(to=16) must produce BFloat16 tensor; got dtype={}",
        out.dtype()
    );
    assert_eq!(out.shape(), &[3]);
    let back = out.as_slice_bf16();
    assert_eq!(back[0], bf16::from_f32(1.5));
    assert_eq!(back[1], bf16::from_f32(-2.0));
    assert_eq!(back[2], bf16::from_f32(0.0));
}
