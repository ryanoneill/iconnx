//! WS-2 GPT-2 M2.7.c blocker: `upload_tensor` panicked on Bool
//! tensors because the docstring promised `Bool → Float32` cast but
//! the implementation called the f32-only `as_slice()` accessor
//! (which panics on non-Float32). The fix performs explicit
//! `bool → f32` conversion (false → 0.0, true → 1.0) and uploads as
//! Float32, matching iconnx's existing f32-as-bool convention used
//! by comparison-op outputs.
//!
//! GPT-2's causal-mask Constant
//! (`value=TENSOR(dtype=BOOL, dims=[1,1,1024,1024])`) was the
//! triggering surface. This test runs a minimal synthetic
//! `Constant(Bool) → Identity passthrough` graph end-to-end so the
//! regression coverage doesn't depend on the 500MB GPT-2 model.

#![cfg(feature = "cuda")]

use std::collections::HashMap;

use iconnx::attributes::NodeAttributes;
use iconnx::cuda::inference::GpuGraphExecutor;
use iconnx::tensor::Tensor;

#[test]
#[ignore = "requires CUDA GPU"]
fn upload_tensor_bool_initializer_uploads_as_f32() {
    let mut executor = GpuGraphExecutor::new().expect("executor");

    // Constant node carrying a 2x2 BOOL value attribute. This is the
    // exact ONNX shape GPT-2 uses for its causal mask (just at a
    // smaller size). Pre-fix: parse_tensor_proto succeeds (since the
    // BOOL parse fix `e2819af`), then upload_tensor panics on
    // `as_slice()`. Post-fix: upload_tensor casts to Float32.
    let mask = Tensor::from_vec_bool(vec![true, false, true, false], vec![2, 2]);

    let mut const_attrs = NodeAttributes::new();
    const_attrs.add_tensor("value".into(), mask);
    executor.add_node(
        "const_mask",
        "Constant",
        Vec::<&str>::new(),
        vec!["mask"],
        const_attrs,
    );
    // Identity-style passthrough so the mask becomes a graph output —
    // unconsumed constants get pruned.
    executor.add_node(
        "passthrough",
        "Identity",
        vec!["mask"],
        vec!["out"],
        NodeAttributes::new(),
    );

    executor.compile().expect("compile");

    let outputs = executor
        .run(HashMap::new(), vec!["out"])
        .expect("run Bool-constant graph");

    let out = outputs.get("out").expect("out present");
    // Bool tensor uploaded as Float32: true → 1.0, false → 0.0.
    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(out.as_slice(), vec![1.0_f32, 0.0, 1.0, 0.0]);
}
