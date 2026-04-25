//! WS-1 M1.4 first piece: gpu_copy / Identity dispatch is now
//! dtype-polymorphic. Pre-fix, an Identity node on an Int64 tensor
//! panicked with `data_f32 called on int64 tensor` deep in gpu_copy.
//! Shape-arithmetic graphs route Int64 through Identity, so this is
//! a real (if narrow) audit-L2.1 closure.
//!
//! This test runs an Identity-on-Int64 graph end-to-end through
//! GpuGraphExecutor. The dispatch_op_multi diagnostic from `b3515b1`
//! would have surfaced any failure as
//! `Identity (output=...): data_f32 called on int64 tensor`.

#![cfg(feature = "cuda")]

use std::collections::HashMap;

use iconnx::attributes::NodeAttributes;
use iconnx::cuda::inference::GpuGraphExecutor;
use iconnx::tensor::Tensor;

#[test]
#[ignore = "requires CUDA GPU"]
fn identity_on_int64_input_passes_through_executor() {
    let mut executor = GpuGraphExecutor::new().expect("executor");
    executor
        .add_initializer(
            "x".into(),
            &Tensor::from_vec_i64(vec![10_i64, -20, 30, -40, 50], vec![5]),
        )
        .expect("init");
    executor.add_node(
        "id",
        "Identity",
        vec!["x"],
        vec!["out"],
        NodeAttributes::new(),
    );
    executor.compile().expect("compile");

    let outputs = executor
        .run(HashMap::new(), vec!["out"])
        .expect("run Identity-on-Int64");
    let out = outputs.get("out").expect("out");
    assert_eq!(out.shape(), &[5]);
    assert_eq!(out.as_slice_i64(), vec![10, -20, 30, -40, 50]);
}
