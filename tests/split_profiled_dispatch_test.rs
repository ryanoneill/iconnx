//! WS-2 M2.5 Split: regression test for the dispatch routing fix.
//!
//! After M2.5 (`65e4ce5`) added Split, GPT-2 unprofiled runs worked
//! because Split was dead-stripped (only the logits output was
//! requested). Profiled runs failed:
//!
//!     CUDA kernel failed: dispatch_layout called with non-layout op 'Split'
//!
//! Cause: the main `Executor::run` loop had inline Split routing, but
//! the 4 instrumented variants (validation / checkpoints / logging /
//! profiling) called `dispatch_op` directly and fell through to the
//! layout dispatch's reject. The fix introduces `dispatch_op_multi` as
//! the single source of truth and uses it in all 5 paths.
//!
//! This test exercises a Split graph through `run_with_profiling`
//! specifically — the path the existing `tests/split_executor_test.rs`
//! integration tests didn't cover.

#![cfg(feature = "cuda")]

use std::collections::HashMap;

use iconnx::attributes::NodeAttributes;
use iconnx::cuda::inference::GpuGraphExecutor;
use iconnx::tensor::Tensor;

#[test]
#[ignore = "requires CUDA GPU"]
fn split_dispatches_through_run_with_profiling() {
    let mut executor = GpuGraphExecutor::new().expect("executor");

    // Same minimal Split graph as split_executor_test, but routed
    // through run_with_profiling rather than run.
    executor.add_input("x".into(), vec![Some(6), Some(2)]);
    executor
        .add_initializer(
            "split_sizes".into(),
            &Tensor::from_vec_i64(vec![1, 3, 2], vec![3]),
        )
        .expect("split sizes");

    let mut attrs = NodeAttributes::new();
    attrs.add_int("axis".into(), 0);
    executor.add_node(
        "split_node",
        "Split",
        vec!["x", "split_sizes"],
        vec!["o0", "o1", "o2"],
        attrs,
    );

    executor.compile().expect("compile");

    let mut inputs = HashMap::new();
    let x_data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    inputs.insert(
        "x".to_string(),
        Tensor::from_vec_f32(x_data, vec![6, 2]),
    );

    // Pre-fix: this returns
    // `CudaError::Kernel("dispatch_layout called with non-layout op 'Split'")`.
    let (outputs, _profile) = executor
        .run_with_profiling(inputs, vec!["o0", "o1", "o2"])
        .expect("run_with_profiling Split graph");

    let o0 = outputs.get("o0").expect("o0");
    let o1 = outputs.get("o1").expect("o1");
    let o2 = outputs.get("o2").expect("o2");
    assert_eq!(o0.shape(), &[1, 2]);
    assert_eq!(o0.as_slice(), vec![0.0, 1.0]);
    assert_eq!(o1.shape(), &[3, 2]);
    assert_eq!(o1.as_slice(), vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    assert_eq!(o2.shape(), &[2, 2]);
    assert_eq!(o2.as_slice(), vec![8.0, 9.0, 10.0, 11.0]);
}
