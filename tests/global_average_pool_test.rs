//! WS-2 M2.3: GlobalAveragePool through GpuGraphExecutor. ResNet-50
//! has one GAP at the classifier head reducing the [N, 2048, 7, 7]
//! feature map to [N, 2048, 1, 1] before the final Flatten + Gemm.

#![cfg(feature = "cuda")]

use std::collections::HashMap;

use iconnx::attributes::NodeAttributes;
use iconnx::cuda::inference::GpuGraphExecutor;
use iconnx::tensor::Tensor;

#[test]
#[ignore = "requires CUDA GPU"]
fn global_average_pool_4d_classifier_head() {
    let mut executor = GpuGraphExecutor::new().expect("executor");
    executor.add_input("x".into(), vec![Some(2), Some(3), Some(4), Some(4)]);
    executor.add_node(
        "gap",
        "GlobalAveragePool",
        vec!["x"],
        vec!["out"],
        NodeAttributes::new(),
    );
    executor.compile().expect("compile");

    // Each (n, c) channel filled with constant n*3+c — mean equals
    // that value, makes per-channel verification trivial.
    let mut data = Vec::with_capacity(2 * 3 * 4 * 4);
    for n in 0..2 {
        for c in 0..3 {
            for _ in 0..16 {
                data.push((n * 3 + c) as f32);
            }
        }
    }
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), Tensor::from_vec_f32(data, vec![2, 3, 4, 4]));

    let outputs = executor.run(inputs, vec!["out"]).expect("run gap");
    let out = outputs.get("out").expect("out");
    assert_eq!(out.shape(), &[2, 3, 1, 1]);
    let vals = out.as_slice();
    for n in 0..2 {
        for c in 0..3 {
            let expected = (n * 3 + c) as f32;
            let got = vals[n * 3 + c];
            assert!(
                (got - expected).abs() < 1e-5,
                "[{},{}]: expected {}, got {}",
                n,
                c,
                expected,
                got
            );
        }
    }
}
