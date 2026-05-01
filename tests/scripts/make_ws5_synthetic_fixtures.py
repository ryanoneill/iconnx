#!/usr/bin/env python3
"""Generate synthetic ONNX models for WS-5 validate_model_test.rs.

Run once to populate tests/fixtures/ws5_synthetic/. Re-run to regenerate.
Outputs are committed to git (small, hand-built models).

Requires: onnx, numpy.
"""
import os

import numpy as np  # noqa: F401  (kept for parity with future shape-init tests)
import onnx
from onnx import TensorProto, helper

OUTDIR = os.path.join(
    os.path.dirname(__file__), "..", "fixtures", "ws5_synthetic"
)
os.makedirs(OUTDIR, exist_ok=True)


def save(model, name):
    path = os.path.join(OUTDIR, f"{name}.onnx")
    onnx.save(model, path)
    print(f"wrote {path}")


# 1. minimal_add: A(f32) + B(f32) -> C
def make_minimal_add():
    a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [3])
    b = helper.make_tensor_value_info("B", TensorProto.FLOAT, [3])
    c = helper.make_tensor_value_info("C", TensorProto.FLOAT, [3])
    node = helper.make_node("Add", ["A", "B"], ["C"])
    # Add a small fp32 initializer so validate_model has something to walk
    # (the heuristics for estimated_peak_bytes assume initializer_count > 0
    # in the common case).
    init = helper.make_tensor(
        name="bias", data_type=TensorProto.FLOAT, dims=[3], vals=[0.0, 0.0, 0.0]
    )
    graph = helper.make_graph(
        [node], "minimal_add", [a, b], [c], initializer=[init]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    save(model, "minimal_add")


# 2. unsupported_op: a custom CustomOp node (won't be in OP_SUPPORT_TABLE).
def make_unsupported_op():
    a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [3])
    c = helper.make_tensor_value_info("C", TensorProto.FLOAT, [3])
    node = helper.make_node("CustomOpFoo", ["A"], ["C"])
    graph = helper.make_graph([node], "unsupported_op", [a], [c])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    save(model, "unsupported_op")


# 3. opset_out_of_range: opset 99 (above any plausible MAX).
def make_opset_out_of_range():
    a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [3])
    b = helper.make_tensor_value_info("B", TensorProto.FLOAT, [3])
    c = helper.make_tensor_value_info("C", TensorProto.FLOAT, [3])
    node = helper.make_node("Add", ["A", "B"], ["C"])
    graph = helper.make_graph([node], "opset_oor", [a, b], [c])
    # `helper.make_model` validates ops against a known opset; we have to
    # sidestep that by hand-building the ModelProto and writing it directly.
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    # Now overwrite the opset version to 99 post-hoc — onnx.checker will
    # reject it but we just want the bytes for our parser to read.
    del model.opset_import[:]
    op = model.opset_import.add()
    op.domain = ""
    op.version = 99
    save(model, "opset_out_of_range")


# 4. unknown_opset_domain: model with a com.example.custom domain import.
def make_unknown_opset_domain():
    a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [3])
    b = helper.make_tensor_value_info("B", TensorProto.FLOAT, [3])
    c = helper.make_tensor_value_info("C", TensorProto.FLOAT, [3])
    node = helper.make_node("Add", ["A", "B"], ["C"])
    graph = helper.make_graph([node], "unknown_domain", [a, b], [c])
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("com.example.custom", 1),
        ],
    )
    save(model, "unknown_opset_domain")


# 5. bf16_with_neg_inf: BF16 initializer with -inf bit pattern.
def make_bf16_with_neg_inf():
    # bf16 -inf bits = 0xFF80 (little-endian raw_data: [0x80, 0xFF]).
    # We bypass `helper.make_tensor` (which round-trips through float)
    # and write raw_data directly so the bf16 bit pattern lands intact.
    init = TensorProto()
    init.name = "bad_const"
    init.data_type = TensorProto.BFLOAT16
    init.dims.append(1)
    init.raw_data = bytes([0x80, 0xFF])  # bf16 -inf
    a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1])
    c = helper.make_tensor_value_info("C", TensorProto.FLOAT, [1])
    # Trivial Identity to make a valid graph; the initializer is the point.
    node = helper.make_node("Identity", ["A"], ["C"])
    graph = helper.make_graph(
        [node], "bf16_inf", [a], [c], initializer=[init]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    save(model, "bf16_with_neg_inf")


# 6. garbage: just write 100 random bytes that won't parse.
def make_garbage():
    path = os.path.join(OUTDIR, "garbage.onnx")
    with open(path, "wb") as f:
        f.write(os.urandom(100))
    print(f"wrote {path}")


# 7. fp64_initializer: a FLOAT64 initializer to exercise the
# `ModelWarning::DTypeDowncast` path in `validate_model`. iconnx maps
# `Tensor::Float64 → DType::Float32` for capability-discovery purposes; the
# warning surfaces the lossy mapping so consumers can detect it.
def make_fp64_initializer():
    a = helper.make_tensor_value_info("A", TensorProto.DOUBLE, [3])
    c = helper.make_tensor_value_info("C", TensorProto.DOUBLE, [3])
    init = helper.make_tensor(
        name="bias_f64",
        data_type=TensorProto.DOUBLE,
        dims=[3],
        vals=[0.0, 0.0, 0.0],
    )
    node = helper.make_node("Add", ["A", "bias_f64"], ["C"])
    graph = helper.make_graph(
        [node], "fp64_init", [a], [c], initializer=[init]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    save(model, "fp64_initializer")


# 7. int64_input_fp32_init: INT64 input + FP32 weight initializer.
# Proves IO-walk contributes a dtype not present in initializers.
def make_int64_input_fp32_init():
    a = helper.make_tensor_value_info("A", TensorProto.INT64, [3])
    c = helper.make_tensor_value_info("C", TensorProto.FLOAT, [3])
    weight = helper.make_tensor(
        name="weight", data_type=TensorProto.FLOAT, dims=[3],
        vals=[1.0, 2.0, 3.0],
    )
    # Identity on INT64 input (we only care about dtypes for capability discovery).
    # Use Cast to convert INT64 to FP32 so the FP32 output validates.
    cast_node = helper.make_node("Cast", ["A"], ["A_f32"], to=TensorProto.FLOAT)
    add_node = helper.make_node("Add", ["A_f32", "weight"], ["C"])
    graph = helper.make_graph(
        [cast_node, add_node], "int64_in_fp32_init",
        [a], [c], initializer=[weight]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    save(model, "int64_input_fp32_init")


if __name__ == "__main__":
    make_minimal_add()
    make_unsupported_op()
    make_opset_out_of_range()
    make_unknown_opset_domain()
    make_bf16_with_neg_inf()
    make_garbage()
    make_fp64_initializer()
    make_int64_input_fp32_init()
