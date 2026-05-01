#!/usr/bin/env python3
"""Generate harness self-test fixtures for tests/conformance_data_self_test/."""
import os
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUTDIR = os.path.join(
    os.path.dirname(__file__), "..", "conformance_data_self_test"
)


def make_test_pass():
    a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [3])
    b = helper.make_tensor_value_info("B", TensorProto.FLOAT, [3])
    c = helper.make_tensor_value_info("C", TensorProto.FLOAT, [3])
    node = helper.make_node("Add", ["A", "B"], ["C"])
    graph = helper.make_graph([node], "self_test_pass", [a, b], [c])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    test_dir = os.path.join(OUTDIR, "test_pass")
    os.makedirs(os.path.join(test_dir, "test_data_set_0"), exist_ok=True)
    onnx.save(model, os.path.join(test_dir, "model.onnx"))

    # Inputs and expected outputs.
    a_val = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b_val = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    c_val = a_val + b_val

    onnx.save_tensor(
        numpy_helper.from_array(a_val, name="A"),
        os.path.join(test_dir, "test_data_set_0", "input_0.pb"),
    )
    onnx.save_tensor(
        numpy_helper.from_array(b_val, name="B"),
        os.path.join(test_dir, "test_data_set_0", "input_1.pb"),
    )
    onnx.save_tensor(
        numpy_helper.from_array(c_val, name="C"),
        os.path.join(test_dir, "test_data_set_0", "output_0.pb"),
    )
    print(f"wrote {test_dir}")


def make_test_unsupported_op():
    a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [3])
    c = helper.make_tensor_value_info("C", TensorProto.FLOAT, [3])
    node = helper.make_node("CustomOpBar", ["A"], ["C"])
    graph = helper.make_graph([node], "self_test_unsup", [a], [c])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    test_dir = os.path.join(OUTDIR, "test_unsupported_op")
    os.makedirs(os.path.join(test_dir, "test_data_set_0"), exist_ok=True)
    onnx.save(model, os.path.join(test_dir, "model.onnx"))

    # Even with the unsupported op, the harness needs valid input/output PBs
    # to exercise the load path.
    a_val = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    c_val = a_val.copy()  # placeholder; harness skips before comparing.
    onnx.save_tensor(
        numpy_helper.from_array(a_val, name="A"),
        os.path.join(test_dir, "test_data_set_0", "input_0.pb"),
    )
    onnx.save_tensor(
        numpy_helper.from_array(c_val, name="C"),
        os.path.join(test_dir, "test_data_set_0", "output_0.pb"),
    )
    print(f"wrote {test_dir}")


if __name__ == "__main__":
    make_test_pass()
    make_test_unsupported_op()
