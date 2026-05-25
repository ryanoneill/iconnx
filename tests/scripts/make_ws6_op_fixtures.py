#!/usr/bin/env python3
"""Generate per-op ONNX test fixtures for WS-6 per-op GPU<->ORT parity tests.

Run once to populate tests/fixtures/ws6_op/. Re-run to regenerate.
Outputs are committed to git (small, hand-built models).

Requires: onnx>=1.14, numpy.
"""
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUTDIR = os.path.join(
    os.path.dirname(__file__), "..", "fixtures", "ws6_op"
)
os.makedirs(OUTDIR, exist_ok=True)

OPSET_14 = [helper.make_opsetid("", 14)]


def save(model, name):
    onnx.checker.check_model(model)
    path = os.path.join(OUTDIR, name)
    onnx.save(model, path)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# 1. Identity 2×2
# ---------------------------------------------------------------------------
def make_identity_2x2():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 2])
    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph([node], "identity_2x2", [x], [y])
    model = helper.make_model(graph, opset_imports=OPSET_14)
    save(model, "identity_2x2.onnx")


# ---------------------------------------------------------------------------
# 2. Relu 1×4
# ---------------------------------------------------------------------------
def make_relu_1x4():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Relu", ["x"], ["y"])
    graph = helper.make_graph([node], "relu_1x4", [x], [y])
    model = helper.make_model(graph, opset_imports=OPSET_14)
    save(model, "relu_1x4.onnx")


# ---------------------------------------------------------------------------
# 3. Conv dynamic dims — input [N,1,H,W] with dim_param
# ---------------------------------------------------------------------------
def make_conv_dyn():
    np.random.seed(0)
    kernel_vals = np.full((1, 1, 3, 3), 1.0 / 9.0, dtype=np.float32)
    kernel_init = numpy_helper.from_array(kernel_vals, name="kernel")

    # Build input ValueInfo with dynamic spatial dims via dim_param.
    x_type = onnx.TypeProto()
    tensor_type = x_type.tensor_type
    tensor_type.elem_type = TensorProto.FLOAT
    shape = tensor_type.shape
    # dim[0]: N (dynamic)
    d0 = shape.dim.add()
    d0.dim_param = "N"
    # dim[1]: C=1 (concrete)
    d1 = shape.dim.add()
    d1.dim_value = 1
    # dim[2]: H (dynamic)
    d2 = shape.dim.add()
    d2.dim_param = "H"
    # dim[3]: W (dynamic)
    d3 = shape.dim.add()
    d3.dim_param = "W"
    x_vi = onnx.ValueInfoProto()
    x_vi.name = "x"
    x_vi.type.CopyFrom(x_type)

    # Output also has dynamic spatial dims (same as input, given pads/strides).
    y_type = onnx.TypeProto()
    y_tensor_type = y_type.tensor_type
    y_tensor_type.elem_type = TensorProto.FLOAT
    y_shape = y_tensor_type.shape
    yd0 = y_shape.dim.add(); yd0.dim_param = "N"
    yd1 = y_shape.dim.add(); yd1.dim_value = 1
    yd2 = y_shape.dim.add(); yd2.dim_param = "H"
    yd3 = y_shape.dim.add(); yd3.dim_param = "W"
    y_vi = onnx.ValueInfoProto()
    y_vi.name = "y"
    y_vi.type.CopyFrom(y_type)

    node = helper.make_node(
        "Conv",
        inputs=["x", "kernel"],
        outputs=["y"],
        pads=[1, 1, 1, 1],
        strides=[1, 1],
        group=1,
    )
    graph = helper.make_graph(
        [node], "conv_dyn", [x_vi], [y_vi], initializer=[kernel_init]
    )
    model = helper.make_model(graph, opset_imports=OPSET_14)
    save(model, "conv_dyn.onnx")


# ---------------------------------------------------------------------------
# 4. Conv concrete 1×1×8×8
# ---------------------------------------------------------------------------
def make_conv_concrete_1x1x8x8():
    np.random.seed(0)
    kernel_vals = np.full((1, 1, 3, 3), 1.0 / 9.0, dtype=np.float32)
    kernel_init = numpy_helper.from_array(kernel_vals, name="kernel")

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 8, 8])
    # With pads=[1,1,1,1] and stride=[1,1], spatial dims are preserved.
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 8, 8])

    node = helper.make_node(
        "Conv",
        inputs=["x", "kernel"],
        outputs=["y"],
        pads=[1, 1, 1, 1],
        strides=[1, 1],
        group=1,
    )
    graph = helper.make_graph(
        [node], "conv_concrete_1x1x8x8", [x], [y], initializer=[kernel_init]
    )
    model = helper.make_model(graph, opset_imports=OPSET_14)
    save(model, "conv_concrete_1x1x8x8.onnx")


# ---------------------------------------------------------------------------
# 5. Conv depthwise group=8
# ---------------------------------------------------------------------------
def make_conv_depthwise_g8():
    np.random.seed(0)
    kernel_vals = np.random.randn(8, 1, 3, 3).astype(np.float32)
    kernel_init = numpy_helper.from_array(kernel_vals, name="kernel")

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8, 16, 16])
    # group=8, kernel [8,1,3,3]: out_channels=8, spatial preserved with pads.
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 16, 16])

    node = helper.make_node(
        "Conv",
        inputs=["x", "kernel"],
        outputs=["y"],
        pads=[1, 1, 1, 1],
        strides=[1, 1],
        group=8,
    )
    graph = helper.make_graph(
        [node], "conv_depthwise_g8", [x], [y], initializer=[kernel_init]
    )
    model = helper.make_model(graph, opset_imports=OPSET_14)
    save(model, "conv_depthwise_g8.onnx")


# ---------------------------------------------------------------------------
# 6. Conv grouped group=2
# ---------------------------------------------------------------------------
def make_conv_grouped_g2():
    np.random.seed(0)
    kernel_vals = np.random.randn(16, 4, 3, 3).astype(np.float32)
    kernel_init = numpy_helper.from_array(kernel_vals, name="kernel")

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8, 16, 16])
    # group=2, kernel [16,4,3,3]: out_channels=16, spatial preserved with pads.
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 16, 16, 16])

    node = helper.make_node(
        "Conv",
        inputs=["x", "kernel"],
        outputs=["y"],
        pads=[1, 1, 1, 1],
        strides=[1, 1],
        group=2,
    )
    graph = helper.make_graph(
        [node], "conv_grouped_g2", [x], [y], initializer=[kernel_init]
    )
    model = helper.make_model(graph, opset_imports=OPSET_14)
    save(model, "conv_grouped_g2.onnx")


# ---------------------------------------------------------------------------
# 7. Conv grouped group=2 with bias
# ---------------------------------------------------------------------------
def make_conv_grouped_g2_biased():
    np.random.seed(0)
    kernel_vals = np.random.randn(16, 4, 3, 3).astype(np.float32)
    kernel_init = numpy_helper.from_array(kernel_vals, name="kernel")
    bias_vals = np.random.randn(16).astype(np.float32)
    bias_init = numpy_helper.from_array(bias_vals, name="bias")

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8, 16, 16])
    # Same as conv_grouped_g2 + bias; output shape unchanged.
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 16, 16, 16])

    node = helper.make_node(
        "Conv",
        inputs=["x", "kernel", "bias"],
        outputs=["y"],
        pads=[1, 1, 1, 1],
        strides=[1, 1],
        group=2,
    )
    graph = helper.make_graph(
        [node], "conv_grouped_g2_biased", [x], [y],
        initializer=[kernel_init, bias_init]
    )
    model = helper.make_model(graph, opset_imports=OPSET_14)
    save(model, "conv_grouped_g2_biased.onnx")


# ---------------------------------------------------------------------------
# 8. Clip opset-11 input form (min/max as inputs)
# ---------------------------------------------------------------------------
def make_clip_input_form_relu6():
    min_init = numpy_helper.from_array(
        np.array(0.0, dtype=np.float32), name="clip_min"
    )
    max_init = numpy_helper.from_array(
        np.array(6.0, dtype=np.float32), name="clip_max"
    )

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 16])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 16])

    node = helper.make_node(
        "Clip",
        inputs=["x", "clip_min", "clip_max"],
        outputs=["y"],
    )
    graph = helper.make_graph(
        [node], "clip_input_form_relu6", [x], [y],
        initializer=[min_init, max_init]
    )
    model = helper.make_model(graph, opset_imports=OPSET_14)
    save(model, "clip_input_form_relu6.onnx")


# ---------------------------------------------------------------------------
# 9. Clip attribute form
#
# The Clip min/max ATTRIBUTE form was removed in opset 11 (it became inputs).
# At opset 14 there are no min/max attributes on Clip, so this fixture uses
# opset 6 — the last opset version that includes the attribute form — to
# demonstrate the alternative calling convention that the per-op parity
# harness must recognise.
# ---------------------------------------------------------------------------
def make_clip_attr_form():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 16])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 16])

    node = helper.make_node(
        "Clip",
        inputs=["x"],
        outputs=["y"],
        min=0.0,
        max=6.0,
    )
    graph = helper.make_graph([node], "clip_attr_form", [x], [y])
    # opset 6: last version with attribute-form min/max on Clip.
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 6)]
    )
    save(model, "clip_attr_form.onnx")


# ---------------------------------------------------------------------------
# 10. ConvTranspose stride=2
# ---------------------------------------------------------------------------
def make_convtranspose2d_s2():
    np.random.seed(0)
    kernel_vals = np.random.randn(4, 2, 2, 2).astype(np.float32)
    kernel_init = numpy_helper.from_array(kernel_vals, name="kernel")

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 8, 8])
    # kernel [4,2,2,2], strides=[2,2], pads=[0,0,0,0]:
    # out_channels=2, H_out=8*2+(2-1)-0=17? Actually ConvTranspose:
    # H_out = (H_in - 1) * stride - 2*pad + kernel = (8-1)*2 + 2 = 16
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 16, 16])

    node = helper.make_node(
        "ConvTranspose",
        inputs=["x", "kernel"],
        outputs=["y"],
        strides=[2, 2],
        pads=[0, 0, 0, 0],
    )
    graph = helper.make_graph(
        [node], "convtranspose2d_s2", [x], [y], initializer=[kernel_init]
    )
    model = helper.make_model(graph, opset_imports=OPSET_14)
    save(model, "convtranspose2d_s2.onnx")


# ---------------------------------------------------------------------------
# 11. HardSigmoid alpha=0.2 beta=0.5
# ---------------------------------------------------------------------------
def make_hardsigmoid_a02_b05_1x64():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 64])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 64])

    node = helper.make_node(
        "HardSigmoid",
        inputs=["x"],
        outputs=["y"],
        alpha=0.2,
        beta=0.5,
    )
    graph = helper.make_graph([node], "hardsigmoid_a02_b05_1x64", [x], [y])
    model = helper.make_model(graph, opset_imports=OPSET_14)
    save(model, "hardsigmoid_a02_b05_1x64.onnx")


# ---------------------------------------------------------------------------
# 12. HardSwish (no attributes)
# ---------------------------------------------------------------------------
def make_hardswish_1x64():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 64])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 64])

    node = helper.make_node("HardSwish", inputs=["x"], outputs=["y"])
    graph = helper.make_graph([node], "hardswish_1x64", [x], [y])
    model = helper.make_model(graph, opset_imports=OPSET_14)
    save(model, "hardswish_1x64.onnx")


# ---------------------------------------------------------------------------
# 13. BatchNormalization 1×3×8×8
# ---------------------------------------------------------------------------
def make_batchnorm_1x3x8x8():
    np.random.seed(0)
    scale_vals = np.random.randn(3).astype(np.float32)
    b_vals = np.random.randn(3).astype(np.float32)
    mean_vals = np.random.randn(3).astype(np.float32)
    var_vals = (np.abs(np.random.randn(3)) + 0.5).astype(np.float32)

    scale_init = numpy_helper.from_array(scale_vals, name="scale")
    b_init = numpy_helper.from_array(b_vals, name="B")
    mean_init = numpy_helper.from_array(mean_vals, name="mean")
    var_init = numpy_helper.from_array(var_vals, name="var")

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 8, 8])

    node = helper.make_node(
        "BatchNormalization",
        inputs=["x", "scale", "B", "mean", "var"],
        outputs=["y"],
        epsilon=1e-5,
    )
    graph = helper.make_graph(
        [node], "batchnorm_1x3x8x8", [x], [y],
        initializer=[scale_init, b_init, mean_init, var_init]
    )
    model = helper.make_model(graph, opset_imports=OPSET_14)
    save(model, "batchnorm_1x3x8x8.onnx")


# ---------------------------------------------------------------------------
# 14. Tile 1×3 repeats=[2,2]
# ---------------------------------------------------------------------------
def make_tile_1x3_r2x2():
    repeats_init = numpy_helper.from_array(
        np.array([2, 2], dtype=np.int64), name="repeats"
    )

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 6])

    node = helper.make_node("Tile", inputs=["x", "repeats"], outputs=["y"])
    graph = helper.make_graph(
        [node], "tile_1x3_r2x2", [x], [y], initializer=[repeats_init]
    )
    model = helper.make_model(graph, opset_imports=OPSET_14)
    save(model, "tile_1x3_r2x2.onnx")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    make_identity_2x2()
    make_relu_1x4()
    make_conv_dyn()
    make_conv_concrete_1x1x8x8()
    make_conv_depthwise_g8()
    make_conv_grouped_g2()
    make_conv_grouped_g2_biased()
    make_clip_input_form_relu6()
    make_clip_attr_form()
    make_convtranspose2d_s2()
    make_hardsigmoid_a02_b05_1x64()
    make_hardswish_1x64()
    make_batchnorm_1x3x8x8()
    make_tile_1x3_r2x2()
