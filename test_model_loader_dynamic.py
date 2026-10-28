
import onnx
from onnx import TensorProto, helper

from models.loader import load_model


def test_dynamic_input_shape(tmp_path):
    input_info = helper.make_tensor_value_info(
        "input",
        TensorProto.FLOAT,
        ["batch_size", 3, 224, 224],
    )

    output_info = helper.make_tensor_value_info(
        "output",
        TensorProto.FLOAT,
        ["batch_size", 3, 224, 224],
    )

    identity = helper.make_node(
        "Identity",
        ["input"],
        ["output"],
    )

    graph = helper.make_graph(
        [identity],
        "dynamic_shape_test",
        [input_info],
        [output_info],
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18)],
    )
    model.ir_version = 10

    model_path = tmp_path / "dynamic_model.onnx"
    onnx.save(model, str(model_path))

    spec = load_model(str(model_path))

    assert spec.input_shapes["input"][0] == "batch_size"
    assert spec.output_shapes["output"][0] == "batch_size"
