#!/usr/bin/env python3

import tvm
from tvm import relay
import tflite
from tvm.relay.build_module import bind_params_by_name
import argparse
import random
import json
from tvm.relay.frontend.tensorflow_parser import TFParser
import tensorflow as tf
import onnx
import onnxruntime
import os

random.seed(10)
INPUT_SHAPE = [1, 8, 8, 8]

def get_onnx_input_dict(model_path):
    shape_dict = {}
    dtype_dict = {}
    interpret_model = onnxruntime.InferenceSession(model_path)
    input_details = interpret_model.get_inputs()
    for input_detail in input_details:
        input_name = input_detail.name
        shape_dict[input_name] = tuple(
            i if isinstance(i, int) else 1 for i in input_detail.shape
        )
        dtype_dict[input_name] = "float32"

    return shape_dict, dtype_dict

def onnxruntime_optimization(onnx_model, temp_name=None):
    if temp_name is None:
        temp_name = f"temp_model_{random.random()}.onnx"
    inputs = onnx_model.graph.input
    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input

    for initializer in onnx_model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])
    onnx.save(onnx_model, temp_name)
    sess_options = onnxruntime.SessionOptions()
    # Set graph optimization level
    sess_options.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
    )

    # To enable model serialization after graph optimization set this
    sess_options.optimized_model_filepath = temp_name
    _ = onnxruntime.InferenceSession(temp_name, sess_options)
    model = onnx.load(temp_name)
    os.remove(temp_name)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=False)

    return model

def get_mod_from_tflite(model_name):
    if model_name.endswith(".tflite"):
        tflite_model = tflite.Model.GetRootAsModel(open(model_name, "rb").read(), 0)
        mod, params = relay.frontend.from_tflite(
            tflite_model,
            shape_dict={"input": INPUT_SHAPE},
            dtype_dict={"input": "float32"},
        )
        mod["main"] = bind_params_by_name(mod["main"], params)

    elif model_name.endswith(".pb"):
        tf_compat_v1 = tf.compat.v1
        with tf_compat_v1.gfile.GFile(model_name, "rb") as f:
            graph_def = tf_compat_v1.GraphDef()
            graph_def.ParseFromString(f.read())
        def get_graph_outputs(graph_def):
            with tf_compat_v1.Graph().as_default() as graph:
                tf_compat_v1.import_graph_def(graph_def)
            ops = graph.get_operations()
            outputs_set = set(ops)
            for op in ops:
                if len(op.inputs) != 0 or op.type == "Const":
                    for input_tensor in op.inputs:
                        if input_tensor.op in outputs_set:
                            outputs_set.remove(input_tensor.op)
            return ["/".join(i.name.split("/")[1:]) for i in list(outputs_set)]
        mod, params = relay.frontend.from_tensorflow(graph_def, layout=None, shape=INPUT_SHAPE, outputs=get_graph_outputs(graph_def))
        mod["main"] = bind_params_by_name(mod["main"], params)
        
    elif model_name.endswith(".json"):
        mod_json = json.load(open(model_name, "r"))
        mod = tvm.ir.load_json(mod_json)
    elif model_name.endswith(".onnx"):
        shape_dict, dtype_dict = get_onnx_input_dict(model_name)
        onnx_model = onnx.load(model_name)
        onnx_model = onnxruntime_optimization(onnx_model=onnx_model)
        mod, params = relay.frontend.from_onnx(onnx_model, shape=shape_dict, dtype=dtype_dict)
    print(mod["main"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="")
    parser.add_argument("--width", "-w", type=int, default=8, help="inp_width")
    parser.add_argument("--height", "-ht", type=int, default=8, help="inp_height")
    parser.add_argument("--channels", "-c", type=int, default=8, help="inp_channels")
    args = parser.parse_args()

    get_mod_from_tflite(args.model)
