#!/usr/bin/env python3

import tvm
from tvm import relay
import tflite
from tvm.relay.build_module import bind_params_by_name
import argparse
import random
import json


random.seed(10)
INPUT_SHAPE = [1, 8, 8, 8]


def get_mod_from_tflite(model_name):
    if model_name.endswith(".tflite"):
        tflite_model = tflite.Model.GetRootAsModel(open(model_name, "rb").read(), 0)
        mod, params = relay.frontend.from_tflite(
            tflite_model,
            shape_dict={"input": INPUT_SHAPE},
            dtype_dict={"input": "float32"},
        )
        mod["main"] = bind_params_by_name(mod["main"], params)
    elif model_name.endswith(".json"):
        mod_json = json.load(open(model_name, "r"))
        mod = tvm.ir.load_json(mod_json)
    print(mod["main"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="")
    parser.add_argument("--width", "-w", type=int, default=8, help="inp_width")
    parser.add_argument("--height", "-ht", type=int, default=8, help="inp_height")
    parser.add_argument("--channels", "-c", type=int, default=8, help="inp_channels")
    args = parser.parse_args()

    get_mod_from_tflite(args.model)
