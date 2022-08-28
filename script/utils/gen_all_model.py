#!/bin/env python3
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
from quantize_utils import quantize_utils
from model_list import model_choices
import gen_model as gm

data_type = "float32"


def gen_all_model():
    out_dir = base_dir + "/models/dump/"
    for m in model_choices:
        print("\033[1;32mRunning model: \033[1;34m" + m + "\033[0m\n")
        bool_type_op = ["logical_not", "logical_or", "reduce_any"]
        en_quant = True
        if m in bool_type_op:
            data_type = "bool"
            en_quant = False
        else:
            data_type = "float32"

        args_list = [1, 8, 8, 8, 8, data_type, 3, 1, "same", 1, 2]
        if m == "batch_to_space":
            args_list = [9, 8, 8, 8, 8, data_type, 3, 1, "same", 1, 2]
        elif m == "space_to_batch":
            args_list = [1, 9, 9, 8, 8, data_type, 3, 1, "same", 1, 2]
        elif m == "squeeze":
            args_list = [1, 1, 8, 8, 8, data_type, 3, 1, "same", 1, 2]

        model = gm.call_gen_model(args_list, m)
        input_shape = (
            model.input_shape
            if isinstance(model.input_shape, list)
            else [model.input_shape]
        )
        qu = quantize_utils(input_shape, len(model.inputs))
        qu.gen_quant_tflite(
            model=model,
            model_name=m,
            save_dir=out_dir,
            op_type=data_type,
            en_quant=en_quant,
            quant_layer=True,
            nbits=8,
        )
        print(
            "\033[1;32mGen model: \033[1;34m" + m + "\033[1;32m successfully!\033[0m\n"
        )


gen_all_model()
