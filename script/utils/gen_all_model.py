#!/bin/env python3
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
from main import gen_tflite
from model_list import model_choices
import gen_model as gm

data_type = "float32"

not_test_list = ["right_shift", "squeeze", "tan", "reduce_all"]


def gen_all_model():
    out_dir = base_dir + "/models/dump/"
    for m in model_choices:
        print("\033[1;32mRunning model: \033[1;34m" + m + "\033[0m\n")
        bool_type_op = ["logical_not", "logical_or", "reduce_any", "reduce_all"]
        int32_type_op = ["right_shift"]

        if m in bool_type_op:
            data_type = "bool"
        elif m in int32_type_op:
            data_type = "int32"
        else:
            data_type = "float32"

        if m in not_test_list:
            continue
        else:
            args_list = [1, 8, 8, 8, 8, data_type, 3, 1, "same", 1, 2]
            if m == "batch_to_space":
                args_list[0] = 4
            model = gm.call_gen_model(args_list, m)
            gen_tflite(model, m, out_dir, data_type, False, None)
            print(
                "\033[1;32mGen model: \033[1;34m"
                + m
                + "\033[1;32m successfully!\033[0m\n"
            )


gen_all_model()
