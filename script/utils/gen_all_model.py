import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
from main import gen_tflite
from model_list import model_choices
import gen_model as gm

data_type = "float32"

not_test_list = ["right_shift", "squeeze", "tan"]


def gen_all_model():
    out_dir = base_dir + "/models/examine/"
    for m in model_choices:
        if m == "logical_not" or m == "logical_or" or m == "reduce_any":
            data_type = "bool"
        elif m == "right_shift":
            data_type = "int32"
        else:
            data_type = "float32"

        if m in not_test_list:
            continue
        else:
            args_list = [8, 8, 8, 8, data_type, 3, 1, "same", 1, 2]
            model = gm.call_gen_model(args_list, m)
            gen_tflite(model, m, out_dir, data_type, False, None)
            print("%s is done" % m)


gen_all_model()
