#!/usr/bin/env python3
import argparse
import gen_model as gm
import numpy as np
import tensorflow as tf
import os

input_shape = [1, 8, 8, 8]
num_inp = 1


def gen_tflite(model, model_name, save_dir, op_type, en_quant):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model_name = save_dir + model_name

    if en_quant:
        tflite_model_name += "_quant"
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if op_type == "float32":
            converter.representative_dataset = representative_float_dataset_gen
        elif op_type == "bool":
            converter.representative_dataset = representative_bool_dataset_gen
        elif op_type == "int32":
            converter.representative_dataset = representative_int_dataset_gen
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.SELECT_TF_OPS,
            tf.lite.OpsSet.TFLITE_BUILTINS,
        ]
    tflite_model = converter.convert()
    tflite_model_name += ".tflite"
    open(tflite_model_name, "wb").write(tflite_model)
    print(tflite_model_name)


def representative_float_dataset_gen():
    input_set = []
    for _ in range(num_inp):
        input_data = np.array(np.random.random_sample(input_shape), dtype="float32")
        input_set.append(input_data)
    yield input_set


def representative_bool_dataset_gen():
    input_set = []
    for _ in range(num_inp):
        input_data = np.array(np.random.random_sample(input_shape), dtype="bool")
        input_set.append(input_data)
    yield input_set


def representative_int_dataset_gen():
    input_set = []
    for _ in range(num_inp):
        input_data = np.array(np.random.random_sample(input_shape), dtype="int32")
        input_set.append(input_data)
    yield input_set


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    model_choices = [
        "abs",
        "arg_max",
        "arg_min",
        "bilinear",
        "concat",
        "conv2d_m",
        "conv2d_trans",
        "dense",
        "depth_to_space",
        "exp",
        "gather",
        "global_average_2d",
        "greater",
        "greater_equal",
        "leaky_relu",
        "less",
        "less_equal",
        "log_softmax",
        "logical_not",
        "logical_or",
        "maximum",
        "minimum",
        "mul_add",
        "multiply",
        "not_equal",
        "power",
        "reduce_any",
        "reduce_max",
        "reduce_min",
        "reduce_prod",
        "relu",
        "relu6",
        "reshape",
        "right_shift",
        "rsqrt",
        "segment_sum",
        "sin",
        "sigmoid",
        "softmax",
        "space_to_depth",
        "square",
        "sum",
        "subtract",
        "tan",
        "tanh",
    ]
    parser.add_argument("--model", "-m", choices=model_choices, type=str, default="")
    parser.add_argument(
        "--out_dir",
        "-o",
        type=str,
        default=os.getcwd() + "/models/",
    )
    parser.add_argument("--num_inp", "-n", type=int, default=1)
    parser.add_argument("--width", "-w", type=int, default=8)
    parser.add_argument("--height", "-ht", type=int, default=8)
    parser.add_argument("--channels", "-c", type=int, default=8)
    parser.add_argument("--en_quant", "-q", type=int, default=1)
    parser.add_argument("--filter", "-f", type=int, default=8)
    parser.add_argument("--kernel", "-k", type=int, default=3)
    parser.add_argument("--stride", "-s", type=int, default=1)
    parser.add_argument("--padding", "-p", type=str, default="valid")
    args = parser.parse_args()
    num_inp = args.num_inp
    input_shape = [1, args.height, args.width, args.channels]

    if (
        args.model.lower() == "logical_not"
        or args.model.lower() == "logical_or"
        or args.model.lower() == "reduce_any"
    ):
        data_type = "bool"
    elif args.model.lower() == "right_shift":
        data_type = "int32"
    else:
        data_type = "float32"

    args_list = [
        args.height,
        args.width,
        args.channels,
        args.filter,
        args.kernel,
        args.stride,
        args.padding,
    ]

    model = gm.call_gen_model(args_list, args.model.lower(), data_type)

    model.summary()
    model_name = (
        args.model.lower()
        + "_h"
        + str(args.height)
        + "_w"
        + str(args.width)
        + "_c"
        + str(args.channels)
    )
    if "conv" in args.model.lower():
        model_name += (
            "_f"
            + str(args.filter)
            + "_k"
            + str(args.kernel)
            + "_s"
            + str(args.stride)
            + "_"
            + args.padding.lower()
        )
    gen_tflite(model, model_name, args.out_dir, data_type, args.en_quant)
