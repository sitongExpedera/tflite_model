#!/usr/bin/env python3
import argparse
import gen_model as gm
import numpy as np
import tensorflow as tf
import os
from utils.model_list import model_choices


input_shape = [1, 8, 8, 8]
num_inp = 1
nbits = 8
is_2d = False


def gen_tflite(model, model_name, save_dir, op_type, en_quant, quant_layer):
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
        if nbits == 8:
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                tf.lite.OpsSet.SELECT_TF_OPS,
                tf.lite.OpsSet.TFLITE_BUILTINS,
            ]
            if quant_layer:
                converter.experimental_new_converter = True
                converter.target_spec.supported_types = [tf.int8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
            tflite_model_name += "_8bits"
        elif nbits == 16:
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
                tf.lite.OpsSet.SELECT_TF_OPS,
                tf.lite.OpsSet.TFLITE_BUILTINS,
            ]
            if quant_layer:
                converter.experimental_new_converter = True
                converter.target_spec.supported_types = [tf.int16]
                converter.inference_input_type = tf.int16
                converter.inference_output_type = tf.int16
            tflite_model_name += "_16bits"
    tflite_model = converter.convert()
    tflite_model_name += ".tflite"
    open(tflite_model_name, "wb").write(tflite_model)
    print(tflite_model_name)


def representative_float_dataset_gen():
    input_set = []
    for _ in range(num_inp):
        input_data = 5 * np.array(np.random.random_sample(input_shape), dtype="float32")
        input_set.append(input_data)
    yield input_set


def representative_bool_dataset_gen():
    input_set = []
    for _ in range(num_inp):
        input_data = 5 * np.array(np.random.random_sample(input_shape), dtype="bool")
        input_set.append(input_data)
    yield input_set


def representative_int_dataset_gen():
    input_set = []
    for _ in range(num_inp):
        input_data = 5 * np.array(np.random.random_sample(input_shape), dtype="int32")
        input_set.append(input_data)
    yield input_set


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", choices=model_choices, type=str, default="")
    parser.add_argument(
        "--out_dir",
        "-o",
        type=str,
        default=os.getcwd() + "/models/test/",
    )
    parser.add_argument("--num_inp", "-n", type=int, default=1)
    parser.add_argument("--width", "-w", type=int, default=8)
    parser.add_argument("--height", "-ht", type=int, default=8)
    parser.add_argument("--channels", "-c", type=int, default=8)
    parser.add_argument("--en_quant", "-q", type=int, default=1)
    parser.add_argument("--filter", "-f", type=int, default=8)
    parser.add_argument("--kernel", "-k", type=int, default=3)
    parser.add_argument("--stride", "-s", type=int, default=1)
    parser.add_argument("--axis", "-a", type=int, default=1)
    parser.add_argument("--padding", "-p", type=str, default="valid")
    parser.add_argument("--nbits", "-nb", type=int, default=8)
    parser.add_argument("--list", "-l", action="store_true")
    parser.add_argument(
        "--quant_layer",
        "-ql",
        action="store_true",
        default=False,
        help="includes quantize and dequantize layers",
    )
    args = parser.parse_args()
    num_inp = args.num_inp
    nbits = args.nbits
    input_shape = [1, args.height, args.width, args.channels]

    if args.list:
        print("Supported models:")
        model_in_line = 1
        for model in sorted(model_choices):
            temp = format(model, "<30s")
            if model_in_line < 6:
                print(temp, end="")
                model_in_line += 1
            else:
                print(temp)
                model_in_line = 1
        print("")
        exit(0)

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

    if "2d" in args.model and "conv2d" not in args.model:
        is_2d = True

    if args.model == "minimum":
        num_inp = 2

    args_list = [
        args.height,
        args.width,
        args.channels,
        args.filter,
        data_type,
        args.kernel,
        args.stride,
        args.padding,
        args.axis,
        args.num_inp,
    ]

    model = gm.call_gen_model(args_list, args.model.lower())

    model.summary()

    if is_2d:
        model_name = args.model.lower() + "_c" + str(args.channels)
    else:
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
    gen_tflite(
        model, model_name, args.out_dir, data_type, args.en_quant, args.quant_layer
    )
