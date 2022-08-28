#!/usr/bin/env python3
import argparse
import gen_model as gm
import numpy as np
import tensorflow as tf
import os
from utils.model_list import model_choices
from utils.quantize_utils import quantize_utils


input_shape = [1, 8, 8, 8]
num_inp = 1
nbits = 8
is_2d = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", choices=model_choices, type=str, default="")
    parser.add_argument(
        "--out_dir",
        "-o",
        type=str,
        default=os.getcwd() + "/models/test/",
    )
    parser.add_argument("--num_inp", "-n", type=int, default=1, help="Number of inputs")
    parser.add_argument("--batch", "-b", type=int, default=1, help="Batch")
    parser.add_argument("--width", "-w", type=int, default=8, help="Width")
    parser.add_argument("--height", "-ht", type=int, default=8, help="Height")
    parser.add_argument("--channels", "-c", type=int, default=8, help="Channel")
    parser.add_argument("--en_quant", "-q", type=int, default=1, help="Quantize model")
    parser.add_argument("--filter", "-f", type=int, default=8, help="Number of filters")
    parser.add_argument("--kernel", "-k", type=int, default=3, help="Kernel size")
    parser.add_argument("--stride", "-s", type=int, default=1, help="Stride")
    parser.add_argument(
        "--axis", "-a", type=int, default=1, help="axis for some ops eg.concat"
    )
    parser.add_argument(
        "--padding", "-p", type=str, default="valid", help="padding mode(same/valid)"
    )
    parser.add_argument(
        "--nbits",
        "-nb",
        type=int,
        default=8,
        help="Number of bits, support 8/16 bits for now",
    )
    parser.add_argument(
        "--list", "-l", action="store_true", help="List all support model"
    )
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
    en_quant = args.en_quant
    input_shape = [args.batch, args.height, args.width, args.channels]

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

    model_type = args.model.lower()

    bool_type_op = ["logical_not", "logical_or", "reduce_any", "reduce_all"]
    int32_type_op = ["right_shift"]

    if model_type in bool_type_op:
        data_type = "bool"
        en_quant = 0
    elif model_type in int32_type_op:
        data_type = "int32"
    else:
        data_type = "float32"
    if "2d" in model_type and "conv2d" not in model_type:
        is_2d = True
        input_shape = [1, args.channels]

    two_input_op = [
        "add",
        "minimum",
        "maximum",
        "matmul",
        "squared_difference",
        "batchmatmul",
    ]

    if model_type in two_input_op:
        num_inp = 2

    args_list = [
        args.batch,
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

    model = gm.call_gen_model(args_list, model_type)

    model.summary()

    if is_2d:
        model_name = model_type + "_b" + str(args.batch) + "_c" + str(args.channels)
    else:
        model_name = (
            model_type
            + "_b"
            + str(args.batch)
            + "_h"
            + str(args.height)
            + "_w"
            + str(args.width)
            + "_c"
            + str(args.channels)
        )
    if "conv" in model_type:
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
    qu = quantize_utils(input_shape, num_inp)
    qu.gen_quant_tflite(
        model=model,
        model_name=model_name,
        save_dir=args.out_dir,
        op_type=data_type,
        en_quant=en_quant,
        quant_layer=args.quant_layer,
        nbits=nbits,
    )
