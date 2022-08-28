#!/usr/bin/env python3
from gen_layer import *
import json
import logging
from utils.quantize_utils import quantize_utils
import argparse
import os


def check_attrs(ltype, attrs, attrs_check):
    correct_attrs = {}
    possible_attrs = attrs_check["AttrsCheck"][ltype]
    for a in attrs.keys():
        if a not in possible_attrs:
            logging.error(
                f"Attribute {a} is not possible attributes, please check layer_attrs.json for more detail!"
            )
    for k in attrs_check["AttrsDefault"][ltype]:
        if k in attrs.keys():
            correct_attrs[k] = attrs[k]
        else:
            correct_attrs[k] = attrs_check["AttrsDefault"][ltype][k]
    return correct_attrs


def gen_model(json_file, attrs_file, input_shape):
    model_info = json.load(open(json_file, encoding="utf-8"))
    attrs_info = json.load(open(attrs_file, encoding="utf-8"))

    model_input = []
    model_output = []
    layer_list = {}
    layer_used = {}
    for layer in model_info["Layer"]:
        ltype = layer["LayerType"]
        lid = layer["LayerID"]
        input_id = layer["InputID"]
        attrs = layer["Attrs"]
        correct_attrs = check_attrs(ltype, attrs, attrs_info)
        layer_input = []
        for i in input_id:
            if i not in layer_list.keys():
                logging.warning(f"LayerID {i} not in layer_list, create new input!")
                l = eval(f"gen_layer_Input")(input_shape[1:], input_shape[0])
                layer_list[i] = l
                model_input.append(l)
            layer_input.append(layer_list[i])
            layer_used[i] = True

        l = eval(f"gen_layer_{ltype}")(layer_input, correct_attrs)
        layer_list[lid] = l
        layer_used[lid] = False

    for i in layer_used.keys():
        if not layer_used[i]:
            model_output.append(layer_list[i])
    model = eval("create_model")(model_input, model_output, model_info["ModelName"])
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_info",
        "-mi",
        type=str,
        default=os.getcwd() + "/model_info.json",
        help="All the information of model, json file!",
    )
    parser.add_argument(
        "--layer_attrs",
        "-la",
        type=str,
        default=os.getcwd() + "/layer_attrs.json",
        help="Possible attributes and defult attributes!",
    )
    parser.add_argument(
        "--out_dir",
        "-o",
        type=str,
        default=os.getcwd() + "/models/test/",
    )
    parser.add_argument("--batch", "-b", type=int, default=1, help="Batch")
    parser.add_argument("--height", "-ht", type=int, default=8, help="Height")
    parser.add_argument("--width", "-w", type=int, default=8, help="Width")
    parser.add_argument("--channels", "-c", type=int, default=8, help="Channel")
    parser.add_argument("--en_quant", "-q", type=int, default=1, help="Quantize model")
    parser.add_argument(
        "--nbits",
        "-nb",
        type=int,
        default=8,
        help="Number of bits, support 8/16 bits for now",
    )
    parser.add_argument(
        "--quant_layer",
        "-ql",
        action="store_true",
        default=False,
        help="includes quantize and dequantize layers",
    )
    args = parser.parse_args()
    model_info = args.model_info
    layer_attrs = args.layer_attrs
    input_shape = [args.batch, args.height, args.width, args.channels]
    model = gen_model(model_info, layer_attrs, input_shape)
    qu = quantize_utils(model.input_shape, len(model.inputs))
    qu.gen_quant_tflite(
        model,
        model.name,
        args.out_dir,
        model.dtype,
        args.en_quant,
        args.quant_layer,
        args.nbits,
    )
