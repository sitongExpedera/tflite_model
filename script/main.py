#!/usr/bin/env python3
import argparse
import gen_model as gm
import numpy as np
import tensorflow as tf

input_shape = [1, 8, 8, 8]
num_inp = 1


def gen_ONNX(model, model_name, save_dir):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    tflite_quant_model = converter.convert()

    tflite_model_name = save_dir + model_name + ".tflite"
    open(tflite_model_name, "wb").write(tflite_quant_model)
    print(tflite_model_name)


def representative_dataset_gen():
    input_set = []
    for _ in range(num_inp):
        input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        input_set.append(input_data)
    yield input_set


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="")
    parser.add_argument(
        "--out_dir",
        "-o",
        type=str,
        default="/home/sitong/tflite_model/models/",
        help="model dir",
    )
    parser.add_argument("--num_inp", "-n", type=int, default=1, help="number of input")
    parser.add_argument("--width", "-w", type=int, default=8, help="inp_width")
    parser.add_argument("--height", "-ht ", type=int, default=8, help="inp_height")
    parser.add_argument("--channels", "-c", type=int, default=8, help="inp_channels")
    args = parser.parse_args()
    num_inp = args.num_inp
    input_shape = [1, args.height, args.width, args.channels]

    model = gm.call_gen_model(
        [args.height, args.width, args.channels], args.model.lower()
    )
    model.summary()
    model_name = (
        args.model.lower()
        + "_"
        + str(args.height)
        + "x"
        + str(args.width)
        + "x"
        + str(args.channels)
    )
    gen_ONNX(model, model_name, args.out_dir)
