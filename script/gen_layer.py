import tensorflow as tf
import tensorflow.keras.layers as Layer
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Conv2DTranspose,
    Concatenate,
    Add,
    Multiply,
    Subtract,
    LeakyReLU,
    GlobalAveragePooling2D,
    Dense,
    SeparableConv2D,
    UpSampling2D,
    DepthwiseConv2D,
    MaxPooling2D,
    Cropping2D,
)
from tensorflow.keras.models import Model


def create_model(input, output, name):
    model = Model(input, output, name=name)
    return model


def gen_layer_Input(shape, batch_size):
    layer = Input(shape=shape, batch_size=batch_size)
    return layer


def gen_layer_Conv2D(input, attrs):
    input = input if len(input) != 1 else input[0]
    layer = Conv2D(
        filters=attrs["filters"],
        kernel_size=attrs["kernel_size"],
        strides=attrs["strides"],
        padding=attrs["padding"],
        dilation_rate=attrs["dilation_rate"],
        groups=attrs["groups"],
    )(input)
    return layer


def gen_layer_DepthwiseConv2D(input, attrs):
    input = input if len(input) != 1 else input[0]
    layer = DepthwiseConv2D(
        kernel_size=attrs["kernel_size"],
        strides=attrs["strides"],
        padding=attrs["padding"],
        dilation_rate=attrs["dilation_rate"],
        groups=attrs["groups"],
    )(input)
    return layer


def gen_layer_Add(input, attrs):
    input = input if len(input) != 1 else input[0]
    if attrs["const"]:
        const_val = 10
        layer = input + const_val
    else:
        layer = Add()(input)
    return layer


def gen_layer_Mul(input, attrs):
    input = input if len(input) != 1 else input[0]
    if attrs["const"]:
        const_val = 10
        layer = input * const_val
    else:
        layer = Multiply()(input)
    return layer
