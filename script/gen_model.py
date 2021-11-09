from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Conv2DTranspose,
    Concatenate,
    Add,
    Subtract,
    LeakyReLU,
)
from tensorflow.keras.activations import tanh, relu, sigmoid
from tensorflow.keras.models import Model
import tensorflow as tf
import logging


def call_gen_model(input_size, modle_type):
    gm = gen_model(input_size)
    if modle_type == "abs":
        model = gm.abs_model()
    elif modle_type == "concat":
        model = gm.concat_model()
    elif modle_type == "conv2d_trans":
        model = gm.conv2d_trans_model()
    elif modle_type == "depth_to_space":
        model = gm.depth_to_space_model()
    elif modle_type == "exp":
        model = gm.exp_model()
    elif modle_type == "leaky_relu":
        model = gm.leaky_relu_model()
    elif modle_type == "mul_add":
        model = gm.mul_add_model()
    elif modle_type == "power":
        model = gm.power_model()
    elif modle_type == "relu":
        model = gm.relu_model()
    elif modle_type == "relu6":
        model = gm.relu6_model()
    elif modle_type == "right_shift":
        model = gm.right_shift_model()
    elif modle_type == "rsqrt":
        model = gm.rsqrt_model()
    elif modle_type == "sin":
        model = gm.sin_model()
    elif modle_type == "sigmoid":
        model = gm.sigmoid_model()
    elif modle_type == "space_to_depth":
        model = gm.space_to_depth_model()
    elif modle_type == "square":
        model = gm.square_model()
    elif modle_type == "subtract":
        model = gm.subtract_model()
    elif modle_type == "tan":
        model = gm.tan_model()
    elif modle_type == "tanh":
        model = gm.tanh_model()
    else:
        logging.error("Cannot support this operator!!!")
        exit(1)
    return model


class gen_model:
    def __init__(self, input_size):
        self.input_size = input_size
        self.input = Input(self.input_size, batch_size=1)

    def abs_model(self):
        abs_out = tf.math.abs(self.input)
        output = Model([self.input], abs_out)
        return output

    def concat_model(self):
        input_set = []
        for i in range(5):
            input_set.append(self.input)
        concat_output = Concatenate()(input_set)
        output = Model(input_set, concat_output)
        return output

    def conv2d_trans_model(self):
        conv2d_trans_out = Conv2DTranspose(filters=8, kernel_size=(3, 3))(self.input)
        output = Model([self.input], conv2d_trans_out)
        return output

    def depth_to_space_model(self):
        depth_to_space_out = tf.nn.depth_to_space(self.input, 2)
        output = Model([self.input], depth_to_space_out)
        return output

    def exp_model(self):
        exp_out = tf.math.exp(self.input)
        output = Model([self.input], exp_out)
        return output

    def leaky_relu_model(self):
        leaky_relu_out = LeakyReLU(alpha=0.3)(self.input)
        output = Model([self.input], leaky_relu_out)
        return output

    def mul_add_model(self):
        input_set = []
        for i in range(5):
            input_set.append(self.input)
        mul_add_output = Add()(input_set)
        output = Model(input_set, mul_add_output)
        return output

    def power_model(self):
        pow_out = tf.math.pow(self.input, 3)
        output = Model([self.input], pow_out)
        return output

    def relu_model(self):
        relu_out = relu(self.input)
        output = Model([self.input], relu_out)
        return output

    def relu6_model(self):
        relu6_out = tf.nn.relu6(self.input)
        output = Model([self.input], relu6_out)
        return output

    def right_shift_model(self):
        input1 = Input(self.input_size, batch_size=1, dtype="int32")
        input2 = Input(self.input_size, batch_size=1, dtype="int32")
        right_shift_output = tf.bitwise.right_shift(input1, input2)
        output = Model([input1, input2], right_shift_output)
        return output

    def rsqrt_model(self):
        rsqrt_out = tf.math.rsqrt(self.input)
        output = Model([self.input], rsqrt_out)
        return output

    def sin_model(self):
        sin_out = tf.math.sin(self.input)
        output = Model([self.input], sin_out)
        return output

    def sigmoid_model(self):
        sigmoid_out = sigmoid(self.input)
        output = Model([self.input], sigmoid_out)
        return output

    def space_to_depth_model(self):
        space_to_depth_out = tf.nn.space_to_depth(self.input, 2)
        output = Model([self.input], space_to_depth_out)
        return output

    def square_model(self):
        square_out = tf.math.square(self.input)
        output = Model([self.input], square_out)
        return output

    def subtract_model(self):
        conv1 = Conv2D(filters=8, kernel_size=3, padding="same")(self.input)
        conv2 = Conv2D(filters=8, kernel_size=3, padding="same")(self.input)
        subtract_out = Subtract()([conv1, conv2])
        output = Model([self.input], subtract_out)
        return output

    def tan_model(self):
        tan_out = tf.math.tan(self.input)
        output = Model([self.input], tan_out)
        return output

    def tanh_model(self):
        tanh_out = tanh(self.input)
        output = Model([self.input], tanh_out)
        return output
