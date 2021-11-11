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
from tensorflow.keras.metrics import Mean
from tensorflow.keras.models import Model
import tensorflow as tf
import logging


def call_gen_model(input_size, model_type, data_type):
    gm = gen_model(input_size, data_type)
    if model_type == "abs":
        model = gm.abs_model()
    elif model_type == "arg_max":
        model = gm.arg_max_model()
    elif model_type == "arg_min":
        model = gm.arg_min_model()
    elif model_type == "concat":
        model = gm.concat_model()
    elif model_type == "conv2d_trans":
        model = gm.conv2d_trans_model()
    elif model_type == "depth_to_space":
        model = gm.depth_to_space_model()
    elif model_type == "exp":
        model = gm.exp_model()
    elif model_type == "greater":
        model = gm.greater_model()
    elif model_type == "greater_equal":
        model = gm.greater_equal_model()
    elif model_type == "leaky_relu":
        model = gm.leaky_relu_model()
    elif model_type == "less":
        model = gm.less_model()
    elif model_type == "less_equal":
        model = gm.less_equal_model()
    elif model_type == "log_softmax":
        model = gm.log_softmax_model()
    elif model_type == "logical_not":
        model = gm.logical_not_model()
    elif model_type == "logical_or":
        model = gm.logical_or_model()
    elif model_type == "maximum":
        model = gm.maximum_model()
    elif model_type == "minimum":
        model = gm.minimum_model()
    elif model_type == "mul_add":
        model = gm.mul_add_model()
    elif model_type == "not_equal":
        model = gm.not_equal_model()
    elif model_type == "power":
        model = gm.power_model()
    elif model_type == "reduce_any":
        model = gm.reduce_any_model()
    elif model_type == "relu":
        model = gm.relu_model()
    elif model_type == "relu6":
        model = gm.relu6_model()
    elif model_type == "right_shift":
        model = gm.right_shift_model()
    elif model_type == "rsqrt":
        model = gm.rsqrt_model()
    elif model_type == "sin":
        model = gm.sin_model()
    elif model_type == "sigmoid":
        model = gm.sigmoid_model()
    elif model_type == "space_to_depth":
        model = gm.space_to_depth_model()
    elif model_type == "square":
        model = gm.square_model()
    elif model_type == "subtract":
        model = gm.subtract_model()
    elif model_type == "tan":
        model = gm.tan_model()
    elif model_type == "tanh":
        model = gm.tanh_model()
    else:
        logging.error("Cannot support this operator!!!")
        exit(1)
    return model


class gen_model:
    def __init__(self, input_size, data_type):
        self.input_size = input_size
        self.input = Input(self.input_size, batch_size=1, dtype=data_type)

    def abs_model(self):
        abs_out = tf.math.abs(self.input)
        output = Model([self.input], abs_out)
        return output

    def arg_max_model(self):
        arg_max_out = tf.math.argmax(self.input)
        output = Model([self.input], arg_max_out)
        return output

    def arg_min_model(self):
        arg_min_out = tf.math.argmin(self.input)
        output = Model([self.input], arg_min_out)
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

    def greater_model(self):
        greater_out = tf.math.greater(self.input, self.input)
        output = Model([self.input], greater_out)
        return output

    def greater_equal_model(self):
        greater_equal_out = tf.math.greater_equal(self.input, self.input)
        output = Model([self.input], greater_equal_out)
        return output

    def leaky_relu_model(self):
        leaky_relu_out = LeakyReLU(alpha=0.3)(self.input)
        output = Model([self.input], leaky_relu_out)
        return output

    def less_model(self):
        less_out = tf.math.less(self.input, self.input)
        output = Model([self.input], less_out)
        return output

    def less_equal_model(self):
        less_equal_out = tf.math.less_equal(self.input, self.input)
        output = Model([self.input], less_equal_out)
        return output

    def log_softmax_model(self):
        log_softmax_out = tf.math.log_softmax(self.input)
        output = Model([self.input], log_softmax_out)
        return output

    def logical_not_model(self):
        logical_not_out = tf.math.logical_not(self.input)
        output = Model([self.input], logical_not_out)
        return output

    def logical_or_model(self):
        logical_or_out = tf.math.logical_or(self.input, self.input)
        output = Model([self.input], logical_or_out)
        return output

    def maximum_model(self):
        maximum_out = tf.math.maximum(self.input, self.input)
        output = Model([self.input], maximum_out)
        return output

    def minimum_model(self):
        minimum_out = tf.math.minimum(self.input, self.input)
        output = Model([self.input], minimum_out)
        return output

    def mul_add_model(self):
        input_set = []
        for i in range(5):
            input_set.append(self.input)
        mul_add_output = Add()(input_set)
        output = Model(input_set, mul_add_output)
        return output

    def not_equal_model(self):
        not_equal_out = tf.math.not_equal(self.input, self.input)
        output = Model([self.input], not_equal_out)
        return output

    def power_model(self):
        pow_out = tf.math.pow(self.input, 3)
        output = Model([self.input], pow_out)
        return output

    def reduce_any_model(self):
        reduce_any_out = tf.math.reduce_any(self.input)
        output = Model([self.input], reduce_any_out)
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
        right_shift_output = tf.bitwise.right_shift(self.input, self.input)
        output = Model([self.input], right_shift_output)
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
