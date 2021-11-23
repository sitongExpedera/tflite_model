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


def call_gen_model(input_size, model_type, data_type):
    gm = gen_model(input_size, data_type)
    if model_type == "abs":
        model = gm.abs_model()
    elif model_type == "arg_max":
        model = gm.arg_max_model()
    elif model_type == "arg_min":
        model = gm.arg_min_model()
    elif model_type == "bilinear":
        model = gm.bilinear()
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
    elif model_type == "reduce_max":
        model = gm.reduce_max_model()
    elif model_type == "reduce_min":
        model = gm.reduce_min_model()
    elif model_type == "reduce_prod":
        model = gm.reduce_prod_model()
    elif model_type == "relu":
        model = gm.relu_model()
    elif model_type == "relu6":
        model = gm.relu6_model()
    elif model_type == "right_shift":
        model = gm.right_shift_model()
    elif model_type == "rsqrt":
        model = gm.rsqrt_model()
    elif model_type == "segment_sum":
        model = gm.segment_sum_model()
    elif model_type == "sin":
        model = gm.sin_model()
    elif model_type == "sigmoid":
        model = gm.sigmoid_model()
    elif model_type == "softmax":
        model = gm.softmax_model()
    elif model_type == "space_to_depth":
        model = gm.space_to_depth_model()
    elif model_type == "square":
        model = gm.square_model()
    elif model_type == "sum":
        model = gm.sum_model()
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
        input_tensor = tf.math.abs(self.input)
        output = Model([self.input], input_tensor)
        return output

    def arg_max_model(self):
        input_tensor = tf.math.argmax(self.input)
        output = Model([self.input], input_tensor)
        return output

    def arg_min_model(self):
        input_tensor = tf.math.argmin(self.input)
        output = Model([self.input], input_tensor)
        return output

    def bilinear(self):
        input_tensor = tf.image.resize(self.input, size=[self.input_size[0] // 2, self.input_size[1] // 2])
        output = Model([self.input], input_tensor)
        return output

    def concat_model(self):
        input_set = []
        for i in range(5):
            input_set.append(self.input)
        input_tensor = Concatenate()(input_set)
        output = Model(input_set, input_tensor)
        return output

    def conv2d_trans_model(self):
        input_tensor = Conv2DTranspose(filters=8, kernel_size=(3, 3))(self.input)
        output = Model([self.input], input_tensor)
        return output

    def depth_to_space_model(self):
        input_tensor = tf.nn.depth_to_space(self.input, 2)
        output = Model([self.input], input_tensor)
        return output

    def exp_model(self):
        input_tensor = tf.math.exp(self.input)
        output = Model([self.input], input_tensor)
        return output

    def greater_model(self):
        input_tensor = tf.math.greater(self.input, self.input)
        output = Model([self.input], input_tensor)
        return output

    def greater_equal_model(self):
        input_tensor = tf.math.greater_equal(self.input, self.input)
        output = Model([self.input], input_tensor)
        return output

    def leaky_relu_model(self):
        input_tensor = LeakyReLU(alpha=0.3)(self.input)
        output = Model([self.input], input_tensor)
        return output

    def less_model(self):
        input_tensor = tf.math.less(self.input, self.input)
        output = Model([self.input], input_tensor)
        return output

    def less_equal_model(self):
        input_tensor = tf.math.less_equal(self.input, self.input)
        output = Model([self.input], input_tensor)
        return output

    def log_softmax_model(self):
        input_tensor = tf.math.log_softmax(self.input)
        output = Model([self.input], input_tensor)
        return output

    def logical_not_model(self):
        input_tensor = tf.math.logical_not(self.input)
        output = Model([self.input], input_tensor)
        return output

    def logical_or_model(self):
        input_tensor = tf.math.logical_or(self.input, self.input)
        output = Model([self.input], input_tensor)
        return output

    def maximum_model(self):
        input_tensor = tf.math.maximum(self.input, self.input)
        output = Model([self.input], input_tensor)
        return output

    def minimum_model(self):
        input_tensor = tf.math.minimum(self.input, self.input)
        output = Model([self.input], input_tensor)
        return output

    def mul_add_model(self):
        input_set = []
        for i in range(5):
            input_set.append(self.input)
        input_tensor = Add()(input_set)
        output = Model(input_set, input_tensor)
        return output

    def not_equal_model(self):
        input_tensor = tf.math.not_equal(self.input, self.input)
        output = Model([self.input], input_tensor)
        return output

    def power_model(self):
        input_tensor = tf.math.pow(self.input, 3)
        output = Model([self.input], input_tensor)
        return output

    def reduce_any_model(self):
        input_tensor = tf.math.reduce_any(self.input)
        output = Model([self.input], input_tensor)
        return output

    def reduce_max_model(self):
        input_tensor = tf.math.reduce_max(self.input)
        output = Model([self.input], input_tensor)
        return output

    def reduce_min_model(self):
        input_tensor = tf.math.reduce_min(self.input)
        output = Model([self.input], input_tensor)
        return output

    def reduce_prod_model(self):
        input_tensor = tf.math.reduce_prod(self.input)
        output = Model([self.input], input_tensor)
        return output

    def relu_model(self):
        input_tensor = relu(self.input)
        output = Model([self.input], input_tensor)
        return output

    def relu6_model(self):
        input_tensor = tf.nn.relu6(self.input)
        output = Model([self.input], input_tensor)
        return output

    def right_shift_model(self):
        input_tensor = tf.bitwise.right_shift(self.input, self.input)
        output = Model([self.input], input_tensor)
        return output

    def rsqrt_model(self):
        input_tensor = tf.math.rsqrt(self.input)
        output = Model([self.input], input_tensor)
        return output

    def segment_sum_model(self):
        segment_idx = tf.zeros(self.input.shape[0], dtype="int32")
        input_tensor = tf.math.segment_sum(self.input, segment_idx)
        output = Model([self.input], input_tensor)
        return output

    def sin_model(self):
        input_tensor = tf.math.sin(self.input)
        output = Model([self.input], input_tensor)
        return output

    def sigmoid_model(self):
        input_tensor = sigmoid(self.input)
        output = Model([self.input], input_tensor)
        return output

    def softmax_model(self):
        input_tensor = tf.nn.softmax(self.input)
        output = Model([self.input], input_tensor)
        return output

    def space_to_depth_model(self):
        input_tensor = tf.nn.space_to_depth(self.input, 2)
        output = Model([self.input], input_tensor)
        return output

    def square_model(self):
        input_tensor = tf.math.square(self.input)
        output = Model([self.input], input_tensor)
        return output

    def sum_model(self):
        input_tensor = tf.keras.backend.sum(self.input)
        output = Model([self.input], input_tensor)
        return output

    def subtract_model(self):
        conv1 = Conv2D(filters=8, kernel_size=3, padding="same")(self.input)
        conv2 = Conv2D(filters=8, kernel_size=3, padding="same")(self.input)
        input_tensor = Subtract()([conv1, conv2])
        output = Model([self.input], input_tensor)
        return output

    def tan_model(self):
        input_tensor = tf.math.tan(self.input)
        output = Model([self.input], input_tensor)
        return output

    def tanh_model(self):
        input_tensor = tanh(self.input)
        output = Model([self.input], input_tensor)
        return output
