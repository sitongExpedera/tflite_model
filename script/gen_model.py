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
from tensorflow.keras.activations import tanh, relu, sigmoid, swish
from tensorflow.keras.models import Model
import tensorflow as tf


def call_gen_model(args_list, model_type):
    gm = gen_model(args_list)
    model_name = f"gm.{model_type}_model"
    model = eval(model_name)()
    return model


class gen_model:
    def __init__(self, args_list):
        self.batch = args_list[0]
        self.height = args_list[1]
        self.width = args_list[2]
        self.channel = args_list[3]
        self.filter = args_list[4]
        self.data_type = args_list[5]
        self.kernel = args_list[6]
        self.stride = args_list[7]
        self.padding = args_list[8]
        self.axis = args_list[9]
        self.num_inputs = args_list[10]

        self.input_size = [self.height, self.width, self.channel]
        self.input_size_2d = [self.channel]
        self.input = Input(self.input_size, batch_size=self.batch, dtype=self.data_type)
        self.input2 = Input(
            self.input_size, batch_size=self.batch, dtype=self.data_type
        )
        self.input_2d = Input(
            self.input_size_2d, batch_size=self.batch, dtype=self.data_type
        )

    def abs_model(self):
        input_tensor = tf.math.abs(self.input)
        output = Model([self.input], input_tensor)
        return output

    def add_model(self):
        input_tensor = Add()([self.input, self.input2])
        output = Model([self.input, self.input2], input_tensor)
        return output

    def add_const_model(self):
        input_tensor = self.input + 1
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

    def batch_to_space_model(self):
        input_tensor = tf.batch_to_space(
            self.input, block_shape=[3, 3], crops=[[0, 0], [0, 0]]
        )
        output = Model([self.input], input_tensor)
        return output

    def batchmatmul_model(self):
        input_tensor = tf.matmul(self.input, self.input2)
        output = Model([self.input, self.input2], input_tensor)
        return output

    def bilinear_resize_model(self):
        input_tensor = tf.keras.layers.Lambda(
            lambda x: tf.compat.v1.image.resize_bilinear(
                x,
                size=[self.input_size[0] // 2, self.input_size[1] // 2],
                align_corners=True,
            )
        )(self.input)
        output = Model([self.input], input_tensor)
        return output

    def boradcast_add_model(self):
        input_size = [self.height, 1, self.channel]
        input = Input(input_size, batch_size=self.batch, dtype=tf.float32)
        input_tensor = tf.math.add(self.input, input)
        output = Model([self.input, input], input_tensor)
        return output

    def concat_model(self):
        input_set = []
        for i in range(self.num_inputs):
            input = Input(self.input_size, batch_size=self.batch, dtype=tf.float32)
            input_set.append(input)
        input_tensor = Concatenate(axis=self.axis)(input_set)
        output = Model(input_set, input_tensor)
        return output

    def conv2d_model(self):
        input_tensor = Conv2D(
            filters=self.filter,
            kernel_size=self.kernel,
            strides=self.stride,
            name="conv0",
            use_bias=True,
            padding=self.padding,
        )(self.input)
        output = Model([self.input], input_tensor)
        return output

    def conv2d_trans_model(self):
        input_tensor = Conv2DTranspose(
            filters=self.filter,
            strides=self.stride,
            kernel_size=self.kernel,
            padding=self.padding,
        )(self.input)
        output = Model([self.input], input_tensor)
        return output

    def cos_model(self):
        input_tensor = tf.math.cos(self.input)
        output = Model([self.input], input_tensor)
        return output

    def crop_model(self):
        input_tensor = Cropping2D(cropping=((1, 1), (2, 2)))(self.input)
        output = Model([self.input], input_tensor)
        return output

    def dense_model(self):
        input_tensor = Dense(
            1, use_bias=True, bias_initializer=tf.keras.initializers.HeNormal()
        )(self.input_2d)
        output = Model([self.input_2d], input_tensor)
        return output

    def depth_to_space_model(self):
        input_tensor = tf.nn.depth_to_space(self.input, 2)
        output = Model([self.input], input_tensor)
        return output

    def dw_conv2d_model(self):
        input_tensor = DepthwiseConv2D(
            kernel_size=self.kernel,
            strides=self.stride,
            padding=self.padding,
            use_bias=True,
            name="dw_conv0",
        )(self.input)
        output = Model([self.input], input_tensor)
        return output

    def exp_model(self):
        input_tensor = tf.math.exp(self.input)
        output = Model([self.input], input_tensor)
        return output

    def expand_dims_model(self):
        input_tensor = tf.expand_dims(self.input, self.axis)
        output = Model([self.input], input_tensor)
        return output

    def gather_model(self):
        input_tensor = tf.gather(self.input, indices=[0, 3, 1, 1, 2, 3], axis=self.axis)
        output = Model([self.input], input_tensor)
        return output

    def global_average_2d_model(self):
        input_tensor = GlobalAveragePooling2D()(self.input)
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

    def log_model(self):
        input_tensor = tf.math.log(self.input)
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

    def matmul_model(self):
        input = Input(self.input_size, self.batch, dtype=self.data_type)
        input2 = Input(self.input_size, self.batch, dtype=self.data_type)
        input_tensor = tf.matmul(input, input2)
        output = Model([input, input2], input_tensor)
        return output

    def maximum_model(self):
        input_tensor = tf.math.maximum(self.input, self.input2)
        output = Model([self.input, self.input2], input_tensor)
        return output

    def maxpool_model(self):
        input_tensor = MaxPooling2D(
            pool_size=self.kernel, strides=self.stride, padding=self.padding
        )(self.input)
        output = Model([self.input], input_tensor)
        return output

    def mean_model(self):
        input_tensor = tf.math.reduce_mean(self.input, self.axis, keepdims=True)
        output = Model([self.input], input_tensor)
        return output

    def minimum_model(self):
        input_tensor = tf.math.minimum(self.input, self.input2)
        output = Model([self.input, self.input2], input_tensor)
        return output

    def mul_add_model(self):
        input_set = []
        for i in range(self.num_inputs):
            input = Input(self.input_size, batch_size=self.batch, dtype=tf.float32)
            input_set.append(input)
        input_tensor = Add()(input_set)
        output = Model(input_set, input_tensor)
        return output

    def multiply_model(self):
        input_tensor = Multiply()([self.input, self.input])
        output = Model([self.input], input_tensor)
        return output

    def nearest_neighbor_resize_model(self):
        input_tensor = tf.keras.layers.Lambda(
            lambda x: tf.compat.v1.image.resize_nearest_neighbor(
                x,
                size=[self.input_size[0] * 2, self.input_size[1] * 2],
                half_pixel_centers=True,
            )
        )(self.input)
        output = Model([self.input], input_tensor)
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
        input_tensor = tf.math.reduce_any(self.input, self.axis, keepdims=True)
        output = Model([self.input], input_tensor)
        return output

    def reduce_max_model(self):
        input_tensor = tf.math.reduce_max(self.input, self.axis, keepdims=True)
        output = Model([self.input], input_tensor)
        return output

    def reduce_min_model(self):
        input_tensor = tf.math.reduce_min(self.input, self.axis, keepdims=True)
        output = Model([self.input], input_tensor)
        return output

    def reduce_prod_model(self):
        input_tensor = tf.math.reduce_prod(self.input, self.axis, keepdims=True)
        output = Model([self.input], input_tensor)
        return output

    def reduce_sum_model(self):
        input_tensor = tf.math.reduce_sum(self.input, self.axis, keepdims=True)
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

    def reshape_model(self):
        input_tensor = tf.reshape(
            self.input, shape=[1, self.input_size[0] // 2, self.input_size[1] * 2, -1]
        )
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

    def separable_conv2d_model(self):
        input_tensor = SeparableConv2D(
            filters=self.filter,
            kernel_size=self.kernel,
            strides=self.stride,
            name="separable_conv0",
            use_bias=True,
            padding=self.padding,
        )(self.input)
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

    def softmax2d_model(self):
        input_tensor = tf.nn.softmax(self.input_2d)
        output = Model([self.input_2d], input_tensor)
        return output

    def space_to_batch_model(self):
        block_size = 3
        pad_h = self.height - self.height // block_size * block_size
        pad_w = self.width - self.width // block_size * block_size
        pad_top = pad_h // 2
        pad_bot = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        input_tensor = tf.nn.space_to_batch(
            self.input,
            block_shape=[block_size, block_size],
            paddings=[[pad_top, pad_bot], [pad_left, pad_right]],
        )
        output = Model([self.input], input_tensor)
        return output

    def space_to_depth_model(self):
        input_tensor = tf.nn.space_to_depth(self.input, 2)
        output = Model([self.input], input_tensor)
        return output

    def split_model(self):
        input_tensor = tf.split(self.input, num_or_size_splits=4, axis=self.axis)
        output = Model([self.input], input_tensor)
        return output

    def split_2d_model(self):
        input_tensor = tf.split(self.input_2d, num_or_size_splits=4, axis=self.axis)
        output = Model([self.input_2d], input_tensor)
        return output

    def square_model(self):
        input_tensor = tf.math.square(self.input)
        output = Model([self.input], input_tensor)
        return output

    def squared_difference_model(self):
        input_tensor = tf.math.squared_difference(self.input, self.input2)
        output = Model([self.input, self.input2], input_tensor)
        return output

    def squeeze_model(self):
        input_tensor = tf.squeeze(self.input, self.axis)
        output = Model([self.input], input_tensor)
        return output

    def sqrt_model(self):
        input_tensor = tf.math.sqrt(self.input)
        output = Model([self.input], input_tensor)
        return output

    def stack_model(self):
        input_set = []
        for i in range(self.num_inputs):
            input = Input(self.input_size, batch_size=self.batch, dtype=tf.float32)
            input_set.append(input)
        input_tensor = tf.stack(input_set, axis=self.axis)
        output = Model(input_set, input_tensor)
        return output

    def stack_2d_model(self):
        input_set = []
        for i in range(self.num_inputs):
            input = Input(self.input_size_2d, batch_size=self.batch, dtype=tf.float32)
            input_set.append(input)
        input_tensor = tf.stack(input_set, axis=self.axis)
        output = Model(input_set, input_tensor)
        return output

    def sum_model(self):
        input_tensor = tf.keras.backend.sum(self.input)
        output = Model([self.input], input_tensor)
        return output

    def subtract_model(self):
        input_tensor = Subtract()([self.input, self.input])
        output = Model([self.input], input_tensor)
        return output

    def swish_model(self):
        input_tensor = swish(self.input)
        output = Model([self.input], input_tensor)
        return output

    def tan_model(self):
        input_tensor = tf.keras.backend.sin(self.input) / tf.keras.backend.cos(
            self.input
        )
        output = Model([self.input], input_tensor)
        return output

    def tanh_model(self):
        input_tensor = tanh(self.input)
        output = Model([self.input], input_tensor)
        return output

    def unstack_model(self):
        input_tensor = tf.unstack(self.input, axis=self.axis)
        output = Model([self.input], input_tensor)
        return output

    def upsample_model(self):
        input_tensor = UpSampling2D(size=(1.37, 1.37))(self.input)
        output = Model([self.input], input_tensor)
        return output

    def custom_model(self):
        x = Conv2D(
            filters=self.filter,
            kernel_size=self.kernel,
            strides=self.stride,
            name="conv2",
            use_bias=True,
            padding=self.padding,
        )(self.input)
        x = tf.keras.layers.Lambda(
            lambda x: tf.compat.v1.image.resize_bilinear(
                x,
                size=[self.input_size[0] // 2, self.input_size[1] // 2],
                align_corners=True,
            )
        )(x)
        x = Conv2D(
            filters=self.filter,
            kernel_size=self.kernel,
            strides=self.stride,
            name="conv0",
            use_bias=True,
            padding=self.padding,
        )(x)
        output = Model([self.input], x)
        return output
