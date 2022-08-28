import tensorflow as tf
import numpy as np


class quantize_utils:
    def __init__(self, input_shape, num_inp) -> None:
        self.input_shape = input_shape
        self.num_inp = num_inp

    def gen_quant_tflite(
        self,
        model,
        model_name,
        save_dir,
        op_type,
        en_quant,
        quant_layer,
        nbits=8,
    ):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model_name = save_dir + model_name
        if en_quant:
            tflite_model_name += "_quant"
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            if op_type == "float32":
                converter.representative_dataset = self.representative_float_dataset_gen
            elif op_type == "bool":
                converter.representative_dataset = self.representative_bool_dataset_gen
            elif op_type == "int32":
                converter.representative_dataset = self.representative_int_dataset_gen
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

    def representative_float_dataset_gen(self):
        input_set = []
        for _ in range(self.num_inp):
            input_data = 5 * np.array(
                np.random.random_sample(self.input_shape), dtype="float32"
            )
            input_set.append(input_data)
        yield input_set

    def representative_bool_dataset_gen(self):
        input_set = []
        for _ in range(self.num_inp):
            input_data = 5 * np.array(
                np.random.random_sample(self.input_shape), dtype="bool"
            )
            input_set.append(input_data)
        yield input_set

    def representative_int_dataset_gen(self):
        input_set = []
        for _ in range(self.num_inp):
            input_data = 5 * np.array(
                np.random.random_sample(self.input_shape), dtype="int32"
            )
            input_set.append(input_data)
        yield input_set
