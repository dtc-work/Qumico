import qumico.common.keras_helper as k_helper
import qumico.common.tensorflow_helper as tf_converter

class BaseFrontend:
    pass


class KerasFrontend(BaseFrontend):

    @classmethod
    def convert_from_keras(cls, output_path, onnx_name, model_name, ckpt_name, pb_name, cache_path, freeze_pb_name,
                           output_op_name, K, binary_flag=True, **kwargs):
        onnx_file = k_helper.keras_save_to_onnx(K=K, output_path=output_path, pb_name=pb_name, ckpt_name=ckpt_name,
                                                cache_path=cache_path, freeze_pb_name=freeze_pb_name,
                                                onnx_name=onnx_name, model_name=model_name,
                                                output_op_name=output_op_name,
                                                binary_flag=binary_flag)
        return onnx_file

    @classmethod
    def convert_from_yaml(cls, output_path, onnx_name, model_name, ckpt_name, pb_name, cache_path, freeze_pb_name,
                          yaml_file, h5_file, binary_flag=True, **kwargs):
        onnx_file = k_helper.load_yamlh5_to_onnx(output_path=output_path, onnx_name=onnx_name, model_name=model_name,
                                                 ckpt_name=ckpt_name, pb_name=pb_name, cache_path=cache_path,
                                                 freeze_pb_name=freeze_pb_name, yaml_file=yaml_file, h5_file=h5_file,
                                                 binary_flag=binary_flag)
        return onnx_file

    @classmethod
    def convert_from_json(cls, output_path, onnx_name, model_name, ckpt_name, pb_name, cache_path, freeze_pb_name,
                          json_file, h5_file, binary_flag=True, **kwargs):
        onnx_file = k_helper.load_jsonh5_to_onnx(output_path=output_path, onnx_name=onnx_name, model_name=model_name,
                                                 ckpt_name=ckpt_name, pb_name=pb_name, cache_path=cache_path,
                                                 freeze_pb_name=freeze_pb_name, json_file=json_file, h5_file=h5_file,
                                                 binary_flag=binary_flag)
        return onnx_file


class TensorflowFrontend(BaseFrontend):

    @classmethod
    def convert_from_tensorflow(cls, output_path, onnx_name, model_name, output_op_name, ckpt_file, pb_file, cache_path, freeze_pb_name,
                                binary_flag=True, **kwargs):
        onnx_file = tf_converter.tensorflow_save_onnx(output_path=output_path, onnx_name=onnx_name,
                                                      model_name=model_name, output_op_name=output_op_name, ckpt_file=ckpt_file, pb_file=pb_file,
                                                      cache_path=cache_path, freeze_pb_name=freeze_pb_name,
                                                      binary_flag=binary_flag)
        return onnx_file
