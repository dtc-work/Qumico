import os
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import model_from_json, model_from_yaml
import qumico.common.tf2onnx_tool as converter

def keras_save_to_onnx(output_path, onnx_name, model_name, ckpt_name, pb_name, cache_path, freeze_pb_name,
                       output_op_name, K, binary_flag=True, white_list='', black_list=''):
    """

    :param output_path: 出力パス
    :param onnx_name: 出力onnx ファイル名
    :param model_name: 出力onnx ファイルに記載されるモデル名
    :param ckpt_name: 中間ckpt ファイル名
    :param pb_name: 中間pb ファイル名
    :param cache_path: 中間ファイル用フォルダパス
    :param freeze_pb_name: freezepb ファイル名
    :param output_op_name: モデル名出力する際にoutput op名
    :param K: Keras backend model graphの扱う場所
    :param binary_flag: 保存ファイル形式
    :param white_list: white_list op名(string default:'') カンマで区切りする
    :param black_list: black_list op名(string default:'') カンマで区切りする
    :return: 保存した onnx ファイルのパスを
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)


    ckpt_file = os.path.join(cache_path, ckpt_name)
    pb_path = cache_path
    pb_file = os.path.join(cache_path, pb_name)
    freeze_pb_file = os.path.join(cache_path, freeze_pb_name)
    onnx_file = os.path.join(output_path, onnx_name)

    tf.train.Saver(write_version=tf.train.SaverDef.V1, max_to_keep=1).save(K.get_session(), ckpt_file)
    tf.train.write_graph(K.get_session().graph, pb_path, pb_name, as_text=not binary_flag)

    converter.load_pb_ckpt_to_onnx(pb_file=pb_file, ckpt_file=ckpt_file, freeze_pb_file=freeze_pb_file,
                                   onnx_file=onnx_file, model_name=model_name,
                                   output_op_name=output_op_name, binary_flag=binary_flag, white_list=white_list,
                                   black_list=black_list)

    return onnx_file


def load_jsonh5_to_onnx(output_path, onnx_name, model_name, ckpt_name, pb_name, cache_path, freeze_pb_name, json_file,
                        h5_file, binary_flag=True, white_list='', black_list=''):
    """
    :param output_path: 出力パス
    :param onnx_name: 出力onnx ファイル名
    :param model_name: 出力onnx ファイルに記載されるモデル名
    :param ckpt_name: 中間ckpt ファイル名
    :param pb_name: 中間pb ファイル名
    :param cache_path: 中間ファイル用フォルダパス
    :param freeze_pb_name: freezepb ファイル名
    :param json_file: input json ファイルパス
    :param h5_file: input hdf5 ファイルパス
    :param binary_flag: 保存ファイル形式
    :param white_list: white_list op名(string default:'') カンマで区切りする
    :param black_list: black_list op名(string default:'') カンマで区切りする
    :return: 保存した onnx ファイルのパス
    """
    assert os.path.exists(json_file), 'jsonファイルが存在しません'
    assert os.path.exists(h5_file), 'hdf5ファイが存在しません'
    K.set_learning_phase(0)
    model = model_from_json(open(json_file).read())
    model.load_weights(h5_file)
    onnx_file = keras_save_to_onnx(K=K, output_path=output_path, ckpt_name=ckpt_name, pb_name=pb_name,
                                   cache_path=cache_path, freeze_pb_name=freeze_pb_name, onnx_name=onnx_name,
                                   model_name=model_name, output_op_name=model.output.op.name,
                                   binary_flag=binary_flag, white_list=white_list, black_list=black_list)
    return onnx_file


def load_yamlh5_to_onnx(output_path, onnx_name, model_name, ckpt_name, pb_name, cache_path, freeze_pb_name, yaml_file,
                        h5_file, binary_flag=True, white_list='', black_list=''):
    """

    :param output_path: 出力パス
    :param onnx_name: 出力onnx ファイル名
    :param model_name: 出力onnx ファイルに記載されるモデル名
    :param ckpt_name: 中間ckpt ファイル名
    :param pb_name: 中間pb ファイル名
    :param cache_path: 中間ファイル用フォルダパス
    :param freeze_pb_name: freezepb ファイル名
    :param yaml_file: input yaml ファイルパス
    :param h5_file: input hdf5 ファイルパス
    :param binary_flag: 保存ファイル形式
    :param white_list: white_list op名(string default:'') カンマで区切りする
    :param black_list: black_list op名(string default:'') カンマで区切りする
    :return: 保存した onnx ファイルのパス
    """
    assert os.path.exists(yaml_file), 'yamlファイルが存在しません'
    assert os.path.exists(h5_file), 'hdf5ファイが存在しません'
    K.set_learning_phase(0)
    model = model_from_yaml(open(yaml_file).read())
    model.load_weights(h5_file)
    onnx_file = keras_save_to_onnx(K=K, output_path=output_path, ckpt_name=ckpt_name, pb_name=pb_name,
                                   cache_path=cache_path, freeze_pb_name=freeze_pb_name, onnx_name=onnx_name,
                                   model_name=model_name, output_op_name=model.output.op.name,
                                   binary_flag=binary_flag, white_list=white_list, black_list=black_list)
    return onnx_file


def keras_save_tf_model(K, output_path, ckpt_name, pb_name, binary_flag=True):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    ckpt_file = os.path.join(output_path, ckpt_name)
    pb_path = output_path
    pb_file = os.path.join(output_path, pb_name)
    tf.train.Saver(write_version=tf.train.SaverDef.V1, max_to_keep=1).save(K.get_session(), ckpt_file)
    tf.train.write_graph(K.get_session().graph, pb_path, pb_name, as_text=not binary_flag)
    return ckpt_file, pb_file


def keras_save_json(model, json_file):
    json_string = model.to_json()
    open(json_file, 'w').write(json_string)


def keras_save_yaml(model, yaml_file):
    yaml_string = model.to_yaml()
    open(yaml_file, 'w').write(yaml_string)
