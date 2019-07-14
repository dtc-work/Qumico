import os
import qumico.common.tf2onnx_tool as converter


def tensorflow_save_onnx(output_path, onnx_name, pb_file, ckpt_file, cache_path, freeze_pb_name, model_name,
                         output_op_name, white_list='', black_list='', binary_flag=True):
    """
    :param output_path: 出力パス
    :param onnx_name: 出力onnx ファイル名
    :param model_name: 出力onnx ファイルに記載されるモデル名
    :param pb_file: input pbファイル名
    :param ckpt_file: input ckptファイル名
    :param cache_path: 中間ファイル用フォルダパス
    :param freeze_pb_name: freezepb ファイル名
    :param output_op_name: モデル名出力する際にoutput op名
    :param K: Keras backend model graphの扱う場所
    :param binary_flag: 保存ファイル形式
    :return: 保存した onnx ファイルのパスを
    :param white_list: white_list op名(string default:'') カンマで区切りする
    :param black_list: black_list op名(string default:'') カンマで区切りする
    :return:
    """
    assert os.path.exists(pb_file), "pbファイルが存在しません"
    assert os.path.exists(ckpt_file), "ckpt_fileが存在しません"

    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    freeze_pb_file = os.path.join(cache_path, freeze_pb_name)
    onnx_file = os.path.join(output_path, onnx_name)
    converter.load_pb_ckpt_to_onnx(pb_file=pb_file, ckpt_file=ckpt_file, freeze_pb_file=freeze_pb_file,
                                   model_name=model_name, onnx_file=onnx_file,
                                   output_op_name=output_op_name, white_list=white_list, black_list=black_list,
                                   binary_flag=binary_flag)
    return onnx_file
