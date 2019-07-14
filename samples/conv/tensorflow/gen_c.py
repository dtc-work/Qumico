from os import path

from qumico.Qumico import Qumico


if __name__ == '__main__':
    dir_path = path.dirname(__file__)
    onnx_path = path.join(dir_path, 'onnx', 'tensorflow_conv.onnx')
    out_c_path = path.join(dir_path, 'out_c')

    Qumico().conv_onnx_to_c(onnx_path, out_c_path)
    print("Cソースを生成しました。出力先:", out_c_path)