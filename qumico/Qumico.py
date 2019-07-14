import onnx

from qumico.frontend import KerasFrontend
from qumico.frontend import TensorflowFrontend
from qumico.backend import QumicoBackend
from qumico.export import ExportType

class Qumico:
    """

    ## Qucmio
    onnxからC言語への変換を行うクラス。

　　使用例

    ```python
    # Qumicoのインスタンス化を行います。
    q = Qumico()
    # onnxファイルからCソースを出力します。
    q.conv_onnx_to_c(onnx_path="../onnx/model.onnx", out_c_path="", compile=False)

    ```

    """
    def conv_tf_to_onnx(self, output_path, model_name, binary_flag=True, **kwargs):
        """
        TenforFlow のpbフォーマットからonnxフォーマットへ変換する。
        #### 引数
        - output_path：出力ファイルへのパス
        - model_name：モデルの名前
        - binary_flag：バイナリフラグ（default = True)
        - **kwargs：
        #### 戻り値
        無し
        """
        tensorflow_frontend = TensorflowFrontend()

        if "onnx_name" in kwargs:
            onnx_name = kwargs.get("onnx_name")
        else:
            onnx_name = model_name + ".onnx"
        if "ckpt_file" in kwargs:
            ckpt_file = kwargs.get("ckpt_file")
        else:
            ckpt_file = None
        if "pb_file" in kwargs:
            pb_file = kwargs.get("pb_file")
        else:
            pb_file = None
        if "cache_path" in kwargs:
            cache_path = kwargs.get("cache_path")
        else:
            cache_path = output_path
        if "freeze_pb_name" in kwargs:
            freeze_pb_name = kwargs.get("freeze_pb_name")
        else:
            freeze_pb_name = "freeze_" + model_name + ".pb"
        if "output_op_name" in kwargs:
            output_op_name = kwargs.get("output_op_name")
        else:
            output_op_name = None

        tensorflow_frontend.convert_from_tensorflow(output_path=output_path, onnx_name=onnx_name, model_name=model_name, output_op_name=output_op_name,
                                                    ckpt_file=ckpt_file, pb_file=pb_file, cache_path=cache_path,
                                                    freeze_pb_name=freeze_pb_name, binary_flag=binary_flag)

    def conv_keras_to_onnx(self, model_type, output_path, model_name, binary_flag=True, **kwargs):
        """
        keras のフォーマットからonnxフォーマットへ変換する。
        #### 引数
        - model_type: kerasモデルのフォーマット('keras'/'json'/'yaml'の３種類のみサポート)
        - output_path：出力ファイルへのパス
        - model_name：モデルの名前
        - binary_flag：バイナリフラグ（default = True)
        - **kwargs：
        #### 戻り値
        保存したONNXファイルパスを返す。
        """

        keras_frontend = KerasFrontend()

        if "onnx_name" in kwargs:
            onnx_name = kwargs.get("onnx_name")
        else:
            onnx_name = model_name + ".onnx"
        if "ckpt_name" in kwargs:
            ckpt_name = kwargs.get("ckpt_name")
        else:
            ckpt_name = model_name + ".ckpt"
        if "pb_name" in kwargs:
            pb_name = kwargs.get("pb_name")
        else:
            pb_name = model_name + ".pb"
        if "cache_path" in kwargs:
            cache_path = kwargs.get("cache_path")
        else:
            cache_path = output_path
        if "freeze_pb_name" in kwargs:
            freeze_pb_name = kwargs.get("freeze_pb_name")
        else:
            freeze_pb_name = "freeze_" + model_name + ".pb"
        if "output_op_name" in kwargs:
            output_op_name = kwargs.get("output_op_name")
        else:
            output_op_name = None
        if "json_file" in kwargs:
            json_file = kwargs.get("json_file")
        else:
            json_file = None
        if "yaml_file" in kwargs:
            yaml_file = kwargs.get("yaml_file")
        else:
            yaml_file = None
        if "h5_file" in kwargs:
            h5_file = kwargs.get("h5_file")
        else:
            h5_file = None
        if "K" in kwargs:
            K = kwargs.get("K")
        else:
            K = None

        if model_type == "keras":
            onnx_file = keras_frontend.convert_from_keras(output_path=output_path, onnx_name=onnx_name,
                                                          model_name=model_name,
                                                          binary_flag=binary_flag, ckpt_name=ckpt_name, pb_name=pb_name,
                                                          cache_path=cache_path, freeze_pb_name=freeze_pb_name,
                                                          output_op_name=output_op_name, K=K)

        elif model_type == "json":
            onnx_file = keras_frontend.convert_from_json(output_path=output_path, onnx_name=onnx_name,
                                                         model_name=model_name,
                                                         binary_flag=binary_flag, ckpt_name=ckpt_name, pb_name=pb_name,
                                                         cache_path=cache_path, freeze_pb_name=freeze_pb_name,
                                                         json_file=json_file,
                                                         h5_file=h5_file)
        elif model_type == "yaml":
            onnx_file = keras_frontend.convert_from_yaml(output_path=output_path, onnx_name=onnx_name,
                                                         model_name=model_name,
                                                         binary_flag=binary_flag, ckpt_name=ckpt_name, pb_name=pb_name,
                                                         cache_path=cache_path, freeze_pb_name=freeze_pb_name,
                                                         yaml_file=yaml_file,
                                                         h5_file=h5_file)

        else:
            onnx_file = None
            print("モデル種類typeを入力してください。")

        return onnx_file

    def conv_onnx_to_c(self, onnx_path, out_c_path=None, device=None,
                       compile=True, export_type=ExportType.C, optimize=None, **kwargs):
        """
        onnxフォーマットからCソースコードへ変換する。
        #### 引数
        - onnx_path:onnxファイルのパス
        - out_c_path：出力ファイルへのパス
        - device:ターゲットデバイス指定
        - compile：コンパイル実行フラグ(default = True)
        - export_type：出力形式（default = ExportType.C)
        - **kwargs：
        #### 戻り値
        無し
        """
        
        model = onnx.load(onnx_path)
        rep = QumicoBackend(out_c_path).prepare(model, device, optimize, **kwargs)
        rep.convert_c(compile=compile,export_type=export_type, device=device)


if __name__ == "__main__":
    q = Qumico()
    q.conv_onnx_to_c(onnx_path="../onnx/model.onnx", out_c_path="", compile=False)


