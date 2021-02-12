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


