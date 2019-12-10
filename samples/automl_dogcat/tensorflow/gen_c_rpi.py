from os import path

from qumico.Qumico import Qumico
from qumico.export import ExportType
from qumico.device import RaspberryPi3
from qumico.optimize import Optimize
import qumico.handlers.optimize as optimizer

if __name__ == "__main__":
    dir_path = path.dirname(__file__)
    onnx_path = path.join(dir_path, "onnx", "model.onnx")
    out_c_path = path.join(dir_path, "out_c")

    # デバイス設定
    d = RaspberryPi3(neon=True,openmp=True,armv7a=True)

    # モデル最適化設定 
    o = Optimize(options=[optimizer.FusePrevTranspose])


    Qumico().conv_onnx_to_c(onnx_path, out_c_path, export_type=ExportType.NPY,
            compile=False, device=d, optimize=o)
    print("Cソースを生成しました。出力先:", out_c_path)
