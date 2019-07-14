from os import path
from qumico.Qumico import Qumico
from qumico.export import ExportType
from qumico.device import RaspberryPi3
from qumico.optimize import Optimize
import qumico.handlers.optimize as optimizer


if __name__ == "__main__":
    # 読み込みoNNXファイルと出力設定
    dir_path = path.dirname(__file__)
    onnx_path = path.join(dir_path, "onnx", "tiny_yolo_v2_yad2k_optimize.onnx")
    out_c_path = path.join(dir_path, "out_c_optimize")

    # デイバス設定
    d = RaspberryPi3(neon=False,openmp=False,armv7a=False)

    # モデル最適化設定 
    o = Optimize(options=[optimizer.FusePrevTranspose])
    
    # Cソース生成 
    Qumico().conv_onnx_to_c(onnx_path, out_c_path,export_type=ExportType.NPY,
                            compile=True, device=d, optimize=o)

    print("Cソースを生成しました。出力先:", out_c_path)
