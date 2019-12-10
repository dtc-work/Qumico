from os import path

from qumico.frontend import TFLiteFrontend


if __name__ == "__main__":
    current_path = path.abspath(path.dirname(__file__))
    tflite_path=path.join(current_path, "model", "model.tflite")
    output_path="onnx"

    frontend = TFLiteFrontend()
    frontend.convert_tflite_to_onnx(tflite_path=tflite_path,
                                    export_path=output_path)
