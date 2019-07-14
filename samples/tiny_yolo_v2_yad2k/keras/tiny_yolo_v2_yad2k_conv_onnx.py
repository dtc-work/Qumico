from qumico.Qumico import Qumico

# qumico インスタンス作成
convert = Qumico()
convert.conv_keras_to_onnx(model_type="json", output_path="onnx", model_name="tiny_yolo_v2_yad2k", cache_path="model",
                           json_file="model/tiny_yolo_v2_yad2k.json", h5_file="model/tiny_yolo_v2_yad2k.h5")
