from os import path

import onnx
from onnx import optimizer

onnx_in_path = path.join(path.dirname(__file__),"onnx", "tiny_yolo_v2_yad2k.onnx")
onnx_out_path = path.join(path.dirname(__file__),"onnx", "tiny_yolo_v2_yad2k_optimize.onnx")

onnx_model = onnx.load(onnx_in_path)

# Pick one pass as example
passes = ['fuse_consecutive_transposes', 'eliminate_nop_transpose', "fuse_bn_into_conv"]

# Apply the optimization on the original model
optimized_model = optimizer.optimize(onnx_model, passes)
onnx.checker.check_model(optimized_model)
# Save the ONNX model
onnx.save(optimized_model, onnx_out_path)