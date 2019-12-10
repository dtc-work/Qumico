from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE
from onnx import helper

from qumico.handlers.frontend.tflitehandler import TFLiteBaseHandler
from qumico.handlers.frontend.tflitehandler import tflite_op
from qumico.handlers.frontend.tflite.tflite_decorator import tflite_op_conf


@tflite_op("ADD")
class ADD(TFLiteBaseHandler):
    
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        pass
    
    @classmethod
    @tflite_op_conf(["fuse_activation", "quantize"])
    def create_onnx_node(cls, operator, inputs, outputs,
                         input_buffers, output_buffers,
                         data_format, *args, **kwargs):
        node = ADD()
        # _onnx_value_infos
        node.onnx_value_infos.append(helper.make_tensor_value_info(inputs[0].name,
                                                                   NP_TYPE_TO_TENSOR_TYPE[inputs[0].np_tensor_type],
                                                                   inputs[0].shape))
        node.onnx_value_infos.append(helper.make_tensor_value_info(inputs[1].name,
                                                                   NP_TYPE_TO_TENSOR_TYPE[inputs[1].np_tensor_type],
                                                                   inputs[1].shape))

        node.onnx_value_infos.append(helper.make_tensor_value_info(outputs[0].name,
                                                                   NP_TYPE_TO_TENSOR_TYPE[outputs[0].np_tensor_type],
                                                                   outputs[0].shape))
        # tensor
        # node
        node.onnx_nodes.append(helper.make_node('Add',
                                                inputs=[inputs[0].name, inputs[1].name],
                                                outputs=[outputs[0].name]))

        return node
