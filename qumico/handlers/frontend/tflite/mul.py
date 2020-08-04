from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE
from onnx import helper

from qumico.handlers.frontend.tflitehandler import TFLiteBaseHandler
from qumico.handlers.frontend.tflitehandler import tflite_op
from qumico.handlers.frontend.tflite.tflite_decorator import tflite_op_conf


@tflite_op("MUL")
class MUL(TFLiteBaseHandler):
    
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        pass
    
    @classmethod
    @tflite_op_conf(["fuse_activation", "quantize"])
    def create_onnx_node(cls, operator, inputs, outputs,
                         input_buffers, output_buffers,
                         data_format, *args, **kwargs):
        node = MUL()
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
        if (len(input_buffers[0])>0):
            node.onnx_tensors.append(helper.make_tensor(inputs[0].name,
                                                        NP_TYPE_TO_TENSOR_TYPE[inputs[0].np_tensor_type],
                                                        inputs[0].shape,
                                                        input_buffers[0].tobytes()))

        if (len(input_buffers[1])>0):
            node.onnx_tensors.append(helper.make_tensor(inputs[1].name,
                                                        NP_TYPE_TO_TENSOR_TYPE[inputs[1].np_tensor_type],
                                                        inputs[1].shape,
                                                        input_buffers[1].tobytes()))

        # node
        node.onnx_nodes.append(helper.make_node('Mul',
                                                inputs=[inputs[0].name, inputs[1].name],
                                                outputs=[outputs[0].name]))

        return node
