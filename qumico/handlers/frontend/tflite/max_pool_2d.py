from onnx import helper

from qumico.handlers.frontend.tflitehandler import TFLiteBaseHandler

from qumico.handlers.frontend.tflite.tflite_decorator import tflite_op_conf
from qumico.handlers.frontend.tflitehandler import tflite_op


@tflite_op("MAX_POOL_2D")
class MAX_POOL_2D(TFLiteBaseHandler):
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        pass

    @classmethod
    @tflite_op_conf(["fuse_activation", "quantize"])
    def create_onnx_node(cls, operator, inputs, outputs,
                         input_buffers, output_buffers,
                         data_format, *args, **kwargs):

        node = MAX_POOL_2D()

        # ToDo: integrate  padd calc in conv
        ops_option = operator.option

        padding = ops_option.padding
        auto_pad = 'SAME_UPPER'
        kernel = (ops_option.filter_height, ops_option.filter_width)
        stride = (ops_option.stride_h, ops_option.stride_w)

        node._onnx_nodes.append(helper.make_node(
                                'MaxPool',
                                auto_pad=auto_pad,
                                inputs=[inputs[0].name],
                                outputs=[outputs[0].name],
                                kernel_shape=kernel,
                                strides=stride,
                                pads=[0, 0, 0, 0]))

        return node
