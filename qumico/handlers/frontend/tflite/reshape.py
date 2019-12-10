import numpy as np
from onnx.mapping import  NP_TYPE_TO_TENSOR_TYPE
from onnx import helper, TensorProto

from qumico.handlers.frontend.tflitehandler import TFLiteBaseHandler
from qumico.handlers.frontend.tflitehandler import tflite_op


@tflite_op("RESHAPE")
class RESHAPE(TFLiteBaseHandler):
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        pass

    @classmethod
    def create_onnx_node(cls, operator, inputs ,outputs,
                         input_buffers, output_buffers,
                          data_format, *args, **kwargs):
                
        node = RESHAPE()
        
        ops_option = operator.option
        new_shape = np.array(ops_option.new_shape) # => conv to onnx tensor

        # input & output name
        input_shape_name = inputs[0].name + "/" + "NewShape"

        # value info
        node._onnx_value_infos.append(helper.make_tensor_value_info(inputs[0].name , 
                                                         NP_TYPE_TO_TENSOR_TYPE[inputs[0].np_tensor_type], 
                                                         inputs[0].shape))

        node._onnx_value_infos.append(helper.make_tensor_value_info(input_shape_name, 
                                                         TensorProto.INT64, 
                                                         new_shape.shape))

        node._onnx_value_infos.append(helper.make_tensor_value_info(outputs[0].name , 
                                                          NP_TYPE_TO_TENSOR_TYPE[outputs[0].np_tensor_type], 
                                                          outputs[0].shape))
        
        # tensor
        node._onnx_tensors.append(helper.make_tensor(input_shape_name, 
                                                     TensorProto.INT64, 
                                                     new_shape.shape, 
                                                     new_shape))

        # node
        node._onnx_nodes.append(helper.make_node('Reshape',
                                                 inputs=[inputs[0].name, input_shape_name],
                                                 outputs=[outputs[0].name]))

        return node