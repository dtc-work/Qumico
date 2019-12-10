import numpy as np

from onnx import helper, TensorProto
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from qumico.handlers.frontend.tflitehandler import TFLiteBaseHandler
from qumico.handlers.frontend.tflite.tflite_decorator import tflite_op_conf
from qumico.handlers.frontend.tflitehandler import tflite_op
from qumico.handlers.frontend.tflite.padding import padding as padder
from qumico.handlers.frontend.tflite import create_property_name


@tflite_op("DEPTHWISE_CONV_2D")
class DEPTHWISE_CONV_2D(TFLiteBaseHandler):
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        pass

    @classmethod
    @tflite_op_conf(["fuse_activation"])
    def create_onnx_node(cls, operator, inputs ,outputs,
                         input_buffers, output_buffers, data_format,
                         *args, **kwargs):

        node = DEPTHWISE_CONV_2D()

        # DepthConv Check
        if not (outputs[0].shape[0] * outputs[0].shape[-1] ==inputs[0].shape[0] * inputs[0].shape[-1] and
                outputs[0].shape[0] * outputs[0].shape[-1] ==inputs[1].shape[0] * inputs[1].shape[-1]):
            raise ValueError("Depthwise Shape Error") 


        ops_option = operator.option
        input_names = [i.name for i in inputs]
        output_names = [o.name for o in outputs]
        kernel_shape =inputs[1].shape[1:3]
        group = outputs[0].shape[0] * outputs[0].shape[-1]

        strides = [ops_option.stride_h, ops_option.stride_w]
        dilations = [ops_option.dilation_h_factor, ops_option.dilation_w_factor] 
        padding = ops_option.padding
        pads = [0] * 2 * 2

        pads, input_shape, output_shape = padder(inputs, outputs, padding, pads, strides, dilations, kernel_shape)

        bias  = not len(input_buffers[2]) == 0
        # tensor value
        shape1 = inputs[1].shape
        input1_reshape_shape = (shape1[0] * shape1[-1], 1, shape1[1], shape1[2])

        # input transpose
        input0_name = input_names[0]
        input1_name = input_names[1]
        if bias:
            input2_name = input_names[2]

        input0_transpose_node_name = create_property_name(input_names[0], "Input0Transpose")
        output0_transpose_node_name = create_property_name(output_names[0], "Output0Transpose")
        output_name = output_names[0]

        # todo: check quant
        # input quant
        if inputs[0].quantization is not None:
            input0_x_scale = create_property_name(input_names[0], "x_scale")
            input0_x_zero_point = create_property_name(input_names[0], "x_zero_point")
 
            input0_scale = inputs[0].quantization.scale
            input0_zero_point = inputs[0].quantization.zero_point
            
            if not(inputs[0].quantization.details ==0 and  # ToDo: Refactoring
                   inputs[0].quantization.quantized_dimension == 0):
                raise ValueError("Custome Quantization not supported")
            
        if inputs[1].quantization is not None:
            input1_x_scale = create_property_name(input_names[1], "w_scale")
            input1_x_zero_point = create_property_name(input_names[1], "w_zero_point")

            input1_scale = inputs[1].quantization.scale
            input1_zero_point = inputs[1].quantization.zero_point

            if not(inputs[1].quantization.details ==0 and  # ToDo: Refactoring 
                   inputs[1].quantization.quantized_dimension==0):
                raise ValueError("Custome Quantization not supported")

        if outputs[0].quantization is not None:
            output0_y_scale = create_property_name(output_names[0], "y_scale")
            output0_y_zero_point = create_property_name(output_names[0], "y_zero_point")
        
            output0_scale = outputs[0].quantization.scale
            output0_zero_point = outputs[0].quantization.zero_point

            if not(outputs[0].quantization.details ==0 and   # ToDo: Refactoring
                   outputs[0].quantization.quantized_dimension==0):
                raise ValueError("Custome Quantization not supported")

        # value info input0
        node._onnx_value_infos.append(helper.make_tensor_value_info(input0_name, 
                                                                    NP_TYPE_TO_TENSOR_TYPE[inputs[0].np_tensor_type],
                                                                    inputs[0].shape))

        node._onnx_value_infos.append(helper.make_tensor_value_info(input0_transpose_node_name, 
                                                                    NP_TYPE_TO_TENSOR_TYPE[inputs[0].np_tensor_type],
                                                                    input_shape.tolist()))

        node._onnx_value_infos.append(helper.make_tensor_value_info(input0_x_scale, 
                                                                    TensorProto.FLOAT,
                                                                    (1,)))

        node._onnx_value_infos.append(helper.make_tensor_value_info(input0_x_zero_point, 
                                                                    NP_TYPE_TO_TENSOR_TYPE[inputs[0].np_tensor_type],
                                                                    (1,)))

        node._onnx_value_infos.append(helper.make_tensor_value_info(input1_x_scale, 
                                                                    TensorProto.FLOAT,
                                                                    (1,)))
        # 
        node._onnx_value_infos.append(helper.make_tensor_value_info(input1_x_zero_point, 
                                                                    NP_TYPE_TO_TENSOR_TYPE[inputs[0].np_tensor_type],
                                                                    (1,)))

        if bias:
            node._onnx_value_infos.append(helper.make_tensor_value_info(input2_name, 
                                                                        NP_TYPE_TO_TENSOR_TYPE[inputs[2].np_tensor_type],
                                                                        inputs[2].shape))        
        
        # value info output0_transpose_node_name
        node._onnx_value_infos.append(helper.make_tensor_value_info(output0_y_scale, 
                                                                    TensorProto.FLOAT,
                                                                    (1,)))
        # 
        node._onnx_value_infos.append(helper.make_tensor_value_info(output0_y_zero_point, 
                                                                    NP_TYPE_TO_TENSOR_TYPE[outputs[0].np_tensor_type],
                                                                    (1,)))

        transpose_output_shape =(outputs[0].shape[0], outputs[0].shape[3], outputs[0].shape[1], outputs[0].shape[2])
        node._onnx_value_infos.append(helper.make_tensor_value_info(output0_transpose_node_name, 
                                                                    NP_TYPE_TO_TENSOR_TYPE[outputs[0].np_tensor_type],
                                                                    transpose_output_shape))

        node._onnx_value_infos.append(helper.make_tensor_value_info(output_name, 
                                                                    NP_TYPE_TO_TENSOR_TYPE[outputs[0].np_tensor_type],
                                                                    outputs[0].shape))
        
        # value info input         
        # tensor quant
        node._onnx_tensors.append(helper.make_tensor(input0_x_scale, TensorProto.FLOAT, (1,), input0_scale))
        node._onnx_tensors.append(helper.make_tensor(input0_x_zero_point, NP_TYPE_TO_TENSOR_TYPE[inputs[0].np_tensor_type], (1,), input0_zero_point))
        node._onnx_tensors.append(helper.make_tensor(input1_x_scale, TensorProto.FLOAT, (1,), input1_scale))
        node._onnx_tensors.append(helper.make_tensor(input1_x_zero_point, NP_TYPE_TO_TENSOR_TYPE[inputs[1].np_tensor_type], (1,), input1_zero_point))
        node._onnx_tensors.append(helper.make_tensor(output0_y_scale, TensorProto.FLOAT, (1,), output0_scale))
        node._onnx_tensors.append(helper.make_tensor(output0_y_zero_point, NP_TYPE_TO_TENSOR_TYPE[outputs[0].np_tensor_type], (1,), output0_zero_point))

        transposed_buf = input_buffers[1].transpose((0, 3, 1, 2))
        reshaped_buf = np.reshape(transposed_buf, input1_reshape_shape)
        node._onnx_tensors.append(helper.make_tensor(input1_name,
                                                     NP_TYPE_TO_TENSOR_TYPE[inputs[1].np_tensor_type],
                                                     reshaped_buf.shape,
                                                     reshaped_buf.tobytes()))# input1 Reshape
        if bias:
            node._onnx_tensors.append(helper.make_tensor(input2_name,
                                                         NP_TYPE_TO_TENSOR_TYPE[inputs[2].np_tensor_type],
                                                         inputs[2].shape,
                                                         input_buffers[2]))# input2(bias)
        
        # node
        node._onnx_nodes.append(helper.make_node('Transpose',
                                                 inputs=[input_names[0]], 
                                                 outputs=[input0_transpose_node_name],
                                                 perm =[0,3,1,2]))


        # node conv
        input = [input0_transpose_node_name, input0_x_scale, input0_x_zero_point,
                 input1_name, input1_x_scale, input1_x_zero_point,
                 output0_y_scale, output0_y_zero_point]

        if bias:
            input.append(input2_name)
        
        node._onnx_nodes.append(helper.make_node("QLinearConv", 
                                     inputs=input,
                                     outputs=[output0_transpose_node_name],
                                     dilations=dilations,
                                     group=group,
                                     kernel_shape=kernel_shape,
                                     pads=pads,
                                     strides=strides)) 

        node._onnx_nodes.append(helper.make_node('Transpose',
                                                 inputs=[output0_transpose_node_name], 
                                                 outputs=[output_names[0]],
                                                 perm =[0,2,3,1]))

        return node

