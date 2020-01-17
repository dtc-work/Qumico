from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE
from onnx import helper, TensorProto

from qumico.handlers.frontend.tflitehandler import TFLiteBaseHandler
from qumico.handlers.frontend.tflitehandler import tflite_op
from qumico.handlers.frontend.tflite.tflite_decorator import tflite_op_conf
from qumico.ir.tflite_builtin_op_options import FullyConnectedOptionsWeightsFormat
from qumico.handlers.frontend.tflite import create_property_name

"""
   input  weight bias
     |      |     |
     |  transpose |   
     |      |     |   
    Qmatmul-|     |
        |         |
   dequantize     |
        |         |
       add -------|
        |
     quantize
        |
      output

  ※transpose support uint8 & int8
  ※bias type is int32
"""


@tflite_op("FULLY_CONNECTED")
class FULLY_CONNECTED(TFLiteBaseHandler):
    
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        pass
    
    @classmethod
    @tflite_op_conf(["fuse_activation"])
    def create_onnx_node(cls, operator, inputs, outputs,
                         input_buffers, output_buffers,
                         data_format, *args, **kwargs):
        node = FULLY_CONNECTED()
        # _onnx_value_infos
        ops_option = operator.option
        weights_format = ops_option.weights_format
        if weights_format != FullyConnectedOptionsWeightsFormat.DEFAULT:
            raise NotImplementedError("SHUFFLED4x16INT8 not supported")

        input_names = [i.name for i in inputs]
        output_names = [o.name for o in outputs]
        bias = not len(input_buffers[2]) == 0

        input0_name = input_names[0]
        input1_name = input_names[1]
        input2_name = input_names[2] if bias else None

        input1_transpose_node_name = create_property_name(input_names[1], "Input1Transpose")
        if bias:
            input2_cast_node_name = create_property_name(input_names[2], "Input2Cast")
            output0_qlinearmatmul_node_name= create_property_name(output_names[0], "Output0QLinearMatmul")
            output0_dequantize_node_name= create_property_name(output_names[0], "Output0Dequantize")
            output0_add_node_name= create_property_name(output_names[0], "Output0Add")

        # quantization 
        if inputs[0].quantization is not None:
            input0_x_scale = create_property_name(input_names[0], "x_scale")
            input0_x_zero_point = create_property_name(input_names[0], "x_zero_point")

            input0_scale = inputs[0].quantization.scale
            input0_zero_point = inputs[0].quantization.zero_point

            if not (inputs[0].quantization.details == 0 and  # ToDo: Refactoring
                    inputs[0].quantization.quantized_dimension == 0):
                raise ValueError("Custom Quantization not supported")


        if inputs[1].quantization is not None:
            input1_x_scale = create_property_name(input_names[1], "w_scale")
            input1_x_zero_point = create_property_name(input_names[1], "w_zero_point")

            input1_scale = inputs[1].quantization.scale
            input1_zero_point = inputs[1].quantization.zero_point

            if not (inputs[1].quantization.details == 0 and  # ToDo: Refactoring
                    inputs[1].quantization.quantized_dimension == 0):
                raise ValueError("Custom Quantization not supported")

        if inputs[2].quantization is not None:
            input2_x_scale = create_property_name(input_names[2], "w_scale")
            input2_x_zero_point = create_property_name(input_names[2], "w_zero_point")

            input2_scale = inputs[2].quantization.scale
            input2_zero_point = inputs[2].quantization.zero_point

            if not (inputs[2].quantization.details == 0 and  # ToDo: Refactoring
                    inputs[2].quantization.quantized_dimension == 0):
                raise ValueError("Custom Quantization not supported")


        if outputs[0].quantization is not None:
            output0_y_scale = create_property_name(output_names[0], "y_scale")
            output0_y_zero_point = create_property_name(output_names[0], "y_zero_point")

            output0_scale = outputs[0].quantization.scale
            output0_zero_point = outputs[0].quantization.zero_point

            if not (outputs[0].quantization.details == 0 and  # ToDo: Refactoring
                    outputs[0].quantization.quantized_dimension == 0):
                raise ValueError("Custom Quantization not supported")
        
        # value_info input  input0
        node.onnx_value_infos.append(helper.make_tensor_value_info(input0_name,
                                                                   NP_TYPE_TO_TENSOR_TYPE[inputs[0].np_tensor_type],
                                                                   inputs[0].shape))

        node.onnx_value_infos.append(helper.make_tensor_value_info(input0_x_scale,
                                                                   TensorProto.FLOAT,
                                                                   (1,)))

        node.onnx_value_infos.append(helper.make_tensor_value_info(input0_x_zero_point,
                                                                   NP_TYPE_TO_TENSOR_TYPE[inputs[0].np_tensor_type],
                                                                   (1,)))

        # weight
        node.onnx_value_infos.append(helper.make_tensor_value_info(input1_name,
                                                                   NP_TYPE_TO_TENSOR_TYPE[inputs[1].np_tensor_type],
                                                                   inputs[1].shape))

        transposed_shape = inputs[1].shape[::-1]
        node.onnx_value_infos.append(helper.make_tensor_value_info(input1_transpose_node_name,
                                                                   NP_TYPE_TO_TENSOR_TYPE[inputs[1].np_tensor_type],
                                                                   transposed_shape))


        node.onnx_value_infos.append(helper.make_tensor_value_info(input1_x_scale,
                                                                   TensorProto.FLOAT,
                                                                   (1,)))

        node.onnx_value_infos.append(helper.make_tensor_value_info(input1_x_zero_point,
                                                                   NP_TYPE_TO_TENSOR_TYPE[inputs[1].np_tensor_type],
                                                                   (1,)))

        # bias
        if bias:
            node.onnx_value_infos.append(helper.make_tensor_value_info(inputs[2].name,
                                                                       NP_TYPE_TO_TENSOR_TYPE[inputs[2].np_tensor_type],
                                                                       inputs[2].shape))

            node.onnx_value_infos.append(helper.make_tensor_value_info(input2_x_scale,
                                                                       TensorProto.FLOAT,
                                                                       (1,)))

            node.onnx_value_infos.append(helper.make_tensor_value_info(input2_x_zero_point,
                                                           NP_TYPE_TO_TENSOR_TYPE[inputs[2].np_tensor_type],
                                                           (1,))) 

            node.onnx_value_infos.append(helper.make_tensor_value_info(input2_cast_node_name,
                                                                       TensorProto.FLOAT,
                                                                       inputs[2].shape))
        # output
        # output: QLinaerMatmul
        node.onnx_value_infos.append(helper.make_tensor_value_info(output0_qlinearmatmul_node_name,
                                                                   NP_TYPE_TO_TENSOR_TYPE[outputs[0].np_tensor_type],
                                                                   outputs[0].shape)) # (1,2)
        # output: Dequantize
        node.onnx_value_infos.append(helper.make_tensor_value_info(output0_dequantize_node_name,
                                                                    TensorProto.FLOAT,
                                                                   outputs[0].shape)) # (1,2)

        # output: Add
        node.onnx_value_infos.append(helper.make_tensor_value_info(output0_add_node_name,
                                                                   TensorProto.FLOAT,
                                                                   outputs[0].shape)) # (1,2)

        # output: Quantize
        node.onnx_value_infos.append(helper.make_tensor_value_info(outputs[0].name,
                                                                   NP_TYPE_TO_TENSOR_TYPE[outputs[0].np_tensor_type],
                                                                   outputs[0].shape)) # (1,2) 

        
        # tensor
        # tensor: input0
        node.onnx_tensors.append(helper.make_tensor(input0_x_scale, TensorProto.FLOAT, (1,), input0_scale))
        node.onnx_tensors.append(helper.make_tensor(input0_x_zero_point, NP_TYPE_TO_TENSOR_TYPE[inputs[0].np_tensor_type],
                                                     (1,),input0_zero_point))


        node.onnx_tensors.append(helper.make_tensor(input1_name,
                                                    NP_TYPE_TO_TENSOR_TYPE[inputs[1].np_tensor_type],
                                                    input_buffers[1].shape,
                                                    input_buffers[1].tobytes()))
        node.onnx_tensors.append(helper.make_tensor(input1_x_scale, TensorProto.FLOAT, (1,), input1_scale))
        node.onnx_tensors.append(helper.make_tensor(input1_x_zero_point, NP_TYPE_TO_TENSOR_TYPE[inputs[1].np_tensor_type],
                                                     (1,),input1_zero_point))

        # tensor: input2
        if bias:
            node.onnx_tensors.append(helper.make_tensor(input2_name,
                                                        NP_TYPE_TO_TENSOR_TYPE[inputs[2].np_tensor_type],
                                                        inputs[2].shape,
                                                        input_buffers[2]))

            node.onnx_tensors.append(helper.make_tensor(input2_x_scale, TensorProto.FLOAT, (1,), input2_scale))
            node.onnx_tensors.append(helper.make_tensor(input2_x_zero_point, NP_TYPE_TO_TENSOR_TYPE[inputs[2].np_tensor_type],
                                                        (1,), input2_zero_point))

        node.onnx_tensors.append(helper.make_tensor(output0_y_scale, TensorProto.FLOAT, (1,), output0_scale))
        node.onnx_tensors.append(helper.make_tensor(output0_y_zero_point, NP_TYPE_TO_TENSOR_TYPE[outputs[0].np_tensor_type],
                                                    (1,), output0_zero_point))

        # node
#         transpose_perm = list(range(len(input_buffers[1])))[::-1]
        node.onnx_nodes.append(helper.make_node('Transpose',
                                                inputs=[inputs[1].name],
                                                outputs=[input1_transpose_node_name],
                                                perm=[1, 0]))
        if bias:
            node.onnx_nodes.append(helper.make_node('QLinearMatMul',
                                                    inputs=[inputs[0].name, input0_x_scale, input0_x_zero_point,
                                                            input1_transpose_node_name, input1_x_scale, input1_x_zero_point,
                                                            output0_y_scale, output0_y_zero_point],
                                                    outputs=[output0_qlinearmatmul_node_name]))

            node.onnx_nodes.append(helper.make_node('DequantizeLinear',
                                                    inputs=[output0_qlinearmatmul_node_name,output0_y_scale, output0_y_zero_point],
                                                    outputs=[output0_dequantize_node_name]))


            node.onnx_nodes.append(helper.make_node('DequantizeLinear',
                                                    inputs=[inputs[2].name,input2_x_scale, input2_x_zero_point],
                                                    outputs=[input2_cast_node_name]))

            node.onnx_nodes.append(helper.make_node('Add',
                                                    inputs=[output0_dequantize_node_name, input2_cast_node_name],
                                                    outputs=[output0_add_node_name]))

            node.onnx_nodes.append(helper.make_node('QuantizeLinear',
                                                    inputs=[output0_add_node_name, output0_y_scale, output0_y_zero_point],
                                                    outputs=[outputs[0].name]))
    
        else:
            node.onnx_nodes.append(helper.make_node('QLinearMatMul',
                                                    inputs=[inputs[0].name, input0_x_scale, input0_x_zero_point,
                                                            input1_transpose_node_name, input1_x_scale, input1_x_zero_point,
                                                            output0_y_scale, output0_y_zero_point],
                                                    outputs=[outputs[0].name]))

        return node
