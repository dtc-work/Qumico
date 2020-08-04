from collections import OrderedDict
from functools import reduce, wraps

from onnx import helper, TensorProto
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from qumico.ir.tflite_builtin_op_options import ActivationFunctionType
from qumico.handlers.frontend.tflite import create_property_name


def tflite_op_conf(decorate_list):
    def _decorate(function):
        @wraps(function)
        def wrapper(*args, **kwargs):

            node = function(*args, **kwargs)
            function_dict = OrderedDict([("quantize", quantizing),
                                         ("fuse_activation", fuse_activation_function)])

            func_list = [function_dict[x] for x in [x for x in list(function_dict.keys()) if x in decorate_list]]
            return reduce(lambda r, f: f(r, **kwargs), set(func_list), node)

        return wrapper
    return _decorate


def fuse_activation_function(node, **kwargs):

    operator = kwargs.get("operator")

    if operator.option.fused_activation_function == ActivationFunctionType.NONE:
        pass
    else:
        raise NotImplementedError("Activation function for operator node is not implemented yet.")

    return node


def quantizing(node, **kwargs):

    transpose = True if (node.TFLITE_OP == "AVERAGE_POOL_2D" or node.TFLITE_OP == 'MAX_POOL_2D') else False

    if transpose:
        inputs = kwargs.get("inputs")
        operator_idx = kwargs.get("operator_idx")
        for i in range(len(node.onnx_nodes[0].input)):
            
            node.onnx_nodes[0].input[i] = create_property_name(inputs[i].name + "_" + str(operator_idx), "Dequantize")
            input_x_scale = create_property_name(inputs[i].name, "x_scale")
            input_x_zero_point = create_property_name(inputs[i].name, "x_zero_point")
            input_dequantize_name = create_property_name(inputs[i].name + "_" + str(operator_idx), "Dequantize") 
               
            node.onnx_value_infos.append(helper.make_tensor_value_info(inputs[i].name,
                                                                       NP_TYPE_TO_TENSOR_TYPE[
                                                                           inputs[i].np_tensor_type],
                                                                       inputs[i].shape))
        
            node.onnx_value_infos.append(helper.make_tensor_value_info(input_x_scale,
                                                                       TensorProto.FLOAT,
                                                                       ()))
        
            node.onnx_value_infos.append(helper.make_tensor_value_info(input_x_zero_point,
                                                                       NP_TYPE_TO_TENSOR_TYPE[
                                                                           inputs[i].np_tensor_type],
                                                                       ()))
            
            node.onnx_tensors.append(helper.make_tensor(input_x_scale,
                                                        TensorProto.FLOAT,
                                                        (),
                                                        inputs[i].quantization.scale))
        
            node.onnx_tensors.append(helper.make_tensor(input_x_zero_point,
                                                        NP_TYPE_TO_TENSOR_TYPE[inputs[i].np_tensor_type],
                                                        (),
                                                        inputs[i].quantization.zero_point)) 
           
        # output
        outputs = kwargs.get("outputs")
        node.onnx_nodes[0].output[0] = create_property_name(outputs[0].name, "Quantize")
    
        output_x_scale = create_property_name(outputs[0].name, "x_scale")
        output_x_zero_point = create_property_name(outputs[0].name, "x_zero_point")
        output_quantize_name = create_property_name(outputs[0].name, "Quantize")
    
        node.onnx_value_infos.append(helper.make_tensor_value_info(outputs[0].name,
                                                                   NP_TYPE_TO_TENSOR_TYPE[
                                                                       outputs[0].np_tensor_type],
                                                                   outputs[0].shape))
    
        node.onnx_value_infos.append(helper.make_tensor_value_info(output_x_scale,
                                                                   TensorProto.FLOAT,
                                                                   ()))
    
        node.onnx_value_infos.append(helper.make_tensor_value_info(output_x_zero_point,
                                                                   NP_TYPE_TO_TENSOR_TYPE[
                                                                       outputs[0].np_tensor_type],
                                                                   ()))
    
        node.onnx_tensors.append(helper.make_tensor(output_x_scale,
                                                    TensorProto.FLOAT,
                                                    (),
                                                    outputs[0].quantization.scale))
    
        node.onnx_tensors.append(helper.make_tensor(output_x_zero_point,
                                                    NP_TYPE_TO_TENSOR_TYPE[outputs[0].np_tensor_type],
                                                    (),
                                                    outputs[0].quantization.zero_point))
    

        input_transpose = create_property_name(inputs[0].name + "_" + str(operator_idx), "Transpose")
        output_transpose = create_property_name(outputs[0].name, "Transpose")

        input_transposed_shape = (inputs[0].shape[0],
                                  inputs[0].shape[3],
                                  inputs[0].shape[1],
                                  inputs[0].shape[2])

        node.onnx_value_infos.append(helper.make_tensor_value_info(input_transpose,
                                                                   NP_TYPE_TO_TENSOR_TYPE[
                                                                       inputs[0].np_tensor_type],
                                                                   input_transposed_shape))

        node.onnx_value_infos.append(helper.make_tensor_value_info(input_dequantize_name,
                                                                   TensorProto.FLOAT,
                                                                   # ToDo: support double or float16
                                                                   input_transposed_shape))

        output_transposed_shape = (outputs[0].shape[0],
                                   outputs[0].shape[3],
                                   outputs[0].shape[2],
                                   outputs[0].shape[1])

        node.onnx_value_infos.append(helper.make_tensor_value_info(output_quantize_name,
                                                                   TensorProto.FLOAT,
                                                                   # ToDo: support double or float16
                                                                   output_transposed_shape))

        node.onnx_value_infos.append(helper.make_tensor_value_info(output_transpose,
                                                                   NP_TYPE_TO_TENSOR_TYPE[
                                                                      outputs[0].np_tensor_type],
                                                                   output_transposed_shape))

        node.onnx_nodes.insert(0, helper.make_node('Transpose',
                                                   inputs=[inputs[0].name],
                                                   outputs=[input_transpose],
                                                   perm=[0, 3, 1, 2]))

        input_dequant = list([input_transpose, input_x_scale, input_x_zero_point])
        node.onnx_nodes.insert(1, helper.make_node('DequantizeLinear',
                                                   inputs=input_dequant,
                                                   outputs=[input_dequantize_name]))

        input_quant = list([output_quantize_name, output_x_scale, output_x_zero_point])
        node.onnx_nodes.append(helper.make_node('QuantizeLinear',
                                                inputs=input_quant,
                                                outputs=[output_transpose]))

        node.onnx_nodes.append(helper.make_node('Transpose',
                                                inputs=[output_transpose],
                                                outputs=[outputs[0].name],
                                                perm=[0, 2, 3, 1]))
    else:
        # input
        inputs = kwargs.get("inputs")
        operator_idx = kwargs.get("operator_idx")

        for i in range(len(node.onnx_nodes[0].input)):
            node.onnx_nodes[i].input[i] = create_property_name(inputs[i].name + "_" + str(operator_idx), "Dequantize")
            input_x_scale = create_property_name(inputs[i].name, "x_scale")
            input_x_zero_point = create_property_name(inputs[i].name, "x_zero_point")
            input_dequantize_name = create_property_name(inputs[i].name + "_" + str(operator_idx), "Dequantize") 
               
            node.onnx_value_infos.append(helper.make_tensor_value_info(inputs[i].name,
                                                                       NP_TYPE_TO_TENSOR_TYPE[
                                                                           inputs[i].np_tensor_type],
                                                                       inputs[i].shape))
        
            node.onnx_value_infos.append(helper.make_tensor_value_info(input_x_scale,
                                                                       TensorProto.FLOAT,
                                                                       ()))
        
            node.onnx_value_infos.append(helper.make_tensor_value_info(input_x_zero_point,
                                                                       NP_TYPE_TO_TENSOR_TYPE[
                                                                           inputs[i].np_tensor_type],
                                                                       ()))

            node.onnx_value_infos.append(helper.make_tensor_value_info(input_dequantize_name,
                                                                       TensorProto.FLOAT,
                                                                       # ToDo: support double or float16
                                                                       inputs[i].shape))
            
            node.onnx_tensors.append(helper.make_tensor(input_x_scale,
                                                        TensorProto.FLOAT,
                                                        (),
                                                        inputs[i].quantization.scale))
        
            node.onnx_tensors.append(helper.make_tensor(input_x_zero_point,
                                                        NP_TYPE_TO_TENSOR_TYPE[inputs[i].np_tensor_type],
                                                        (),
                                                        inputs[i].quantization.zero_point)) 
           

            input_dequant = list([inputs[i].name, input_x_scale, input_x_zero_point])

            add_node= helper.make_node('DequantizeLinear',
                                                       inputs=input_dequant,
                                                       outputs=[input_dequantize_name])
            node.onnx_nodes.insert(0, add_node)


        # output
        outputs = kwargs.get("outputs")
        node.onnx_nodes[-1].output[0] = create_property_name(outputs[0].name, "Quantize")
    
        output_x_scale = create_property_name(outputs[0].name, "x_scale")
        output_x_zero_point = create_property_name(outputs[0].name, "x_zero_point")
        output_quantize_name = create_property_name(outputs[0].name, "Quantize")
    
        node.onnx_value_infos.append(helper.make_tensor_value_info(outputs[0].name,
                                                                   NP_TYPE_TO_TENSOR_TYPE[
                                                                       outputs[0].np_tensor_type],
                                                                   outputs[0].shape))
    
        node.onnx_value_infos.append(helper.make_tensor_value_info(output_x_scale,
                                                                   TensorProto.FLOAT,
                                                                   ()))
    
        node.onnx_value_infos.append(helper.make_tensor_value_info(output_x_zero_point,
                                                                   NP_TYPE_TO_TENSOR_TYPE[
                                                                       outputs[0].np_tensor_type],
                                                                   ()))
    
        node.onnx_tensors.append(helper.make_tensor(output_x_scale,
                                                    TensorProto.FLOAT,
                                                    (),
                                                    outputs[0].quantization.scale))
    
        node.onnx_tensors.append(helper.make_tensor(output_x_zero_point,
                                                    NP_TYPE_TO_TENSOR_TYPE[outputs[0].np_tensor_type],
                                                    (),
                                                    outputs[0].quantization.zero_point))

        node.onnx_value_infos.append(helper.make_tensor_value_info(output_quantize_name,
                                                                   TensorProto.FLOAT,
                                                                   # ToDo: support double or float16
                                                                   outputs[0].shape))

        input_quant = list([output_quantize_name, output_x_scale, output_x_zero_point])
        node.onnx_nodes.append(helper.make_node('QuantizeLinear',
                                                inputs=input_quant,
                                                outputs=[outputs[0].name]))

    return node
