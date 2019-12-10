import importlib
from enum import Enum

import numpy as np

import qumico.ir.tflite_builtin_op_options as OP_OPTIONS



"""
this ir is based on schema.fbs at 201904025 
schema.fbs: https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/schema/schema.fbs

TFLiteModelv3
 |-version
 |-operator_codes
 |-subgraphs
 |-description
 |-buffers
 |-metadata_buffer
"""


class TFLiteModelv3:# root_type Model

    def __init__(self, version: int, operator_codes: list, subgraphs: list,
                 description: str, buffers: list, metadata_buffer=None):
        self._version= version
        self._operator_codes = operator_codes
        self._subgraphs = subgraphs
        self._description = description
        self._buffers = buffers
        self._metadata_buffer = metadata_buffer
        
        # additional
        self._np_buffers = [0] * len(buffers)
        for tensor in subgraphs.tensors:
            buffer_index = tensor.buffer
            buffer_shape = tensor.shape


            buf = buffers[buffer_index].data
            element_cnt = np.prod(buffer_shape)
            array = None

            if 4 * element_cnt == len(buf) and tensor.tensor_type == TensorType.INT32:# shoud add float 8 and float32
                # should detect whether type is uint8 or int8
                array =np.reshape(np.array(buf, dtype=TensorType.conv_np_dtype(TensorType.UINT8)).view(np.int32), buffer_shape)
            elif len(buf) !=0:
                array =np.reshape(np.array(buf, dtype=TensorType.conv_np_dtype(tensor.tensor_type)), buffer_shape)
            else:
                array =np.array(buf, dtype=TensorType.conv_np_dtype(tensor.tensor_type))

            self._np_buffers[buffer_index] = array


    @classmethod
    def parse(cls, json_content):
        version  = json_content["version"]

        operator_codes =OperatorCode.parse(json_content.get("operator_codes"))
        subgraphs  =SubGraph.parse( json_content["subgraphs"])
        
        description  = json_content["description"]
        buffers  = Buffer.parse(json_content["buffers"])
        metadata_buffer = json_content.get("metadata_buffer")
        
        return TFLiteModelv3(version, operator_codes, subgraphs,
                             description, buffers, metadata_buffer)
        
    @property
    def np_buffers(self):
        return self._np_buffers
    
    @property
    def get_input_output_tensor_names(self):
        res = []
        for op in self.subgraphs.operators:
            res.extend(op.inputs)
            res.extend(op.outputs)        
        return res

    @property
    def version(self):
        return self._version
    
    @property
    def operator_codes(self):
        return self._operator_codes
    
    @property
    def subgraphs(self):
        return self._subgraphs
    
    @property
    def description(self):
        return self._description

    @property
    def buffers(self):
        return self._buffers

    @property
    def metadata_buffer(self):
        return self._metadata_buffer


    def print(self):
        print("----- version ----")
        print(self.version)
        print("----- builtin_code ----")
        for c in self.operator_codes:
            print(c.builtin_code, c.version)

        print("----- TENSOR ----")
        for i, t in enumerate(self.subgraphs.tensors):
            print(i," ", t.shape, " ", t.tensor_type, " ", t.name, )

        print("----- INPUT and OUTPUT ----")
        print("input shape", self.subgraphs.tensors[self.subgraphs.inputs[0]].shape, self.subgraphs.inputs[0])
        print("output shape", self.subgraphs.tensors[self.subgraphs.outputs[0]].shape, self.subgraphs.outputs[0])

        print("----- OPCODE  ----")
        for o in self.subgraphs.operators:
            if o.builtin_options_type == 'Conv2DOptions':
                input_idx = o.inputs[1]
                weight_tensor = self.subgraphs.tensors[input_idx].shape
                filter_size = (weight_tensor[1], weight_tensor[2])
                filter_num = weight_tensor[0]
                print(o.option.name, " ", o.builtin_options_type, weight_tensor, " inputs: ", o.inputs, " outputs = ", o.outputs)
            elif  o.builtin_options_type == 'DepthwiseConv2DOptions':
                input_idx = o.inputs[1]
                weight_tensor = self.subgraphs.tensors[input_idx].shape
                filter_size = (weight_tensor[1], weight_tensor[2])
                filter_num = weight_tensor[3]
                print(o.option.name, " ", o.builtin_options_type,  weight_tensor, " inputs: ", o.inputs, " outputs = ", o.outputs)
            else:
                print(o.option.name, " ", o.builtin_options_type, " inputs: ", o.inputs, " outputs = ", o.outputs)


class OperatorCode:
    def __init__(self, builtin_code, custom_code=None, version=1):
        self._builtin_code = builtin_code
        self._custom_code = custom_code
        self._version = version

    @classmethod
    def parse(cls, json_content : list):
        res = []

        for op_code in json_content:
            builtin_code =op_code.get("builtin_code")
            custom_code =op_code.get("custom_code")
            op_code_version = op_code.get("version")
            res.append(OperatorCode(builtin_code, custom_code,op_code_version))

        return res
    
    @property
    def builtin_code(self):
        return self._builtin_code

    @property
    def custom_code(self):
        return self._custom_code
    
    @property
    def version(self):
        return self._version


class Buffer: # Buffer
    def __init__(self, data):
        self._data = data
    
    @classmethod
    def parse(cls, json_content):
        res = []

        for buf in json_content:
            if len(buf) == 0:
                res.append(Buffer(data=[]))
            elif len(buf) == 1:
                for _, v in buf.items():
                    res.append(Buffer(data=v))
            else:
                raise ValueError()

        return res

    @property
    def data(self):
        return self._data


class Operator: # Operator
    MODULE_PATH = importlib.import_module(OP_OPTIONS.__name__)

    def __init__(self, opcode_index, inputs, outputs, builtin_options, builtin_options_type,
                 custom_options, custom_options_format, mutating_variable_inputs=None):
        self._opcode_index = opcode_index
        self._inputs = inputs
        self._outputs = outputs
        self._builtin_options = builtin_options # dict
        self._builtin_options_type = builtin_options_type
        self._custom_options = custom_options
        self._custom_options_format = custom_options_format # CustomOptionsFormat
        self._mutating_variable_inputs = mutating_variable_inputs

        # additinal
        self._option = getattr(self.MODULE_PATH, builtin_options_type)(**builtin_options)
        
    @classmethod
    def parse(cls, json_content):

        res = []

        for op in json_content:
            opcode_index =op["opcode_index"]
            inputs = op["inputs"]
            outputs = op["outputs"]
            builtin_options_type =  op["builtin_options_type"]
    
            builtin_options = op["builtin_options"]
            custom_options = op.get("custom_options")
            custom_options_format = op["custom_options_format"]

            mutating_variable_inputs = op.get("mutating_variable_inputs")
            res.append(Operator(opcode_index=opcode_index, inputs=inputs, outputs=outputs, 
                                      builtin_options=builtin_options,builtin_options_type=builtin_options_type,
                                      custom_options=custom_options,
                                      custom_options_format=custom_options_format,
                                      mutating_variable_inputs=mutating_variable_inputs))
        return res

    @property
    def option(self):
        return self._option 
       
    @property
    def opcode_index(self):
        return self._opcode_index

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def builtin_options(self):
        return self._builtin_options

    @property
    def builtin_options_type(self):
        return self._builtin_options_type

    @property
    def custom_options(self):
        return self._custom_options

    @property
    def custom_options_format(self):
        return self._custom_options_format

    @property
    def mutating_variable_inputs(self):
        return self._mutating_variable_inputs
    
        
class SubGraph: # SubGraph
    def __init__(self, tensors, inputs, outputs, operators, name):
        self._tensors = tensors
        self._inputs = inputs
        self._outputs = outputs
        self._operators = operators
        self._name = name

    @classmethod
    def parse(cls, json_content): # only parse first subgraph

        subgraphs_content = json_content[0]
        tensors = Tensor.parse(subgraphs_content["tensors"])

        inputs = subgraphs_content["inputs"] # list
        outputs = subgraphs_content["outputs"] # list
        
        
        operators = Operator.parse(subgraphs_content["operators"])

        name = subgraphs_content.get("name")

        return SubGraph(tensors=tensors,inputs=inputs, outputs=outputs,
                        operators=operators, name=name)

    @property
    def tensors(self):
        return self._tensors
    
    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def operators(self):
        return self._operators

    @property
    def name(self):
        return self._name
        

class Tensor: # Tensor
    def __init__(self, shape, tensor_type, buffer, name, quantization=None, is_variable=False): # tensor_type is originally type in schema.fbs
        self._shape= shape
        self._tensor_type = TensorType[tensor_type]
        self._buffer = buffer
        self._name = name
        self._quantization = quantization
        self._is_variable = is_variable
        # additional
        self._np_tensor_type =TensorType.conv_np_dtype(TensorType[tensor_type])  

    @classmethod
    def parse(cls, json_content):
        res = []
        for tensor in json_content:
            shape = tensor["shape"]
            tensor_type = tensor["type"]
            buffer = tensor["buffer"]
            name = tensor["name"]

            quantization = tensor.get("quantization")
            if quantization is not None:
                quantization=QuantizationParameters.parse(quantization)

            is_variable = tensor.get("is_variable")

            res.append(Tensor(shape=shape, tensor_type=tensor_type, buffer=buffer, name=name, 
                              quantization=quantization, is_variable=is_variable))

        return res
    
    @property
    def np_tensor_type(self):
        return self._np_tensor_type
    
    @property
    def np_buffer(self):
        return self._np_buffer    

    @property
    def shape(self):
        return self._shape

    @property
    def tensor_type(self):
        return self._tensor_type

    @property
    def buffer(self):
        return self._buffer

    @property
    def name(self):
        return self._name
    
    @property
    def quantization(self):
        return self._quantization

    @property
    def is_variable(self):
        return self._is_variable


class TensorType(Enum):
    FLOAT32 = 0
    FLOAT16 = 1
    INT32 = 2
    UINT8 = 3
    INT64  = 4
    STRING = 5
    BOOL = 6
    INT16 = 7
    COMPLEX64 = 8
    INT8 = 9

    @classmethod
    def conv_np_dtype(cls, tt):
        if tt == cls.FLOAT32:
            return np.dtype('float32') 
        elif tt == cls.FLOAT16:
            return np.dtype('float16')
        elif tt == cls.INT32:
            return np.dtype('int32') 
        elif tt == cls.UINT8:
            return np.dtype('uint8') 
        elif tt == cls.STRING:
            return np.dtype(np.object)
        elif tt == cls.BOOL:
            return np.dtype('bool')
        elif tt == cls.INT16:
            return np.dtype('int16')
        elif tt == cls.COMPLEX64:
            return  np.dtype('complex64')
        elif tt == cls.INT8:
            return np.dtype('int8')

class CustomOptionsFormat(Enum): # enum
    FLEXBUFFERS = 0


# Quantization
class QuantizationParameters:
    def __init__(self, qmin, qmax, scale, zero_point, details, quantized_dimension):
        self._qmin = qmin
        self._qmax = qmax
        self._scale = scale
        self._zero_point = zero_point
        self._details = details
        self._quantized_dimension = quantized_dimension

    @classmethod
    def parse(cls, json_content):
        qmin = json_content.get("min")
        qmax = json_content.get("max")
        scale = json_content.get("scale")
        zero_point = json_content.get("zero_point")
        details = json_content.get("details_type")

        if details is not None and details !=0:
            details = QuantizationDetails.CustomQuantization

        quantized_dimension = json_content.get("quantized_dimension")

        return QuantizationParameters(qmin, qmax, scale, zero_point, details, quantized_dimension)
    
    @property
    def qmax(self):
        self._qmax

    @property
    def qmin(self):
        return self._qmin
    
    @property
    def scale(self):
        return self._scale

    @property
    def zero_point(self):
        return self._zero_point

    @property
    def details(self):
        return self._details

    @property
    def quantized_dimension(self):
        return self._quantized_dimension


class CustomQuantization:
    def __init__(self, custom):
        self._custom = custom

    @property
    def custom(self):
        return self._custom


class QuantizationDetails:
    CustomQuantization = None


if __name__ == "__main__":
    pass
        