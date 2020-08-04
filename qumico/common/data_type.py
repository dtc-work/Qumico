from numbers import Number

from onnx import mapping
from onnx import TensorProto


def onnx2np(dtype):
    return mapping.TENSOR_TYPE_TO_NP_TYPE[_onnx_dtype(dtype)]


def np2c(dtype):
    tensor_type = mapping.NP_TYPE_TO_TENSOR_TYPE[dtype]
    return TENSOR_TYPE_TO_C_TYPE_STRING[tensor_type]


TENSOR_TYPE_TO_C_TYPE_STRING = {
    TensorProto.FLOAT: "float",
    TensorProto.UINT8: "uint8_t",
    TensorProto.INT8: "int8_t",
    TensorProto.UINT16: "unsigned short int",
    TensorProto.INT16: "short int",
    TensorProto.INT32: "int",
    TensorProto.INT64: "long long int",
    TensorProto.BOOL: "bool",
    TensorProto.FLOAT16: "float",
    TensorProto.DOUBLE: "double",
    TensorProto.COMPLEX64: "float",
    TensorProto.COMPLEX128: "double",
    TensorProto.UINT32: "unsigned int",
    TensorProto.UINT64: "unsigned long int",
    TensorProto.STRING: "char*",
}

C_TYPE_STRING_TO_TENSOR_TYPE = {v: k for k, v in TENSOR_TYPE_TO_C_TYPE_STRING.items()}

def onnx2c(dtype):
    return  TENSOR_TYPE_TO_C_TYPE_STRING[_onnx_dtype(dtype)]


def _onnx_dtype(dtype):
    if isinstance(dtype, Number):
        onnx_dype = dtype
    elif isinstance(dtype, str):
        onnx_dype = TensorProto.DataType.Value(dtype)
    else:
        raise RuntimeError("dtype should be number or str.")
    return onnx_dype
