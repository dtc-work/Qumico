from enum import Enum, auto
import numpy as np

from qumico.ir.tflite import TensorType as tflite_type

 
TFLITE_TENSOR_TYPE_TO_NP_TYPE = {
    tflite_type.FLOAT32: np.dtype('float32'),
    tflite_type.FLOAT16: np.dtype('float16'),
    tflite_type.INT32: np.dtype('int32'),
    tflite_type.UINT8: np.dtype('uint8'),
    tflite_type.INT64: np.dtype('int64'),    
    tflite_type.STRING: np.dtype(np.object), 
    tflite_type.BOOL: np.dtype('bool'),
    tflite_type.INT16: np.dtype('int16'),
    tflite_type.COMPLEX64: np.dtype('complex64'),
    tflite_type.INT8: np.dtype('int8')
}

class DataFormat(Enum):
    channels_first = auto()
    channels_last = auto()


def get_version(node):
    return getattr(node, "version_1", None)

# little endian
# def conv_4uint8_to_int32(array):
#     return array.view(np.int32)
# 
# def conv_4int8_to_int32(array):
#     return array.view(np.int32)
# 
# def conv_int32_to_4uint8(array):
#     return array.view(np.uint8)
# 
# def conv_int32_to_4int8(array):
#     return array.view(np.int8)