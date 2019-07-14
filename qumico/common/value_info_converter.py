
import numpy as np

from onnx import mapping

from qumico.common import data_type

def conv(value_info_proto):
    return __convert_onnx_value_info_proto(value_info_proto)

def __convert_onnx_value_info_proto(value_info_proto):
    shape = list(d.dim_value if (d.dim_value > 0 and d.dim_param == "") else 1
        for d in value_info_proto.type.tensor_type.shape.dim)
    print(shape)
    return np.empty(shape=shape,dtype=data_type.onnx2np(value_info_proto.type.tensor_type.elem_type))
 