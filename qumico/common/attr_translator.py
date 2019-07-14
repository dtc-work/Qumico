

from qumico.common import data_type

__onnx_attr_translator = {
    "axis": lambda x: int(x),
    "axes": lambda x: [int(a) for a in x],
    "dtype": lambda x: data_type.onnx2np(x),
    "keepdims": lambda x: bool(x),
    "to": lambda x: data_type.onnx2np(x),
}



def translate_onnx(key, val):
    return __onnx_attr_translator.get(key, lambda x: x)(val)