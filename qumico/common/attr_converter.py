from qumico.common import IS_PYTHON3

def convert_onnx(attr):
    return __convert_onnx_attribute_proto(attr)


def __convert_onnx_attribute_proto(attr_proto):
    """
      Convert an ONNX AttributeProto into an appropriate Python object
      for the type.
      NB: Tensor attribute gets returned as the straight proto.
    """
    if attr_proto.HasField('f'):
        return attr_proto.f
    elif attr_proto.HasField('i'):
        return attr_proto.i
    elif attr_proto.HasField('s'):
        return str(attr_proto.s, 'utf-8') if IS_PYTHON3 else attr_proto.s
    elif attr_proto.HasField('t'):
        return attr_proto.t  # this is a proto!
    elif attr_proto.floats:
        return list(attr_proto.floats)
    elif attr_proto.ints:
        return list(attr_proto.ints)
    elif attr_proto.strings:
        str_list = list(attr_proto.strings)
        if IS_PYTHON3:
            str_list = map(lambda x: str(x, 'utf-8'), str_list)
        return str_list
    else:
        raise ValueError("Unsupported ONNX attribute: {}".format(attr_proto))