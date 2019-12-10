import inspect


from qumico.common import IS_PYTHON3


class TFLiteBaseHandler:
    def __init__(self):
        self._onnx_nodes = []
        self._onnx_value_infos = []
        self._onnx_tensors = []
    
    @property    
    def onnx_nodes(self):
        return self._onnx_nodes
    
    @property
    def onnx_value_infos(self):
        return self._onnx_value_infos

    @property
    def onnx_tensors(self):
        return self._onnx_tensors

    def onnx_nodes_append(self, value):
        self._onnx_nodes.append(value)

    def onnx_value_infos_append(self, value):
        self._onnx_value_infos.append(value)

    def onnx_tensors_append(self, value):
        self._onnx_tensors.append(value)

    @classmethod
    def create_onnx_node(cls, operator, inputs, outputs,
                         input_buffers, output_buffers, data_format,
                         *args, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def tflite_op(op):
        return TFLiteBaseHandler.property_register("TFLITE_OP", op)

    @staticmethod
    def property_register(name, value):

        def deco(cls):
            if inspect.isfunction(value) and not IS_PYTHON3:
                setattr(cls, name, staticmethod(value))
            else:
                setattr(cls, name, value)
            return cls

        return deco


tflite_op = TFLiteBaseHandler.tflite_op
