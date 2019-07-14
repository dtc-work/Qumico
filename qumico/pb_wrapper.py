

from qumico.common import attr_converter
from qumico.common import attr_translator
from qumico.common import IS_PYTHON3


# TODO: Move this into ONNX main library
class OnnxNode:
  """
  Reimplementation of NodeProto from ONNX, but in a form
  more convenient to work with from Python.
  """

  def __init__(self, node):
    self.name = str(node.name)
    self.op_type = str(node.op_type)
    self.domain = str(node.domain)
    self.attrs = dict([(attr.name,
                        attr_translator.translate_onnx(
                            attr.name, attr_converter.convert_onnx(attr)))
                       for attr in node.attribute])
    self.inputs = list(node.input)
    self.outputs = list(node.output)
    self.node_proto = node