import numpy
import os
import onnx
import unittest

from onnx import helper
from onnx import TensorProto
from onnx.backend.base import BackendRep

from qumico import backend_rep
from qumico import backend
from qumico.export import ExportType


class TestBackendRep(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.current_path = os.path.dirname(os.path.realpath(__file__))
        cls.input_path = os.path.join(cls.current_path, "input")
        cls.onnx_path = os.path.join(cls.input_path, "onnx")

        cls.input_file_name = "sample"
        cls.model_path = os.path.join(cls.onnx_path, cls.input_file_name + ".onnx")
        cls.model = onnx.load(cls.model_path)

        cls.out_c_path = os.path.join(cls.current_path, 'out_c')
        cls.output_path = os.path.join(cls.current_path, 'output')

        cls.qumico_rep = backend.QumicoBackend().prepare(cls._create_model())

    @classmethod
    def _create_model(cls):
        shape = [3, 2]

        values1 = numpy.array([[0, 1], [2, -3], [-4, 5.6]])
        values2 = numpy.array([[0, 1], [-2, 3], [-4, 5.6]])

        const2_onnx1 = helper.make_tensor("const2", TensorProto.DOUBLE, shape, values1.flatten().astype(float))
        const2_onnx2 = helper.make_tensor("const2", TensorProto.DOUBLE, shape, values2.flatten().astype(float))
        node_def1 = helper.make_node("Constant", [], ["X"], value=const2_onnx1)
        node_def2 = helper.make_node("Constant", [], ["Y"], value=const2_onnx2)
        node_def3 = helper.make_node("Add", ["X", "Y"], ["Z"])
        graph_def = helper.make_graph([node_def1, node_def2, node_def3], name="test",
                                      inputs=[],
                                      outputs=[helper.make_tensor_value_info("Z", TensorProto.DOUBLE, [3, 2])])
        return helper.make_model(graph_def)

    def test_backend_rep_qumico_rep_instance(self):
        self.assertIs(type(self.qumico_rep), backend_rep.QumicoRep)
        self.assertTrue(issubclass(type(self.qumico_rep), BackendRep))

    def test_backend_rep_qumico_rep_convert_c(self):
        self.qumico_rep.convert_c(export_type=ExportType.C)

    def test_backend_rep_qumico_rep_run(self):
        result = self.qumico_rep.run(inputs=[])
        self.assertTrue(numpy.array_equal(numpy.array(result[0]), numpy.array([[0.0, 2.0],
                                                                               [0.0, 0.0],
                                                                               [-8.0, 11.2]])))


if __name__ == "__main__":
    unittest.main()