import numpy
import os
import onnx
import unittest

from onnx import helper
from onnx import TensorProto

from qumico import backend
from qumico import QUMICO_EXPORT_ROOT_PATH


class TestBackend(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.qumico_backend = backend.QumicoBackend()

        cls.current_path = os.path.dirname(os.path.realpath(__file__))
        cls.input_path = os.path.join(cls.current_path, "input")
        cls.onnx_path = os.path.join(cls.input_path, "onnx")

        cls.input_file_name = "sample"
        cls.model_path = os.path.join(cls.onnx_path, cls.input_file_name + ".onnx")

        cls.model = onnx.load(cls.model_path)

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

    @classmethod
    def _create_node(cls):
        shape = [3, 2]
        values1 = numpy.array([[0, 1], [2, -3], [-4, 5.6]])
        const2_onnx1 = helper.make_tensor("const2", TensorProto.DOUBLE, shape, values1.flatten().astype(float))
        return helper.make_node("Constant", [], ["X"], value=const2_onnx1)

    def test_qumico_backend_instance(self):
        self.assertIs(type(self.qumico_backend), backend.QumicoBackend)
        self.assertEqual(self.qumico_backend.out_c_path, os.path.abspath(QUMICO_EXPORT_ROOT_PATH))

    def test_qumico_backend_prepare_no_model(self):
        self.assertRaises(AttributeError, lambda: self.qumico_backend.prepare(model=None))

    def test_qumico_backend_prepare(self):
        result = self.qumico_backend.prepare(model=self.model)
        self.assertIs(type(result), backend.QumicoRep)

    def test_qumico_backend_onnx_model_to_qumico_rep_no_model(self):
        self.assertRaises(AttributeError, lambda: self.qumico_backend.onnx_model_to_qumico_rep(model=None,
                                                                                               device=None,
                                                                                               strict=True,
                                                                                               optimize=None))

    def test_qumico_backend_onnx_model_to_qumico_rep(self):
        result = self.qumico_backend.onnx_model_to_qumico_rep(model=self.model,
                                                              device=None,
                                                              strict=True,
                                                              optimize=None)
        self.assertIs(type(result), backend.QumicoRep)

    def test_qumico_backend_run_model_no_model(self):
        self.assertRaises(AttributeError, lambda: self.qumico_backend.run_model(model=None, inputs=None))

    ## postpone to further release
    #def test_qumico_backend_run_model(self):
    #    result = self.qumico_backend.run_model(model=self._create_model(), inputs=None)
    #    self.assertTrue(numpy.array_equal(numpy.array(result[0]), numpy.array([[0.0, 2.0],
    #                                                                           [0.0, 0.0],
    #                                                                           [-8.0, 11.2]])))

    def test_qumico_backend_run_node_no_Node(self):
        self.assertRaises(RuntimeError, lambda: self.qumico_backend.run_node(node=None, inputs=None))

    ## postpone to further release
    #def test_qumico_backend_run_node(self):
    #    result = self.qumico_backend.run_node(node=self._create_node(), inputs=None)
    #    self.assertTrue(numpy.array_equal(numpy.array(result[0]), numpy.array([[0.0, 1.0],
    #                                                                           [2.0, -3.0],
    #                                                                           [-4.0, 5.6]])))

    def test_qumico_backend_supports_device(self):
        self.assertTrue(self.qumico_backend.supports_device(device="CPU"))
        self.assertFalse(self.qumico_backend.supports_device(device="CUDA"))
        self.assertFalse(self.qumico_backend.supports_device(device="Test"))
        self.assertFalse(self.qumico_backend.supports_device(device=""))


if __name__ == "__main__":
    unittest.main()
