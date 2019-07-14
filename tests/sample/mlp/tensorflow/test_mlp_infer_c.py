import ctypes
import os
import unittest
import numpy

from qumico.Qumico import Qumico
from qumico.export import ExportType

from samples.mlp.tensorflow import mlp_infer_c
from samples.utils import common_tool

import tests
from tests.sample import prepare_infer_dataset


@unittest.skipIf(tests.IS_CIRCLE_CI, "Skip this test because large memory")
class TestMLPInferC(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.current_path = os.path.dirname(os.path.realpath(__file__))
        cls.input_path = os.path.join(cls.current_path, "input")
        cls.output_path = os.path.join(cls.current_path, "output")
        cls.model_path = os.path.join(cls.input_path, "model.onnx")

        cls.so_lib_path = os.path.join(cls.output_path, 'qumico.so')
        cls.input_info = numpy.ctypeslib.ndpointer(dtype=numpy.float32, ndim=2,
                                                   shape=(1, 784), flags='CONTIGUOUS')
        cls.output_info = numpy.ctypeslib.ndpointer(dtype=numpy.float32, ndim=2,
                                                    shape=(1, 10), flags='CONTIGUOUS')

        cls.dll_input = numpy.expand_dims(1, 0).astype(numpy.float32)
        cls._generate_c()

    @classmethod
    def tearDownClass(cls) -> None:
        tests.remove_folder(cls.output_path)

    @classmethod
    def _generate_c(cls):
        Qumico().conv_onnx_to_c(cls.model_path, cls.output_path, export_type=ExportType.C, compile=True)

    def test_init_no_input_info(self):
        self.assertRaises(AttributeError, lambda: mlp_infer_c.init(so_lib_path=None, input_info=None, output_info=None))

    def test_init_no_output_info(self):
        self.assertRaises(AttributeError, lambda: mlp_infer_c.init(so_lib_path=None,
                                                                   input_info=self.input_info,
                                                                   output_info=None))

    def test_init_no_so_lib_path_info(self):
        self.assertRaises(AttributeError, lambda: mlp_infer_c.init(so_lib_path=None,
                                                                   input_info=self.input_info,
                                                                   output_info=self.output_info))

    def test_init(self):
        dll = mlp_infer_c.init(so_lib_path=self.so_lib_path, input_info=self.input_info, output_info=self.output_info)

        self.assertEqual(type(dll), ctypes.CDLL)
        self.assertTrue(hasattr(dll, "qumico"))

    def test_infer_c_no_dll(self):
        self.assertRaises(AttributeError, lambda: mlp_infer_c.infer_c(dll=None, input=None, output=None))

    def test_infer_c_no_input(self):
        dll = mlp_infer_c.init(so_lib_path=self.so_lib_path, input_info=self.input_info, output_info=self.output_info)
        self.assertRaises(ctypes.ArgumentError, lambda: mlp_infer_c.infer_c(dll=dll, input=None, output=None))

    def test_infer_c_no_output(self):
        dll = mlp_infer_c.init(so_lib_path=self.so_lib_path, input_info=self.input_info, output_info=self.output_info)
        self.assertRaises(ctypes.ArgumentError, lambda: mlp_infer_c.infer_c(dll=dll, input=self.dll_input, output=None))

    def test_infer_c(self):
        dll = mlp_infer_c.init(self.so_lib_path, self.input_info, self.output_info)
        res = []

        for i in prepare_infer_dataset():
            output = numpy.zeros(dtype=numpy.float32, shape=(1, 10))
            mlp_infer_c.infer_c(dll, numpy.expand_dims(i, 0).astype(numpy.float32), output)
            classification = common_tool.softmax(output)
            y = common_tool.onehot_decoding(classification)
            res.append(y[0])

        self.assertCountEqual(res, [7, 2, 1, 0, 4, 1, 4, 9, 2, 9])


if __name__ == "__main__":
    unittest.main()