import tensorflow as tf
import os
import unittest

from qumico import Qumico
from qumico import QUMICO_LIB, QUMICO_INCLUDE

import tests


class TestQumico (unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.qumico_object = Qumico.Qumico()
        cls.model_name = "model"
        cls.current_path = os.path.dirname(os.path.realpath(__file__))

        cls.input_path = os.path.join(cls.current_path, "input")
        cls.input_file_name = "sample"

        cls.output_path = os.path.join(cls.current_path, "output")
        cls.output_file = os.path.join(cls.output_path, cls.model_name + ".onnx")

    def tearDown(self) -> None:
        tests.remove_folder(self.output_path)

    def test_instance(self):
        self.assertIsNotNone(self.qumico_object)

    def test_conv_onnx_to_c_no_onnx(self):
        self.assertRaises(FileNotFoundError, lambda: self.qumico_object.conv_onnx_to_c(onnx_path=""))

    def test_conv_onnx_to_c_with_onnx(self):
        self.input_path = os.path.join(self.input_path, "onnx")
        onnx_file = os.path.join(self.input_path, self.input_file_name + ".onnx")
        self.qumico_object.conv_onnx_to_c(onnx_path=onnx_file, out_c_path=self.output_path, compile=False)

        file_list = ["qumico.c", "qumico.h", "qumico_type.h"]
        # file_list = ["qumico.c", "qumico.so", "qumico.h", "qumico_type.h"]
        dir_list = [QUMICO_INCLUDE, QUMICO_LIB]
        self.assertTrue(tests.is_dir_contains(self.output_path, file_list, dir_list))


if __name__ == '__main__':
    unittest.main()
