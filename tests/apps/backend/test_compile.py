import os
import unittest
import shutil

from qumico import compile
from qumico.Qumico import Qumico


class TestCompile(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.current_path = os.path.dirname(os.path.realpath(__file__))
        cls.input_path = os.path.join(cls.current_path, "input")

        cls.onnx_path = os.path.join(cls.input_path, 'onnx', 'sample.onnx')
        cls.out_c_path = os.path.join(cls.current_path, 'out_c')
        cls.output_path = os.path.join(cls.current_path, 'output')

    def tearDown(self) -> None:
        try:
            shutil.rmtree(self.output_path)
        except FileNotFoundError:
            pass

    def _convert_onnx_to_c(self):
        Qumico().conv_onnx_to_c(self.onnx_path, self.out_c_path)

    def test_node_compile_no_c_path(self):
        self.assertRaises(TypeError, lambda: compile.node_compile(c_path=None, device=None))

    def test_node_compile_no_device(self):
        try:
            self._convert_onnx_to_c()
            os.rename(os.path.join(self.current_path, "out_c"), os.path.join(self.current_path, "output"))
            compile_path = os.path.join(self.output_path, "qumico.c")
            compile.node_compile(c_path=compile_path, device=None)
        except Exception:
            self.fail("Compile test is failed.")


if __name__ == "__main__":
    unittest.main()
