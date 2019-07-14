import os
import unittest

from qumico.Qumico import Qumico
from qumico.export import ExportType
from qumico.device import RaspberryPi3
from qumico.optimize import Optimize
import qumico.handlers.optimize as optimizer

from samples.tiny_yolo_v2_yad2k.keras import build
from samples.tiny_yolo_v2_yad2k.keras import tiny_yolo_v2_yad2k_conv_onnx

import tests


class TestBuild(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.onnx_file_name = "tiny_yolo_v2_yad2k.onnx"
        cls.conv_onnx_path = os.path.abspath(tiny_yolo_v2_yad2k_conv_onnx.__file__)

        cls.current_path = os.path.dirname(os.path.abspath(__file__))
        cls.onnx_folder = os.path.join(cls.current_path, "onnx")
        cls.onnx_file = os.path.join(cls.onnx_folder, cls.onnx_file_name)
        cls.out_c_path = os.path.join(cls.current_path, "out_c")
        cls.c_path = os.path.join(cls.out_c_path, "qumico.c")

        tests.execute_file(cls.conv_onnx_path)
        cls._generate_c()

    @classmethod
    def tearDownClass(cls) -> None:
        tests.remove_folder(cls.out_c_path)
        tests.remove_folder(cls.onnx_folder)

    @classmethod
    def _generate_c(cls):
        _device = RaspberryPi3(neon=False, openmp=False, armv7a=False)
        _optimizer = Optimize(options=[optimizer.FusePrevTranspose])
        Qumico().conv_onnx_to_c(cls.onnx_file, cls.out_c_path, export_type=ExportType.NPY,
                                compile=True, device=_device, optimize=_optimizer)

    def test_build_no_path(self):
        self.assertRaises(TypeError, lambda: build.build(c_path=None))

    def test_build(self):
        try:
            build.build(c_path=self.c_path)
        except Exception:
            self.fail("Error while compiling C File")


if __name__ == "__main__":
    unittest.main()
