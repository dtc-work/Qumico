import os
import shutil
import unittest

from qumico.Qumico import Qumico
from qumico.export import ExportType
from qumico.device import RaspberryPi3
from qumico.optimize import Optimize
import qumico.handlers.optimize as optimizer

from samples.tiny_yolo_v2_yad2k.keras import gen_c_optimize
from samples.tiny_yolo_v2_yad2k.keras import tiny_yolo_v2_yad2k_infer_c
from samples.tiny_yolo_v2_yad2k.keras import tiny_yolo_v2_yad2k_optimize_onnx

import tests


class TestTinyYoloV2Yad2KInferC(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        tests.execute_file(os.path.abspath(gen_c_optimize.__file__))

        cls.optimizer_file_name = "tiny_yolo_v2_yad2k_optimize.onnx"

        cls.module_file_dir = os.path.abspath(tiny_yolo_v2_yad2k_infer_c.__file__)

        cls.current_path = os.path.dirname(os.path.abspath(__file__))
        cls.onnx_file = os.path.join(cls.current_path, "onnx", cls.optimizer_file_name)
        cls.out_c_path = os.path.join(cls.current_path, "out_c_optimize")

        cls.optimizer_module_path = os.path.abspath(tiny_yolo_v2_yad2k_optimize_onnx.__file__)
        cls.optimizer_output_path = os.path.join(os.path.dirname(cls.optimizer_module_path), "onnx")
        cls.optimizer_output_file = os.path.join(cls.optimizer_output_path, cls.optimizer_file_name)

        if not os.path.exists(os.path.join(cls.current_path, "onnx")):
            os.mkdir(os.path.join(cls.current_path, "onnx"))

        cls._generate_optimized_onnx()
        cls._generate_c()

    @classmethod
    def tearDownClass(cls) -> None:
        tests.remove_folder(cls.out_c_path)
        tests.remove_folder(os.path.join(os.path.dirname(cls.module_file_dir), "out_c_optimize"))

    @classmethod
    def _generate_c(cls):

        _device = RaspberryPi3(neon=False, openmp=False, armv7a=False)
        _optimizer = Optimize(options=[optimizer.FusePrevTranspose])
        Qumico().conv_onnx_to_c(cls.onnx_file, cls.out_c_path, export_type=ExportType.NPY,
                                compile=True, device=_device, optimize=_optimizer)

    @classmethod
    def _generate_optimized_onnx(cls):
        tests.execute_file(cls.optimizer_module_path)
        shutil.copyfile(cls.optimizer_output_file, os.path.join(cls.current_path, "onnx", cls.optimizer_file_name))

    def test_infer_c(self):
        output = tests.read_from_output(lambda: tests.execute_file(self.module_file_dir))
        self.assertIn("sheep", output)
        self.assertIn("person", output)
        self.assertIn("cow", output)


if __name__ == "__main__":
    unittest.main()