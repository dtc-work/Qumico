import os
import unittest

from samples.tiny_yolo_v2_yad2k.keras import tiny_yolo_v2_yad2k_optimize_onnx

import tests


class TestTinyYoloV2Yad2KOptimizeOnnx(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.module_path = os.path.abspath(tiny_yolo_v2_yad2k_optimize_onnx.__file__)
        cls.module_dir = os.path.dirname(cls.module_path)
        cls.output_path = os.path.join(cls.module_dir, "onnx")

    def test_optimize_onnx(self):
        tests.execute_file(self.module_path)
        file_list = ["tiny_yolo_v2_yad2k_optimize.onnx"]
        self.assertTrue(tests.is_dir_contains(dirs=self.output_path, file_list=file_list))


if __name__ == "__main__":
    unittest.main()
