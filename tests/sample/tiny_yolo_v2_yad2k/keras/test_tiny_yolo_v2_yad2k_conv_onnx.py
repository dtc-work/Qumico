import os
import unittest

from samples.tiny_yolo_v2_yad2k.keras import tiny_yolo_v2_yad2k_conv_onnx

import tests


class TestTinyYoloV2Yad2KConvONNX(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.module_path = os.path.abspath(tiny_yolo_v2_yad2k_conv_onnx.__file__)
        cls.current_path = os.path.dirname(os.path.abspath(__file__))
        cls.output_path = os.path.join(cls.current_path, "onnx")

    @classmethod
    def tearDownClass(cls) -> None:
        tests.remove_folder(cls.output_path)

    def test_conv_onnx(self):
        output = tests.read_from_output(lambda: tests.execute_file(self.module_path))

        folder_list = ["model"]
        file_list = ["tiny_yolo_v2_yad2k.onnx"]

        self.assertTrue(tests.is_dir_contains(dirs=self.current_path, folder_list=folder_list))
        self.assertTrue(tests.is_dir_contains(dirs=self.output_path, file_list=file_list))
        self.assertIn("tiny_yolo_v2_yad2k.onnxを作成しました。", output)


if __name__ == "__main__":
    unittest.main()
