import os
import unittest

from samples.vgg16.keras import vgg16_to_onnx

import tests


class TestVGG16toONNX(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.vgg16_to_onnx_path = os.path.abspath(vgg16_to_onnx.__file__)
        cls.current_path = os.path.abspath(__file__)
        cls.output_path = os.path.join(os.path.dirname(cls.current_path), "onnx")

    def test_vgg16_to_onnx(self):
        output = tests.read_from_output(lambda: tests.execute_file(self.vgg16_to_onnx_path))

        self.assertTrue(tests.is_dir_contains(dirs=self.output_path, file_list=["vgg16.onnx"]))
        self.assertIn("onnx/vgg16.onnxを作成しました。", output)


if __name__ == "__main__":
    unittest.main()