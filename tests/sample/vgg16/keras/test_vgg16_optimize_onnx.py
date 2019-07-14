import os
import unittest

from samples.vgg16.keras import vgg16_optimize_onnx

import tests


class TestVGG16OptimizeONNX(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.vgg16_optimize_onnx_path = os.path.abspath(vgg16_optimize_onnx.__file__)

    def test_vgg16_optimize_onnx(self):
        try:
            tests.execute_file(self.vgg16_optimize_onnx_path)
        except Exception:
            self.fail("The was an Exception when executing " + vgg16_optimize_onnx.__name__)


if __name__ == "__main__":
    unittest.main()
