import os
import unittest

from samples.mobilenet.tensorflow import build


class TestMobilenetBuild(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.current_path = os.path.abspath(os.path.dirname(__file__))
        cls.input_path = os.path.join(cls.current_path, "input")
        cls.output_path = os.path.join(cls.current_path, "output")

        cls.image_input = os.path.join(cls.input_path, "images", "test.jpg")
        cls.c_path = os.path.join(cls.input_path, "out_c", "qumico.c")

    def test_build_no_c_path(self):
        self.assertRaises(TypeError, lambda: build.build(c_path=None))

    def test_build(self):
        try:
            build.build(c_path=self.c_path)
        except Exception:
            self.fail("There are error when building C files")


if __name__ == "__main__":
    unittest.main()