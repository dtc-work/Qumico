import os
import unittest

from samples.mobilenet.tensorflow import mobilenet_infer
import tests


class TestMobilenetInfer(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.current_path = os.path.abspath(os.path.dirname(__file__))
        cls.input_path = os.path.join(cls.current_path, "input")
        cls.output_path = os.path.join(cls.current_path, "output")

        cls.image_input = os.path.join(cls.input_path, "images", "test.jpeg")
        cls.tflite_input = os.path.join(cls.input_path, "model", "mobilenet.tflite")

    def test_infer_no_image(self):
        self.assertRaises(AttributeError, lambda: mobilenet_infer.infer(tflite_model_path=None, image_path=None))

    def test_infer_no_tflite_model_path(self):
        self.assertRaises(ValueError, lambda: mobilenet_infer.infer(tflite_model_path=None,
                                                                    image_path=self.image_input))

    def test_infer(self):
        res = tests.read_from_output(lambda: mobilenet_infer.infer(tflite_model_path=self.tflite_input,
                                                                   image_path=self.image_input))

        self.assertIn("tiger", res)


if __name__ == "__main__":
    unittest.main()
