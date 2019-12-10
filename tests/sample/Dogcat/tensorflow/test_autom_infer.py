import os
import unittest

from samples.automl_dogcat.tensorflow import automl_infer
import tests


class TestautomltInfer(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.current_path = os.path.abspath(os.path.dirname(__file__))
        cls.output_path = os.path.join(cls.current_path, "output")
        cls.input_path = os.path.join(cls.current_path, "input")

        cls.image_input = os.path.join(cls.input_path, "images", "test1.jpeg")
        #print(cls.image_input)
        cls.tflite_input = os.path.join(cls.input_path, "model", "model.tflite")

    def test_infer_no_image(self):
        self.assertRaises(AttributeError, lambda: automl_infer.infer(tflite_model_path=None, image_path=None))

    def test_infer_no_tflite_model_path(self):
        self.assertRaises(ValueError, lambda: automl_infer.infer(tflite_model_path=None,
                                                                    image_path=self.image_input))

    def test_infer(self):
        res = tests.read_from_output(lambda: automl_infer.infer(tflite_model_path=self.tflite_input,
                                                                   image_path=self.image_input))

        self.assertIn("dog", res)


if __name__ == "__main__":
    unittest.main()
