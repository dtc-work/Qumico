import os
import unittest

from samples.conv.tensorflow import conv_model
from samples.conv.tensorflow import conv_test

import tests
from tests.sample import prepare_test_dataset


class TestConvTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = conv_model.CONV()
        cls.dataset_test = prepare_test_dataset()

        cls.current_path = os.path.dirname(os.path.realpath(__file__))
        cls.input_file = os.path.join(cls.current_path, "input")
        cls.output_path = os.path.join(cls.current_path, "model")

        cls.ckpt_file = os.path.join(cls.input_file, "sample.ckpt")

    def tearDown(self) -> None:
        tests.remove_folder(self.output_path)

    def test_conv_test_no_model(self):
        self.assertRaises(AttributeError, lambda: conv_test.conv_test(model=None,
                                                                      test_data=None,
                                                                      ckpt_file=None,
                                                                      batch_size=1000))

    def test_conv_test_no_ckpt_file(self):
        self.assertRaises(ValueError, lambda: conv_test.conv_test(model=self.model,
                                                                  test_data=None,
                                                                  ckpt_file=None,
                                                                  batch_size=1000))

    def test_conv_test_no_test_data_file(self):
        self.assertRaises(AttributeError, lambda: conv_test.conv_test(model=self.model,
                                                                      test_data=None,
                                                                      ckpt_file=self.ckpt_file,
                                                                      batch_size=1000))

    def test_conv_test(self):
        output = tests.read_from_output(lambda: conv_test.conv_test(model=self.model,
                                                                    test_data=self.dataset_test,
                                                                    ckpt_file=self.ckpt_file,
                                                                    batch_size=1000))
        self.assertIn("Total", output)
        self.assertIn("Accuracy", output)


if __name__ == "__main__":
    unittest.main()
