import numpy
import os
import unittest

from samples.mlp.tensorflow import mlp_infer
from samples.mlp.tensorflow import mlp_model

import tests
from tests.sample import prepare_infer_dataset

class TestMLPInfer(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = mlp_model.MLP()

        cls.current_path = os.path.dirname(os.path.realpath(__file__))
        cls.output_path = os.path.join(cls.current_path, "model")
        cls.input_file = os.path.join(cls.current_path, "input")
        cls.ckpt_file = os.path.join(cls.input_file, "sample.ckpt")

    def tearDown(self) -> None:
        tests.remove_folder(self.output_path)

    def test_mlp_infer_no_model(self):
        self.assertRaises(AttributeError, lambda: mlp_infer.mlp_infer(model=None, infer_data=None, ckpt_file=None))

    def test_mlp_infer_no_ckpt_file(self):
        self.assertRaises(ValueError, lambda: mlp_infer.mlp_infer(model=self.model,
                                                                  infer_data=None,
                                                                  ckpt_file=None))

    def test_mlp_infer_no_infer_data(self):
        self.assertRaises(ValueError, lambda: mlp_infer.mlp_infer(model=self.model,
                                                                  infer_data=None,
                                                                  ckpt_file=self.ckpt_file))

    def test_mlp_infer(self):
        res = mlp_infer.mlp_infer(model=self.model, infer_data=prepare_infer_dataset(), ckpt_file=self.ckpt_file)
        self.assertCountEqual(numpy.ndarray.tolist(res[0]), [7, 2, 1, 0, 4, 1, 4, 9, 2, 9])


if __name__ == "__import__":
    unittest.main()
