import os
import unittest

from samples.mlp.tensorflow import mlp_model
from samples.mlp.tensorflow import mlp_train

import tests
from tests.sample import prepare_train_dataset


class TestMLPTrain(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = mlp_model.MLP()
        cls.dataset = prepare_train_dataset()

        cls.current_path = os.path.dirname(os.path.realpath(__file__))
        cls.output_path = os.path.join(cls.current_path, "model")

    def tearDown(self) -> None:
        tests.remove_folder(self.output_path)

    def test_mlp_train_no_train_data(self):
        self.assertRaises(AttributeError, lambda: mlp_train.mlp_train(model=self.model,
                                                                      train_data=None,
                                                                      epoch=3,
                                                                      batch_size=50))

    def test_mlp_train_no_model(self):
        self.assertRaises(AttributeError, lambda: mlp_train.mlp_train(model=None,
                                                                      train_data=self.dataset,
                                                                      epoch=3,
                                                                      batch_size=50))

    @unittest.skipIf(tests.IS_CIRCLE_CI, "Skip this test because large memory")
    def test_mlp_train(self):
        ckpt, pb = mlp_train.mlp_train(model=self.model, train_data=self.dataset, epoch=3, batch_size=50)
        file_list = ["checkpoint", "sample.ckpt", "sample.ckpt.meta", "sample.pb"]
        self.assertEqual(ckpt, os.path.join("model", "sample.ckpt"))
        self.assertEqual(pb, os.path.join("model", "sample.pb"))
        self.assertTrue(tests.is_dir_contains(self.output_path, file_list))


if __name__ == "__main__":
    unittest.main()
