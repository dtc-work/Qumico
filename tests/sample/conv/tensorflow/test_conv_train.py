import os
import unittest

from samples.conv.tensorflow import conv_model
from samples.conv.tensorflow import conv_train

import tests
from tests.sample import prepare_train_dataset


class TestConvTrain(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = conv_model.CONV()
        cls.dataset = prepare_train_dataset()

        cls.current_path = os.path.dirname(os.path.realpath(__file__))
        cls.output_path = os.path.join(cls.current_path, "model")

    def tearDown(self) -> None:
        tests.remove_folder(self.output_path)

    def test_conv_train_no_train_data(self):
        self.assertRaises(AttributeError, lambda: conv_train.conv_train(model=self.model,
                                                                        train_data=None,
                                                                        epoch=3,
                                                                        batch_size=50))

    def test_conv_train_no_model(self):
        self.assertRaises(AttributeError, lambda: conv_train.conv_train(model=None,
                                                                        train_data=self.dataset,
                                                                        epoch=3,
                                                                        batch_size=50))

    @unittest.skipIf(tests.IS_CIRCLE_CI, "Skip this test because large memory")
    def test_conv_train(self):
        ckpt, pb = conv_train.conv_train(model=self.model, train_data=self.dataset, epoch=3, batch_size=50)
        file_list = ["checkpoint", "sample.ckpt", "sample.ckpt.meta", "sample.pb"]
        self.assertEqual(ckpt, os.path.join("model", "sample.ckpt"))
        self.assertEqual(pb, os.path.join("model", "sample.pb"))
        self.assertTrue(tests.is_dir_contains(self.output_path, file_list))


if __name__ == "__main__":
    unittest.main()
