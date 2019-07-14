import cv2
import numpy
import os
import unittest

from samples.vgg16.keras import vgg16_train
from keras.datasets import mnist
from keras.utils import to_categorical

import tests


class TestVGG16Train(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.current_path = os.path.dirname(os.path.abspath(__file__))
        cls.output_path = os.path.join(cls.current_path, "model")
        cls.vgg16_train = vgg16_train

        cls.num_data = 1000
        cls.classes = list(map(str, list(range(10))))
        cls.classes_num = len(cls.classes)

    @classmethod
    def _prepare_dataset(cls):
        (x_train, y_train), _ = mnist.load_data()
        train_y_one_hot = to_categorical(y_train)[:cls.num_data]

        res = []
        for img in x_train[:cls.num_data]:
            res.append(cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC))

        x_train = numpy.asarray(res)
        x_train = numpy.stack([x_train]*3, axis=3)
        x_train = x_train / 255
        x_train = x_train.astype('float32')
        return x_train, train_y_one_hot

    def test_train_no_train_generator(self):
        self.assertRaises(TypeError, lambda: vgg16_train.train(train_generator=None))

    @unittest.skipIf(tests.IS_CIRCLE_CI, "Skip this test because large memory")
    def test_vgg16_train(self):

        x_train, y_train = self._prepare_dataset()
        self.vgg16_train.data_config(_classes_num=self.classes_num,
                                     _batch_size=50,
                                     _epochs=20,
                                     _classes=self.classes,
                                     _all_data_num=self.num_data,
                                     _train_dir=None,
                                     _x_data=x_train,
                                     _y_data=y_train,
                                     _test_flag=True)

        output = tests.read_from_output(lambda: self.vgg16_train.main())
        file_list = ["sample.json", "sample.yaml", "sample.hdf5"]

        self.assertTrue(tests.is_dir_contains(dirs=self.output_path, file_list=file_list))
        self.assertIn("sample.hdf5", output)
        self.assertIn("を作成しました。", output)


if __name__ == "__main__":
    unittest.main()
