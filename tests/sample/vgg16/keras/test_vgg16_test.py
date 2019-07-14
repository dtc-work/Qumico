import os
import unittest

from keras.models import model_from_json
from keras.datasets import mnist
from keras.utils import to_categorical
from samples.vgg16.keras import vgg16_test

import cv2
import numpy

import tests


class TestVGG16Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:

        cls.current_path = os.path.dirname(os.path.abspath(__file__))
        cls.input_path = os.path.join(cls.current_path, "input")

        cls.classes = list(map(str, list(range(10))))
        cls.num_data = len(cls.classes)

        cls.vgg_test_instance = vgg16_test

        cls.vgg_test_instance.classes = cls.classes
        cls.vgg_test_instance.model_folder = os.path.join(cls.current_path, "model")
        cls.vgg_test_instance.json_file = os.path.join(cls.vgg_test_instance.model_folder, 'sample.json')
        cls.vgg_test_instance.h5_file = os.path.join(cls.vgg_test_instance.model_folder, 'sample.hdf5')

        cls.vgg_test_instance.model = model_from_json(open(cls.vgg_test_instance.json_file).read())
        cls.vgg_test_instance.test_dir = cls.input_path

        x_test, y_test = cls._prepare_dataset()
        cls.vgg_test_instance.predict_generator = cls.vgg_test_instance.predict_datagen.flow(x=x_test,
                                                                                             y=y_test,
                                                                                             batch_size=1)

        if not os.path.exists(os.path.join(cls.current_path, "test_data")):
            os.mkdir(os.path.join(cls.current_path, "test_data"))

    # @classmethod
    # def tearDownClass(cls) -> None:
    #     tests.remove_folder(os.path.join(cls.current_path, "test_data"))

    @classmethod
    def _prepare_dataset(cls):
        _, (x_test, y_test) = mnist.load_data()
        test_y_one_hot = to_categorical(y_test)[:cls.num_data]

        res = []
        for img in x_test[:cls.num_data]:
            res.append(cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC))

        x_test = numpy.asarray(res)
        x_test = numpy.stack([x_test] * 3, axis=3)
        x_test = x_test / 255
        x_test = x_test.astype('float32')
        return x_test, test_y_one_hot

    def test_vgg16_test(self):
        output = tests.read_from_output(lambda: self.vgg_test_instance.main())
        self.assertIn("image : ", output)
        self.assertIn("1", output)


if __name__ == "__main__":
    unittest.main()