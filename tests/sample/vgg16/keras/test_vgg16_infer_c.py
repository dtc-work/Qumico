import ctypes
import numpy
import os
import unittest

from keras.preprocessing.image import load_img, img_to_array

from qumico.Qumico import Qumico
from qumico.export import ExportType

from samples.vgg16.keras import vgg16_infer_c
from samples.vgg16.keras import gen_c

import tests


@unittest.skipIf(tests.IS_CIRCLE_CI, "Skip this test because large memory")
class TestVGG16InferC(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.vgg16_infer_path = os.path.abspath(vgg16_infer_c.__file__)
        cls.gen_c_path = os.path.abspath(gen_c.__file__)

        cls.current_path = os.path.dirname(os.path.realpath(__file__))
        cls.input_path = os.path.join(cls.current_path, "onnx")
        cls.output_path = os.path.join(cls.current_path, "out_c")
        cls.model_path = os.path.join(cls.input_path, "vgg16.onnx")

        cls.so_lib_path = os.path.join(os.path.dirname(__file__), 'out_c', 'qumico.so')
        cls.input_info = numpy.ctypeslib.ndpointer(dtype=numpy.float32, ndim=4,
                                                   shape=(1, 224, 224, 3), flags='CONTIGUOUS')
        cls.output_info = numpy.ctypeslib.ndpointer(dtype=numpy.float32, ndim=2,
                                                    shape=(1, 10), flags='CONTIGUOUS')

        cls.classes_test = list(map(str, list(range(10))))

        cls._generate_c()

    @classmethod
    def tearDownClass(cls) -> None:
        tests.remove_folder(cls.output_path)

    @classmethod
    def _generate_c(cls):
        Qumico().conv_onnx_to_c(cls.model_path, cls.output_path, export_type=ExportType.C, compile=True)

    def test_init_no_input_info(self):
        self.assertRaises(AttributeError, lambda: vgg16_infer_c.init(so_lib_path=None, input_info=None, output_info=None))

    def test_init_no_output_info(self):
        self.assertRaises(AttributeError, lambda: vgg16_infer_c.init(so_lib_path=None,
                                                                     input_info=self.input_info,
                                                                     output_info=None))

    def test_init_no_so_lib_path_info(self):
        self.assertRaises(AttributeError, lambda: vgg16_infer_c.init(so_lib_path=None,
                                                                     input_info=self.input_info,
                                                                     output_info=self.output_info))

    def test_init(self):
        dll = vgg16_infer_c.init(so_lib_path=self.so_lib_path, input_info=self.input_info, output_info=self.output_info)

        self.assertEqual(type(dll), ctypes.CDLL)
        self.assertTrue(hasattr(dll, "qumico"))

    def test_vgg16_infer_c(self):

        count_correct = 0

        for x in self.classes_test:
            file_name = x + ".jpg"
            img_file = os.path.join(os.path.dirname(__file__), "input", file_name)

            img = load_img(img_file, grayscale=False, color_mode='rgb', target_size=(224, 224))
            img = img_to_array(img)

            dll = vgg16_infer_c.init(self.so_lib_path, self.input_info, self.output_info)

            input = numpy.expand_dims(img, axis=0)
            output = numpy.zeros(dtype=numpy.float32, shape=(1, 10))
            dll.qumico(input, output)

            result_index = numpy.argmax(output, axis=-1)

            for i in result_index:
                if x in self.classes_test[i]:
                    count_correct += 1

        accuracy = count_correct / len(self.classes_test)
        self.assertGreaterEqual(accuracy, 0.7)


if __name__ == "__main__":
    unittest.main()

