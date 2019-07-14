import keras
import os
import unittest

from qumico import Qumico
from qumico import QUMICO_LIB, QUMICO_INCLUDE

import tests


class TestQumico (unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.qumico_object = Qumico.Qumico()
        cls.model_name = "model"
        cls.current_path = os.path.dirname(os.path.realpath(__file__))

        cls.input_path = os.path.join(cls.current_path, "input")
        cls.input_file_name = "sample"

        cls.output_path = os.path.join(cls.current_path, "output")
        cls.output_file = os.path.join(cls.output_path, cls.model_name + ".onnx")

    def tearDown(self) -> None:
        tests.remove_folder(self.output_path)

    def test_instance(self):
        self.assertIsNotNone(self.qumico_object)

    def test_conv_tf_to_onnx_no_dataset(self):
        self.assertRaises(TypeError, lambda: self.qumico_object.conv_tf_to_onnx(self.output_path, self.model_name))

    def test_conv_tf_to_onnx_no_output_path(self):
        self.assertRaises(TypeError, lambda: self.qumico_object.conv_tf_to_onnx(output_path=None,
                                                                                model_name=self.model_name,
                                                                                output_op_name='output',
                                                                                cache_path=self.output_path))

    def test_conv_tf_to_onnx_no_model_name(self):
        self.assertRaises(TypeError, lambda: self.qumico_object.conv_tf_to_onnx(output_path=self.output_path,
                                                                                model_name=None,
                                                                                output_op_name='output',
                                                                                cache_path=self.output_path))

    def test_conv_tf_to_onnx_with_dataset(self):
        self.input_path = os.path.join(self.input_path, "tensorflow")
        pb_file = os.path.join(self.input_path, self.input_file_name + ".pb")
        ckpt_file = os.path.join(self.input_path, self.input_file_name + ".ckpt")

        self.qumico_object.conv_tf_to_onnx(output_path=self.output_path, model_name=self.model_name,
                                           output_op_name='output', cache_path=self.output_path,
                                           ckpt_file=ckpt_file, pb_file=pb_file)

        self.assertTrue(os.path.isfile(self.output_file))

    def test_keras_to_onnx_no_dataset(self):
        self.assertRaises(TypeError, lambda: self.qumico_object.conv_keras_to_onnx(self, self.model_name))

    def test_keras_to_onnx_with_dataset_no_model_type(self):
        output = self.qumico_object.conv_keras_to_onnx(model_type=None, output_path=self.output_path,
                                                       model_name=self.model_name, output_op_name='output',
                                                       cache_path=self.output_path)

        self.assertIsNone(output)
        self.assertFalse(os.path.isfile(self.output_file))

    @unittest.skipIf(tests.IS_CIRCLE_CI, "Skip this test because large memory")
    def test_keras_to_onnx_with_dataset_keras(self):

        self.input_path = os.path.join(self.input_path, "keras")
        model = keras.applications.vgg16.VGG16(weights='imagenet', include_top=True, input_tensor=None,
                                               input_shape=(224, 224, 3))

        output = self.qumico_object.conv_keras_to_onnx(model_type="keras", output_path=self.output_path,
                                                       model_name=self.model_name,  output_op_name=model.output.op.name,
                                                       cache_path=self.output_path, K=keras.backend)
        self.assertIsNotNone(output)
        self.assertTrue(os.path.isfile(self.output_file))

    def test_keras_to_onnx_with_dataset_json(self):
        self.input_path = os.path.join(self.input_path, "keras")
        json_file = os.path.join(self.input_path, self.input_file_name + ".json")
        h5_file = os.path.join(self.input_path, self.input_file_name + ".hdf5")

        output = self.qumico_object.conv_keras_to_onnx(model_type="json", output_path=self.output_path,
                                                       model_name=self.model_name, output_op_name='output',
                                                       cache_path=self.output_path, json_file=json_file,
                                                       h5_file=h5_file)
        self.assertIsNotNone(output)
        self.assertTrue(os.path.isfile(self.output_file))

    def test_keras_to_onnx_with_dataset_yaml(self):
        self.input_path = os.path.join(self.input_path, "keras")
        yaml_file = os.path.join(self.input_path, self.input_file_name + ".yaml")
        h5_file = os.path.join(self.input_path, self.input_file_name + ".hdf5")

        output = self.qumico_object.conv_keras_to_onnx(model_type="yaml", output_path=self.output_path,
                                                       model_name=self.model_name, output_op_name='output',
                                                       cache_path=self.output_path, yaml_file=yaml_file,
                                                       h5_file=h5_file)

        self.assertIsNotNone(output)
        self.assertTrue(os.path.isfile(self.output_file))

    def test_conv_onnx_to_c_no_onnx(self):
        self.assertRaises(FileNotFoundError, lambda: self.qumico_object.conv_onnx_to_c(onnx_path=""))

    def test_conv_onnx_to_c_with_onnx(self):
        self.input_path = os.path.join(self.input_path, "onnx")
        onnx_file = os.path.join(self.input_path, self.input_file_name + ".onnx")
        self.qumico_object.conv_onnx_to_c(onnx_path=onnx_file, out_c_path=self.output_path)

        file_list = ["qumico.c", "qumico.h", "qumico_type.h"]
        # file_list = ["qumico.c", "qumico.so", "qumico.h", "qumico_type.h"]
        dir_list = [QUMICO_INCLUDE, QUMICO_LIB]
        self.assertTrue(tests.is_dir_contains(self.output_path, file_list, dir_list))


if __name__ == '__main__':
    unittest.main()
