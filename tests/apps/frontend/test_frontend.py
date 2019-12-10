import keras
import os
import shutil
import unittest

from qumico import QumicoRootPath
from qumico import frontend

import tests


class TestFrontend (unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.model_name = "model"
        cls.pb_name = cls.model_name + ".pb"
        cls.ckpt_name = cls.model_name + ".ckpt"
        cls.freeze_pb_name = "freeze_" + cls.model_name + ".pb"

        cls.current_path = os.path.dirname(os.path.realpath(__file__))

        cls.input_path = os.path.join(cls.current_path, "input")
        cls.input_file_name = "sample"

        cls.output_path = os.path.join(cls.current_path, "output")
        cls.output_file = os.path.join(cls.output_path, cls.model_name + ".onnx")

        cls.tflite_path = os.path.join(cls.input_path, "tflite", "model.tflite")
        cls.flat_c_path = os.path.join(QumicoRootPath, "..", "external", "flatbuffers", "flatc")
        cls.schema_path = os.path.join(QumicoRootPath, "..", "external", "flatbuffers", "schema.fbs")

    def tearDown(self) -> None:
        try:
            shutil.rmtree(self.output_path)
        except FileNotFoundError:
            pass

    def test_base_frontend_instance(self):
        base_frontend = frontend.BaseFrontend()
        self.assertIs(type(base_frontend), frontend.BaseFrontend)
 
    def test_keras_frontend_instance(self):
        keras_frontend = frontend.KerasFrontend()
        self.assertIs(type(keras_frontend), frontend.KerasFrontend)
        self.assertTrue(issubclass(type(keras_frontend), frontend.BaseFrontend))
 
    @unittest.skipIf(tests.IS_CIRCLE_CI, "Skip this test because large memory")
    def test_keras_frontend_convert_from_keras(self):
        keras_frontend = frontend.KerasFrontend()
        model = keras.applications.vgg16.VGG16(weights='imagenet', include_top=True, input_tensor=None,
                                               input_shape=(224, 224, 3))
 
        onnx_file = keras_frontend.convert_from_keras(output_path=self.output_path, onnx_name=self.model_name,
                                                      model_name=self.model_name,
                                                      binary_flag=True, ckpt_name=self.ckpt_name, pb_name=self.pb_name,
                                                      cache_path=self.output_file, freeze_pb_name=self.freeze_pb_name,
                                                      output_op_name=model.output.op.name, K=keras.backend)
 
        self.assertTrue(os.path.isfile(onnx_file))
 
    def test_keras_frontend_convert_from_json(self):
 
        self.input_path = os.path.join(self.input_path, "keras")
 
        json_file = os.path.join(self.input_path, self.input_file_name + ".json")
        h5_file = os.path.join(self.input_path, self.input_file_name + ".hdf5")
 
        keras_frontend = frontend.KerasFrontend()
        onnx_file = keras_frontend.convert_from_json(output_path=self.output_path, onnx_name=self.model_name,
                                                     model_name=self.model_name, binary_flag=True,
                                                     ckpt_name=self.ckpt_name, pb_name=self.pb_name,
                                                     cache_path=self.output_file, freeze_pb_name=self.freeze_pb_name,
                                                     json_file=json_file, h5_file=h5_file)
 
        self.assertTrue(os.path.isfile(onnx_file))
 
    def test_keras_frontend_convert_from_yaml(self):
        self.input_path = os.path.join(self.input_path, "keras")
 
        yaml_file = os.path.join(self.input_path, self.input_file_name + ".yaml")
        h5_file = os.path.join(self.input_path, self.input_file_name + ".hdf5")
 
        keras_frontend = frontend.KerasFrontend()
        onnx_file = keras_frontend.convert_from_yaml(output_path=self.output_path, onnx_name=self.model_name,
                                                     model_name=self.model_name, binary_flag=True,
                                                     ckpt_name=self.ckpt_name, pb_name=self.pb_name,
                                                     cache_path=self.output_file, freeze_pb_name=self.freeze_pb_name,
                                                     yaml_file=yaml_file, h5_file=h5_file)
 
        self.assertTrue(os.path.isfile(onnx_file))
 
    def test_tensorflow_frontend_instance(self):
        tensorflow_frontend = frontend.TensorflowFrontend()
        self.assertIs(type(tensorflow_frontend), frontend.TensorflowFrontend)
        self.assertTrue(issubclass(type(tensorflow_frontend), frontend.BaseFrontend))
 
    def test_tenserflow_frontend_convert_to_frontend(self):
        self.input_path = os.path.join(self.input_path, "tensorflow")
        pb_file = os.path.join(self.input_path, self.input_file_name + ".pb")
        ckpt_file = os.path.join(self.input_path, self.input_file_name + ".ckpt")
 
        tensorflow_frontend = frontend.TensorflowFrontend()
        onnx_file = tensorflow_frontend.convert_from_tensorflow(output_path=self.output_path, onnx_name=self.model_name,
                                                                model_name=self.model_name, output_op_name="output",
                                                                ckpt_file=ckpt_file, pb_file=pb_file,
                                                                cache_path=self.output_file,
                                                                freeze_pb_name=self.freeze_pb_name, binary_flag=True)
 
        self.assertTrue(os.path.isfile(onnx_file))


    def test_tflite_frontend_no_tflite_path(self):
        converter = frontend.TFLiteFrontend(self.flat_c_path, self.schema_path)
        self.assertRaises(FileNotFoundError, lambda: converter.convert_tflite_to_onnx(tflite_path=None,
                                                                                      export_path=self.output_path))


    def test_tflite_frontend_no_flatc_path(self):
        converter = frontend.TFLiteFrontend("", self.schema_path)
        self.assertRaises(FileNotFoundError, lambda: converter.convert_tflite_to_onnx(tflite_path=self.tflite_path,
                                                                                      export_path=self.output_path))


    def test_tflite_frontend_no_schema_path(self):
        converter = frontend.TFLiteFrontend(flatc_path=self.flat_c_path,schema_path="")
        self.assertRaises(FileNotFoundError, lambda: converter.convert_tflite_to_onnx(tflite_path=self.tflite_path,                                                                                      
                                                                                      export_path=None))
 

    def test_tflite_frontend_no_export_path(self):
        converter = frontend.TFLiteFrontend(self.flat_c_path, self.schema_path)
        self.assertRaises(FileNotFoundError, lambda: converter.convert_tflite_to_onnx(tflite_path=self.tflite_path,
                                                                                      export_path=None))
 

    def test_tflite_frontend(self):
        converter = frontend.TFLiteFrontend(self.flat_c_path, self.schema_path)
        converter.convert_tflite_to_onnx(tflite_path=self.tflite_path,
                                         export_path=self.output_path)
  
        self.assertTrue(tests.is_dir_contains(self.output_path, file_list=["model.json", "model.onnx"]))



if __name__ == '__main__':
    unittest.main()
