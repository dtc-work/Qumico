import tensorflow as tf
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
  
    def test_tensorflow_frontend_instance(self):
        tensorflow_frontend = frontend.TensorflowFrontend()
        self.assertIs(type(tensorflow_frontend), frontend.TensorflowFrontend)
        self.assertTrue(issubclass(type(tensorflow_frontend), frontend.BaseFrontend))

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
