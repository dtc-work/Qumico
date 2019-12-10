import os
import unittest

from qumico import QumicoRootPath
from qumico.frontend import TFLiteFrontend

import tests


class TestMobilenetTFliteToONNX(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.current_path = os.path.abspath(os.path.dirname(__file__))
        cls.input_path = os.path.join(cls.current_path, "input")
        cls.output_path = os.path.join(cls.current_path, "output")

        cls.tflite_input = os.path.join(cls.input_path, "model", "mobilenet.tflite")
        cls.flat_c_path = os.path.join(QumicoRootPath, "..", "external", "flatbuffers", "flatc")
        cls.schema_path = os.path.join(QumicoRootPath, "..", "external", "flatbuffers", "schema.fbs")
        

    def tearDown(self) -> None:
        tests.remove_folder(self.output_path)


    def test_tflite_to_onnx_no_tflite_path(self):
        frontend = TFLiteFrontend()

        self.assertRaises(FileNotFoundError,
                          lambda:frontend.convert_tflite_to_onnx(tflite_path=None,
                                                                 export_path=self.output_path))

    def test_tflite_to_onnx_no_output_path(self):
        frontend = TFLiteFrontend()

        self.assertRaises(FileNotFoundError,
                          lambda:frontend.convert_tflite_to_onnx(tflite_path=self.tflite_input,
                                                                 export_path=None))

 
    def test_tflite_to_onnx_no_flatc_path(self):
        frontend = TFLiteFrontend(flatc_path="")

        self.assertRaises(FileNotFoundError,
                          lambda:frontend.convert_tflite_to_onnx(tflite_path=self.tflite_input,
                                                                 export_path=self.output_path))
 

    def test_tflite_to_onnx_no_schema_path(self):
        frontend = TFLiteFrontend(schema_path="")

        self.assertRaises(FileNotFoundError,
                          lambda:frontend.convert_tflite_to_onnx(tflite_path=self.tflite_input,
                                                                 export_path=self.output_path))
 

    def test_tflite_to_onnx_default_args(self):
        file_list = ["mobilenet.json", "mobilenet.onnx"]
        frontend = TFLiteFrontend()
        frontend.convert_tflite_to_onnx(tflite_path=self.tflite_input,
                                        export_path=self.output_path)
 
        self.assertTrue(tests.is_dir_contains(dirs=self.output_path, file_list=file_list))
    

    def test_tflite_to_onnx(self):
        file_list = ["mobilenet.json", "mobilenet.onnx"]
        frontend = TFLiteFrontend(flatc_path=self.flat_c_path,
                                  schema_path=self.schema_path)
        frontend.convert_tflite_to_onnx(tflite_path=self.tflite_input,
                                        export_path=self.output_path)
 
        self.assertTrue(tests.is_dir_contains(dirs=self.output_path, file_list=file_list))


if __name__ == "__main__":
    unittest.main()
