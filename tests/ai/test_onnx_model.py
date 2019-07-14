import unittest
import os
import shutil

from onnx.backend.test.case.test_case import TestCase as tc
from qumico.backend import QumicoBackend

from tests.ai import QumicoBackendTest

import tests


class TestOnnxModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.input_folder = "input"
        cls.input_path = os.path.abspath(os.path.join(os.path.dirname(__file__), cls.input_folder))

    def tearDown(self) -> None:
        tests.remove_folder(self.input_path)

    def test_mnist_model(self):

        _model_name = "mnist"

        try:
            tc_mnist = tc(name=f"test_{_model_name}",
                          url="https://onnxzoo.blob.core.windows.net/models/opset_8/mnist/mnist.tar.gz",
                          model_name=_model_name,
                          model_dir=os.path.join(self.input_path, _model_name),
                          model=None,
                          data_sets=None,
                          kind=None,
                          rtol=1e-3,
                          atol=1e-7)

            QumicoBackendTest.download_model(model_test=tc_mnist, model_dir="", models_dir=self.input_path)

            backend_test = QumicoBackendTest(QumicoBackend, __name__)
            backend_test._add_model_test(tc_mnist, self.input_folder)

            runner = unittest.TextTestRunner(verbosity=2)
            runner.run(backend_test.test_suite)

        except Exception:
            self.fail(f"Model Test for {_model_name} is failed")

    def test_mobilenet_model(self):
        
        _model_name = "mobilenet"
        _model_name_ver = "mobilenetv2-1.0"

        try:
            tc_mobile = tc(name=f"test_{_model_name}",
                           url="https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-1.0/mobilenetv2-1.0.tar.gz",
                           model_name=_model_name,
                           model_dir=os.path.join(self.input_path, _model_name),
                           model=None,
                           data_sets=None,
                           kind=None,
                           rtol=1e-2,
                           atol=1e-7)

            QumicoBackendTest.download_model(model_test=tc_mobile, model_dir="", models_dir=self.input_path)

            os.rename(os.path.join(self.input_path, _model_name_ver), os.path.join(self.input_path, _model_name))

            for filename in os.listdir(os.path.join(self.input_path, _model_name)):
                if filename == f"{_model_name_ver}.onnx":
                    os.rename(os.path.join(self.input_path, _model_name, f"{_model_name_ver}.onnx"),
                              os.path.join(self.input_path, _model_name, "model.onnx"))

            backend_test = QumicoBackendTest(QumicoBackend, __name__)
            backend_test._add_model_test(tc_mobile, self.input_folder)

            runner = unittest.TextTestRunner(verbosity=2)
            runner.run(backend_test.test_suite)

        except FileNotFoundError:
            self.fail(f"Model Test for {_model_name} is failed")


if __name__ == '__main__':
    unittest.main()