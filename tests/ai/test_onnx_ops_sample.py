import unittest
import onnx.backend.test

from qumico.backend import QumicoBackend

pytest_plugins = 'onnx.backend.test.report'
backend_test = onnx.backend.test.BackendTest(QumicoBackend, __name__)

backend_test.exclude('test_expand_shape_model1_cpu')
backend_test.exclude('test_expand_shape_model2_cpu')
backend_test.exclude('test_expand_shape_model3_cpu')
backend_test.exclude('test_expand_shape_model4_cpu')
backend_test.exclude('test_shrink_cpu')
backend_test.exclude('test_sign_model_cpu')
backend_test.include('test_single_relu_model_cpu')

globals().update(backend_test.test_cases)

if __name__ == '__main__':
    unittest.main()