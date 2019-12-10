from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest

import onnx.backend.test

from qumico.backend import QumicoBackend
pytest_plugins = 'onnx.backend.test.report'
backend_test = onnx.backend.test.BackendTest(QumicoBackend, __name__)
backend_test.include('test_quantizelinear_cpu')


globals().update(backend_test
                 .test_cases)

if __name__ == '__main__':
    unittest.main()
