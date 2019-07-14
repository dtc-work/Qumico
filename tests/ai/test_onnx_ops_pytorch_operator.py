import unittest
import onnx.backend.test

from qumico.backend import QumicoBackend

pytest_plugins = 'onnx.backend.test.report'
backend_test = onnx.backend.test.BackendTest(QumicoBackend, __name__)

backend_test.include('test_operator_add_broadcast_cpu')
backend_test.include('test_operator_add_size1_broadcast_cpu')
backend_test.include('test_operator_add_size1_right_broadcast_cpu')
backend_test.include('test_operator_add_size1_singleton_broadcast_cpu')
backend_test.include('test_operator_addconstant_cpu')
backend_test.include('test_operator_addmm_cpu')
backend_test.exclude('test_operator_basic_cpu')
backend_test.exclude('test_operator_chunk_cpu')
backend_test.exclude('test_operator_clip_cpu')
backend_test.include('test_operator_concat2_cpu')
backend_test.include('test_operator_conv_cpu')
backend_test.exclude('test_operator_convtranspose_cpu')
backend_test.exclude('test_operator_exp_cpu')
backend_test.include('test_operator_flatten_cpu')
backend_test.include('test_operator_index_cpu')
backend_test.include('test_operator_max_cpu')
backend_test.include('test_operator_maxpool_cpu')
backend_test.include('test_operator_min_cpu')
backend_test.include('test_operator_mm_cpu')
backend_test.include('test_operator_non_float_params_cpu')
# backend_test.include('test_operator_pad_cpu')                 PAD
backend_test.exclude('test_operator_params_cpu')
backend_test.include('test_operator_permute2_cpu')
backend_test.exclude('test_operator_pow_cpu')
backend_test.include('test_operator_reduced_mean_cpu')
backend_test.include('test_operator_reduced_mean_keepdim_cpu')
backend_test.exclude('test_operator_reduced_sum_cpu')
backend_test.exclude('test_operator_reduced_sum_keepdim_cpu')
backend_test.exclude('test_operator_repeat_cpu')
backend_test.exclude('test_operator_repeat_dim_overflow_cpu')
backend_test.exclude('test_operator_selu_cpu')
backend_test.exclude('test_operator_sqrt_cpu')
backend_test.exclude('test_operator_symbolic_override_cpu')
backend_test.exclude('test_operator_symbolic_override_nested_cpu')
backend_test.include('test_operator_view_cpu')


globals().update(backend_test.test_cases)

if __name__ == '__main__':
    unittest.main()