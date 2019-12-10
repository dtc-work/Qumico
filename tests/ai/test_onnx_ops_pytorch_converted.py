import unittest
import onnx.backend.test

from qumico.backend import QumicoBackend

pytest_plugins = 'onnx.backend.test.report'
backend_test = onnx.backend.test.BackendTest(QumicoBackend, __name__)

# backend_test.include('test_AvgPool1d_cpu')
# backend_test.include('test_AvgPool1d_stride_cpu')                     # BUG
# backend_test.include('test_AvgPool2d_cpu')                            # BUG
# backend_test.include('test_AvgPool2d_stride_cpu')                     # BUG
# backend_test.include('test_AvgPool3d_cpu')                            # BUG
# backend_test.include('test_AvgPool3d_stride_cpu')                     # BUG
# backend_test.include('test_AvgPool3d_stride1_pad0_gpu_input_cpu')     # BUG
# backend_test.include('test_BatchNorm1d_3d_input_eval_cpu')            # BUG
# backend_test.include('test_BatchNorm2d_eval_cpu')                     # BUG
# backend_test.include('test_BatchNorm2d_momentum_eval_cpu')            # BUG
# backend_test.include('test_BatchNorm3d_eval_cpu')                     # BUG
# backend_test.include('test_BatchNorm3d_momentum_eval_cpu')            # BUG
# backend_test.include('test_ConstantPad2d_cpu')                        # BUG
# backend_test.include('test_Conv1d_cpu')                               # BUG
# backend_test.include('test_Conv1d_dilated_cpu')                       # BUG
# backend_test.include('test_Conv1d_groups_cpu')
# backend_test.include('test_Conv1d_pad1_cpu')
# backend_test.include('test_Conv1d_pad1size1_cpu')
# backend_test.include('test_Conv1d_pad2_cpu')
# backend_test.include('test_Conv1d_pad2size1_cpu')
# backend_test.include('test_Conv1d_stride_cpu')
# backend_test.include('test_Conv2d_cpu')
# backend_test.include('test_Conv2d_depthwise_cpu')
# backend_test.include('test_Conv2d_depthwise_padded_cpu')
# backend_test.include('test_Conv2d_depthwise_strided_cpu')
# backend_test.include('test_Conv2d_depthwise_with_multiplier_cpu')
# backend_test.include('test_Conv2d_dilated_cpu')
# backend_test.include('test_Conv2d_groups_cpu')
# backend_test.include('test_Conv2d_groups_thnn_cpu')
# backend_test.include('test_Conv2d_no_bias_cpu')
# backend_test.include('test_Conv2d_padding_cpu')
# backend_test.include('test_Conv2d_strided_cpu')
# backend_test.include('test_Conv3d_cpu')
# backend_test.include('test_Conv3d_dilated_cpu')
# backend_test.include('test_Conv3d_dilated_strided_cpu')
# backend_test.include('test_Conv3d_groups_cpu')
# backend_test.include('test_Conv3d_no_bias_cpu')
# backend_test.include('test_Conv3d_stride_cpu')
# backend_test.include('test_Conv3d_stride_padding_cpu')
backend_test.exclude('test_ConvTranspose2d_cpu')
backend_test.exclude('test_ConvTranspose2d_no_bias_cpu')
backend_test.exclude('test_ELU_cpu')
backend_test.exclude('test_Embedding_cpu')
backend_test.exclude('test_Embedding_sparse_cpu')
backend_test.exclude('test_GLU_cpu')
backend_test.exclude('test_GLU_dim_cpu')
backend_test.exclude('test_LeakyReLU_cpu')
backend_test.exclude('test_LeakyReLU_with_negval_cpu')
backend_test.exclude('test_Linear_cpu')
backend_test.exclude('test_Linear_no_bias_cpu')
backend_test.exclude('test_LogSoftmax_cpu')
#backend_test.include('test_MaxPool1d_cpu')                        # BUG
#backend_test.include('test_MaxPool1d_stride_cpu')                 # BUG
#backend_test.include('test_MaxPool2d_cpu')                        # BUG
#backend_test.include('test_MaxPool3d_cpu')                        # BUG
#backend_test.include('test_MaxPool3d_stride_cpu')                 # BUG
#backend_test.include('test_MaxPool3d_stride_padding_cpu')         # BUG
backend_test.exclude('test_PReLU_1d_cpu')
backend_test.exclude('test_PReLU_1d_multiparam_cpu')
backend_test.exclude('test_PReLU_2d_cpu')
backend_test.exclude('test_PReLU_2d_multiparam_cpu')
backend_test.exclude('test_PReLU_3d_cpu')
backend_test.exclude('test_PReLU_3d_multiparam_cpu')
backend_test.exclude('test_PixelShuffle_cpu')
backend_test.exclude('test_PoissonNLLLLoss_no_reduce_cpu')
backend_test.include('test_ReLU_cpu')
backend_test.exclude('test_ReflectionPad2d_cpu')
backend_test.exclude('test_ReplicationPad2d_cpu')
backend_test.exclude('test_SELU_cpu')
backend_test.exclude('test_Sigmoid_cpu')
backend_test.exclude('test_Softmin_cpu')
backend_test.exclude('test_Softplus_cpu')
backend_test.exclude('test_Softsign_cpu')
backend_test.exclude('test_Tanh_cpu')
backend_test.exclude('test_ZeroPad2d_cpu')
backend_test.exclude('test_log_softmax_dim3_cpu')
backend_test.exclude('test_log_softmax_lastdim_cpu')
backend_test.include('test_softmax_functional_dim3_cpu')
backend_test.include('test_softmax_lastdim_cpu')


globals().update(backend_test.test_cases)

if __name__ == '__main__':
    unittest.main()