import numpy as np
from qumico.ir.tflite_builtin_op_options import Padding


def padding(inputs, outputs, _padding, pads, strides, dilations, kernel_shape):

    if _padding == Padding.SAME:
        input_shape = np.array(inputs[0].shape)[[0, 3, 1, 2]]  # NHWC_TO_NCHW
        output_shape = np.array(outputs[0].shape)[[0, 3, 1, 2]]  # NHWC_TO_NCHW
        if any(input_shape[i + 2] == -1 or output_shape[i + 2] == -1 for i in range(2)):
            raise NotImplementedError("autopad")
        else:
            for i in range(2):
                pad = (output_shape[i + 2] - 1) * strides[i] + dilations[i] * kernel_shape[i] - input_shape[i + 2]
                pad = max(pad, 0)
                pads[i] = pad // 2
                pads[i + 2] = pad - pad // 2
    elif _padding == Padding.VALID:
        pass
    else:
        raise ValueError("invalid padding value: " + _padding)

    return pads, input_shape, output_shape
