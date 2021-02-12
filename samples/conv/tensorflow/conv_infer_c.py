import ctypes
from os import path

import numpy as np
from tensorflow.keras.datasets import mnist

import samples.utils.common_tool as common


def init(so_lib_path, input_info, output_info):
    ModelDLL = ctypes.CDLL(so_lib_path)
    ModelDLL.qumico.argtypes = [input_info, output_info]
    ModelDLL.qumico.restype = ctypes.c_int
    return ModelDLL


def infer_c(dll, input, output):
    dll.qumico(input, output)


if __name__ == '__main__':
    # prepare the infer date 28px * 28px image
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_test = x_test.reshape(10000, 784) / 255.
    image_data = x_test[:10, ...]
    

    # model path
    so_lib_path= path.join(path.dirname(__file__), 'out_c', 'qumico.so')
 
    # Load
    input_info = np.ctypeslib.ndpointer(dtype=np.float32, ndim=2,
                                    shape=(1,784), flags='CONTIGUOUS')

    output_info = np.ctypeslib.ndpointer(dtype=np.float32, ndim=2,
                                    shape=(1,10), flags='CONTIGUOUS')

    dll = init(so_lib_path, input_info, output_info)

    # infer
    res = []
    for i in image_data:
        output =np.zeros(dtype=np.float32, shape=(1,10))
        infer_c(dll, np.expand_dims(i,0).astype(np.float32), output)
        classification = common.softmax(output)
        y = common.onehot_decoding(classification)
        res.append(y[0])
    print('Predict Index', res) # [7, 2, 1, 0, 4, 1, 4, 9, 5, 9]
