import time
import ctypes
import sys
from os import path
from datetime import datetime

from PIL import Image
import numpy as np

dir_path = path.dirname(__file__)
if dir_path not in sys.path:
    sys.path.append(dir_path)

from mobilenet_common import WIDTH, HEIGHT, LABEL_CNT, LABELS


def init(so_lib_path, input_info, output_info):
    ModelDLL = ctypes.CDLL(so_lib_path)
    ModelDLL.qumico.argtypes = [input_info, output_info]
    ModelDLL.qumico.restype = ctypes.c_int
    ModelDLL.run.argtypes = [input_info, output_info]
    ModelDLL.run.restype = ctypes.c_int
    
    return ModelDLL


def run_c(dll, input, output):
    dll.run(input, output)


def infer_c(dll, input, output):
    dll.qumico(input, output)


def infer(image_path, so_lib_path):

    image = Image.open(image_path)
    resized_image = image.resize((WIDTH, HEIGHT), Image.BICUBIC)
    #
    image_data = np.array(resized_image, dtype='uint8')

    # model path
    # load & config
    input_info = np.ctypeslib.ndpointer(dtype=np.uint8, ndim=4,
                                        shape=(1, WIDTH, HEIGHT, 3), flags='CONTIGUOUS')
    output_info = np.ctypeslib.ndpointer(dtype=np.uint8, ndim=2,
                                         shape=(1, LABEL_CNT), flags='CONTIGUOUS')

    dll = init(so_lib_path, input_info, output_info)
    start = time.time()

    # infer
    input = np.ascontiguousarray(np.expand_dims(image_data, axis=0))
    output = np.zeros(dtype=np.uint8, shape=(1, LABEL_CNT))  # (1, 125, 13, 13)

    print("init:start", datetime.now())
    dll.init()
    print("load:start", datetime.now())
    dll.load_initializers()
    print("run:start", datetime.now())
    dll.run(input, output)
    print("run:end", datetime.now())
    print("elapsed_time:{0}".format(time.time() - start) + "[sec]")
    label_index = np.argmax(output)
    print(LABELS[label_index])



if __name__ == '__main__':
    infer(image_path=path.join(path.dirname(__file__), "images", "tiger.jpeg"),
          so_lib_path=path.join(path.dirname(__file__), "out_c", "qumico.so"))
