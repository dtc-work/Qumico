import time
import ctypes
import sys
from os import path
from datetime import datetime

from PIL import Image
import numpy as np
import cv2
from camera import capture_stream

dir_path = path.dirname(__file__)
if dir_path not in sys.path:
    sys.path.append(dir_path)

from automl_common import WIDTH, HEIGHT, LABEL_CNT, LABELS


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
    # image window
    cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("symbol", cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow("image", 80, 80)
    cv2.moveWindow("symbol", 380, 80)


    # model path
    # load & config
    input_info = np.ctypeslib.ndpointer(dtype=np.uint8, ndim=4,
                                        shape=(1, WIDTH, HEIGHT, 3), flags='CONTIGUOUS')
    output_info = np.ctypeslib.ndpointer(dtype=np.uint8, ndim=2,
                                         shape=(1, LABEL_CNT), flags='CONTIGUOUS')

    dll = init(so_lib_path, input_info, output_info)
    dll.init()
    dll.load_initializers()

    while True:
    # image
#        image = Image.open(image_path)
    # camera
        image = Image.fromarray(capture_stream(resolution=(224,224)))
        #
        resized_image = image.resize((224, 224), Image.BICUBIC)
        image_data = np.array(resized_image, dtype='uint8')


    # infer
        input = np.ascontiguousarray(np.expand_dims(image_data, axis=0))
        output = np.zeros(dtype=np.uint8, shape=(1, LABEL_CNT))

        print("run:start", datetime.now())
        start = time.time()
        dll.run(input, output)
        print("run:end", datetime.now())
        print("elapsed_time:{0}".format(time.time() - start) + "[sec]")
        label_index = np.argmax(output)
        print("prediction",LABELS[label_index], output)
        print(label_index)
        print("finish")


        if (label_index == 0): # cat
            im2 = cv2.imread(path.join(path.dirname(__file__), "symbol", "cat.png"), cv2.IMREAD_COLOR)
        else: # dog
            im2 = cv2.imread(path.join(path.dirname(__file__), "symbol", "dog.png"), cv2.IMREAD_COLOR)

        cv2.imshow("image", cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
        im2out = cv2.resize(im2, (200,200))
        cv2.imshow("symbol", im2out)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    return ()



if __name__ == '__main__':
    image_path = path.join(path.dirname(__file__), "images", "dog1.jpeg")

    infer(image_path=image_path,
          so_lib_path=path.join(path.dirname(__file__), "out_c", "qumico.so"))
