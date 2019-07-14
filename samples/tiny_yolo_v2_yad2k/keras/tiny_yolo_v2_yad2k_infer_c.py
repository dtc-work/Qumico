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

from tiny_yolo_v2_yad2k_common import (width, height, r_w, r_h, r_n,
                                      thresh,iou_threshold, voc_label,
                                      classes, yolo_eval)
from images import draw_label


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


if __name__ == '__main__':

    image_path = path.join(path.dirname(__file__), "images", "000001.jpg")

    image = Image.open(image_path)
    resized_image = image.resize((width, height), Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32') / 255.0
    # model path
    so_lib_path= path.join(path.dirname(__file__), "out_c_optimize", "qumico.so")

    # load & config
    input_info = np.ctypeslib.ndpointer(dtype=np.float32, ndim=4,
                                    shape=(1, width, height, 3), flags='CONTIGUOUS')
    output_info = np.ctypeslib.ndpointer(dtype=np.float32, ndim=4,
                                    shape=(1,r_w, r_h, 125), flags='CONTIGUOUS')
    dll = init(so_lib_path, input_info, output_info)

    start = time.time()

    # infer
    input = np.ascontiguousarray(np.expand_dims(image_data, axis=0))
    output =np.zeros(dtype=np.float32, shape=(1, r_w, r_h, 125)) # (1, 125, 13, 13)

    print("init:start", datetime.now())
    dll.init()
    print("load:start", datetime.now())
    dll.load_initializers()
    print("run:start", datetime.now())
    dll.run(input, output)
    print("run:end", datetime.now())    
    out_boxes, out_scores, out_classes = yolo_eval(output, image.size, score_threshold = thresh,
                                                   iou_threshold = iou_threshold, classes = classes)
    
    print("post:end",datetime.now())
    draw_image = draw_label(image, out_boxes, out_scores, out_classes, voc_label)  

    
    draw_image.save(path.join(path.dirname(image_path),'out_infer_c.png'))
    print("draw", datetime.now())
    print ("elapsed_time:{0}".format(time.time() - start) + "[sec]")
    
    """
    person 0.732742 (213, 16) (445, 357)
    horse 0.5306421 (53, 146) (217, 350)
    """
    print("finish")

