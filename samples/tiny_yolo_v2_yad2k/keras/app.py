import time
import sys
from os import path

import numpy as np
from PIL import Image
import cv2

# Only Support on RaspberryPI
dir_path = path.dirname(__file__)
if dir_path not in sys.path:
    sys.path.append(dir_path)

from camera import capture_stream
from tiny_yolo_v2_yad2k_infer_c import init 

from tiny_yolo_v2_yad2k_common import (width, height, r_w, r_h, r_n,
                                      thresh,iou_threshold, voc_label,
                                      classes, yolo_eval)
from images import draw_label


def app(dll_path):

    input_info = np.ctypeslib.ndpointer(dtype=np.float32, ndim=4,
                                    shape=(1, width, height, 3), flags='CONTIGUOUS')
    output_info = np.ctypeslib.ndpointer(dtype=np.float32, ndim=4,
                                    shape=(1,r_w, r_h, 125), flags='CONTIGUOUS')

    dll = init(dll_path, input_info, output_info)    
    dll.init()
    dll.load_initializers()
    
    try:
        cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)

        while True:
            image =Image.fromarray(capture_stream(resolution=(720, 540))) 
            resized_image = image.resize((width, height), Image.BICUBIC)
            image_data = np.array(resized_image, dtype='float32') / 255.0

            input = np.ascontiguousarray(np.expand_dims(image_data, axis=0))
            output =np.zeros(dtype=np.float32, shape=(1, r_w, r_h, 125)) # (1, 125, 13, 13)
            start = time.time()
            dll.run(input, output)
            print ("elapsed_time:{0}".format(time.time() - start) + "[sec]")

            out_boxes, out_scores, out_classes = yolo_eval(output, image.size, score_threshold = thresh,
                                                           iou_threshold = iou_threshold, classes = classes)

            labeled_image = draw_label(image, out_boxes, out_scores, out_classes, voc_label)

            cv2.imshow("image", cv2.cvtColor(np.array(labeled_image), cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
             
        cv2.destroyAllWindows()
    
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        exit()


if __name__ =="__main__":

    dll_path= path.join(path.dirname(__file__), "out_c_optimize", "qumico.so")
    app(dll_path)
    