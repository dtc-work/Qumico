# This program was ported from the YAD2K project.
# https://github.com/allanzelener/YAD2K
#
# copyright
# https://github.com/allanzelener/YAD2K/blob/master/LICENSE
#
import sys
import os
from os import path
from PIL import Image
import numpy as np

dir_path = path.dirname(__file__)
if dir_path not in sys.path:
    sys.path.append(dir_path)

from tiny_yolo_v2_yad2k_common import (width, height, r_w, r_h, r_n,
                                      voc_label, classes, yolo_eval,
                                      thresh,iou_threshold)
from tiny_yolo_v2_yad2k_model import tiny_yolo_model, layer_dump
from images import draw_label


if __name__ == "__main__":
    image_path = path.join(path.dirname(__file__), "images", "000001.jpg")
    h5_path = path.join(path.dirname(__file__), "model", "tiny_yolo_v2_yad2k.h5")

    output_path = path.join(path.dirname(__file__), "output")

    file_post_fix = ''
    
    # モデルの構築
    tiny_yolo_model = tiny_yolo_model()
    tiny_yolo_model.load_weights(h5_path)
    
    if not path.exists(output_path):
        os.mkdir(output_path)
    
    tiny_yolo_model.summary()
        
    # run yolo
    image = Image.open(image_path)
    resized_image = image.resize((width, height), Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32') / 255.0
    x = np.expand_dims(image_data, axis=0)

    preds = tiny_yolo_model.predict(x) # shape is 1, 13, 13,125
    
    probs = np.zeros((r_h * r_w * r_n, classes+1), dtype=np.float)
    
    for i, l in enumerate(range(31)):
        layer_dump(tiny_yolo_model, x, l)
    
    np.save(path.join(output_path, 'preds%s.npy' % file_post_fix) , preds)
    
    out_boxes, out_scores, out_classes = yolo_eval(preds, image.size, score_threshold = thresh,
                                                   iou_threshold = iou_threshold, classes = classes)

    draw_image = draw_label(image, out_boxes, out_scores, out_classes, voc_label)
    draw_image.save(path.join(path.dirname(image_path),'out_infer_keras.png'))
    """
    person 0.7327421 (213, 16) (445, 357)
    horse 0.530643 (53, 146) (217, 350)
    """
    print("finish")
