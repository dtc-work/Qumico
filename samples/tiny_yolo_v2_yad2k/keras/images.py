import numpy as np
from PIL import ImageDraw

def draw_label(image, out_boxes, out_scores, out_classes, label):
    dr = ImageDraw.Draw(image)

    for i in range(len(out_classes)):
        cls = out_classes[i]
        score = out_scores[i]
        box = out_boxes[i]
    
        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

        print(label[cls], score, (left, top), (right, bottom))

        lt = (left, top)
        rt = (right, top)
        lb = (left, bottom)
        rb = (right, bottom)
        red = (255, 0, 0)
        dr.line((lt, rt), red, 2)
        dr.line((lt, lb), red, 2)
        dr.line((rt, rb), red, 2)
        dr.line((lb, rb), red, 2)
    return image