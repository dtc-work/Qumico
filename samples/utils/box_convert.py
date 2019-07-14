import numpy as np

class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, c = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        
        self.c     = c
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        
        return self.label
    
    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
            
        return self.score

def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])  
    
    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
    
    union = w1*h1 + w2*h2 - intersect
    
    return float(intersect) / union

def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2,x4) - x3       

def bbox_to_anbox(bbox):
    anbox = np.zeros_like(bbox)
    anbox[..., 0] = (bbox[..., 0] + bbox[..., 2]) / 2
    anbox[..., 1] = (bbox[..., 1] + bbox[..., 3]) / 2
    anbox[..., 2] = bbox[..., 2] - bbox[..., 0]
    anbox[..., 3] = bbox[..., 3] - bbox[..., 1]
    return anbox


def anbox_to_bbox(anbox):
    bbox = np.zeros_like(anbox)
    bbox[..., 0] = anbox[..., 0] - anbox[..., 2] / 2
    bbox[..., 2] = anbox[..., 0] + anbox[..., 2] / 2
    bbox[..., 1] = anbox[..., 1] - anbox[..., 3] / 2
    bbox[..., 3] = anbox[..., 1] + anbox[..., 3] / 2
    return bbox
