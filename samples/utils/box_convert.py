import numpy as np

class BoundBox:
    
    def __init__(self, xmin, ymin, xmax, ymax, c = None, classes = None):
        """
            初期化
    
            ###引数
            xmin : int   ボックスのxmin座標
            ymin : int   ボックスのymin座標
            xmax : int   ボックスのxmax座標
            ymax : int   ボックスのymax座標
            c: 現在未使用
            classes: dict  ボックスの分類ラベルとスコア(key ラベル, value スコア)
            ###戻り値
            なし
        """

        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.c = c
        self.classes = classes
        self.label = -1
        self.score = -1

    def get_label(self):
        """
        "ラベル取得する（予約済み）"
        "###引数"
        "なし"
        "###戻り値"
        "スコア値一番大きなラベル"
        """
        if self.label == -1:
            self.label = np.argmax(self.classes)
        
        return self.label
    
    def get_score(self):
        """
        "スコア取得する（予約済み）"
        "###引数"
        "なし"
        "###戻り値"
        "ラベルのスコア値"
        """
        if self.score == -1:
            self.score = self.classes[self.get_label()]
        return self.score

def bbox_iou(box1, box2):
    """
    "スコア取得する（予約済み）"
    "box1: array Boundary Box1"
    "box2: array Boundary Box2"
    "###戻り値"
    "IoU"
    """

    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])  
    
    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
    
    union = w1*h1 + w2*h2 - intersect
    
    return float(intersect) / union

def _interval_overlap(interval_a, interval_b):
    """
        "二つボックスのxまたyの座標間隔"
    
        "###引数"
        "interval a: array  ボックスaのxまたy min座標、　xまたy max座標"
        "interval b: array  ボックスbのxまたy min座標、 xまたy max座標"
        "###戻り値"
        "二つボックスのxまたyの座標間隔"
    """
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

    """
        "Boundary BoxからAnchor boxに変換する "
        "###引数"
        "bbox: array Boundary Box"
        "###戻り値"
        "Anchor: array Anchor Box"
    """

    anbox = np.zeros_like(bbox)
    anbox[..., 0] = (bbox[..., 0] + bbox[..., 2]) / 2
    anbox[..., 1] = (bbox[..., 1] + bbox[..., 3]) / 2
    anbox[..., 2] = bbox[..., 2] - bbox[..., 0]
    anbox[..., 3] = bbox[..., 3] - bbox[..., 1]
    return anbox


def anbox_to_bbox(anbox):

    """
    "Anchor BoxからBounding Boxに変換する "

    "###引数"
    "Anchor: array Anchor Box"
    "###戻り値"
    "Bbox: array Bounding Box"
    """

    bbox = np.zeros_like(anbox)
    bbox[..., 0] = anbox[..., 0] - anbox[..., 2] / 2
    bbox[..., 2] = anbox[..., 0] + anbox[..., 2] / 2
    bbox[..., 1] = anbox[..., 1] - anbox[..., 3] / 2
    bbox[..., 3] = anbox[..., 1] + anbox[..., 3] / 2
    return bbox
