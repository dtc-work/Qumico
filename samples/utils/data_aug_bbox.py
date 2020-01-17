import cv2 
import numpy as np


def draw_rect(im, cords, color = None):
    """ 
        #### 
        画像の中で矩形を作る
    
        ----------
        ####引数
        im : image　ファイル
        cords : 　array 　bounding boxesの座標　
    　  color : array 　デフォルトはnone
       　####戻り値
        -------
        im:　list 
        (画像ファイル、point(x1,y1), point2(x2,y2), 画像の色、画像の枠) 　　
    """

    im = im.copy()
    
    cords = cords[:,:4]
    cords = cords.reshape(-1,4)
    if not color:
        color = [255,255,255]
    for cord in cords:
        
        pt1, pt2 = (cord[0], cord[1]) , (cord[2], cord[3])
                
        pt1 = int(pt1[0]), int(pt1[1])
        pt2 = int(pt2[0]), int(pt2[1])
    
        im = cv2.rectangle(im.copy(), pt1, pt2, color, int(max(im.shape[:2])/200))
    return im


def bbox_area(bbox):

    """
    ####
    ボックスの面積を取得

    ----------
    ####引数
    bbox:　 array         　bounding box集
　
   　####戻り値
    -------
    計算した面積の結果　
    """

    return (bbox[:,2] - bbox[:,0])*(bbox[:,3] - bbox[:,1])
        
def clip_box(bbox, clip_box, alpha):

    """
        ####
        clipboxと一番重なられるbounding boxを探す
    
        ----------
        ####引数
        bbox:　 array         　bounding box集
    　  clip_box:　array 　 　 　実際のボックス   
        alpha: float         　　alpha値(重ねる敷居値)   
       　####戻り値
        -------
        bbox: array 　一番重なるbounding boxの値残す(他のは0になる)　
    """

    ar_ = (bbox_area(bbox))

    x_min = np.maximum(bbox[:,0], clip_box[0]).reshape(-1,1)
    y_min = np.maximum(bbox[:,1], clip_box[1]).reshape(-1,1)
    x_max = np.minimum(bbox[:,2], clip_box[2]).reshape(-1,1)
    y_max = np.minimum(bbox[:,3], clip_box[3]).reshape(-1,1)
    bbox = np.hstack((x_min, y_min, x_max, y_max, bbox[:,4:]))


    delta_area = np.nan_to_num((ar_ - bbox_area(bbox))/ar_)    
#     delta_area = ((ar_ - bbox_area(bbox))/ar_)    
    mask = (delta_area < (1 - alpha)).astype(int)

    bbox = bbox * mask[...,np.newaxis]
#     maskbbox = bbox[mask == 1,:]

    return bbox


def rotate_im(image, angle):

    """
        ####
       　画像を回転する。像数は黒いの場合は回転しない。
        ----------
        ####引数
        image:　 array           bounding box集
    　   angle:　array 　 　 　   回転するimageの角度 
       　####戻り値
        -------
        image: array 　回転したimage　
    """

    # grab the dimensions of the image and then determine the
    # centre
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    image = cv2.warpAffine(image, M, (nW, nH))

    return image

def get_corners(bboxes):

    """
        bounding boxesの４点を取得する    
        ####
       　bounding boxのコーナー取得する
        ----------
        ####引数
        bboxes:　 array   bounding　box集
    　  　
       ####戻り値
    　　-------
        corners: array    画像のコーナー  　　
    """
    width = (bboxes[:,2] - bboxes[:,0]).reshape(-1,1)
    height = (bboxes[:,3] - bboxes[:,1]).reshape(-1,1)
    
    x1 = bboxes[:,0].reshape(-1,1)
    y1 = bboxes[:,1].reshape(-1,1)
    
    x2 = x1 + width
    y2 = y1 
    
    x3 = x1
    y3 = y1 + height
    
    x4 = bboxes[:,2].reshape(-1,1)
    y4 = bboxes[:,3].reshape(-1,1)
    
    corners = np.hstack((x1,y1,x2,y2,x3,y3,x4,y4))
    
    return corners

def rotate_box(corners,angle,  cx, cy, h, w):

    """
        ####
       　bounding boxを回転する。
        ----------
        ####引数
        corners:　 array    bounding box集
    　  angle:　array 　 　 回転するimageの角度
        cx: int    　中心のx座標
        cy:　int　 　中心のy座標
        h:  int　    画像の高さ
        w:  int      画像の広さ
       　####戻り値
        -------
        calculated: array 　回転したbounding boxのコーナー座標（`x1 y1 x2 y2 x3 y3 x4 y4`）　
    """


    corners = corners.reshape(-1,2)
    corners = np.hstack((corners, np.ones((corners.shape[0],1), dtype = type(corners[0][0]))))
    
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    
    
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    # Prepare the vector to be transformed
    calculated = np.dot(M,corners.T).T
    
    calculated = calculated.reshape(-1,8)
    
    return calculated


def get_enclosing_box(corners):

    """
        ####
       　回転した boundingboxのenclosing box取得する。
        ----------
        ####引数
        corners:　 array    bounding box集
        
       　####戻り値
        -------
        final: array: array  enclosing box(`x1 y1 x2 y2`)　　
    """

    x_ = corners[:,[0,2,4,6]]
    y_ = corners[:,[1,3,5,7]]
    
    xmin = np.min(x_,1).reshape(-1,1)
    ymin = np.min(y_,1).reshape(-1,1)
    xmax = np.max(x_,1).reshape(-1,1)
    ymax = np.max(y_,1).reshape(-1,1)
    
    final = np.hstack((xmin, ymin, xmax, ymax,corners[:,8:]))
    
    return final


def letterbox_image(img, inp_dim):

    """
        ####
       　imageを固定な比率でリサイズする（パティング）
        ----------
        ####引数
        img:　 array    bounding box集
    
        inp_dim: tuple(int)　:　リサイズしたimageのサイズ
    
       　####戻り値
        -------
        canvas:  array  enclosing box(`x1 y1 x2 y2`)　　
    """

    inp_dim = (inp_dim, inp_dim)
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h))
    
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 0)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    
    return canvas