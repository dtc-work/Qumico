import random
import numpy as np
import cv2

from samples.utils import data_aug_bbox


class RandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        """
            ####
            初期化
    
            ----------
            ####引数
            p: float 確率
            ####戻り値
            p: 確率
        """
        self.p = p

    def __call__(self, img, bboxes):
        """
            ####
            ランダムに水平フリップする
    
            ####----------
            引数
            　
            image:   ndaaray   画像
            bboxes:  nparray   boundling boxes
    
            ####戻り値
            回転したimage:   ndaaray   画像
            回転したbboxes:  ndarray   boundling boxes
        """

        img_center = np.array(img.shape[:2])[::-1]/2
        img_center = np.hstack((img_center, img_center))
        if random.random() < self.p:
            img = img[:, ::-1, :]
            bboxes[:, [0, 2]] += 2*(img_center[[0, 2]] - bboxes[:, [0, 2]])

            box_w = abs(bboxes[:, 0] - bboxes[:, 2])

            bboxes[:, 0] -= box_w
            bboxes[:, 2] += box_w

        return img, bboxes


class HorizontalFlip(object):

    def __init__(self):
        pass

    def __call__(self, img, bboxes):
        """　　　　
            ####
            水平フリップ
    
            ####----------
            引数
            　
            image:   ndaaray   画像
            bboxes:  nparray   bounding boxes
    
            ####戻り値
            回転したimage:   ndaaray   画像
            回転したbboxes:  nparray   bounding boxes
        """

        img_center = np.array(img.shape[:2])[::-1]/2
        img_center = np.hstack((img_center, img_center))

        img = img[:, ::-1, :]
        bboxes[:, [0, 2]] += 2*(img_center[[0, 2]] - bboxes[:, [0, 2]])

        box_w = abs(bboxes[:, 0] - bboxes[:, 2])

        bboxes[:, 0] -= box_w
        bboxes[:, 2] += box_w

        return img, bboxes


class RandomScale(object):

    def __init__(self, scale=0.2, diff=False):
        self.scale = scale
        """
            ####
            初期化
    
            ----------
            ####引数
            scale: float or tuple(float) scale率
            diff:　boolean デフォルトはFalse
            ####戻り値
            -------
            なし
        """
        
        if type(self.scale) == tuple:
            assert len(self.scale) == 2, "Invalid range"
            assert self.scale[0] > -1, "Scale factor can't be less than -1"
            assert self.scale[1] > -1, "Scale factor can't be less than -1"
        else:
            assert self.scale > 0, "Please input a positive float"
            self.scale = (max(-1, -self.scale), self.scale)
            self.diff = diff


        

    def __call__(self, img, bboxes):
        """
            ランダムに画像を スケールする
            元画像の25％未満の面積を持つバウンディングボックス
            変換された画像はドロップされます。
            解像度は維持され、残りは領域が黒色で塗りつぶされている場合。

            ####
            引数
            　
            - scale: float or tuple(float)　　　スケール
            floatの場合: 　(1 - `scale` , 1 + `scale`)の範囲にスケールする
            tupleの場合: 　その値スケールする
            image:   ndaaray   画像
            bboxes:  nparray   boundling boxes

                    　
            ####戻り値

            - numpy.ndaaray
                スケールしたimage
            - numpy.ndarray
                スケールしたboundingbox

        """
        
        #Chose a random digit to scale by 
        
        img_shape = img.shape
        
        if self.diff:
            scale_x = random.uniform(*self.scale)
            scale_y = random.uniform(*self.scale)
        else:
            scale_x = random.uniform(*self.scale)
            scale_y = scale_x
            
    
        
        resize_scale_x = 1 + scale_x
        resize_scale_y = 1 + scale_y
        
        img=  cv2.resize(img, None, fx = resize_scale_x, fy = resize_scale_y)
        bboxes[:,:4] *= [resize_scale_x, resize_scale_y, resize_scale_x, resize_scale_y]
        
        
        canvas = np.zeros(img_shape, dtype = np.uint8)
        
        y_lim = int(min(resize_scale_y,1)*img_shape[0])
        x_lim = int(min(resize_scale_x,1)*img_shape[1])
        
        
        canvas[:y_lim,:x_lim,:] =  img[:y_lim,:x_lim,:]
        
        img = canvas

        bboxes = data_aug_bbox.clip_box(bboxes, [0,0,1 + img_shape[1], img_shape[0]], 0.25)        
    
        return img, bboxes


class Scale(object):


    def __init__(self, scale_x = 0.2, scale_y = 0.2):
        """
            ####
            初期化
        
            ----------
            ####引数
            scale_x: scale_x 率   デフォルトは0.2
            scale_:　scale_y 率 デフォルトは0.2
            ####戻り値
            -------
            なし y
        """
        self.scale_x = scale_x
        self.scale_y = scale_y



    def __call__(self, img, bboxes):
        """
            画像スケールする
                残りの25％未満の面積を持つバウンディングボックス
                変換された画像はドロップされます。
                解像度は維持され、残りは領域が黒色で塗りつぶされている場合


            ####
            引数
            - scale_x: float

                画像を水平方向にスケールする係数

            - scale_y: float
                画像を垂直に方向にスケールする係数

            -image:   ndaaray   画像
            -bboxes:  nparray   boundling boxes

                ####戻り値

                - image
                    スケールしたimage
                - boxes
                    スケールしたbounding box
        """

        img_shape = img.shape
        resize_scale_x = 1 + self.scale_x
        resize_scale_y = 1 + self.scale_y
        
        img=  cv2.resize(img, None, fx = resize_scale_x, fy = resize_scale_y)
        
        bboxes[:,:4] *= [resize_scale_x, resize_scale_y, resize_scale_x, resize_scale_y]
        
        
        
        canvas = np.zeros(img_shape, dtype = np.uint8)
        
        y_lim = int(min(resize_scale_y,1)*img_shape[0])
        x_lim = int(min(resize_scale_x,1)*img_shape[1])
        
        
        canvas[:y_lim,:x_lim,:] =  img[:y_lim,:x_lim,:]
        
        img = canvas
        bboxes = data_aug_bbox.clip_box(bboxes, [0,0,1 + img_shape[1], img_shape[0]], 0.25)
        return img, bboxes
    

class RandomTranslate(object):

    def __init__(self, translate = 0.2, diff = False):

        """
            ####
            初期化
    
            ----------
            ####引数
            translate: float or tuple(float)　translate 率   デフォルトは0.2
            diff:　　boonlean デフォルトはFalse
            ####戻り値
            -------
            なし
        """
        self.translate = translate
        
        if type(self.translate) == tuple:
            assert len(self.translate) == 2, "Invalid range"  
            assert self.translate[0] > 0 & self.translate[0] < 1
            assert self.translate[1] > 0 & self.translate[1] < 1


        else:
            assert self.translate > 0 and self.translate < 1
            self.translate = (-self.translate, self.translate)
            
            
        self.diff = diff


    def __call__(self, img, bboxes):
        """
            ####ランダムにimageを移動する
            残りの25％未満の面積を持つバウンディングボックス
            変換された画像はドロップされます。
            解像度は維持され、残りは領域が黒色で塗りつぶされている場合

            ####引数
            ----------
            - translate: float or tuple(float)
                floatの場合: 　(1 - `scale` , 1 + `scale`)の範囲に移動する
                tupleの場合: 　その値する

            image:   ndaaray   画像
            bboxes:  nparray   boundling boxes


          　####戻り値
            -------

            - image
                移動したimage
            - boxes
                移動したbounding box
        """
        #Chose a random digit to scale by 
        img_shape = img.shape
        
        #translate the image
        
        #percentage of the dimension of the image to translate
        translate_factor_x = random.uniform(*self.translate)
        translate_factor_y = random.uniform(*self.translate)
        
        if not self.diff:
            translate_factor_y = translate_factor_x
            
        canvas = np.zeros(img_shape).astype(np.uint8)
    
    
        corner_x = int(translate_factor_x*img.shape[1])
        corner_y = int(translate_factor_y*img.shape[0])
        
        
        #change the origin to the top-left corner of the translated box
        orig_box_cords =  [max(0,corner_y), max(corner_x,0), min(img_shape[0], corner_y + img.shape[0]), min(img_shape[1],corner_x + img.shape[1])]       
    
        mask = img[max(-corner_y, 0):min(img.shape[0], -corner_y + img_shape[0]), max(-corner_x, 0):min(img.shape[1], -corner_x + img_shape[1]),:]
        canvas[orig_box_cords[0]:orig_box_cords[2], orig_box_cords[1]:orig_box_cords[3],:] = mask
        img = canvas
        
        bboxes[:,:4] += [corner_x, corner_y, corner_x, corner_y]
        bboxes = data_aug_bbox.clip_box(bboxes, [0,0,img_shape[1], img_shape[0]], 0.25)    
        
        return img, bboxes
    

class Translate(object):


    def __init__(self, translate_x = 0.2, translate_y = 0.2, diff = False):
        """
            ####
            初期化

            ----------
            ####引数
            translate_x: float or tuple(float)　translate 率   デフォルトは0.2
            translate_y: float or tuple(float)　translate 率   デフォルトは0.2
            diff:　boonlean デフォルトはFalse
            ####戻り値
            -------
            なし
        """
        self.translate_x = translate_x
        self.translate_y = translate_y

        assert self.translate_x > 0 and self.translate_x < 1
        assert self.translate_y > 0 and self.translate_y < 1



    def __call__(self, img, bboxes):
        """画像移動する

           残りの25％未満の面積を持つバウンディングボックス
           変換された画像はドロップされる。
           解像度は維持され、残りは領域が黒色で塗りつぶされている場合。

           ###
           引数
           ----------
           - translate: float or tuple(float)
               floatの場合: 　(1 - `scale` , 1 + `scale`)の範囲に移動する
               tupleの場合: 　その値する
           image:   ndaaray   画像
           bboxes:  nparray   boundling boxes

           ####戻り値
           -------

           - image
               移動したimage
           - boxes
               移動したbounding box
        """

        #Chose a random digit to scale by 
        img_shape = img.shape
        
        #translate the image
        
        #percentage of the dimension of the image to translate
        translate_factor_x = self.translate_x
        translate_factor_y = self.translate_y        
            
        canvas = np.zeros(img_shape).astype(np.uint8)

        #get the top-left corner co-ordinates of the shifted box 
        corner_x = int(translate_factor_x*img.shape[1])
        corner_y = int(translate_factor_y*img.shape[0])        
        
        #change the origin to the top-left corner of the translated box
        orig_box_cords =  [max(0,corner_y), max(corner_x,0), min(img_shape[0], corner_y + img.shape[0]), min(img_shape[1],corner_x + img.shape[1])]

        mask = img[max(-corner_y, 0):min(img.shape[0], -corner_y + img_shape[0]), max(-corner_x, 0):min(img.shape[1], -corner_x + img_shape[1]),:]
        canvas[orig_box_cords[0]:orig_box_cords[2], orig_box_cords[1]:orig_box_cords[3],:] = mask
        img = canvas
        
        bboxes[:,:4] += [corner_x, corner_y, corner_x, corner_y]        
        bboxes = data_aug_bbox.clip_box(bboxes, [0,0,img_shape[1], img_shape[0]], 0.25)
        
        return img, bboxes
    
    
class RandomRotate(object):

    def __init__(self, angle = 10):
        """
            ####
            初期化
    
            ----------
            ####引数
            - angle: float or tuple(float)　デフォルトは10
            ####戻り値
            -------
            なし
        """
        self.angle = angle
        
        if type(self.angle) == tuple:
            assert len(self.angle) == 2, "Invalid range"  
            
        else:
            self.angle = (-self.angle, self.angle)



    def __call__(self, img, bboxes):
        """
            ランダムに画像回転する

             残りの25％未満の面積を持つバウンディングボックス
             変換された画像はドロップされます。
             解像度は維持され、残りは領域が黒色で塗りつぶされている場合。

             ###
             引数
             ----------
             - angle: float or tuple(float)
                 floatの場合: 　(1 - `scale` , 1 + `scale`)の範囲に回転する
                 tupleの場合: 　その値する
            image:   ndaaray   画像
            bboxes:  nparray   boundling boxes

             ####戻り値
             -------

             - image
                 回転したimage
             - boxes
                 回転したboundingbox
        """
        angle = random.uniform(*self.angle)
    
        w,h = img.shape[1], img.shape[0]
        cx, cy = w//2, h//2
    
        img = data_aug_bbox.rotate_im(img, angle)
    
        corners = data_aug_bbox.get_corners(bboxes)
    
        corners = np.hstack((corners, bboxes[:,4:]))

            
        corners[:,:8] = data_aug_bbox.rotate_box(corners[:,:8], angle, cx, cy, h, w)
    
        new_bbox = data_aug_bbox.get_enclosing_box(corners)

    
        scale_factor_x = img.shape[1] / w
    
        scale_factor_y = img.shape[0] / h
    
        img = cv2.resize(img, (w,h))
    
        new_bbox[:,:4] /= [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y] 
    
        bboxes  = new_bbox


        bboxes = data_aug_bbox.clip_box(bboxes, [0,0,w, h], 0.25)
    
        return img, bboxes

    
class Rotate(object):


    def __init__(self, angle):
        """
            ####
            初期化

            ----------
            ####引数
            - angle: float or tuple(float)
            ####戻り値
            -------
            なし
        """
        self.angle = angle
        

    def __call__(self, img, bboxes):
        """画像回転する

             残りの25％未満の面積を持つバウンディングボックス
             変換された画像はドロップされます。
             解像度は維持され、残りは領域が黒色で塗りつぶされている場合。

             ###
             引数
            ----------
            - angle: float or tuple(float)
            floatの場合: 　(1 - `scale` , 1 + `scale`)の範囲に回転する
            tupleの場合: 　その値する    angle: int   角度 デフォルトは10
            image:   ndaaray   画像
            bboxes:  nparray   boundling boxes
     　

             ####戻り値
             -------
             - image
                 回転したimage
             - boxes
                 回転したbounding box
        """

        
        angle = self.angle
        
        w,h = img.shape[1], img.shape[0]
        cx, cy = w//2, h//2
        
        corners = data_aug_bbox.get_corners(bboxes)
        
        corners = np.hstack((corners, bboxes[:,4:]))

        img = data_aug_bbox.rotate_im(img, angle)
        
        corners[:,:8] = data_aug_bbox.rotate_box(corners[:,:8], angle, cx, cy, h, w)
                
        
        new_bbox = data_aug_bbox.get_enclosing_box(corners)
        
        
        scale_factor_x = img.shape[1] / w
        
        scale_factor_y = img.shape[0] / h
        
        img = cv2.resize(img, (w,h))
        
        new_bbox[:,:4] /= [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y] 
        
        
        bboxes  = new_bbox

        bboxes = data_aug_bbox.clip_box(bboxes, [0,0,w, h], 0.25)
        
        return img, bboxes
        


class RandomShear(object):

    def __init__(self, shear_factor = 0.2):

        """
            ####
            初期化

            ----------
            ####引数
            - shear_factor: float or tuple(float)　　shear率
            ####戻り値
            -------
            なし
        """

        self.shear_factor = shear_factor
        
        if type(self.shear_factor) == tuple:
            assert len(self.shear_factor) == 2, "Invalid range for scaling factor"   
        else:
            self.shear_factor = (-self.shear_factor, self.shear_factor)
        
        shear_factor = random.uniform(*self.shear_factor)



        
    def __call__(self, img, bboxes):

        """
            ランダムに画像カットする

            残りの25％未満の面積を持つバウンディングボックス
            変換された画像はドロップされます。
            解像度は維持され、残りは領域が黒色で塗りつぶされている場合。

            ###
            引数
            ----------
       　　- angle: float or tuple(float)
           floatの場合: 　(1 - `scale` , 1 + `scale`)の範囲にカットする
           tupleの場合: 　その値する    angle: int   角度 デフォルトは10　
            image:   ndaaray   画像
       　　 bboxes:  nparray   boundling boxes

            ####戻り値
            -------
    　
            - image
                カットしたimage
            - boxes
                カットしたboundingbox
        """
    
        shear_factor = random.uniform(*self.shear_factor)
    
        w,h = img.shape[1], img.shape[0]
    
        if shear_factor < 0:
            img, bboxes = HorizontalFlip()(img, bboxes)
    
        M = np.array([[1, abs(shear_factor), 0],[0,1,0]])
    
        nW =  img.shape[1] + abs(shear_factor*img.shape[0])
    
        bboxes[:,[0,2]] += ((bboxes[:,[1,3]]) * abs(shear_factor) ).astype(int) 
    
    
        img = cv2.warpAffine(img, M, (int(nW), img.shape[0]))
    
        if shear_factor < 0:
            img, bboxes = HorizontalFlip()(img, bboxes)
    
        img = cv2.resize(img, (w,h))
    
        scale_factor_x = nW / w
    
        bboxes[:,:4] /= [scale_factor_x, 1, scale_factor_x, 1] 
    
    
        return img, bboxes
        
class Shear(object):

    def __init__(self, shear_factor = 0.2):
        """
            ####
            初期化
        
            ----------
            ####引数
            - shear_factor: float or tuple(float)　　shear率
            ####戻り値
            -------
            なし
        """
        self.shear_factor = shear_factor


    def __call__(self, img, bboxes):
        """
            画像カットする

             残りの25％未満の面積を持つバウンディングボックス
             変換された画像はドロップされます。
             解像度は維持され、残りは領域が黒色で塗りつぶされている場合。

             ###
             引数
             ----------
             - shear: float or tuple(float)
                 floatの場合: 　(1 - `scale` , 1 + `scale`)の範囲にカットする
                 tupleの場合: 　その値する
            　
            -image:   ndaaray   画像
           　-bboxes:  nparray   boundling boxes


             ####戻り値
             -------

             - image
                 カットしたimage
             - boxes
                 カットしたbounding box
        """


        shear_factor = self.shear_factor
        if shear_factor < 0:
            img, bboxes = HorizontalFlip()(img, bboxes)

        M = np.array([[1, abs(shear_factor), 0],[0,1,0]])
                
        nW =  img.shape[1] + abs(shear_factor*img.shape[0])
        
        bboxes[:,[0,2]] += ((bboxes[:,[1,3]])*abs(shear_factor)).astype(int)
        img = cv2.warpAffine(img, M, (int(nW), img.shape[0]))
        
        if shear_factor < 0:
            img, bboxes = HorizontalFlip()(img, bboxes)

        return img, bboxes
    
class Resize(object):

    
    def __init__(self, inp_dim):
        """
            ####
            初期化

            ----------
            ####引数
            - inp_dim: float 　

            ####戻り値
            -------
            なし
        """
        self.inp_dim = inp_dim


    def __call__(self, img, bboxes):
        """画像リサイズする
            `image_letter_box`関数に従って画像のサイズを変更します` function in darknet
            アスペクト比は維持されます。長い辺は入力に合わせてサイズ変更されます
            ネットワークのサイズ、短辺の残りのスペースはいっぱい
            黒い色で。 **これが最後の変換でなければなりません***

             ###
             引数
            - inp_dim : tuple(int)　リサイズの比率
        　  - image:   ndaaray   画像
            - bboxes:  nparray   boundling boxes

            ####戻り値
            -------
            　- image
                 カットしたimage
             - boxes
                 カットしたbounding box

        """
        w,h = img.shape[1], img.shape[0]
        img = data_aug_bbox.letterbox_image(img, self.inp_dim)
    
    
        scale = min(self.inp_dim/h, self.inp_dim/w)
        bboxes[:,:4] *= (scale)
    
        new_w = scale*w
        new_h = scale*h
        inp_dim = self.inp_dim   
    
        del_h = (inp_dim - new_h)/2
        del_w = (inp_dim - new_w)/2
    
        add_matrix = np.array([[del_w, del_h, del_w, del_h]]).astype(int)
    
        bboxes[:,:4] += add_matrix
    
        img = img.astype(np.uint8)
    
        return img, bboxes 
    

class RandomHSV(object):

    
    def __init__(self, hue = None, saturation = None, brightness = None):
        """
            ####
            初期化

            ----------
            ####引数
            -色相：なしまたはintまたはtuple（int）
                    なしの場合、  画像の色相は変更されません。
                    intの場合、     ンダムなintは（-hue、hue）から均一にサンプリングされ、
                    画像の色相。
                    tuple（int）の場合、  intは範囲からサンプリングされます
                    タプルによって指定されます。

            -彩度：なしまたはintまたはtuple（int）
                    なしの場合、画像の彩度は変更されません。
                    intの場合、ランダムなintは（-saturation、saturation）から一様にサンプリングされます
                    彩度の色相に追加されます。
                    tuple（int）の場合、intがサンプリングされます
                    タプルで指定された範囲から。

            -明度：なしまたはintまたはtuple（int）
                    なしの場合、画像の明るさは変更されません。
                    intの場合、
                    ランダムなintは（-brightness、brightness）から均一にサンプリングされます
                    画像の色相に追加されます。
                    tuple（int）の場合、intがサンプリングされます
                    タプルで指定された範囲から。

            ####戻り値
            -------
            なし
        """
        if hue:
            self.hue = hue 
        else:
            self.hue = 0
            
        if saturation:
            self.saturation = saturation 
        else:
            self.saturation = 0
            
        if brightness:
            self.brightness = brightness
        else:
            self.brightness = 0
            
            

        if type(self.hue) != tuple:
            self.hue = (-self.hue, self.hue)
            
        if type(self.saturation) != tuple:
            self.saturation = (-self.saturation, self.saturation)
        
        if type(brightness) != tuple:
            self.brightness = (-self.brightness, self.brightness)


    def __call__(self, img, bboxes):
        """
            色相の彩度と明度を変えるHSV変換

            色相の範囲は0〜179です。
            彩度と明るさの範囲は0〜255です。
            それに応じて、上記の数量を変更する金額を選択します。

             ###
             引数
            image:   ndaaray   画像
            bboxes:  nparray   boundling boxes

            ####戻り値
             -------
            　- image
                 カットしたimage
             - boxes
                 カットしたbounding box

        """

        hue = random.randint(*self.hue)
        saturation = random.randint(*self.saturation)
        brightness = random.randint(*self.brightness)
        
        img = img.astype(int)
        
        a = np.array([hue, saturation, brightness]).astype(int)
        img += np.reshape(a, (1,1,3))
        
        img = np.clip(img, 0, 255)
        img[:,:,0] = np.clip(img[:,:,0],0, 179)
        
        img = img.astype(np.uint8)

        return img, bboxes
    
class Sequence(object):


    def __init__(self, augmentations, probs = 1):

        """
            ####
            初期化

            ----------
            ####引数
            - augemnetations: list
            変換オブジェクトを順番に含むリスト
            適用された


            - probs: int or list

            ** int ** の場合、各変換が発生する確率
            適用されます。
            ** list ** の場合、長さは * augmentations * と等しくなければなりません。
            このリストの各要素は、それぞれが
            対応する変換が適用されます
            ####戻り値
            -------
            なし
        """
        self.augmentations = augmentations
        self.probs = probs


    def __call__(self, images, bboxes):
        """
            シーケンスオブジェクトの初期化

            変換のシーケンスを画像/ボックスに適用します。

            ###引数
            ----------

            -image:   ndaaray   画像
            -bboxes:  nparray   boundling boxes

            ###戻り値
            -------
            　- image
                 変換したimage
             - boxes
                 変換したbounding box
        """
        for i, augmentation in enumerate(self.augmentations):
            if type(self.probs) == list:
                prob = self.probs[i]
            else:
                prob = self.probs
                
            if random.random() < prob:
                images, bboxes = augmentation(images, bboxes)
        return images, bboxes

def Transform(transformer, x_train, y_train):
    """画像データのタイプ変換
　
     ###
     引数
　   transformer:
　   x_train:
    　y_train:
    ####戻り値
    -------
    　- image
         変換したimage
     - boxes
         変換したbounding box

    """
    for i, (x, y) in enumerate(zip(x_train, y_train)):
        print("Transform",x, y)
        tra_x, tra_y = transformer(x.transpose(1, 2, 0), np.array(y)[...,:4].astype(np.float))
        for tra, org in zip(tra_y, y):# override xywh from transform

            org[0],org[1],org[2],org[3] = tra[0],tra[1],tra[2],tra[3] 

        x_train[i] = tra_x.transpose(2,0,1)
    
    return x_train, y_train
