import numpy as np
import random
from PIL import Image
import cv2
import json
import xml.etree.ElementTree as ET
class AnnotationDatasetTool:
    '###分類クラス'
    category_class = ['Bicycle', 'Bus', 'Car', 'Motorbike', 'Pedestrian', 'SVehicle', 'Signal', 'Signs', 'Train',
                      'Truck']
    '###分類クラスの拡張'
    category_class_extend = ['day', 'morning', 'night']
    """"""
    "###初期化"
    """"""
    "###引数"
    """"""
    "training_flag : bool   学習"
    "data_list : list   画像の入力データ"
    "label_list : list  ラベルの入力データ"
    "repeat : bool   重複の有無"
    "resize_flag: bool 画像リサイズの有無"
    "target_h: int  出力画像サイズ(縦)"
    "target_w: int 出力画像サイズ(横)"
    "data_rescale: bool 画素値のリスケーリングの有無"
    "label_resclar ラベルのリスケーリングの有無"
    "box_format_trans:   現在未使用"
    "category_class: list  分類クラス"
    "label_file_type: string  ラベルフィイルのフィイルタイプ"
    "format: string     画像のフォーマットタイプ"
    "transformer: bool  HCWHからHWCHに変換"
    "####training_flag=True の場合　入力データサイズとラベルデータサイズの値をチェックする"
    "学習データサイズが異なる, 入力データサイズ = {0} ラベルサイズ = {1}のエラーメッセージを表示する"

    "###戻り値"
    "なし"

    def __init__(self, training_flag=False, data_list=None, label_list=None, repeat=False, one_hot_classes=None,
                 resize_flag=False, target_h=None, target_w=None, data_rescale=None, label_resclar=None,
                 box_format_trans=None, category_class=None, label_file_type=None, format="NHWC",
                 transformer=None, **kwargs):
        self.training = training_flag
        self.data_list = data_list
        self.label_list = label_list
        self.one_hot_classes = one_hot_classes
        self.repeat = repeat
        self.total_size = data_list.shape[0]
        self.index_list = list(np.arange(0, self.total_size))
        self.resize_flag = resize_flag
        self.target_h = target_h
        self.target_w = target_w
        self.label_file_type = label_file_type
        self.format = format
        self.data_rescale = data_rescale
        self.label_rescale = label_resclar
        self.transformer = transformer
        if category_class is not None:
            self.category_class = category_class
        else:
            pass
        np.random.shuffle(self.index_list)

        if training_flag:
            try:
                assert data_list.shape[0] == label_list.shape[0], "学習データサイズが異なる, 入力データサイズ = {0} ラベルサイズ = {1}".format(
                    data_list.shape[0], label_list.shape[0])
            except AssertionError as err:
                print("Error : ", err)




    def next_batch(self, batch_size):
        """
            ランダム順にバッチ学習を行う。
            DatasetToolクラスのインスタンス生成時に
            repeat=True設定をしていれば重複あり、repeat=Falseを設定していれば重複なしで、バッチ実行データを選択する。
            #### 引数
            - batch_size: int  バッチサイズ
            #### 戻り値
            バッチ実行回数
        """
        try:
            assert batch_size < self.total_size, "batch_sizeがtotal_batchを超えている。"
        except AssertionError as err:
            print("Error : ", err)
        if self.repeat:
            batch_mask = random.choices(self.index_list, k=batch_size)
        else:
            batch_mask = random.sample(self.index_list, k=batch_size)

        if self.training:
            x_path_batch = self.data_list[batch_mask]
            y_path_batch = self.label_list[batch_mask]
            x_batch, y_batch = self.get_train_batch(x_path_batch, y_path_batch, data_rescale=self.data_rescale,
                                                    label_rescale=self.label_rescale,
                                                    one_hot_classes=self.one_hot_classes)
            return x_batch, y_batch, x_path_batch, y_path_batch
        else:
            x_path_batch = self.data_list[batch_mask]
            x_batch = self.get_infer_batch(x_path_batch)
            return x_batch, x_path_batch

    def next_batch_once(self, batch_size):
        """
            あらかじめ決めた順にバッチ学習を行う。
            #### 引数
            - batch_size: バッチサイズ
            #### 戻り値
            バッチ実行回数
        """
        if len(self.index_list) == 0:
            self.index_reset()
        batch_mask = self.index_list[:batch_size]
        self.index_list = self.index_list[batch_size:]

        if self.training:
            x_path_batch = self.data_list[batch_mask]
            y_path_batch = self.label_list[batch_mask]
            x_batch, y_batch = self.get_train_batch(x_path_batch, y_path_batch, data_rescale=self.data_rescale,
                                                    label_rescale=self.label_rescale,
                                                    one_hot_classes=self.one_hot_classes)
            return x_batch, y_batch, x_path_batch, y_path_batch
        else:
            x_path_batch = self.data_list[batch_mask]
            x_batch = self.get_infer_batch(x_path_batch)
            return x_batch, x_path_batch

    def index_reset(self):
        """
            バッチ実行リストの初期化を行う。
            #### 引数
            なし
            #### 戻り値
            なし
        """
        self.index_list = list(np.arange(0, self.total_size))

    def get_train_batch(self, data_path_list, label_path_list, data_rescale=False, label_rescale=False,
                        one_hot_classes=None):
        """
            入力データバッチの取得。
            #### 引数
            "data_path_list：　list"　  画像データディレクトリのファイルリスト
            "label_path_list：　list"  　ラベルディレクトリのファイルリスト
            "data_rescale: bool 画素値のリスケーリングの有無"
            "label_resclar ラベルのリスケーリングの有無"
            "one_hot_classes ： int　　OneHotクラス"
            #### 戻り値
            x_train　　入力データ
            y_train　　ラベルデータ　
        """
        x_train = []
        y_train = []
        for index, data_path in enumerate(data_path_list):
            img_array, resize_rate_h, resize_rate_w = self.image_generator(data_path_list[index],
                                                                           target_h=self.target_h,
                                                                           target_w=self.target_w,
                                                                           format=self.format)
            train_label = self.get_annotation_label(label_path_list[index],
                                                    label_file_type=self.label_file_type,
                                                    resize_rate_h=resize_rate_h,
                                                    resize_rate_w=resize_rate_w,
                                                    one_hot_classes=one_hot_classes)
            
            if self.training and self.transformer is not None:
                tra_x, tra_y = self.transformer(img_array.transpose(1, 2, 0), np.array(train_label)[...,:4].astype(np.float))

                for i in range(len(train_label)):
                    train_label[i][0] = tra_y[i][0]
                    train_label[i][1] = tra_y[i][1]
                    train_label[i][2] = tra_y[i][2]
                    train_label[i][3] = tra_y[i][3]
 
                img_array = tra_x.transpose(2,0,1)

            if data_rescale:
                img_array = img_array / 255.

            if label_rescale:
                for i in range(len(train_label)):
                    train_label[i]=train_label[i].astype(np.float32)
                    train_label[i][0] /= self.target_w
                    train_label[i][1] /= self.target_h
                    train_label[i][2] /= self.target_w
                    train_label[i][3] /= self.target_h
            
            x_train.append(img_array)
            y_train.append(train_label)
        return x_train, y_train

    def get_infer_batch(self, data_path_list):
        """
            入力データバッチの取得。
            #### 引数
            "data_path_list:  list"  入力データディレクトリのファイルリスト
    
            #### 戻り値
            x_infer　　画像データ
        """
        x_infer = []
        for index, data_path in enumerate(data_path_list):
            img_valid, resize_rate_h, resize_rate_w = self.image_generator(data_path, target_h=self.target_h,
                                                                           target_w=self.target_w, format=self.format)
            x_infer.append(img_valid)
        return x_infer

    def image_generator(self, full_path, target_h=224, target_w=224, is_opencv=True, format="NHWC",
                        histogram=True):
        """
            画像の生成を行う。
            #### 引数
            - full_path: list　入力する画像ファイルのパス
            - target_h: float　　出力画像サイズ(縦)
            - target_w: float　　出力画像サイズ(横)
            - rescale: 　bool　リサイズ
            - is_opencv:  bool　　Trueの場合OpenCVから読み込み、Falseの場合画像ファイルから読み込みを行う(default=True)
            - format:  string  出力画像tensorのディメンジョン順(default=NHWC: バッチ数/縦/横/チャネル)
            - histogram:     現在未使用(default=True)
            #### 戻り値
            画像データ、リサイズ比率(縦)、リサイズ比率(横)  のリスト
        """

        if is_opencv:
            # Use Opencv
            image_bgr = cv2.imread(full_path, 1)
            height, width = image_bgr.shape[0], image_bgr.shape[1]
            image_cv = cv2.resize(image_bgr, (target_w, target_h))
            image_array = image_cv

            # print(image_bgr)
            # cv2.imshow("image", image_bgr)

        else:
            # Use PIL
            image_pil = Image.open(full_path)
            height, width = image_pil.height, image_pil.width

            image_rgb = image_pil.convert("RGB")
            image_resize = image_rgb.resize((target_h, target_w), Image.LANCZOS)
            image_array = np.asarray(image_resize, dtype="uint8")

            # image_pil.show()

        resize_rate_h = target_h / height
        resize_rate_w = target_w / width


        if format == "NCHW":
            image_array = np.transpose(image_array, (2, 0, 1))

        return image_array, resize_rate_h, resize_rate_w

    def get_annotation_label(self, full_path, label_file_type=None,
                             resize_rate_h=None, resize_rate_w=None, one_hot_classes=None):
        """
            ラベルデータの取得。
            jsonとxmlの違い
            #### 引数
            "full_path": list  入力データディレクトリのファイルリスト
            "label_file_type": list  　ラベルディレクトリのファイルリスト
            "resize_rate_h:　　float  出力画像サイズ率(縦)　"　
            "resize_rate_w:　　float   出力画像リイサイズ率(横)　"　　
            "one_hot_classes:  int  OneHotクラス"
            #### 戻り値
            x_train　　入力データ
            y_train　　ラベルデータ　
        """

        if label_file_type == "json":
            bbox_classes, day_time = self.get_bbox_json_ext(json_file_path=full_path, resize_rate_h=resize_rate_h,
                                                            resize_rate_w=resize_rate_w,
                                                            one_hot_classes=one_hot_classes)
            return bbox_classes

        elif label_file_type == "voc_xml":
            class_list = self.get_bbox_xml_ext(full_path, resize_rate_h=resize_rate_h,
                                               resize_rate_w=resize_rate_w,
                                               one_hot_classes=one_hot_classes)

            return class_list
        else:
            return None


    def get_bbox_json_ext(self, json_file_path, resize_rate_h=None, resize_rate_w=None,
                          one_hot_classes=None):

        """
            入力データのboounding box作る(ラベルフィイルタイプはjsonの場合)
            #### 引数
            - json_file_path: String jsonラベルフィイルのpath
            - resize_rate_h:  float  出力画像サイズ率(縦)
            - resize_rate_w:  float   出力画像リイサイズ率(横)
            - one_hot_classes: int OneHotクラス
            #### 戻り値
            学習ラベル、時間（朝、午後、夜）
        """

        with open(json_file_path, encoding="utf-8") as json_file:
            json_data = json.loads(json_file.read())
        classes_truth = []
        ground_truth = []
        day_time = self.category_class_extend.index(json_data['attributes']['timeofday'])
        labels = json_data['labels']
        for label in labels:
            classes_truth.append(self.category_class.index(label['category']))
            ground_truth.append(
                [label['box2d']['x1'], label['box2d']['y1'],
                 label['box2d']['x2'], label['box2d']['y2']])
        classes_array = np.asarray(classes_truth)
        if one_hot_classes is not None:
            classes_array = np.identity(one_hot_classes)[classes_array]
        else:
            classes_array = np.expand_dims(classes_array, 1)

        bbox_array = np.array(ground_truth, dtype=np.float32)
        if resize_rate_h is not None:
            bbox_array[:, 1::2] *= resize_rate_h
        else:
            pass
        if resize_rate_w is not None:
            bbox_array[:, 0::2] *= resize_rate_w
        else:
            pass

        train_label = np.hstack((bbox_array, classes_array)).astype(np.uint16)

        return train_label, day_time

    def get_bbox_xml_ext(self, xml_file, resize_rate_h, resize_rate_w, one_hot_classes=None):
        """
            入力データのbounding box作る(ラベルフィイルタイプはxmlの場合)
            #### 引数
            - xml_file: file object   xmlのフィイル
            - resize_rate_h:  float  出力画像サイズ率(縦)
            - resize_rate_w:  float   出力画像リイサイズ率(横)
            - one_hot_classes: int　 OneHotクラス
            #### 戻り値
            学習XMLデータ　　
        """

        tree = ET.parse(xml_file)
        root = tree.getroot()
        train_labels = []

        for boxes in root.iter('object'):
            class_name = boxes.find('name').text
            class_id = self.category_class.index(class_name)
            if one_hot_classes is not None:
                classes_array = np.identity(one_hot_classes)[class_id]
            else:
                classes_array = np.expand_dims(class_id, 1)
            ymin, xmin, ymax, xmax = None, None, None, None

            for box in boxes.findall("bndbox"):
                xmin = int(box.find("xmin").text) * resize_rate_w
                ymin = int(box.find("ymin").text) * resize_rate_h
                xmax = int(box.find("xmax").text) * resize_rate_w
                ymax = int(box.find("ymax").text) * resize_rate_h

            boxes_item = [xmin, ymin, xmax, ymax]
            train_label = np.hstack((boxes_item, classes_array)).astype(np.uint16)
            train_labels.append(train_label)
        return train_labels


    def json_reader(self, json_file_path: str):
        """
            入力データのbounding box作る(ラベルフィイルタイプはxmlの場合)
            #### 引数
            - json_file_path: string jsonラベルフィイルのpath
            #### 戻り値
            学習JSONデータ　　
        """
        with open(json_file_path, encoding="utf-8") as json_file:
            json_data = json.loads(json_file.read())
        return json_data
