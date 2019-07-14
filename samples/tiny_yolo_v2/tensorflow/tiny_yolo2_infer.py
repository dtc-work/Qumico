import colorsys
import numpy as np
import tensorflow as tf
from samples.tiny_yolo_v2.tensorflow.tiny_yolo2_model import TINY_YOLO_v2
from samples.utils.common_tool import sigmoid
import cv2
from samples.utils.box_convert import bbox_to_anbox, anbox_to_bbox
import samples.utils.pre_process_tool as list_reader
from samples.utils.annotation_dataset_tool import AnnotationDatasetTool


voc2007_classes = ['chair', 'bird', 'sofa', 'bicycle', 'cat', 'motorbike', 'bus', 'boat', 'sheep', 'bottle', 'cow',
                   'person', 'horse', 'diningtable', 'pottedplant', 'aeroplane', 'car', 'train', 'dog', 'tvmonitor']


def random_colors(N, bright=True):
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    np.random.shuffle(colors)
    return colors


def non_max_suppression(boxes, overlapThresh=0.6):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
 
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
 
    # initialize the list of picked indexes    
    pick = []
 
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
 
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
 
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
 
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
 
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
 
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
 
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick]


def draw(image_bgr, features, classes, labels=None, threshold=0.4, to_draw=True):
    image_bgr_1 = image_bgr
    image_bgr_2 = np.copy(image_bgr)
    font = cv2.FONT_HERSHEY_SIMPLEX

    detected_boxes = []
    res = []
    for box in features:
        if box[4] > threshold:
            box[0:4:2] = box[0:4:2] * image_bgr_1.shape[1]
            box[1:4:2] = box[1:4:2] * image_bgr_1.shape[0]
            detected_boxes.append(box)

    for box in non_max_suppression(np.array(detected_boxes)):
        classes_index = np.argmax(box[5:25])
        label = voc2007_classes[classes_index]
        box = np.asarray(box, dtype="int32")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.rectangle(image_bgr_1, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        cv2.putText(image_bgr_1, label, (box[0], box[1]+10), font, 0.4, (10, 200, 10), 1, cv2.LINE_AA)
 
        res.append((label, box[0:4]))

    if labels is not None:
        for tag in labels:
            classes_id_array = np.copy(tag[4:])
            tag[0::2] = tag[0::2] * image_bgr_2.shape[1]
            tag[1::2] = tag[1::2] * image_bgr_2.shape[0]
            classes_id = np.argmax(classes_id_array)
            class_label_true = voc2007_classes[classes_id]
            cv2.rectangle(image_bgr_2, (tag[0], tag[1]), (tag[2], tag[3]), (0, 0, 255), 2)
            cv2.putText(image_bgr_2, class_label_true, (int(tag[0]), int(tag[1]) + 10), font, 0.4, (10, 200, 10), 1, cv2.LINE_AA) 

    if to_draw:
        image_bgr_1 = np.concatenate((image_bgr_1, image_bgr_2))
        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.imshow("image", image_bgr_1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return res


def prepare_boxes(feature, anchors, grid_h, grid_w, offset, classes, block_size):
    num_anchors = len(anchors)
    num_classes = len(classes)
    feature = np.transpose(feature, (2, 1, 0))
    feature = np.reshape(feature, (grid_h, grid_w, num_anchors, 5 + num_classes))
    feature_xy, feature_wh, feature_conf, feature_classes = np.split(feature, [2, 4, 5], axis=-1)
    feature_xy = (sigmoid(feature_xy) + offset) / [grid_w, grid_h]
    feature_wh = np.exp(feature_wh)
    feature_wh = feature_wh * anchors / [grid_w, grid_h]
    feature_boxes = np.concatenate((feature_xy, feature_wh), axis=-1)
    feature_bbox = anbox_to_bbox(feature_boxes)
    feature_conf = sigmoid(feature_conf)
    feature_concat = np.concatenate((feature_bbox, feature_conf, feature_classes), axis=-1)
    feature_prepared = feature_concat.reshape((-1, 5 + num_classes))
    return feature_prepared


def infer(model, infer_data, ckpt_file, classes, batch_size, to_draw=True):
    data_size = infer_data.total_size
    total_batch = data_size // batch_size
    for i in range(total_batch):
        batch_x, batch_y, x_path, y_path = infer_data.next_batch_once(batch_size=batch_size)

        image_data = cv2.imread(x_path[0], 1)
        grid_height = model.grid_h
        grid_width = model.grid_w
        offset = model.get_offset_yx(grid_h=grid_height, grid_w=grid_width)
        block_size = model.block_size
        anchors = model.anchors

        tf.reset_default_graph()
        with tf.Session(graph=model.graph) as sess_predict:
            model.saver.restore(sess_predict, ckpt_file)
            output = sess_predict.run(model.output, feed_dict={model.inputs: batch_x})

        feature = prepare_boxes(output[0], anchors, grid_height, grid_width, offset, classes, block_size)

        return draw(image_data, feature, classes, labels=batch_y[0], to_draw=to_draw)


if __name__ == '__main__':
    # クラス　ラベル
    voc2007_classes = ['chair', 'bird', 'sofa', 'bicycle', 'cat', 'motorbike', 'bus', 'boat', 'sheep', 'bottle', 'cow',
                       'person', 'horse', 'diningtable', 'pottedplant', 'aeroplane', 'car', 'train', 'dog', 'tvmonitor']
    # クラスの数　Ont-Hot表現用パラメータ
    num_classes = len(voc2007_classes)

    # データルートパス(絶対パス)を設定する
    # root_path = ""

    # 画像ファイルフォルダ
    data_list_path = root_path + "JPEGImages"
    # タグデータフォルダ（サンプルでは xml に対応しています、
    # 他のデータで学習させたい場合は、utilsフォルダ中のannotation_dataset_toolファイルに読み込むロジックを追加してください。
    label_list_path = root_path + "Annotations"

    # infer 画像のid を設定する
    pic_num = 0

    # フォルダからinfer画像ファイルのパスを取得する
    data_list = np.asarray(list_reader.get_data_path_list(data_list_path)[pic_num:pic_num+1])
    label_list = np.asarray(list_reader.get_data_path_list(label_list_path)[pic_num:pic_num+1])

    # 読み方、バッチ処理などを設定し、データを供給するツール設定する
    annotation_dataset_tool = AnnotationDatasetTool(training_flag=True, data_list=data_list, label_list=label_list,
                                                    category_class=voc2007_classes, one_hot_classes=num_classes,
                                                    resize_flag=True, target_h=416, target_w=416,
                                                    label_file_type="voc_xml", format="NCHW", data_rescale=True, label_resclar=True)
    batch_size = 1
    # init model
    tiny_yolo2_model = TINY_YOLO_v2(output_op_name="output", num_classes=20, is_train=False, width=416, height=416)

    # model weights path
    ckpt_file = "model/tiny_yolo2.ckpt"
    infer(tiny_yolo2_model, annotation_dataset_tool, ckpt_file, voc2007_classes, batch_size)
