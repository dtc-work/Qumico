from os import path
import ctypes
import cv2
import numpy as np


import samples.utils.pre_process_tool as list_reader
from samples.tiny_yolo_v2.tensorflow.tiny_yolo2_model import TINY_YOLO_v2
from samples.tiny_yolo_v2.tensorflow.tiny_yolo2_infer import voc2007_classes, draw, prepare_boxes
from samples.utils.annotation_dataset_tool import AnnotationDatasetTool


def infer_c(model, infer_data, c_path, classes, batch_size, to_draw=True):
    # config
    input_info = np.ctypeslib.ndpointer(dtype=np.float32, ndim=4,
                                        shape=(1, 3, 416, 416), flags='CONTIGUOUS')
    output_info = np.ctypeslib.ndpointer(dtype=np.float32, ndim=4,
                                         shape=(1,125, 13, 13), flags='CONTIGUOUS')

    dll = ctypes.CDLL(c_path)
    dll.qumico.argtypes = [input_info, output_info]
    dll.qumico.restype = ctypes.c_int
    dll.run.argtypes = [input_info, output_info]
    dll.run.restype = ctypes.c_int

    
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

        input = np.ascontiguousarray(np.expand_dims(batch_x[0], 0).astype(np.float32))
        output =np.zeros(dtype=np.float32, shape=(1, 125, 13, 13)) # (1, 125, 13, 13)
        dll.qumico(input, output)
        feature = prepare_boxes(output[0], anchors, grid_height, grid_width, offset, classes, block_size)

        return draw(image_data, feature, classes, labels=batch_y[0], threshold=model.threshold, to_draw=to_draw)


if __name__ == '__main__':
    # Cライブラリパス
    c_path = path.join(path.dirname(path.abspath(__file__)), "out_c", "qumico.so") 

    # クラスの数　Ont-Hot表現用パラメータ
    num_classes = len(voc2007_classes)

    # データルートパス(絶対パス)を設定する
    root_path = "./data/"

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
    # print(tiny_yolo2_model.threshold)

    # model weights path
    ckpt_file = "model/tiny_yolo2.ckpt"
    infer_c(tiny_yolo2_model, annotation_dataset_tool, c_path, voc2007_classes, batch_size)
