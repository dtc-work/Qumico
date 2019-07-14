import numpy
import os
import shutil
import unittest

from qumico import Qumico
from qumico import compile
from qumico.export import ExportType

from samples.tiny_yolo_v2.tensorflow import tiny_yolo2_infer_c
from samples.tiny_yolo_v2.tensorflow import tiny_yolo2_infer
from samples.tiny_yolo_v2.tensorflow import tiny_yolo2_model
from samples.utils.annotation_dataset_tool import AnnotationDatasetTool
from samples.utils import pre_process_tool
from samples.utils import box_convert

import tests


class TestTinyYolo2InferC(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.current_path = os.path.dirname(os.path.abspath(__file__))
        cls.input_path = os.path.join(cls.current_path, "input")
        cls.image_path = os.path.join(cls.input_path, "images")

        cls.onnx_path = os.path.join(cls.current_path, "onnx")
        cls.onnx_file = os.path.join(cls.onnx_path, "tiny_yolo_2.onnx")
        cls.out_c_path = os.path.join(cls.current_path, "out_c")
        cls.out_c_file = os.path.join(cls.out_c_path, "qumico.so")

        cls.ckpt_file = os.path.join(cls.current_path, "input", "tiny_yolo2.ckpt")
        cls.pb_file = os.path.join(cls.current_path, "input", "tiny_yolo2.pb")

        cls.classes = tiny_yolo2_infer.voc2007_classes
        cls.num_classes = len(cls.classes)

        cls.classes = tiny_yolo2_infer.voc2007_classes
        cls.num_classes = len(cls.classes)
        cls.batch_size = 1

        pic_num = 0
        data_list_path = os.path.join(cls.input_path, "images")
        cls.label_list_path = os.path.join(cls.input_path, "annotations")

        data_list = numpy.asarray(pre_process_tool.get_data_path_list(data_list_path)[pic_num:pic_num + 1])
        label_list = numpy.asarray(pre_process_tool.get_data_path_list(cls.label_list_path)[pic_num:pic_num + 1])

        cls.annotation_dataset_tool = AnnotationDatasetTool(training_flag=True, data_list=data_list,
                                                            label_list=label_list, category_class=cls.classes,
                                                            one_hot_classes=cls.num_classes, resize_flag=True,
                                                            target_h=416, target_w=416, label_file_type="voc_xml",
                                                            format="NCHW", data_rescale=True, label_resclar=True)

        cls.model = tiny_yolo2_model.TINY_YOLO_v2(output_op_name="output",
                                                  num_classes=cls.num_classes,
                                                  is_train=False,
                                                  width=416,
                                                  height=416)

        cls._infer_prepare()

    @classmethod
    def tearDown(cls) -> None:
        tests.remove_folder(cls.out_c_path)
        tests.remove_folder(cls.onnx_path)
        tests.remove_folder(os.path.join(cls.current_path, "model"))

    @classmethod
    def _generate_boxes(cls, box_data):

        boxes = []
        if type(box_data) is list:
            for x in box_data:
                if type(x) is numpy.ndarray:
                    tmp = x.tolist()
                    boxes.append(box_convert.BoundBox(xmin=tmp[0], xmax=tmp[2], ymin=tmp[1], ymax=tmp[3]))
                elif type(x) is tuple:
                    tmp = list(x)[1]
                    boxes.append(box_convert.BoundBox(xmin=tmp[0], xmax=tmp[2], ymin=tmp[1], ymax=tmp[3]))
        else:
            return None

        return boxes

    @classmethod
    def _get_average_iuo_value(cls, boxes_1, boxes_2):

        val = 0

        for box_1 in boxes_1:
            max = 0
            for box_2 in boxes_2:
                iuo = box_convert.bbox_iou(box_1, box_2)
                if iuo > max:
                    max = iuo

            val += max

        return val / len(boxes_1)

    @classmethod
    def _generate_c(cls):
        converter = Qumico.Qumico()
        converter.conv_tf_to_onnx(output_path=cls.onnx_path, model_name="tiny_yolo_2", output_op_name="output",
                                  cache_path="model", ckpt_file=cls.ckpt_file, pb_file=cls.pb_file)

        Qumico.Qumico().conv_onnx_to_c(cls.onnx_file, cls.out_c_path, compile=True, export_type=ExportType.NPY)

    @classmethod
    def _infer_prepare(cls):

        cls._generate_c()

        batch_file_name = "batchnormalization.c"
        maxpool_file_name = "maxpool.c"
        qumico_c_file_name = "qumico.c"

        dest_batch_file = os.path.join(cls.out_c_path, "lib", batch_file_name)
        dest_maxpool_file = os.path.join(cls.out_c_path, "lib", maxpool_file_name)
        souce_batch_file = os.path.join(os.path.dirname(tiny_yolo2_infer_c.__file__),
                                         "known_issue",
                                         batch_file_name)
        souce_maxpool_file = os.path.join(os.path.dirname(tiny_yolo2_infer_c.__file__),
                                           "known_issue",
                                           maxpool_file_name)

        shutil.copyfile(souce_batch_file, dest_batch_file)
        shutil.copyfile(souce_maxpool_file, dest_maxpool_file)

        compile.node_compile(os.path.join(cls.out_c_path, qumico_c_file_name), device=None)

    @unittest.skipIf(tests.IS_CIRCLE_CI, "Skip this test because large memory")
    def test_infer_c(self):

        label_path = os.path.join(self.label_list_path, "test.xml")
        origin = self.annotation_dataset_tool.get_annotation_label(full_path=label_path,
                                                                   label_file_type="voc_xml",
                                                                   resize_rate_h=1,
                                                                   resize_rate_w=1)

        output = tiny_yolo2_infer_c.infer_c(model=self.model,
                                            infer_data=self.annotation_dataset_tool,
                                            c_path=self.out_c_file,
                                            classes=self.classes,
                                            batch_size=self.batch_size,
                                            to_draw=False)

        iou_val = self._get_average_iuo_value(self._generate_boxes(origin), self._generate_boxes(output))
        self.assertGreaterEqual(iou_val, 0.6)


if __name__ == "__main__":
    unittest.main()
