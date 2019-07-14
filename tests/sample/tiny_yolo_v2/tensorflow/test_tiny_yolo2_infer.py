import cv2
import numpy
import os
import unittest

from samples.tiny_yolo_v2.tensorflow import tiny_yolo2_infer
from samples.tiny_yolo_v2.tensorflow import tiny_yolo2_model

from samples.utils import box_convert
from samples.utils import pre_process_tool
from samples.utils.annotation_dataset_tool import AnnotationDatasetTool

import tests


class TestTinyYolo2Infer(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.current_path = os.path.dirname(os.path.abspath(__file__))
        cls.input_path = os.path.join(cls.current_path, "input")
        cls.image_path = os.path.join(cls.input_path, "images")

        cls.boxes_test = numpy.asarray([[0, 1, 2, 3], [-4, -5, -6, -7], [8, 9, -10, 11], [12, -13, -14, 15]])
        cls.feature_test = numpy.asarray([[0, 1, 2, 3, 0],
                                         [-4, -5, -6, -7, 0],
                                         [8, 9, -10, 11, 0],
                                         [12, -13, -14, 15, 0],
                                         [16, -17, 18, 19, 0]])

        cls.classes = tiny_yolo2_infer.voc2007_classes
        cls.num_classes = len(cls.classes)
        cls.batch_size = 1

        cls.data_list_path = os.path.join(cls.input_path, "images")
        cls.label_list_path = os.path.join(cls.input_path, "annotations")

        pic_num = 0
        data_list = numpy.asarray(pre_process_tool.get_data_path_list(cls.data_list_path)[pic_num:pic_num + 1])
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

        cls.ckpt_file = os.path.join(cls.current_path, "input", "tiny_yolo2.ckpt")

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

    def test_random_colors_no_n(self):
        self.assertRaises(TypeError, lambda: tiny_yolo2_infer.random_colors(N=None))

    def test_random_colors(self):
        res = tiny_yolo2_infer.random_colors(N=5)
        self.assertEqual(len(res), 5)
        self.assertEqual(len(res[0]), 3)

        res = tiny_yolo2_infer.random_colors(N=0)
        self.assertEqual(len(res), 0)

        res = tiny_yolo2_infer.random_colors(N=-1)
        self.assertEqual(len(res), 0)

    def test_non_max_suppression_no_boxes(self):
        self.assertRaises(TypeError, lambda: tiny_yolo2_infer.non_max_suppression(boxes=None))

    def test_non_max_suppression_empty_boxes(self):
        self.assertEqual(tiny_yolo2_infer.non_max_suppression(boxes=[]), [])

    def test_non_max_suppression(self):
        res = tiny_yolo2_infer.non_max_suppression(boxes=self.boxes_test)

        self.assertCountEqual(res[0], [12, -13, -14, 15])
        self.assertCountEqual(res[1], [8, 9, -10, 11])
        self.assertCountEqual(res[2], [0, 1, 2, 3])
        self.assertCountEqual(res[3], [-4, -5, -6, -7])

    def test_draw_no_classes(self):
        self.assertRaises(TypeError, lambda: tiny_yolo2_infer.draw(image_bgr=None, features=None, classes=None))

    def test_draw_no_features(self):
        self.assertRaises(TypeError, lambda: tiny_yolo2_infer.draw(image_bgr=None,
                                                                   features=None,
                                                                   classes=self.classes))

    def test_draw_no_image_bgr(self):
        self.assertRaises(TypeError, lambda: tiny_yolo2_infer.draw(image_bgr=None,
                                                                   features=[0, 1, 2, 3, 4, 5],
                                                                   classes=self.classes))

    def test_draw_success(self):
        image_test = cv2.imread(os.path.join(self.image_path, "test.jpg"))
        try:
            tiny_yolo2_infer.draw(image_bgr=image_test,
                                  features=self.feature_test,
                                  classes=self.classes,
                                  to_draw=False)

        except Exception:
            self.fail("Error while running test_draw()")

    def test_prepare_boxes_no_anchors(self):
        self.assertRaises(TypeError, lambda: tiny_yolo2_infer.prepare_boxes(feature=None,
                                                                            anchors=None,
                                                                            grid_h=None,
                                                                            grid_w=None,
                                                                            offset=None,
                                                                            classes=None,
                                                                            block_size=None))

    def test_prepare_boxes_no_classes(self):
        self.assertRaises(TypeError, lambda: tiny_yolo2_infer.prepare_boxes(feature=None,
                                                                            anchors=[1, 2, 3],
                                                                            grid_h=None,
                                                                            grid_w=None,
                                                                            offset=None,
                                                                            classes=None,
                                                                            block_size=None))

    def test_prepare_boxes_no_feature(self):
        self.assertRaises(ValueError, lambda: tiny_yolo2_infer.prepare_boxes(feature=None,
                                                                             anchors=[1, 2, 3],
                                                                             grid_h=None,
                                                                             grid_w=None,
                                                                             offset=None,
                                                                             classes=self.classes,
                                                                             block_size=None))

    def test_infer_no_infer_data(self):
        self.assertRaises(AttributeError, lambda: tiny_yolo2_infer.infer(model=None,
                                                                         infer_data=None,
                                                                         ckpt_file=None,
                                                                         classes=None,
                                                                         batch_size=None))

    def test_infer_no_batch_size(self):
        self.assertRaises(TypeError, lambda: tiny_yolo2_infer.infer(model=None,
                                                                    infer_data=self.annotation_dataset_tool,
                                                                    ckpt_file=None,
                                                                    classes=None,
                                                                    batch_size=None))

    def test_infer_no_model(self):
        self.assertRaises(AttributeError, lambda: tiny_yolo2_infer.infer(model=None,
                                                                         infer_data=self.annotation_dataset_tool,
                                                                         ckpt_file=None,
                                                                         classes=None,
                                                                         batch_size=self.batch_size))

    def test_infer_no_ckpt_file(self):
        self.assertRaises(ValueError, lambda: tiny_yolo2_infer.infer(model=self.model,
                                                                     infer_data=self.annotation_dataset_tool,
                                                                     ckpt_file=None,
                                                                     classes=None,
                                                                     batch_size=self.batch_size))

    def test_infer_no_classes(self):
        self.assertRaises(TypeError, lambda: tiny_yolo2_infer.infer(model=self.model,
                                                                    infer_data=self.annotation_dataset_tool,
                                                                    ckpt_file=self.ckpt_file,
                                                                    classes=None,
                                                                    batch_size=self.batch_size))

    def test_infer(self):

        label_path = os.path.join(self.label_list_path, "test.xml")
        origin = self.annotation_dataset_tool.get_annotation_label(full_path=label_path,
                                                                   label_file_type="voc_xml",
                                                                   resize_rate_h=1,
                                                                   resize_rate_w=1)

        output = tiny_yolo2_infer.infer(model=self.model,
                                        infer_data=self.annotation_dataset_tool,
                                        ckpt_file=self.ckpt_file,
                                        classes=self.classes,
                                        batch_size=self.batch_size,
                                        to_draw=False)

        iou_val = self._get_average_iuo_value(self._generate_boxes(origin), self._generate_boxes(output))
        self.assertGreaterEqual(iou_val, 0.6)


if __name__ == "__main__":
    unittest.main()
