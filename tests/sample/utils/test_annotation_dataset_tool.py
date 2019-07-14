import json
import os
import unittest

from samples.utils import annotation_dataset_tool
from samples.utils import pre_process_tool

import numpy


class TestAnnotationDatasetTool(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:

        cls.current_path = os.path.dirname(os.path.abspath(__file__))
        cls.input_path = os.path.join(cls.current_path, "input")

        cls.image_path = os.path.join(cls.input_path, "images")
        cls.annotation_path = os.path.join(cls.input_path, "annotations")

        cls.classes = ["car", "person"]
        cls.num_classes = len(cls.classes)

        pic_num = 0

        cls.data_list = numpy.asarray(pre_process_tool.get_data_path_list(cls.image_path)[pic_num:pic_num + 1])
        cls.label_list = numpy.asarray(pre_process_tool.get_data_path_list(cls.annotation_path)[pic_num:pic_num + 1])

        cls.annotation_dataset_tool = annotation_dataset_tool.AnnotationDatasetTool(training_flag=True,
                                                                                    data_list=cls.data_list,
                                                                                    label_list=cls.label_list,
                                                                                    category_class=cls.classes,
                                                                                    one_hot_classes=cls.num_classes,
                                                                                    resize_flag=True,
                                                                                    target_h=416, target_w=416,
                                                                                    label_file_type="voc_xml",
                                                                                    format="NCHW",
                                                                                    data_rescale=True,
                                                                                    label_resclar=True)

        cls.test_annotation = os.path.join(cls.annotation_path, "test.xml")
        cls.json_annotation = os.path.join(cls.input_path, "json_annotations", "test.json")


    def test_annotation_dataset_tool_instance_no_datalist(self):
        self.assertRaises(AttributeError, lambda: annotation_dataset_tool.AnnotationDatasetTool(data_list=None))

    def test_annotation_dataset_tool_instance(self):
        self.assertIs(type(self.annotation_dataset_tool), annotation_dataset_tool.AnnotationDatasetTool)
        self.assertEqual(self.annotation_dataset_tool.training, True)
        self.assertEqual(self.annotation_dataset_tool.data_list,
                         numpy.asarray(os.path.join(self.image_path, "test.jpg")))
        self.assertEqual(self.annotation_dataset_tool.label_list,
                         numpy.asarray(os.path.join(self.annotation_path, "test.xml")))
        self.assertEqual(self.annotation_dataset_tool.category_class, ["car", "person"])
        self.assertEqual(self.annotation_dataset_tool.one_hot_classes, 2)
        self.assertEqual(self.annotation_dataset_tool.resize_flag, True)
        self.assertEqual(self.annotation_dataset_tool.target_h, 416)
        self.assertEqual(self.annotation_dataset_tool.target_w, 416)
        self.assertEqual(self.annotation_dataset_tool.label_file_type, "voc_xml")
        self.assertEqual(self.annotation_dataset_tool.format, "NCHW")
        self.assertEqual(self.annotation_dataset_tool.data_rescale, True)
        self.assertEqual(self.annotation_dataset_tool.label_rescale, True)

    def test_annotation_dataset_tool_next_batch_no_batch_size(self):
        self.assertRaises(TypeError, lambda: self.annotation_dataset_tool.next_batch(batch_size=None))

    def test_annotation_dataset_tool_next_batch_sample_greater_than_population(self):
        self.assertRaises(ValueError, lambda: self.annotation_dataset_tool.next_batch(batch_size=3))

    def test_annotation_dataset_tool_next_batch(self):
        x_batch, y_batch, x_path_batch, y_path_batch = self.annotation_dataset_tool.next_batch(batch_size=1)
        self.assertEqual(x_batch[0].shape, (3, 416, 416))
        self.assertEqual(len(y_batch[0]), 8)
        self.assertEqual(x_path_batch[0], os.path.join(self.image_path, "test.jpg"))
        self.assertEqual(y_path_batch[0], os.path.join(self.annotation_path, "test.xml"))

    def test_next_batch_once(self):
        x_batch, y_batch, x_path_batch, y_path_batch = self.annotation_dataset_tool.next_batch_once(batch_size=1)
        self.assertEqual(x_batch[0].shape, (3, 416, 416))
        self.assertEqual(len(y_batch[0]), 8)
        self.assertEqual(x_path_batch[0], os.path.join(self.image_path, "test.jpg"))
        self.assertEqual(y_path_batch[0], os.path.join(self.annotation_path, "test.xml"))

    def test_index_reset(self):
        self.annotation_dataset_tool.index_reset()
        self.assertEqual(self.annotation_dataset_tool.index_list, [0])

    def test_get_train_batch_no_data_path_list(self):
        self.assertRaises(TypeError, lambda: self.annotation_dataset_tool.get_train_batch(data_path_list=None,
                                                                                          label_path_list=None))

    def test_get_train_batch_no_label_path_list(self):
        self.assertRaises(TypeError, lambda: self.annotation_dataset_tool.get_train_batch(data_path_list=self.data_list,
                                                                                          label_path_list=None))

    def test_get_train_batch(self):
        x_train, y_train = self.annotation_dataset_tool.get_train_batch(data_path_list=self.data_list,
                                                                        label_path_list=self.label_list,
                                                                        one_hot_classes=self.num_classes)

        self.assertEqual(x_train[0].shape, (3, 416, 416))
        self.assertEqual(len(y_train[0]), 8)

    def test_get_infer_batch_no_data_path_list(self):
        self.assertRaises(TypeError, lambda: self.annotation_dataset_tool.get_infer_batch(data_path_list=None))

    def test_get_infer_batch(self):
        x_infer = self.annotation_dataset_tool.get_infer_batch(data_path_list=self.data_list)
        self.assertEqual(x_infer[0].shape, (3, 416, 416))

    def test_image_generator_default_resize_size(self):
        test_image_path = os.path.join(self.image_path, "test.jpg")
        img_array, resize_h, resize_w = self.annotation_dataset_tool.image_generator(full_path=test_image_path)
        self.assertEqual(img_array.shape, (224, 224, 3))
        self.assertAlmostEqual(resize_h, 224/3024)
        self.assertAlmostEqual(resize_w, 224/4032)

    def test_image_generator_custom_resize_size(self):
        test_image_path = os.path.join(self.image_path, "test.jpg")
        img_array, resize_h, resize_w = self.annotation_dataset_tool.image_generator(full_path=test_image_path,
                                                                                     target_w=403, target_h=302)
        self.assertEqual(img_array.shape, (302, 403, 3))
        self.assertAlmostEqual(resize_h, 0.1, places=2)
        self.assertAlmostEqual(resize_w, 0.1, places=2)

    def test_get_annotation_label_type_voc_xml_no_full_path(self):
        self.assertRaises(TypeError,
                          lambda: self.annotation_dataset_tool.get_annotation_label(full_path=None,
                                                                                    label_file_type="voc_xml"))

    def test_get_annotation_label_type_voc_xml_no_resize_rate_w(self):
        self.assertRaises(TypeError,
                          lambda: self.annotation_dataset_tool.get_annotation_label(full_path=self.test_annotation,
                                                                                    label_file_type="voc_xml",
                                                                                    one_hot_classes=None,
                                                                                    resize_rate_w=None,
                                                                                    resize_rate_h=None))

    def test_get_annotation_label_type_voc_xml_no_resize_rate_h(self):
        self.assertRaises(TypeError,
                          lambda: self.annotation_dataset_tool.get_annotation_label(full_path=self.test_annotation,
                                                                                    label_file_type="voc_xml",
                                                                                    one_hot_classes=None,
                                                                                    resize_rate_w=0.1,
                                                                                    resize_rate_h=None))

    def test_get_annotation_label_type_voc_xml(self):
        class_list = self.annotation_dataset_tool.get_annotation_label(full_path=self.test_annotation,
                                                                       label_file_type="voc_xml",
                                                                       one_hot_classes=self.num_classes,
                                                                       resize_rate_w=0.1,
                                                                       resize_rate_h=0.1)
        self.assertEqual(len(class_list), 8)

    def test_get_bbox_xml_ext_no_xml_file(self):
        self.assertRaises(TypeError, lambda: self.annotation_dataset_tool.get_bbox_xml_ext(xml_file=None,
                                                                                           resize_rate_h=None,
                                                                                           resize_rate_w=None))

    def test_get_bbox_xml_ext_no_resize_rate_w(self):
        self.assertRaises(TypeError,
                          lambda: self.annotation_dataset_tool.get_bbox_xml_ext(xml_file=self.test_annotation,
                                                                                resize_rate_h=None,
                                                                                resize_rate_w=None))

    def test_get_bbox_xml_ext_no_resize_rate_h(self):
        self.assertRaises(TypeError,
                          lambda: self.annotation_dataset_tool.get_bbox_xml_ext(xml_file=self.test_annotation,
                                                                                resize_rate_h=None,
                                                                                resize_rate_w=0.1))

    def test_get_bbox_xml_ext(self):
        train_labels = self.annotation_dataset_tool.get_bbox_xml_ext(xml_file=self.test_annotation,
                                                                     resize_rate_h=0.1,
                                                                     resize_rate_w=0.1)

        self.assertEqual(len(train_labels), 8)

    def test_json_reader_invalid_json_file_path(self):
        self.assertRaises(FileNotFoundError, lambda: self.annotation_dataset_tool.json_reader(json_file_path=""))

    def test_json_reader(self):
        json_data = self.annotation_dataset_tool.json_reader(json_file_path=self.json_annotation)
        self.assertEqual(json_data["annotation"]["folder"], "ダウンロード")
        self.assertEqual(json_data["annotation"]["filename"], "test.jpg")
        self.assertEqual(json_data["annotation"]["path"], "/home/guest/ダウンロード/test.jpg")
        self.assertEqual(json_data["annotation"]["size"]["width"], "4032")
        self.assertEqual(json_data["annotation"]["size"]["height"], "3024")
        self.assertEqual(json_data["annotation"]["size"]["depth"], "3")


if __name__ == "__main__":
    unittest.main()
