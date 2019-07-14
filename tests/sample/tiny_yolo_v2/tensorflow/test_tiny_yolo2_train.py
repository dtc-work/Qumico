import os
import numpy
import unittest

from pathlib import Path

from samples.tiny_yolo_v2.tensorflow import tiny_yolo2_train
from samples.tiny_yolo_v2.tensorflow import tiny_yolo2_model
from samples.tiny_yolo_v2.tensorflow import tiny_yolo2_infer
from samples.utils import annotation_dataset_tool
from samples.utils import pre_process_tool

import tests


class TestTinyYolo2Train(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.current_path = os.path.dirname(os.path.realpath(__file__))
        cls.input_path = os.path.join(cls.current_path, "input")
        cls.model_path = os.path.join(cls.current_path, "model")

        cls.data_list_path = os.path.join(cls.input_path, "images")
        cls.label_list_path = os.path.join(cls.input_path, "annotations")

        cls.batch_size = 1
        cls.classes = tiny_yolo2_infer.voc2007_classes
        cls.data_list = numpy.asarray(pre_process_tool.get_data_path_list(cls.data_list_path)[:])
        cls.label_list = numpy.asarray(pre_process_tool.get_data_path_list(cls.label_list_path)[:])
        cls.transformer = None

        cls.dataset = annotation_dataset_tool.AnnotationDatasetTool(training_flag=True,
                                                                    data_list=cls.data_list,
                                                                    label_list=cls.label_list,
                                                                    category_class=cls.classes,
                                                                    one_hot_classes=len(cls.classes),
                                                                    resize_flag=True, target_h=416, target_w=416,
                                                                    label_file_type="voc_xml", format="NCHW",
                                                                    data_rescale=True, label_resclar=True,
                                                                    transformer=cls.transformer)

        cls.model = tiny_yolo2_model.TINY_YOLO_v2(height=416, width=416, output_op_name="output",
                                                  num_classes=len(cls.classes), is_train=True,
                                                  batch_size=cls.batch_size)

        cls.epochs = 999
        cls._prepare()

    @classmethod
    def tearDownClass(cls) -> None:
        tests.remove_folder(cls.model_path)

    @classmethod
    def _prepare(cls):
        if not os.path.exists(cls.model_path):
            os.mkdir(cls.model_path)

        Path(os.path.join(cls.model_path, "tiny_yolo2.ckpt")).touch()

    def test_retrain_no_train_data(self):
        self.assertRaises(AttributeError, lambda: tiny_yolo2_train.retrain(model=None,
                                                                           train_data=None,
                                                                           ckpt_file=None,
                                                                           epochs=None,
                                                                           batch_size=None))

    def test_retrain_no_batch_size(self):
        self.assertRaises(TypeError, lambda: tiny_yolo2_train.retrain(model=None, train_data=self.dataset,
                                                                      ckpt_file=None, epochs=None, batch_size=None))

    def test_retrain_no_model(self):
        self.assertRaises(AttributeError, lambda: tiny_yolo2_train.retrain(model=None,
                                                                           train_data=self.dataset,
                                                                           ckpt_file=None,
                                                                           epochs=None,
                                                                           batch_size=1))

    def test_retrain_no_ckpt_file(self):
        self.assertRaises(ValueError, lambda: tiny_yolo2_train.retrain(model=self.model,
                                                                       train_data=self.dataset,
                                                                       ckpt_file=None,
                                                                       epochs=None,
                                                                       batch_size=self.batch_size))

    def test_train_no_train_data(self):
        self.assertRaises(AttributeError, lambda: tiny_yolo2_train.train(model=None,
                                                                         train_data=None,
                                                                         epochs=None,
                                                                         batch_size=None))

    def test_train_no_batch_size(self):
        self.assertRaises(TypeError, lambda: tiny_yolo2_train.train(model=None, train_data=self.dataset,
                                                                    epochs=None, batch_size=None))

    def test_train_no_model(self):
        self.assertRaises(AttributeError, lambda: tiny_yolo2_train.train(model=None,
                                                                         train_data=self.dataset,
                                                                         epochs=None,
                                                                         batch_size=1))

    def test_train_no_epoch(self):
        self.assertRaises(TypeError, lambda: tiny_yolo2_train.train(model=self.model,
                                                                    train_data=self.dataset,
                                                                    epochs=None,
                                                                    batch_size=self.batch_size))

    @unittest.skipIf(tests.IS_CIRCLE_CI, "Skip this test because large memory")
    def test_train(self):
        ckpt, pb = tiny_yolo2_train.train(model=self.model, train_data=self.dataset,
                                          epochs=self.epochs, batch_size=self.batch_size)

        file_list = ["tiny_yolo2.ckpt", "tiny_yolo2.pb"]
        self.assertIn(file_list[0], ckpt)
        self.assertIn(file_list[1], pb)
        self.assertTrue(tests.is_dir_contains(dirs=self.model_path, file_list=file_list))


if __name__ == "__main__":
    unittest.main()
