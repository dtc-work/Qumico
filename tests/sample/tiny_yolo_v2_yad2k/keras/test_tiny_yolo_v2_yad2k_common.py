import os
import numpy
import unittest

from samples.tiny_yolo_v2_yad2k.keras import tiny_yolo_v2_yad2k_common


class TinyYoloV2Yad2KCommon(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.module_instance = tiny_yolo_v2_yad2k_common
        pass

    def test_tiny_yolo2k_v2_yad_2k_common_variables(self):
        self.assertEqual(self.module_instance, tiny_yolo_v2_yad2k_common)
        self.assertEqual(self.module_instance.width, 416)
        self.assertEqual(self.module_instance.height, 416)
        self.assertEqual(self.module_instance.r_w, 13)
        self.assertEqual(self.module_instance.r_h, 13)
        self.assertEqual(self.module_instance.r_n, 5)
        self.assertEqual(self.module_instance.classes, 20)
        self.assertEqual(self.module_instance.thresh, 0.3)
        self.assertEqual(self.module_instance.iou_threshold, 0.5)

        self.assertEqual(self.module_instance.region_biases, (1.080000, 1.190000, 3.420000, 4.410000, 6.630000,
                                                              11.380000, 9.420000, 5.110000, 16.620001, 10.520000))

        self.assertTrue(numpy.array_equal(self.module_instance.voc_anchors, numpy.array([[1.08, 1.19], [3.42, 4.41],
                                                                                         [6.63, 11.38], [9.42, 5.11],
                                                                                         [16.62, 10.52]])))

        self.assertEqual(self.module_instance.voc_label, ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                                                          'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                                                          'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                                                          'train', 'tvmonitor'])

    def test_sigmoid_no_x(self):
        self.assertRaises(TypeError, lambda: tiny_yolo_v2_yad2k_common.sigmoid(x=None))

    def test_sigmoid(self):
        self.assertAlmostEqual(tiny_yolo_v2_yad2k_common.sigmoid(1), 0.7310585786300049)
        self.assertAlmostEqual(tiny_yolo_v2_yad2k_common.sigmoid(0), 0.5)
        self.assertAlmostEqual(tiny_yolo_v2_yad2k_common.sigmoid(-1), 0.2689414213699951)

        self.assertAlmostEqual(tiny_yolo_v2_yad2k_common.sigmoid(0.5), 0.6224593312018546)
        self.assertAlmostEqual(tiny_yolo_v2_yad2k_common.sigmoid(-0.5), 0.3775406687981454)

    def test_softmax_no_x(self):
        self.assertRaises(AttributeError, lambda: tiny_yolo_v2_yad2k_common.softmax(x=None))

    def test_softmax(self):
        self.assertEqual(tiny_yolo_v2_yad2k_common.softmax(x=numpy.random.rand(1, 13, 13, 5, 20)).shape,
                         (1, 13, 13, 5, 20))


if __name__ == "__main__":
    unittest.main()