import unittest
import numpy

from samples.utils import common_tool


class TestCommonTool(unittest.TestCase):

    @classmethod
    def _softmax(cls, x):
        e_x = numpy.exp(x - numpy.max(x))
        return e_x / e_x.sum()

    def test_sigmoid_type_error(self):
        self.assertRaises(TypeError, lambda: common_tool.sigmoid({}))
        self.assertRaises(TypeError, lambda: common_tool.sigmoid(()))
        self.assertRaises(TypeError, lambda: common_tool.sigmoid([]))
        self.assertRaises(TypeError, lambda: common_tool.sigmoid(None))
        self.assertRaises(TypeError, lambda: common_tool.sigmoid(""))

    def test_sigmoid(self):
        test_number = [-30, -24, -10, -5, 0, 1, 7, 33, 40, 51, 75]
        for x in test_number:
            self.assertEqual(1 / (1 + numpy.exp(-x)), common_tool.sigmoid(x))

    def test_softmax_error(self):
        self.assertRaises(ValueError, lambda: common_tool.softmax([]))
        self.assertRaises(ValueError, lambda: common_tool.softmax(()))

        self.assertRaises(TypeError, lambda: common_tool.softmax({}))
        self.assertRaises(TypeError, lambda: common_tool.softmax(None))
        self.assertRaises(TypeError, lambda: common_tool.softmax(""))

    def test_softmax(self):
        self.assertCountEqual(common_tool.softmax([1, 2, 3, 4, 5]), self._softmax([1, 2, 3, 4, 5]))
        self.assertCountEqual(common_tool.softmax([0, 0, 0, 0, 0]), self._softmax([0, 0, 0, 0, 0]))
        self.assertCountEqual(common_tool.softmax([-1, -2, -3, -4, -5]), self._softmax([-1, -2, -3, -4, -5]))

    def test_one_hot_encoding_error(self):
        self.assertRaises(IndexError, lambda: common_tool.onehot_encoding(-6, 5))
        self.assertRaises(IndexError, lambda: common_tool.onehot_encoding(2, 2))

        self.assertRaises(ValueError, lambda: common_tool.onehot_encoding(0, -1))

        self.assertRaises(TypeError, lambda: common_tool.onehot_encoding((), []))
        self.assertRaises(TypeError, lambda: common_tool.onehot_encoding("", False))
        self.assertRaises(TypeError, lambda: common_tool.onehot_encoding({}, None))

    def test_one_hot_encoding(self):
        self.assertCountEqual(common_tool.onehot_encoding(0, 5), [1, 0, 0, 0, 0])
        self.assertCountEqual(common_tool.onehot_encoding(-1, 5), [0, 0, 0, 0, 1])
        self.assertCountEqual(common_tool.onehot_encoding(2, 5), [0, 0, 1, 0, 0])

    def test_one_hot_decoding_error(self):
        self.assertRaises(ValueError, lambda: common_tool.onehot_decoding([]))
        self.assertRaises(ValueError, lambda: common_tool.onehot_decoding(()))

    def test_one_hot_decoding(self):
        self.assertEqual(common_tool.onehot_decoding([1, 0, 0, 0, 0]), 0)
        self.assertEqual(common_tool.onehot_decoding([0, 1, 0, 0, 0]), 1)
        self.assertEqual(common_tool.onehot_decoding([0, 0, 0, 0, 1]), 4)

        self.assertEqual(common_tool.onehot_decoding([0, 0, 1, 0, 1]), 2)
        self.assertEqual(common_tool.onehot_decoding([0, 1, 0, 1, 1]), 1)


if __name__ == "__main__":
    unittest.main()