import unittest
import numpy

from samples.utils import dataset_tool

import tests


class TestDatasetTool(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = numpy.array([1, 2, 3])
        cls.label = numpy.array([1, 2, 3])
        cls.dataset_tool = dataset_tool.DatasetTool(training_flag=True, data=cls.dataset, label=cls.label)

    def test_dataset_tool_instance_training_flag_false(self):
        res = dataset_tool.DatasetTool(training_flag=False, data=self.dataset)
        self.assertIs(type(res), dataset_tool.DatasetTool)

    def test_dataset_tool_instance_training_flag_true_no_data(self):
        self.assertRaises(AttributeError, lambda: dataset_tool.DatasetTool(training_flag=True, data=None))

    def test_dataset_tool_instance_training_flag_true_no_label(self):
        self.assertRaises(AttributeError, lambda: dataset_tool.DatasetTool(training_flag=True, data=self.dataset,
                                                                           label=None))

    def test_dataset_tool_instance_training_flag_true_data_label_different(self):
        result = tests.read_from_output(lambda: dataset_tool.DatasetTool(training_flag=True,
                                                                         data=self.dataset,
                                                                         label=numpy.array([1, 2, 3, 5, 6])))

        self.assertIn("学習データサイズが異なる, 入力データサイズ = 3 ラベルサイズ = 5", result)

    def test_next_batch_no_batch_size(self):
        self.assertRaises(TypeError, lambda: self.dataset_tool.next_batch(batch_size=None))

    def test_next_batch(self):
        a, b = self.dataset_tool.next_batch(batch_size=1)
        self.assertIs(type(a), numpy.ndarray)
        self.assertIs(type(b), numpy.ndarray)

    def test_next_batch_once(self):
        a, b = self.dataset_tool.next_batch_once(batch_size=1)
        self.assertIs(type(a), numpy.ndarray)
        self.assertIs(type(b), numpy.ndarray)

    def test_dataset_tool_index_reset(self):
        self.dataset_tool.index_reset()
        self.assertCountEqual(self.dataset_tool.index_list, list(range(len(self.dataset))))


if __name__ == "__main__":
    unittest.main()