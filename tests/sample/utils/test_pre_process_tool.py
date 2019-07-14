import os
import shutil
import unittest

from samples.utils import pre_process_tool


class TestPreProcessTool(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.dir_test_str = "ABC"
        cls.current_path = os.path.dirname(os.path.realpath(__file__))
        cls.input_path = os.path.join(cls.current_path, "dir_test")

    def setUp(self) -> None:
        os.mkdir(os.path.join(self.input_path))
        self._make_dir_rec(self.input_path, 0)

    def tearDown(self) -> None:
        try:
            shutil.rmtree(self.input_path)
        except FileNotFoundError:
            pass

    @classmethod
    def _make_dir_rec(cls, curdir, level):
        if level >= 1:
            for letter in cls.dir_test_str:
                cls._make_file(curdir, letter + str(level))
            return

        for letter in cls.dir_test_str:
            cls._make_dir(os.path.join(curdir, f"{letter}"), level)
            cls._make_dir_rec(os.path.join(curdir, f"{letter}{level}"), level + 1)

    @classmethod
    def _make_dir(cls, loc, x):
        os.mkdir(loc + str(x))

    @classmethod
    def _make_file(cls, loc, x):
        with open(os.path.join(loc, f"{x}.txt"), "w+") as file:
            file.write(str(x))

    def test_get_data_path_list_no_data_root_path(self):
        self.assertRaises(TypeError, lambda: pre_process_tool.get_data_path_list(None))

    def test_get_data_path_list_depth_1(self):
        depth = 1
        result = pre_process_tool.get_data_path_list(self.input_path, depth=depth)
        for c in self.dir_test_str:
            for i in range(depth):
                self.assertTrue([s for s in result if f"{c}{i}" in s], msg=f"Directory {c}{i} not found")

    def test_get_data_path_list_depth_2(self):
        depth = 2
        result = pre_process_tool.get_data_path_list(self.input_path, depth=depth)
        for c in self.dir_test_str:
            for i in range(depth):
                self.assertTrue([s for s in result if f"{c}{i}" in s], msg=f"Directory {c}{i} not found")

        for c in self.dir_test_str:
            for i in range(depth-1, depth):
                self.assertTrue([s for s in result if f"{c}{i}.txt" in s], msg=f"File {c}{i}.txt not found")

    def test_get_data_path_list_depth_other_depth(self):
        self.assertFalse(pre_process_tool.get_data_path_list(self.input_path, depth=0))
        self.assertFalse(pre_process_tool.get_data_path_list(self.input_path, depth=3))
        self.assertFalse(pre_process_tool.get_data_path_list(self.input_path, depth=-1))


if __name__ == "__main__":
    unittest.main()