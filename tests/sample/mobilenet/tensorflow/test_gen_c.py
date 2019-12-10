import os
import unittest

from samples.mobilenet.tensorflow import gen_c
import tests


class TestGenC(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.gen_c_path = os.path.abspath(gen_c.__file__)
        cls.output_path = os.path.join(os.path.dirname(cls.gen_c_path), "out_c")
        cls.is_exists = os.path.exists(cls.output_path)

        tests.remove_folder(cls.output_path)

    @classmethod
    def tearDownClass(cls) -> None:
        if not cls.is_exists:
            tests.remove_folder(cls.output_path)

    def test_gen_c(self):
        res = tests.read_from_output(lambda: tests.execute_file(self.gen_c_path))

        file_list = ["numpy.c", "qumico.c", "qumico.so"]
        folder_list = ["include", "initializers", "lib"]

        self.assertIn("Cソースを生成しました。出力先:", res)
        self.assertTrue(tests.is_dir_contains(self.output_path, file_list=file_list, folder_list=folder_list))


if __name__ == "__main__":
    unittest.main()
