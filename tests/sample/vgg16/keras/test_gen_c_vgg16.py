import os
import unittest

from qumico import QUMICO_LIB, QUMICO_INCLUDE
from samples.vgg16.keras import gen_c

import tests


@unittest.skipIf(tests.IS_CIRCLE_CI, "Skip this test because large memory")
class TestGenCVGG16(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.gen_c_path = os.path.abspath(gen_c.__file__)
        cls.output_path = os.path.join(os.path.dirname(cls.gen_c_path), "out_c")

    @classmethod
    def tearDownClass(cls) -> None:
        tests.remove_folder(cls.output_path)
        pass

    def test_gen_C(self):
        output = tests.read_from_output(lambda: tests.execute_file(self.gen_c_path))

        file_list = ["qumico.c", "qumico.so", "qumico.h", "qumico_type.h"]
        folder_list = [QUMICO_LIB, QUMICO_INCLUDE]

        self.assertTrue(tests.is_dir_contains(dirs=self.output_path, folder_list=folder_list, file_list=file_list))
        self.assertIn("Cソースを生成しました。出力先:", output)


if __name__ == "__main__":
    unittest.main()