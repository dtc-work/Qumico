import shutil
import os
import unittest


from samples.mobilenet.tensorflow import mobilenet_infer_c
import tests


class TestMobilenetInferC(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.current_path = os.path.abspath(os.path.dirname(__file__))
        cls.input_path = os.path.join(cls.current_path, "input")
        cls.output_path = os.path.join(cls.current_path, "output")

        cls.image_input = os.path.join(cls.input_path, "images", "test.jpeg")

        cls.c_path = os.path.join(cls.current_path, "out_c", "qumico.so")

        cls.c_dir_old = os.path.join(cls.input_path, "out_c")
        cls.c_dir_new = os.path.join(cls.current_path, "out_c")

        cls._prepare()

    @classmethod
    def tearDownClass(cls) -> None:
        tests.remove_folder(cls.c_dir_new)

    @classmethod
    def _prepare(cls):
        shutil.copytree(cls.c_dir_old, cls.c_dir_new)

    def test_infer_c(self):
        res = tests.read_from_output(lambda: mobilenet_infer_c.infer(image_path=self.image_input,
                                                                     so_lib_path=self.c_path))

        self.assertIn("tiger", res)


if __name__ == "__main__":
    unittest.main()
