import os
import unittest

from samples.tiny_yolo_v2_yad2k.keras import tiny_yolo_v2_yad2k_infer

import tests


class TestTinyYoloV2Yad2KInfer(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.module_path = os.path.abspath(tiny_yolo_v2_yad2k_infer.__file__)
        cls.current_path = os.path.dirname(os.path.abspath(__file__))
        cls.output_path = os.path.join(cls.current_path, "output")

        if not os.path.exists(cls.output_path):
            os.makedirs(cls.output_path)

    @classmethod
    def tearDownClass(cls) -> None:
        tests.remove_folder(cls.output_path)

    def test_infer(self):
        output = tests.read_from_output(lambda: tests.execute_file(self.module_path))
        self.assertIn("sheep", output)
        self.assertIn("person", output)
        self.assertIn("cow", output)


if __name__ == "__main__":
    unittest.main()