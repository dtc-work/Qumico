import os
import unittest

from samples.vgg16.keras import vgg16_infer

import tests


class TestVGG16Infer(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.vgg16_infer_path = os.path.abspath(vgg16_infer.__file__)
        cls.current_path = os.path.abspath(__file__)
        cls.output_path = os.path.join(os.path.dirname(cls.current_path), "onnx")
        cls.input_path = os.path.join(os.path.dirname(cls.current_path), "input")

        cls.vgg16_infer_test = vgg16_infer
        cls.classes_test = list(map(str, list(range(10))))

    def test_vgg16_infer(self):
        vgg16_infer.classes = self.classes_test
        count_correct = 0
        for x in vgg16_infer.classes:
            vgg16_infer.img_file = os.path.join(self.input_path, (x + ".jpg"))
            output = tests.read_from_output(lambda: vgg16_infer.main())
            if x in output:
                count_correct += 1

        accuracy = count_correct / len(self.classes_test)
        self.assertGreaterEqual(accuracy, 0.7)


if __name__ == "__main__":
    unittest.main()

