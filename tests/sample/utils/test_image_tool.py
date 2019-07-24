import numpy
import os
import unittest

from pathlib import Path
from samples.utils import image_tool


from PIL import Image

class TestImageTool(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.current_path = os.path.dirname(os.path.abspath(__file__))
        cls.image_path = os.path.join(cls.current_path, "input")
        cls.images = cls._load_images()[:3]

    @classmethod
    def _load_images(cls):
        images = []
        for img in Path(cls.current_path).glob('**/*.jpg'):
            images.append(numpy.array(Image.open(str(img)).convert('RGB')))
        return numpy.array(images)

    def test_resize_image_array_set_no_image_sets(self):
        self.assertRaises(AttributeError, lambda: image_tool.resize_image_array_set(image_sets=None, w_in=0, h_in=0))

    def test_resize_image_array_set_channel_error(self):
        self.assertRaises(ValueError, lambda: image_tool.resize_image_array_set(image_sets=self.images,
                                                                                w_in=224, h_in=224,
                                                                                w_resize=224, h_resize=224,
                                                                                input_mode="RGB", channel_out=1,
                                                                                resize=False))

    def test_resize_image_array_set_resize_false_error(self):
        self.assertRaises(ValueError, lambda: image_tool.resize_image_array_set(image_sets=self.images,
                                                                                w_in=224, h_in=224,
                                                                                w_resize=200, h_resize=200,
                                                                                input_mode="RGB", channel_out=3,
                                                                                resize=False))

    def test_resize_image_array_set_without_resize(self):
        output = image_tool.resize_image_array_set(image_sets=self.images, w_in=224, h_in=224,
                                                   w_resize=224, h_resize=224,
                                                   input_mode="RGB", channel_out=3, resize=False)
        self.assertEqual(output.shape, (3, 224, 224, 3))

    def test_resize_image_array_set_with_resize(self):
        output = image_tool.resize_image_array_set(image_sets=self.images, w_in=224, h_in=224,
                                                   w_resize=200, h_resize=200,
                                                   input_mode="RGB", channel_out=3, resize=True)
        self.assertEqual(output.shape, (3, 200, 200, 3))

    def test_resize_image_array_no_image_array(self):
        self.assertRaises(AttributeError, lambda: image_tool.resize_image_array(image_array=None,
                                                                                w_in=224, h_in=224,
                                                                                w_resize=224, h_resize=224,
                                                                                input_mode="RGB", channel_out=1,
                                                                                resize=False))

    def test_resize_image_array_without_resize(self):
        output = image_tool.resize_image_array(image_array=self.images[0], w_in=224, h_in=224,
                                               w_resize=224, h_resize=224,
                                               input_mode="RGB", channel_out=3, resize=False)
        self.assertEqual(output.shape, (224, 224, 3))

    def test_resize_image_array_with_resize(self):
        output = image_tool.resize_image_array(image_array=self.images[0], w_in=224, h_in=224,
                                               w_resize=200, h_resize=200,
                                               input_mode="RGB", channel_out=3, resize=True)

        self.assertEqual(output.shape, (200, 200, 3))

    def test_image_encode_no_image_array(self):
        self.assertRaises(AttributeError, lambda: image_tool.image_encode(image_array=None))

    def test_image_encode(self):
        output = image_tool.image_encode(image_array=self.images[0])
        self.assertIs(type(output), Image.Image)
        self.assertEqual(output.width, 224)
        self.assertEqual(output.height, 224)

    def test_image_encode_1_d(self):
        output = image_tool.image_encode(image_array=numpy.reshape(self.images[0], 224 * 224 * 3), H=224, W=224)
        self.assertIs(type(output), Image.Image)
        self.assertEqual(output.width, 224)
        self.assertEqual(output.height, 224)

    def test_decode(self):
        output = image_tool.image_decode(image=self.images[0])
        self.assertEqual(output.shape, (224, 224, 3))


if __name__ == "__main__":
    unittest.main()
