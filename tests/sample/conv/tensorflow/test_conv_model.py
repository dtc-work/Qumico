import unittest

from samples.conv.tensorflow import conv_model


class TestConvModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.conv = conv_model.CONV()

    def test_conv_instance(self):
        self.assertIs(type(self.conv), conv_model.CONV)
        self.assertEqual(self.conv.input_size, 784)
        self.assertEqual(self.conv.mid_units, 100)
        self.assertEqual(self.conv.output_size, 10)
        self.assertEqual(self.conv.lr, 1e-4)
        self.assertEqual(self.conv.output_node_name, "output")


if __name__ == "__main__":
    unittest.main()
