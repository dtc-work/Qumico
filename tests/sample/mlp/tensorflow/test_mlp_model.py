import unittest

from samples.mlp.tensorflow import mlp_model


class TestMLPModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.mlp = mlp_model.MLP()

    def test_mlp_instance(self):
        self.assertIs(type(self.mlp), mlp_model.MLP)
        self.assertEqual(self.mlp.input_size, 784)
        self.assertEqual(self.mlp.mid_units, 100)
        self.assertEqual(self.mlp.output_size, 10)
        self.assertEqual(self.mlp.lr, 0.1)
        self.assertEqual(self.mlp.output_op_name, "output")


if __name__ == "__main__":
    unittest.main()
