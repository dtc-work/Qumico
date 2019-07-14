import unittest

from qumico import device


class TestDevice(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.qumico_device = device.QumicoDevice(device="CPU")
        cls.qumico_device_type = device.QumicoDeviceType()

        cls.raspberry_device = device.RaspberryPi3()
        cls.raspberry_neon = device.RaspberryPi3(neon=True)
        cls.raspberry_openmp = device.RaspberryPi3(openmp=True)

    def test_device_qumico_device_type_instance(self):
        self.assertIs(type(self.qumico_device_type), device.QumicoDeviceType)
        self.assertEqual(self.qumico_device_type.OpenCL, 935)
        self.assertEqual(self.qumico_device_type.OpenMP, 936)
        self.assertEqual(self.qumico_device_type.ARMNeon, 937)

    def test_device_qumico_device_instance_invalid_device(self):
        self.assertRaises(AttributeError, lambda: device.QumicoDevice(device="Test"))

    def test_device_qumico_device_instance(self):
        self.assertIs(type(self.qumico_device), device.QumicoDevice)
        self.assertIs(type(self.qumico_device.options), list)
        self.assertEqual(len(self.qumico_device.options), 0)

    def test_device_qumico_device_is_valid_device_type(self):
        self.assertRaises(NotImplementedError, lambda: self.qumico_device.is_valid_device_type())

    def test_device_qumico_device_get_compile_option(self):
        self.assertRaises(NotImplementedError, lambda: self.qumico_device.get_compile_option())

    def test_device_raspberry_pi_instance(self):

        self.assertIs(type(self.raspberry_device), device.RaspberryPi3)
        self.assertTrue(device.QumicoDeviceType.ARMNeon in self.raspberry_neon.options)
        self.assertTrue(device.QumicoDeviceType.OpenMP in self.raspberry_openmp.options)

    def test_device_raspberry_pi_get_compile_option(self):
        result = self.raspberry_device.get_compile_option()
        self.assertIn("-march=armv7-a", result)

        result = self.raspberry_neon.get_compile_option()
        param_list = ["-mfpu=neon", "-ftree-vectorize"]
        for item in param_list:
            self.assertIn(item, result)

        result = self.raspberry_openmp.get_compile_option()
        self.assertIn("-fopenmp", result)


if __name__ == "__main__":
    unittest.main()
