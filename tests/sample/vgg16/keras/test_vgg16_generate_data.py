import os
import shutil
import unittest

from pathlib import Path
from unittest.mock import patch

from samples.vgg16.keras import vgg16_generate_data

import tests


class TestVGG16GenerateData(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.current_path = os.path.dirname(os.path.abspath(__file__))
        cls.input_path = os.path.join(cls.current_path, "input")
        cls.output_path = os.path.join(cls.current_path, "output")

        cls.data_generator = vgg16_generate_data
        cls.data_generator.base_len = 2
        cls.data_generator.input_path = cls.input_path
        cls.data_generator.train_data_path = cls.output_path
        cls.data_generator.output_path = os.path.join(cls.output_path, "output")
        cls.data_generator.all_classes = ["1", "2", "3", "4", "5"]
        cls.data_generator.extract_classes = ["1", "3", "4"]

        cls.tar_file = os.path.join(cls.input_path, "test.tgz")
        cls.jpg_path = os.path.join(cls.current_path, "jpg")

    def tearDown(self) -> None:
        tests.remove_folder(self.jpg_path)
        tests.remove_folder(self.output_path)

    def test_download_file_no_url(self):
        self.assertRaises(AttributeError, lambda: self.data_generator.download_file(url=None))

    @patch("test_vgg16_generate_data.vgg16_generate_data.download_file", return_value="downloaded.tar.gz")
    def test_download_file(self, _):
        self.assertEqual(self.data_generator.download_file("http://www.com"), "downloaded.tar.gz")

    def test_extract_file_no_path(self):
        self.assertRaises(ValueError, lambda: self.data_generator.extract_file(path=None))

    def test_extract_file(self):
        self.data_generator.extract_file(path=self.tar_file)
        self.assertTrue(os.path.exists(self.jpg_path))

    def test_classifying_data(self):
        self.data_generator.classifying_data()
        self.assertTrue(tests.is_dir_contains(self.data_generator.output_path, folder_list=["1", "3", "4"]))
        self.assertTrue(tests.is_dir_contains(os.path.join(self.data_generator.output_path, "1"),
                                              file_list=["0.jpg", "1.jpg"]))
        self.assertTrue(tests.is_dir_contains(os.path.join(self.data_generator.output_path, "3"),
                                              file_list=["4.jpg", "5.jpg"]))
        self.assertTrue(tests.is_dir_contains(os.path.join(self.data_generator.output_path, "4"),
                                              file_list=["6.jpg", "7.jpg"]))

    def test_workdir_cleanup_no_downloaded_file(self):
        file_delete_path = os.path.join(self.current_path, "Temp")
        self.data_generator.input_path = file_delete_path
        self.assertRaises(TypeError, lambda: self.data_generator.workdir_cleanup(downloaded_file=None))

    def test_workdir_cleanup(self):
        file_delete_path = os.path.join(self.current_path, "Temp")
        file_path = os.path.join(self.current_path, "Temp.tgz")

        if os.path.exists(file_delete_path):
            os.mkdir(file_delete_path)

        Path(file_path).touch()

        self.data_generator.input_path = file_delete_path

        try:
            self.data_generator.workdir_cleanup(downloaded_file=file_path)
        except Exception as e:
            self.fail(e)


if __name__ == "__main__":
    unittest.main()
