import datetime
import os
import shutil
import unittest

from qumico import export
from qumico import device
from qumico import QUMICO_TEMPLATE_PATH, QUMICO_MAIN, QUMICO_VERSION, QUMICO_LIB, QUMICO_INCLUDE

import tests


class TestExport(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.current_path = os.path.dirname(os.path.realpath(__file__))
        cls.output_path = os.path.join(cls.current_path, "output")

        cls.export_type = export.ExportType()
        cls.export_c = export.Export(template_root_path=QUMICO_TEMPLATE_PATH, export_root_path=cls.output_path,
                                     export_type=cls.export_type.C)
        cls.export_npy = export.Export(template_root_path=QUMICO_TEMPLATE_PATH, export_root_path=cls.output_path,
                                       export_type=cls.export_type.NPY)

        cls.output_file_name = "test"
        cls.output_file_content = "This is test"

    def tearDown(self) -> None:
        try:
            shutil.rmtree(self.output_path)
            pass
        except FileNotFoundError:
            pass

    def test_export_type_instance(self):
        self.assertTrue(issubclass(type(self.export_type), export.ExportType))
        self.assertEqual(self.export_type.C, "C")
        self.assertEqual(self.export_type.NPY, "NPY")

    def test_export_instance_no_export_type(self):
        self.assertRaises(ValueError, lambda: export.Export(template_root_path=QUMICO_TEMPLATE_PATH,
                                                            export_root_path=self.output_path, export_type=None))

    def test_export_instance_no_template_root_path(self):
        self.assertRaises(TypeError, lambda: export.Export(template_root_path=None, export_root_path=self.output_path,
                                                           export_type=self.export_type.C))

    def test_export_instance_no_export_root_path(self):
        self.assertRaises(TypeError, lambda: export.Export(template_root_path=QUMICO_TEMPLATE_PATH,
                                                           export_root_path=None, export_type=self.export_type.C))

    def test_export_instance_type_c(self):
        self.assertTrue(self.export_c.export_qumico_path.lower().endswith(".c"))

    def test_export_instance_type_npy(self):
        self.assertTrue(self.export_npy.export_qumico_path.lower().endswith(""))

    def test_export_crete_out_path_type_c(self):
        self.export_c.crete_out_path()

        dir_list = [QUMICO_INCLUDE, QUMICO_LIB]
        self.assertTrue(tests.is_dir_contains(dirs=self.output_path, folder_list=dir_list))

    def test_export_crete_out_path_type_npy(self):
        self.export_npy.crete_out_path()

        dir_list = [QUMICO_INCLUDE, QUMICO_LIB, "initializers"]
        self.assertTrue(tests.is_dir_contains(dirs=self.output_path, folder_list=dir_list))

    def test_export_export_ops_header(self):
        self.export_npy.crete_out_path()
        self.export_npy.export_ops_header(device=device.RaspberryPi3())

        file_list = ["qumico.h", "qumico_type.h", "numpy.c", "numpy.h"]
        self.assertTrue(tests.is_dir_contains(dirs=self.output_path, file_list=file_list))

    def test_export_export_initializers_type_c(self):
        self.export_c.crete_out_path()
        self.export_c.export_initializers("This is test")
        output_file = os.path.join(self.output_path, QUMICO_LIB, "initializers.c")

        self.assertTrue(os.path.isfile(output_file) and output_file.lower().endswith(".c"))

    def test_export_export_initializers_type_npy(self):
        self.export_npy.crete_out_path()
        self.export_npy.export_initializers({self.output_file_name: self.output_file_content})

        output_file = os.path.join(self.output_path, "initializers", (self.output_file_name + ".npy"))

        self.assertTrue(os.path.isfile(output_file) and output_file.lower().endswith(".npy"))

    def test_export_export_lib_ops(self):
        self.export_c.crete_out_path()
        self.export_c.export_lib_ops(self.output_file_name, self.output_file_content)

        output_file = os.path.join(self.output_path, QUMICO_LIB, (self.output_file_name + ".c"))

        self.assertTrue(os.path.isfile(output_file) and output_file.lower().endswith(".c"))

    def test_export_export_qumico(self):
        self.export_c.crete_out_path()
        self.export_c.export_qumico(self.output_file_content)

        output_file = os.path.join(self.output_path, (QUMICO_MAIN + ".c"))

        self.assertTrue(os.path.isfile(output_file) and output_file.lower().endswith(".c"))

    def test_export_copyright_date_version(self):
        copyright_date_test = self.export_c.copyright_date_version()
        str_date = datetime.datetime.now().date()
        str_hour = datetime.datetime.now().hour
        str_minute = datetime.datetime.now().minute

        self.assertTrue("Copyright (c) 2019 Pasona Tech Inc. http://pasona.tech" in copyright_date_test)
        self.assertTrue("(the \"Software\")," in copyright_date_test)
        self.assertTrue("THE SOFTWARE IS PROVIDED \"AS IS\"," in copyright_date_test)
        self.assertTrue(f"{str_date} {str_hour}:{str_minute}")
        self.assertTrue(QUMICO_VERSION in copyright_date_test)


if __name__ == "__main__":
    unittest.main()
