from os import path,  mkdir 
import datetime
import shutil
import os
import re

import numpy as np

from qumico import QUMICO_MAIN
from qumico import QUMICO_VERSION
from qumico.device import QumicoDeviceType

class ExportType:
    C = "C"
    NPY = "NPY"


class Export:
    def __init__(self, template_root_path, export_root_path, export_type=ExportType.C):
        # template
        self._temlate_root_path = template_root_path
        self._template_ops_path = path.join(template_root_path,"ops")
        # export
        self._export_root_path = export_root_path
        self._export_include_path =path.join(export_root_path, "include")
        self._export_lib_path = path.join(export_root_path, "lib")
        self._export_qumico_path = path.join(self._export_root_path, QUMICO_MAIN + ".c")

        self._export_type =  export_type
        if self._export_type == ExportType.C:
            self._export_initializers_path = path.join(self._export_lib_path, "initializers.c") # file path
        elif self._export_type == ExportType.NPY:
            self._export_initializers_path =  path.join(self._export_root_path, "initializers") # folder path
        else:
            raise ValueError()


    @property
    def export_qumico_path(self):
        return self._export_qumico_path

    @export_qumico_path.setter
    def export_qumico_path(self, p):
        self._export_qumico_path = p
    
    def crete_out_path(self, cleanup=True):
    
        if cleanup and path.exists(self._export_root_path):
            shutil.rmtree(self._export_root_path)

        if not path.exists(self._export_root_path):
            mkdir(self._export_root_path)
    
        if not path.exists(self._export_lib_path):
            mkdir(self._export_lib_path)
    
        if not path.exists(self._export_include_path):
            mkdir(self._export_include_path)

        if (self._export_type==ExportType.NPY and
            not path.exists(self._export_initializers_path)):
            mkdir(self._export_initializers_path)

    # include/qumico.h, include/qumico_type.h
    def _cp_file(self, fname,_from=None, _to=None):

        if _from is None:
            _from =path.join(self._temlate_root_path,fname)
        else:
            _from =path.join(_from,fname)

        if _to is None:
            _to = path.join(self._export_include_path, fname)
        else:
            _to = path.join(_to, fname)

        shutil.copyfile(_from, _to)

        with open(_to, 'r', encoding='utf-8') as original_file:
            with open('tmp.txt', 'w', encoding='utf-8') as temp_file:
                temp_file.write(self.copyright_date_version())
                temp_file.write(original_file.read())

        os.remove(_to)
        os.rename('tmp.txt', _to)

    # include/ops.h
    def export_ops_header(self, device=None):
        self._cp_file("qumico.h")
        self._cp_file("qumico_type.h")

        if self._export_type == ExportType.NPY:
            self._cp_file("numpy.h")
            self._cp_file("numpy.c", _to=self._export_root_path)


    # lib/initializers.c
    def export_initializers(self, content): # content is OrderedDict

        if self._export_type == ExportType.C:
            with open(self._export_initializers_path, "w", encoding='utf-8') as fout:
                fout.write(self.copyright_date_version())
                fout.write(content)

        elif self._export_type == ExportType.NPY:            
            for k, v in content.items():
                export_path = path.join(self._export_initializers_path, k + ".npy")
                np.save(export_path, v, allow_pickle=False)

    # lib/Constant.c, GEMM.c .etc
    def export_lib_ops(self, op_name, content):
        with open(path.join(self._export_lib_path, op_name.lower() + ".c"), "w", encoding='utf-8') as fout:
            fout.write(self.copyright_date_version())
            fout.write(content)

    # qumico.c
    def export_qumico(self, content):
        with open(self._export_qumico_path, "w", encoding='utf-8') as fout:
            fout.write(self.copyright_date_version())
            fout.write(content)

    @staticmethod
    def copyright_date_version():
        str_now = datetime.datetime.now().replace(microsecond=0)

        qumico_dir = os.path.dirname(os.path.abspath(__file__))
        license_dir = os.path.join(os.path.dirname(qumico_dir), "LICENSE")

        with open(license_dir, "r", encoding='utf-8') as licence_file:
            license_content = licence_file.read()
            licence_file.close()

        str_copyright_date_ver = "/**\n"
        str_copyright_date_ver += "\n".join([" * {0}".format(l) for l in license_content.split("\n")])
        str_copyright_date_ver += "\n */\n\n"

        str_copyright_date_ver += "/**\n"
        str_copyright_date_ver += f" * Date Created : {str_now}\n"
        str_copyright_date_ver += f" * Version      : {QUMICO_VERSION}\n"
        str_copyright_date_ver += " */\n\n"
        return str_copyright_date_ver
