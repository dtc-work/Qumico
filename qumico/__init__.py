from os import path, pardir

QumicoRootPath = path.dirname(__file__)

"""
example: export layout
export_root(defalut: out_c)
 |-include
 |  |- qumico.h
 |  |- qumico_type.h
 |-lib
 |  |-add.c
 |  |-constant.c
 |-qumico.c
"""

STD_INCLUDES = ["stdio.h","string.h", "stdbool.h"]
QUANT_INCLUDES = ["stdint.h","inttypes.h"]
ADD_INCLUDES = ["qumico.h", "qumico_type.h"]

QUMICO_MAIN = "qumico"
QUMICO_EXPORT_ROOT ="out_c"
QUMICO_INCLUDE ="include"
QUMICO_LIB ="lib"
QUMICO_TEMPLATE ="templates"

QUMICO_EXPORT_ROOT_PATH = path.join(QumicoRootPath, pardir, QUMICO_EXPORT_ROOT)
QUMICO_INCLUDE_PATH = path.join(QUMICO_EXPORT_ROOT_PATH, QUMICO_INCLUDE)
QUMICO_LIB_PATH = path.join(QUMICO_EXPORT_ROOT_PATH, QUMICO_LIB)
QUMICO_TEMPLATE_PATH = path.join(QumicoRootPath, QUMICO_TEMPLATE)

SUPPORT_ONNX_OPSET = 10

__version__ = "1.2.1"
QUMICO_VERSION = __version__
