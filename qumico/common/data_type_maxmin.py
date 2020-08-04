import numpy as np

MAXMIN_LIST = {
        np.dtype('float32') : ("DBL_MAX", "DBL_MIN"),
        np.dtype('uint8'): ("255", "0"),
        np.dtype('int8'): ("127", "-128"),
        np.dtype('uint16'): ("USHRT_MAX", "0"),
        np.dtype('int16'):("SHRT_MAX", "SHRT_MIN"),
        np.dtype('int32'):("INT_MAX", "INT_MIN"),
        np.dtype('int64'):("LONG_MAX", "LONG_MIN"),
}


def get_max(dtype):
    return MAXMIN_LIST[dtype][0]


def get_min(dtype):
    return MAXMIN_LIST[dtype][1]
