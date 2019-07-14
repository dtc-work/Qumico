import sys

from onnx.backend.base import DeviceType

IS_PYTHON3 = sys.version_info > (3,)


def get_perm_from_formats(from_, to_):
    """ Get perm from data formats.
        For example:
        get_perm_from_formats('NHWC', 'NCHW') = [0, 3, 1, 2]
        :param from_: From data format string.
        :param to_: To data format string.
        :return: Perm. Int list.
      """
    return list(map(lambda x: from_.find(x), to_))


def get_device_option(device):
    m = {DeviceType.CPU: '/cpu', DeviceType.CUDA: '/gpu'}
    return m[device.type]


def get_data_format(x_rank):
    """ Get data format by input rank.
      Channel first if support CUDA.
      :param x_rank: Input rank.
      :return: Data format.
    """
    sp_dim_names = ["D", "H", "W"]
    sp_dim_lst = []
    for i in range(x_rank - 2):
        sp_dim_lst.append(sp_dim_names[-i - 1])

    sp_dim_string = "".join(reversed(sp_dim_lst))
    storage_format = "NC" + sp_dim_string

    if supports_device("CUDA"):
        compute_format = "NC" + sp_dim_string
    else:
        compute_format = "N" + sp_dim_string + "C"
    return storage_format, compute_format


def supports_device(device):
    """ Check if support target device.
    :param device: CUDA or CPU.
    :return: If supports.
    """

    if device == "CUDA":
        return False
    #     local_device_protos = device_lib.list_local_devices()
    #     return len([x.name for x in local_device_protos if x.device_type == 'GPU'
    #                ]) > 0
    elif device == "CPU":
        return True
    return False