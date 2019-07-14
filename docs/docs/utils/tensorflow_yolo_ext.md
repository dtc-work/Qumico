
# tensorflow_yolo_ext.py

### conv2d_1x1_down
- :param x:
- :param filter:
- :param format:
- :param name:
- :param batch_normalization:
- :param layer_list:
- :return:

### conv2d_1x1_up
- :param x:
- :param filter:
- :param format:
- :param name:
- :param batch_normalization:
- :param layer_list:
- :return:

### conv2d_3x3_down
- :param x:
- :param filter:
- :param format:
- :param name:
- :param batch_normalization:
- :param layer_list:
- :return:

### conv2d_3x3_up
- :param x:
- :param filter:
- :param format:
- :param name:
- :param batch_normalization:
- :param layer_list:
- :return:

### conv2d_5l_block
- :param x:
- :param filter:
- :param format:
- :param name:
- :param batch_normalization:
- :param layer_list:
- :return:

### conv2d_before_residual
- :param x:
- :param filter:
- :param format:
- :param name:
- :param batch_normalization:
- :param layer_list:
- :return:

### conv2d_input
- :param x:
- :param filter:
- :param format:
- :param name:
- :param batch_normalization:
- :param layer_list:
- :return:

### deepcopy
Deep copy operation on arbitrary Python objects.

See the module's __doc__ string for more info.

### feature_out
- :param x:
- :param classes_num:
- :param format:
- :param name:
- :param batch_normalization:
- :param layer_list:
- :return:

### residual_block
- :param x: input tensor
- :param filter: first conv2d layer filter size. if None, it will be a half of the input tensor channel size.
- :param format: "NHWC" for channel last and "NCHW" for channel first. default is 'NHWC'
- :param name:
- :param batch_normalization:
- :param layer_list:
- :return:

### route2d
- :param x:
- :param sub_tensor:
- :param route_dim:
- :param format:
- :param name:
- :param layer_list:
- :return:

### up_sampling2d
- :param x:
- :param format:
- :param name:
- :param layer_list:
- :return:


