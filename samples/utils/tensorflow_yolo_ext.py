import tensorflow as tf
import numpy as np
from copy import deepcopy
import samples.utils.tensorflow_ext as tf_helper



def residual_block(x, filter=None, format="NHWC", name="residual_block", batch_normalization=True,
                   layer_list=None):
    """
    :param x: input tensor
    :param filter: first conv2d layer filter size. if None, it will be a half of the input tensor channel size.
    :param format: "NHWC" for channel last and "NCHW" for channel first. default is 'NHWC'
    :param name:
    :param batch_normalization:
    :param layer_list:
    :return:
    """
    with tf.name_scope(name) as scope:
        x = tf.convert_to_tensor(x)
        shortcut = x
        input_shape = x.get_shape()
        N, H, W, C = (0, 0, 0, 0)
        if format == "NHWC":
            N, H, W, C = input_shape[0].value, input_shape[1].value, input_shape[2].value, input_shape[3].value
        elif format == "NCHW":
            N, C, H, W = input_shape[0].value, input_shape[1].value, input_shape[2].value, input_shape[3].value

        filter_1 = filter if filter is not None else int(C / 2)
        filter_2 = C

        block_conv_1 = tf_helper.add_conv2d(x, filter_1, h_kernel=1, w_kernel=1, name="layer_1", h_stride=1,
                                            w_stride=1, format=format, batch_normalization=batch_normalization,
                                            activation="leaky_relu", leaky_relu_alpha=0.1, padding="SAME")
        print("conv_residual : ", block_conv_1.shape)
        block_conv_2 = tf_helper.add_conv2d(block_conv_1, filter_2, h_kernel=3, w_kernel=3, name="layer_2",
                                            h_stride=1,
                                            w_stride=1, format=format, batch_normalization=batch_normalization,
                                            activation="leaky_relu", leaky_relu_alpha=0.1, padding="SAME")
        print("conv_residual : ", block_conv_2.shape)
        y = tf_helper.add_shortcut(block_conv_2, shortcut, name=scope)
        print("shortcut : ", y.shape)
        if layer_list is not None:
            layer_list.append(block_conv_1)
            layer_list.append(block_conv_2)
        return y


def conv2d_before_residual(x, filter=None, format="NHWC", name="conv_before_residual",
                           batch_normalization=True,
                           layer_list=None):
    """

    :param x:
    :param filter:
    :param format:
    :param name:
    :param batch_normalization:
    :param layer_list:
    :return:
    """
    with tf.name_scope(name) as scope:
        x = tf.convert_to_tensor(x)
        input_shape = x.get_shape()
        N, H, W, C = (0, 0, 0, 0)
        if format == "NHWC":
            N, H, W, C = input_shape[0].value, input_shape[1].value, input_shape[2].value, input_shape[3].value
        elif format == "NCHW":
            N, C, H, W = input_shape[0].value, input_shape[1].value, input_shape[2].value, input_shape[3].value

        filter_1 = filter if filter is not None else int(C * 2)

        y = tf_helper.add_conv2d(x, filter_1, h_kernel=3, w_kernel=3, name=scope, h_stride=2,
                                 w_stride=2, format=format, batch_normalization=batch_normalization,
                                 activation="leaky_relu", leaky_relu_alpha=0.1, padding="SAME")
        print("conv_before_residual : ", y.shape)
        if layer_list is not None:
            layer_list.append(y)
        return y


def conv2d_input(x, filter=32, format="NHWC", name="conv_input", batch_normalization=True,
                 layer_list=None):
    """

    :param x:
    :param filter:
    :param format:
    :param name:
    :param batch_normalization:
    :param layer_list:
    :return:
    """
    with tf.name_scope(name) as scope:
        x = tf.convert_to_tensor(x)
        input_shape = x.get_shape()
        N, H, W, C = (0, 0, 0, 0)
        if format == "NHWC":
            N, H, W, C = input_shape[0].value, input_shape[1].value, input_shape[2].value, input_shape[3].value
        elif format == "NCHW":
            N, C, H, W = input_shape[0].value, input_shape[1].value, input_shape[2].value, input_shape[3].value

        filter_1 = filter if filter is not None else C
        y = tf_helper.add_conv2d(x, filter_1, h_kernel=3, w_kernel=3, name=scope, h_stride=1,
                                 w_stride=1, format=format, batch_normalization=batch_normalization,
                                 activation="leaky_relu", leaky_relu_alpha=0.1, padding="SAME")
        print("conv_input : ", y.shape)
        if layer_list is not None:
            layer_list.append(y)
        return y


def conv2d_1x1_down(x, filter=None, format="NHWC", name="conv_1x1_down", batch_normalization=True,
                    layer_list=None):
    """

    :param x:
    :param filter:
    :param format:
    :param name:
    :param batch_normalization:
    :param layer_list:
    :return:
    """
    with tf.name_scope(name) as scope:
        x = tf.convert_to_tensor(x)
        input_shape = x.get_shape()
        N, H, W, C = (0, 0, 0, 0)
        if format == "NHWC":
            N, H, W, C = input_shape[0].value, input_shape[1].value, input_shape[2].value, input_shape[3].value
        elif format == "NCHW":
            N, C, H, W = input_shape[0].value, input_shape[1].value, input_shape[2].value, input_shape[3].value

        filter_1 = filter if filter is not None else int(C / 2)

        y = tf_helper.add_conv2d(x, filter_1, h_kernel=1, w_kernel=1, name=scope, h_stride=1,
                                 w_stride=1, format=format, batch_normalization=batch_normalization,
                                 activation="leaky_relu", leaky_relu_alpha=0.1, padding="SAME")
        print("conv_1x1_down : ", y.shape)
        if layer_list is not None:
            layer_list.append(y)
        return y


def conv2d_1x1_up(x, filter=None, format="NHWC", name="conv_1x1_up", batch_normalization=True,
                  layer_list=None):
    """

    :param x:
    :param filter:
    :param format:
    :param name:
    :param batch_normalization:
    :param layer_list:
    :return:
    """
    with tf.name_scope(name) as scope:
        x = tf.convert_to_tensor(x)
        input_shape = x.get_shape()
        N, H, W, C = (0, 0, 0, 0)
        if format == "NHWC":
            N, H, W, C = input_shape[0].value, input_shape[1].value, input_shape[2].value, input_shape[3].value
        elif format == "NCHW":
            N, C, H, W = input_shape[0].value, input_shape[1].value, input_shape[2].value, input_shape[3].value

        filter_1 = filter if filter is not None else int(C * 2)

        y = tf_helper.add_conv2d(x, filter_1, h_kernel=1, w_kernel=1, name=scope, h_stride=1,
                                 w_stride=1, format=format, batch_normalization=batch_normalization,
                                 activation="leaky_relu", leaky_relu_alpha=0.1, padding="SAME")
        print("conv_1x1_up : ", y.shape)
        if layer_list is not None:
            layer_list.append(y)
        return y


def conv2d_3x3_down(x, filter=None, format="NHWC", name="conv_3x3_down", batch_normalization=True,
                    layer_list=None):
    """

    :param x:
    :param filter:
    :param format:
    :param name:
    :param batch_normalization:
    :param layer_list:
    :return:
    """
    with tf.name_scope(name) as scope:
        x = tf.convert_to_tensor(x)
        input_shape = x.get_shape()
        N, H, W, C = (0, 0, 0, 0)
        if format == "NHWC":
            N, H, W, C = input_shape[0].value, input_shape[1].value, input_shape[2].value, input_shape[3].value
        elif format == "NCHW":
            N, C, H, W = input_shape[0].value, input_shape[1].value, input_shape[2].value, input_shape[3].value

        filter_1 = filter if filter is not None else int(C / 2)

        y = tf_helper.add_conv2d(x, filter_1, h_kernel=1, w_kernel=1, name=scope, h_stride=1,
                                 w_stride=1, format=format, batch_normalization=batch_normalization,
                                 activation="leaky_relu", leaky_relu_alpha=0.1, padding="SAME")
        print("conv_3x3_down : ", y.shape)
        if layer_list is not None:
            layer_list.append(y)
        return y


def conv2d_3x3_up(x, filter=None, format="NHWC", name="conv_1x1_up", batch_normalization=True,
                  layer_list=None):
    """

    :param x:
    :param filter:
    :param format:
    :param name:
    :param batch_normalization:
    :param layer_list:
    :return:
    """
    with tf.name_scope(name) as scope:
        x = tf.convert_to_tensor(x)
        input_shape = x.get_shape()
        N, H, W, C = (0, 0, 0, 0)
        if format == "NHWC":
            N, H, W, C = input_shape[0].value, input_shape[1].value, input_shape[2].value, input_shape[3].value
        elif format == "NCHW":
            N, C, H, W = input_shape[0].value, input_shape[1].value, input_shape[2].value, input_shape[3].value

        filter_1 = filter if filter is not None else int(C * 2)

        y = tf_helper.add_conv2d(x, filter_1, h_kernel=3, w_kernel=3, name=scope, h_stride=1,
                                 w_stride=1, format=format, batch_normalization=batch_normalization,
                                 activation="leaky_relu", leaky_relu_alpha=0.1, padding="SAME")
        print("conv_3x3_up : ", y.shape)
        if layer_list is not None:
            layer_list.append(y)
        return y


def feature_out(x, classes_num, format="NHWC", name="feature_out", batch_normalization=True,
                layer_list=None):
    """

    :param x:
    :param classes_num:
    :param format:
    :param name:
    :param batch_normalization:
    :param layer_list:
    :return:
    """

    with tf.name_scope(name) as scope:
        x = tf.convert_to_tensor(x)
        input_shape = x.get_shape()
        if format == "NHWC":
            N, H, W, C = input_shape[0].value, input_shape[1].value, input_shape[2].value, input_shape[3].value
        elif format == "NCHW":
            N, C, H, W = input_shape[0].value, input_shape[1].value, input_shape[2].value, input_shape[3].value

        filter_1 = (classes_num + 5) * 3

        feature = tf_helper.add_conv2d(x, filter_1, h_kernel=1, w_kernel=1, name=scope, h_stride=1,
                                       w_stride=1, format=format, batch_normalization=batch_normalization,
                                       activation="linear", leaky_relu_alpha=0.1, padding="SAME")
        print("conv_feature : ", feature.shape)
    if layer_list is not None:
        layer_list.append(feature)
    return feature


def up_sampling2d(x, format="NHWC", name="up_sampling",
                  layer_list=None):
    """

    :param x:
    :param format:
    :param name:
    :param layer_list:
    :return:
    """
    with tf.name_scope(name) as scope:
        x = tf.convert_to_tensor(x)
        input_shape = x.get_shape()
        if format == "NHWC":
            N, H, W, C = input_shape[0].value, input_shape[1].value, input_shape[2].value, input_shape[3].value
            x = tf.image.resize_images(x, [H * 2, W * 2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        elif format == "NCHW":
            N, C, H, W = input_shape[0].value, input_shape[1].value, input_shape[2].value, input_shape[3].value
            x = tf.transpose(x, perm=[0, 2, 3, 1])
            x = tf.image.resize_images(x, [H * 2, W * 2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            x = tf.transpose(x, perm=[0, 3, 1, 2])
        y = x
        print("up_sampling : ", y.shape)
    if layer_list is not None:
        layer_list.append(y)
    return y


def route2d(x, sub_tensor=None, route_dim=3, format="NHWC", name="route", layer_list=None):
    """

    :param x:
    :param sub_tensor:
    :param route_dim:
    :param format:
    :param name:
    :param layer_list:
    :return:
    """
    with tf.name_scope(name) as scope:
        x = tf.convert_to_tensor(x)
        input_shape = x.get_shape()
        if format == "NHWC":
            N, H, W, C = input_shape[0].value, input_shape[1].value, input_shape[2].value, input_shape[3].value
        elif format == "NCHW":
            N, C, H, W = input_shape[0].value, input_shape[1].value, input_shape[2].value, input_shape[3].value
        if sub_tensor is not None:
            x = tf.concat([x, sub_tensor], route_dim, name=scope)
        y = x
        print("route2d : ", y.shape)
    if layer_list is not None:
        layer_list.append(y)
    return y


def conv2d_5l_block(x, filter=None, format="NHWC", name="conv_5l", batch_normalization=True,
                    layer_list=None):
    """

    :param x:
    :param filter:
    :param format:
    :param name:
    :param batch_normalization:
    :param layer_list:
    :return:
    """
    with tf.name_scope(name) as scope:
        x = tf.convert_to_tensor(x)
        input_shape = x.get_shape()
        N, H, W, C = (0, 0, 0, 0)
        if format == "NHWC":
            N, H, W, C = input_shape[0].value, input_shape[1].value, input_shape[2].value, input_shape[3].value
        elif format == "NCHW":
            N, C, H, W = input_shape[0].value, input_shape[1].value, input_shape[2].value, input_shape[3].value

        filter_1 = filter if filter is not None else int(C / 2)
        filter_2 = int(filter_1 * 2)
        filter_3 = filter_1
        filter_4 = int(filter_1 * 2)
        filter_5 = filter_1

        block_conv_1 = tf_helper.add_conv2d(x, filter_1, h_kernel=1, w_kernel=1, name="layer_1", h_stride=1,
                                            w_stride=1, format=format, batch_normalization=batch_normalization,
                                            activation="leaky_relu", leaky_relu_alpha=0.1, padding="SAME")
        print("conv5l : ", block_conv_1.shape)

        block_conv_2 = tf_helper.add_conv2d(block_conv_1, filter_2, h_kernel=3, w_kernel=3, name="layer_2",
                                            h_stride=1,
                                            w_stride=1, format=format, batch_normalization=batch_normalization,
                                            activation="leaky_relu", leaky_relu_alpha=0.1, padding="SAME")
        print("conv5l : ", block_conv_2.shape)

        block_conv_3 = tf_helper.add_conv2d(block_conv_2, filter_3, h_kernel=1, w_kernel=1, name="layer_3",
                                            h_stride=1,
                                            w_stride=1, format=format, batch_normalization=batch_normalization,
                                            activation="leaky_relu", leaky_relu_alpha=0.1, padding="SAME")
        print("conv5l : ", block_conv_3.shape)

        block_conv_4 = tf_helper.add_conv2d(block_conv_3, filter_4, h_kernel=3, w_kernel=3, name="layer_4",
                                            h_stride=1,
                                            w_stride=1, format=format, batch_normalization=batch_normalization,
                                            activation="leaky_relu", leaky_relu_alpha=0.1, padding="SAME")
        print("conv5l : ", block_conv_4.shape)

        block_conv_5 = tf_helper.add_conv2d(block_conv_4, filter_5, h_kernel=1, w_kernel=1, name="layer_5",
                                            h_stride=1,
                                            w_stride=1, format=format, batch_normalization=batch_normalization,
                                            activation="leaky_relu", leaky_relu_alpha=0.1, padding="SAME")
        print("conv5l : ", block_conv_5.shape)
        y = block_conv_5

        if layer_list is not None:
            layer_list.append(block_conv_1)
            layer_list.append(block_conv_2)
            layer_list.append(block_conv_3)
            layer_list.append(block_conv_4)
            layer_list.append(block_conv_5)
        return y


def pooling2x(x, format="NHWC", name="maxpooling", layer_list=None):
    with tf.name_scope(name) as scope:
        x = tf.convert_to_tensor(x)
        input_shape = x.get_shape()
        N, H, W, C = (0, 0, 0, 0)
        if format == "NHWC":
            N, H, W, C = input_shape[0].value, input_shape[1].value, input_shape[2].value, input_shape[3].value
        elif format == "NCHW":
            N, C, H, W = input_shape[0].value, input_shape[1].value, input_shape[2].value, input_shape[3].value
        pool_layer = tf_helper.add_pool(x, name=scope)
        return pool_layer


def darknetconv2d(x, output_size, h_kernel, w_kernel, name, h_stride=1, w_stride=1, padding="SAME",
                  activation="relu", leaky_relu_alpha=0.1, format="NCHW", batch_normalization=False, training=True):
    x = tf.convert_to_tensor(x)
    if format == "NHWC":
        input_size = x.get_shape()[-1].value
        batch_normalization_axis = -1
        bias_shape = [output_size]
    else:
        input_size = x.get_shape()[-3].value
        batch_normalization_axis = 1
        bias_shape = [output_size, 1, 1]

    with tf.name_scope(name) as scope:
        weight = tf_helper.weight_variable([h_kernel, w_kernel, input_size, output_size], name="W_" + scope)
        bias = tf_helper.bias_variable(bias_shape, name="b_" + scope)
        z = tf.nn.conv2d(x, weight, strides=[1, h_stride, w_stride, 1], padding=padding, data_format=format)

        if batch_normalization:
            z = tf.layers.batch_normalization(z, training=training, axis=batch_normalization_axis)
        else:
            z = z + bias

        if activation == "relu":
            z = tf.nn.relu(z, name=scope)
        elif activation == "sigmoid":
            z = tf.nn.sigmoid(z, name=scope)
        elif activation == "leaky_relu":
            z = tf.nn.leaky_relu(z, alpha=leaky_relu_alpha, name=scope)
        elif activation == "linear":
            z = tf.identity(z, name=scope)
        return z


def darknetpool(x, name, h_kernel=2, w_kernel=2, h_stride=2, w_stride=2, padding="SAME", format="NCHW"):
    if format == "NCHW":
        ksize = [1, 1, h_kernel, w_kernel]
        strides = [1, 1, h_stride, w_stride]
    else:
        ksize = [1, h_kernel, w_kernel, 1]
        strides = [1, h_stride, w_stride, 1]

    with tf.name_scope(name) as scope:
        z = tf.nn.max_pool(x, ksize=ksize, strides=strides, padding=padding,
                           data_format=format, name="p_" + scope)
        return z


def reorg_layer(feature_out_layer, num_classes, anchors, model_h, model_w):
    x = tf.convert_to_tensor(feature_out_layer)
    input_shape = x.get_shape()

    batch, grid_w, grid_h, grid_out = input_shape[0].value, input_shape[1].value, input_shape[2].value, input_shape[
        3].value
    num_anchors = len(anchors)
    lout = int(grid_out / num_anchors)
    stride_h = model_h / grid_h
    stride_w = model_w / grid_w
    print(stride_h)
    x = tf.reshape(x, [-1, grid_w, grid_h, num_anchors, lout])

    feature_class, feature_xy, feature_wh, feature_conf = tf.split(x, [num_classes, 2, 2, 1], axis=-1)

    xy_offset = get_offset_xy(grid_w, grid_h)
    feature_xy = tf.nn.sigmoid(feature_xy)

    feature_xy = tf.add(feature_xy, xy_offset)
    box_xy = tf.multiply(feature_xy, stride_h)
    box_wh = tf.exp(feature_wh) * anchors
    boxes = tf.concat([box_xy, box_wh], axis=-1)

    return boxes, feature_conf, feature_class, xy_offset


def get_offset_xy(grid_w, grid_h):
    grid_x = np.arange(grid_w)
    grid_y = np.arange(grid_h)
    x, y = np.meshgrid(grid_x, grid_y)
    x = np.reshape(x, (grid_w, grid_h, -1))
    y = np.reshape(y, (grid_w, grid_h, -1))
    x_y_offset = np.concatenate((x, y), -1)
    x_y_offset = np.reshape(x_y_offset, [grid_w, grid_h, 1, 2])
    return x_y_offset

def get_offset_yx(grid_h, grid_w):
    grid_x = np.arange(grid_w)
    grid_y = np.arange(grid_h)
    x, y = np.meshgrid(grid_y, grid_x)
    x = np.reshape(x, (grid_h, grid_w, -1))
    y = np.reshape(y, (grid_h, grid_w, -1))
    x_y_offset = np.concatenate((y, x), -1)
    x_y_offset = np.reshape(x_y_offset, [grid_h, grid_w, 1, 2])
    return x_y_offset


def bbox_to_anbox(bbox):
    anbox = deepcopy(bbox)
    anbox[..., 0] = (bbox[..., 0] + bbox[..., 2]) / 2
    anbox[..., 1] = (bbox[..., 1] + bbox[..., 3]) / 2
    anbox[..., 2] = bbox[..., 2] - bbox[..., 0]
    anbox[..., 3] = bbox[..., 3] - bbox[..., 1]
    return anbox


def anbox_to_bbox(anbox):
    bbox = deepcopy(anbox)
    bbox[..., 0] = anbox[..., 0] - anbox[..., 2] / 2
    bbox[..., 2] = anbox[..., 0] + anbox[..., 2] / 2
    bbox[..., 1] = anbox[..., 1] - anbox[..., 3] / 2
    bbox[..., 3] = anbox[..., 1] + anbox[..., 3] / 2
    return bbox


def extract_feature(feature_map, num_classes, anchors, model_h=None, model_w=None):
    feature_map = tf.transpose(feature_map, [0, 2, 1, 3])

    print(anchors)

    boxes, feature_conf, feature_class, xy_offset = reorg_layer(feature_map, num_classes, anchors, model_h,
                                                                model_w)

    print(type(boxes), boxes)
    print(type(feature_conf), feature_conf)
    print(type(feature_class), feature_class)
    return boxes, feature_conf, feature_class, xy_offset
