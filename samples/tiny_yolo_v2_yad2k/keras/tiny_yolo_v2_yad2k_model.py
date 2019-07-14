import os
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, InputLayer
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
import numpy as np


from tiny_yolo_v2_yad2k_common import (width, height, r_w, r_h, r_n,
                                      thresh,iou_threshold, voc_label,
                                      classes, yolo_eval,region_biases, voc_anchors)


def tiny_yolo_model():
    model = Sequential()
    model.add(InputLayer(input_shape=(width, height, 3)))
    model.add(Conv2D(16, use_bias=False, data_format="channels_last",
                     padding='same', kernel_size=(3, 3), strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same', strides=(2, 2)))

    model.add(Conv2D(32, use_bias=False, data_format="channels_last", padding='same', kernel_size=(3, 3), strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same', strides=(2, 2)))

    model.add(Conv2D(64, use_bias=False, data_format="channels_last", padding='same', kernel_size=(3, 3), strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same', strides=(2, 2)))

    model.add(Conv2D(128, use_bias=False, data_format="channels_last", padding='same', kernel_size=(3, 3), strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same', strides=(2, 2)))

    model.add(Conv2D(256, use_bias=False, data_format="channels_last", padding='same', kernel_size=(3, 3), strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same', strides=(2, 2)))

    model.add(Conv2D(512, use_bias=False, data_format="channels_last", padding='same', kernel_size=(3, 3), strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same', strides=(1, 1)))

    model.add(Conv2D(1024, use_bias=False, data_format="channels_last", padding='same', kernel_size=(3, 3), strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    model.add(Conv2D(1024, use_bias=False, data_format="channels_last", padding='same', kernel_size=(3, 3), strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    model.add(Conv2D(125, use_bias=True, data_format="channels_last", padding='same', kernel_size=(1, 1), strides=(1, 1)))

    return model


def layer_dump(model, x, l, postfix=''):
    """
    :param model: Keras model
    :param x: input data
    :param l: layer number
    :param postfix: postfix for numpy file name
    :return: None
    """
    get_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[l].output])

    layer_output = get_layer_output([x, 0])[0]
    last_dim = layer_output.shape[-1] - 1

    if l == 0:
        d0 = layer_output[0,:, :, 0]
        d1 = layer_output[0,:, :, 1]
        d2 = layer_output[0,:, :, 2]
        np.save('output/l%02d%s_0.npy' % (l, postfix), d0, allow_pickle=False)
        np.save('output/l%02d%s_1.npy' % (l, postfix), d1, allow_pickle=False)
        np.save('output/l%02d%s_2.npy' % (l, postfix), d2, allow_pickle=False)
    else:
        d0 = layer_output[0,:, :, 0]
        d1 = layer_output[0,:, :, 1]
        d2 = layer_output[0,:, :, 2]
        d3 = layer_output[0,:, :, 3]
        d4 = layer_output[0,:, :, 4]
        dl = layer_output[0,:, :, last_dim]
        np.save('output/l%02d%s_all.npy' % (l, postfix), layer_output, allow_pickle=False)
        np.save('output/l%02d%s_0.npy' % (l, postfix), d0, allow_pickle=False)
        np.save('output/l%02d%s_1.npy' % (l, postfix), d1, allow_pickle=False)
        np.save('output/l%02d%s_2.npy' % (l, postfix), d2, allow_pickle=False)
        np.save('output/l%02d%s_3.npy' % (l, postfix), d3, allow_pickle=False)
        np.save('output/l%02d%s_4.npy' % (l, postfix), d4, allow_pickle=False)

        np.save('output/l%02d%s_%d.npy' % (l, postfix, last_dim), dl, allow_pickle=False)


