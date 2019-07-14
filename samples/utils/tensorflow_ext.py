import tensorflow as tf

def weight_variable(shape, name='W', stddev=0.1):
    initial = tf.truncated_normal(shape=shape, stddev=stddev)
    return tf.Variable(initial, name=name, trainable=True)


def bias_variable(shape, name='b', stddev=0.1):
    initial = tf.constant(stddev, shape=shape)
    return tf.Variable(initial, name=name, trainable=True)

def add_conv2d(input, output_size, h_kernel, w_kernel, name, h_stride=1, w_stride=1, padding='SAME',
               activation='relu', param=None):
    input = tf.convert_to_tensor(input)
    input_size = input.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        weight = weight_variable([h_kernel, w_kernel, input_size, output_size], name='W_' + scope)
        bias = bias_variable([output_size], name='b_' + scope)
        conv = tf.nn.conv2d(input, weight, strides=[1, h_stride, w_stride, 1], padding=padding)
        z = tf.nn.bias_add(conv, bias)
        if activation == 'relu':
            z = tf.nn.relu(z, name=scope)
        elif activation == 'sigmoid':
            z = tf.nn.sigmoid(z, name=scope)
        elif activation == 'leakyrelu':
            z = tf.nn.leaky_relu(z, name=scope)

        if param is not None:
            param += [weight, bias]
        return z


def add_fc(input, output_size, name, activation='relu', param=None):
    input_size = input.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        weight = weight_variable([input_size, output_size], name='W_' + scope)
        bias = bias_variable([output_size], name='b_' + scope)
        mul = tf.matmul(input, weight)
        z = tf.add(mul, bias)
        if activation == 'relu':
            z = tf.nn.relu(z, name=scope)
        elif activation == 'sigmoid':
            z = tf.nn.sigmoid(z, name=scope)
        elif activation == 'leakyrelu':
            z = tf.nn.leaky_relu(z, name=scope)

        if param is not None:
            param += [weight, bias]
        return z


def add_pool(input, name, h_kernel=2, w_kernel=2, h_stride=2, w_stride=2, padding='SAME'):
    with tf.name_scope(name) as scope:
        z = tf.nn.max_pool(input, ksize=[1, h_kernel, w_kernel, 1], strides=[1, h_stride, w_stride, 1], padding=padding,
                           name='p_' + scope)
        return z


def add_flatten(input, name):
    input_size = input.get_shape()
    flat_shape = input_size[1].value * input_size[2].value * input_size[3].value
    z = tf.reshape(input, [-1, flat_shape], name=name)
    return z


def add_dropout(input, name, keep_prob=0.5, flag=True):
    if flag:
        dropout = tf.nn.dropout(input, keep_prob, name=name)
        return dropout
    else:
        return input
