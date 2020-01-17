import tensorflow as tf

def weight_variable(shape, name='W', stddev=0.1):
    """
        weight_variableを取得する
        ### 引数
        shape: 　　1次元整数のテンソルまたはPython配列。出力テンソルの形状
        name:   　　変数のオプション名。デフォルトは `'Variable'`で、取得する
                    自動的に一意化する。　　　　
        stddev:　　　タイプ `dtype`の0-DテンソルまたはPython値。標準偏差
                    切り捨て前の正規分布の。
        ### 戻り値
        tf.Variable:
        tensorflowの新しい変数
    """
    initial = tf.truncated_normal(shape=shape, stddev=stddev)
    return tf.Variable(initial, name=name, trainable=True)


def bias_variable(shape, name='b', stddev=0.1):
    """
        bias_variableを取得する
        ### 引数
        shape: list　　1次元整数のテンソルまたはPython配列。出力テンソルの形状
        name:  string 　　変数のオプション名。デフォルトは `'Variable'`で、取得する
                    自動的に一意化する。　　　　
        stddev:　float　　タイプ `dtype`の0-DテンソルまたはPython値。標準偏差
                    切り捨て前の正規分布の。
        ### 戻り値
        tf.Variable:
        tensorflowの新しい変数
    """
    initial = tf.constant(stddev, shape=shape)
    return tf.Variable(initial, name=name, trainable=True)

def add_conv2d(input, output_size, h_kernel, w_kernel, name, h_stride=1, w_stride=1, padding='SAME',
               activation='relu', param=None):
    """
      　convolution追加する
        ### 引数
        input: 　object　　タイプ `Tensor`として登録された変換関数を持つオブジェクト。　
        output_size: int        weight_variableの出力サイズ        　　　　
        h_kernel:　 int　　      tensorのh_kernel　　　             　
        w_kernel： int  　       tensorのw_kernel
        name： string        変数のオプション名。デフォルトは `'Variable'`で、取得する
                        自動的に一意化されます。　　
        h_stride：   　 strideのh_stride　
        w_stride：      strideのw_stride
        　(stride: `int`のリスト。
          長さ1の1次元テンソル。それぞれのスライディングウィンドウの歩幅
          「入力」の次元。次元の順序は、次の値によって決定する
          `data_format`。。)
        padding：　'string'   パディングアルゴリズムのタイプ
        activation： string   reluとsigmoidとleakyrelu三つの選択肢がある
        param: string     存在する場合は　weight変数とbias変数追加します。
        ### 戻り値
        z: object「tensor」。 `features`と同じ型を持つ
    """
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
    """  　
        ### 引数
        input: 　object　　タイプ `Tensor`として登録された変換関数を持つオブジェクト。　
        output_size: int        weight_variableの出力サイズ        　　　　
        name： string        変数のオプション名。デフォルトは `'Variable'`で、取得します
                        自動的に一意化されます。　　
        activation： string   reluとsigmoidとleakyrelu三つの選択肢がある
        param: string     存在する場合は　weight変数とbias変数追加します。
        ### 戻り値
        z: object「tensor」。 `features`と同じ型を持ちます
    """
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
    """
      　pool追加する
        ### 引数
        input: 　object　　タイプ `Tensor`として登録された変換関数を持つオブジェクト。　
        output_size: int        weight_variableの出力サイズ
        name： string        変数のオプション名。デフォルトは `'Variable'`で、取得します
                        自動的に一意化されます。　　　　
        h_kernel:　 int　　      tensorのh_kernel　　　             　
        w_kernel： int  　       tensorのw_kernel　　
        h_stride：   　 strideのh_stride　
        w_stride：      strideのw_stride
        　(stride: `ints`のリスト。
          長さ1の1次元テンソル。それぞれのスライディングウィンドウの歩幅
          「入力」の次元。次元の順序は、次の値によって決定されます
          `data_format`。。)
        padding：　'String'   パディングアルゴリズムのタイプ
        ### 戻り値
        z: object「tensor」。 最大プール出力テンソル
    """
    with tf.name_scope(name) as scope:
        z = tf.nn.max_pool(input, ksize=[1, h_kernel, w_kernel, 1], strides=[1, h_stride, w_stride, 1], padding=padding,
                           name='p_' + scope)
        return z


def add_flatten(input, name):
    """
      　flattenを追加する
        ### 引数
        input: 　object　　タイプ `Tensor`として登録された変換関数を持つオブジェクト。　
        name： string        変数のオプション名。デフォルトは `'Variable'`で、取得する
                        自動的に一意化されます。　　　　
        ### 戻り値
        z: object「tensor」。 変換したtensor
    """
    input_size = input.get_shape()
    flat_shape = input_size[1].value * input_size[2].value * input_size[3].value
    z = tf.reshape(input, [-1, flat_shape], name=name)
    return z


def add_dropout(input, name, keep_prob=0.5, flag=True):
    """
      　tensorをドロップアウト
        ### 引数
        input: 　object　　タイプ `Tensor`として登録された変換関数を持つオブジェクト。　
        name： string        変数のオプション名。デフォルトは `'Variable'`で、取得する
                        自動的に一意化されます。　　　　
        keep_prob:　　残される確率
        flag:   ドロップアウト
        ### 戻り値
        z: object「tensor」。 変換したtensor
    """
    if flag:
        dropout = tf.nn.dropout(input, keep_prob, name=name)
        return dropout
    else:
        return input
