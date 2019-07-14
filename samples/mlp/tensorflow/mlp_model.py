import tensorflow as tf


class MLP():
    def __init__(self, input_size=784, mid_units=100, output_size=10, lr=0.1, output_op_name='output'):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.mid_units = mid_units
        self.output_size = output_size
        self.lr = lr
        self.output_op_name = output_op_name

        mlp_graph = tf.Graph()
        with mlp_graph.as_default():
            x = tf.placeholder(tf.float32, [None, self.input_size], name='input')
            y_t = tf.placeholder(tf.float32, [None, self.output_size], name='valid')
            W1 = tf.Variable(tf.random_normal([self.input_size, self.mid_units], stddev=0.03), name='W1')
            b1 = tf.Variable(tf.random_normal([self.mid_units]), name='b1')
            W2 = tf.Variable(tf.random_normal([self.mid_units, self.output_size], stddev=0.03), name='W2')
            b2 = tf.Variable(tf.random_normal([self.output_size]), name='b2')

            mid_layer = tf.add(tf.matmul(x, W1), b1, name='hidden')
            mid_layer = tf.nn.relu(mid_layer, name='relu')

            output_layer = tf.add(tf.matmul(mid_layer, W2), b2, name='hidden')
            y = tf.nn.softmax(output_layer, name=output_op_name)

            with tf.name_scope('train'):
                y_clipped = tf.clip_by_value(y, 1e-10, 0.9999999)
                cross_entropy = -tf.reduce_mean(
                    tf.reduce_sum(y_t * tf.log(y_clipped) + (1 - y_t) * tf.log(1 - y_clipped), axis=1))
                train_op = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(cross_entropy)

            with tf.name_scope('predict'):
                correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_t, 1))
                predict_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            # version V2 ファイル新規作成する際にエラーになります。
            # max_to_keep 出力ckpt フィアル数

            saver = tf.train.Saver(write_version=tf.train.SaverDef.V1, max_to_keep=1)
            pb_saver = tf.train
            init = tf.global_variables_initializer()

        self.graph = mlp_graph
        self.train_op = train_op
        self.predict_op = predict_op
        self.loss_op = cross_entropy
        self.x = x
        self.y = y
        self.y_t = y_t

        self.init = init
        self.saver = saver
        self.pb_saver = pb_saver
