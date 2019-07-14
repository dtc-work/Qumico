import tensorflow as tf
import samples.utils.tensorflow_ext as tf_ext


class CONV():

    def __init__(self, input_size=784, mid_units=100, output_size=10, lr=1e-4, keep_prob=0.5,
                 output_op_name='output'):
        super(CONV, self).__init__()
        self.input_size = input_size
        self.mid_units = mid_units
        self.output_size = output_size
        self.lr = lr
        self.output_node_name = output_op_name

        mlp_graph = tf.Graph()
        with mlp_graph.as_default():
            x = tf.placeholder(tf.float32, [None, self.input_size], name='input')
            y_t = tf.placeholder(tf.float32, [None, self.output_size], name='valid')
            x_image = tf.reshape(x, [-1, 28, 28, 1], name='reshape')
            conv1 = tf_ext.add_conv2d(x_image, output_size=32, h_kernel=5, w_kernel=5, activation='relu',
                                      name='conv1')
            pool1 = tf_ext.add_pool(conv1, name='pool1')
            conv2 = tf_ext.add_conv2d(pool1, output_size=64, h_kernel=5, w_kernel=5, activation='relu', name='conv2')
            pool2 = tf_ext.add_pool(conv2, name='pool2')

            flat = tf_ext.add_flatten(pool2, name='flatten')
            fc1 = tf_ext.add_fc(flat, output_size=1024, activation='relu', name='fc1')
            drop = tf_ext.add_dropout(fc1, keep_prob=keep_prob, name='dropout1')
            fc2 = tf_ext.add_fc(drop, self.output_size, activation='relu', name='fc2')
            y = tf.nn.softmax(fc2, name=self.output_node_name)

            with tf.name_scope('train'):
                cross_entropy = -tf.reduce_sum(y_t * tf.log(y))
                train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(cross_entropy)

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
