import tensorflow as tf
from keras.datasets import mnist
from samples.utils.dataset_tool import DatasetTool
from samples.mlp.tensorflow.mlp_model import MLP


def mlp_test(model, test_data, ckpt_file, batch_size):
    tf.reset_default_graph()

    with tf.Session(graph=model.graph) as sess_predict:
        model.saver.restore(sess_predict, ckpt_file)
        total_size = test_data.total_size
        total_batch = int(total_size / batch_size)
        acc = 0
        for i in range(total_batch):
            batch_x, batch_y = test_data.next_batch_once(batch_size=batch_size)
            _, c = sess_predict.run([model.train_op, model.predict_op],
                                    feed_dict={model.x: batch_x, model.y_t: batch_y})

            acc += c / total_batch

        print('Total: ', total_size, ' Accuracy: ', acc)


if __name__ == '__main__':
    # prepare the test date
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_test = x_test.reshape(10000, 784) / 255.
    test_data = DatasetTool(data=x_test, label=y_test, training_flag=True, repeat=False, one_hot_classes=10)

    # init model
    mlp_example = MLP(output_op_name='output')

    # model weights path
    ckpt_file = 'model/sample.ckpt'

    # load weights and test model
    mlp_test(mlp_example, test_data, ckpt_file, 1000)
