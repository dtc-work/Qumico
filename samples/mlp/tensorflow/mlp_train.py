import tensorflow as tf
from keras.datasets import mnist
from samples.utils.dataset_tool import DatasetTool
from samples.mlp.tensorflow.mlp_model import MLP
from qumico.Qumico import Qumico
import os


def mlp_train(model, train_data, epoch, batch_size, save_flag=True, save_path=None, save_file_name=None):
    tf.reset_default_graph()
    total_batch = int(train_data.total_size / batch_size)
    ckpt_file = None
    pb_file = None
    with tf.Session(graph=model.graph) as sess_train:
        sess_train.run(model.init)
        for epoch in range(epoch):
            avg_loss = 0
            acc = 0
            for i in range(total_batch):
                batch_x, batch_y = train_data.next_batch_once(batch_size=batch_size)
                _, l, c = sess_train.run([model.train_op, model.loss_op, model.predict_op],
                                         feed_dict={model.x: batch_x, model.y_t: batch_y})

                avg_loss += l / total_batch
                acc += c / total_batch

            print('Epoch: ', (epoch + 1), 'Loss: ', avg_loss, 'Accuracy: ', acc)

        if save_flag:
            if save_path is None:
                save_path = 'model'
            if save_file_name is None:
                save_file_name = 'sample'
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            ckpt_name = save_file_name + '.ckpt'
            pb_name = save_file_name + '.pb'
            ckpt_file = os.path.join(save_path, ckpt_name)
            pb_file = os.path.join(save_path, pb_name)
            model.saver.save(sess_train, ckpt_file)
            model.pb_saver.write_graph(sess_train.graph, save_path, pb_name, as_text=False)

    return ckpt_file, pb_file


if __name__ == '__main__':
    # prepare the train date
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784) / 255.
    dataset_train = DatasetTool(data=x_train, label=y_train, training_flag=True, repeat=False, one_hot_classes=10)

    # load model
    mlp_example = MLP(output_op_name='output')

    # train and save ckpt pb file
    ckpt_file, pb_file = mlp_train(mlp_example, dataset_train, epoch=10, batch_size=50)
    print(ckpt_file)
    print(pb_file)

    # prepare Qumico Convertor
    converter = Qumico()
    converter.conv_tf_to_onnx(output_path='onnx', model_name='tensorflow_mlp', output_op_name='output',
                              cache_path='model', ckpt_file=ckpt_file, pb_file=pb_file)
