import tensorflow as tf
from keras.datasets import mnist
import samples.utils.common_tool as common
from samples.mlp.tensorflow.mlp_model import MLP


def mlp_infer(model, infer_data, ckpt_file):
    tf.reset_default_graph()

    with tf.Session(graph=model.graph) as sess_predict:
        model.saver.restore(sess_predict, ckpt_file)
        output = sess_predict.run([model.y], feed_dict={model.x: infer_data})
        classification = common.softmax(output)
        y = common.onehot_decoding(classification)
        return y

if __name__ == '__main__':
    # prepare the infer date 28px * 28px image
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_test = x_test.reshape(10000, 784) / 255.
    image_data = x_test[:10, ...]

    # init model
    mlp_example = MLP(output_op_name='output')

    # model weights path
    ckpt_file = 'model/sample.ckpt'

    # load weights and infer image
    # return : classes index
    result = mlp_infer(mlp_example, image_data, ckpt_file)

    # output the result
    print('Predict Index ', result)
