from keras.datasets import mnist
from samples.utils import dataset_tool


def prepare_infer_dataset():
    (_, _), (x_test, y_test) = mnist.load_data()
    x_test = x_test.reshape(10000, 784) / 255.
    return x_test[:10, ...]


def prepare_test_dataset():
    (_, _), (x_test, y_test) = mnist.load_data()
    x_test = x_test.reshape(10000, 784) / 255.
    return dataset_tool.DatasetTool(data=x_test, label=y_test, training_flag=True, repeat=False, one_hot_classes=10)


def prepare_train_dataset():
    (x_train, y_train), (_, _) = mnist.load_data()
    x_train = x_train.reshape(60000, 784) / 255.
    return dataset_tool.DatasetTool(data=x_train, label=y_train, training_flag=True, repeat=False, one_hot_classes=10)
