import os
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from samples.utils.dataset_tool import DatasetTool
import onnx
import keras2onnx
import qumico


if __name__ == '__main__':
    # prepare the train date
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784) / 255.
    dataset_train = DatasetTool(data=x_train, label=y_train, training_flag=True, repeat=False, one_hot_classes=10)

    model = keras.Sequential(
        [
            keras.layers.Flatten(input_shape=[28, 28], name='input'),
            keras.layers.Dense(100, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ]
    )
    model.summary()
    model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=30, validation_data=(x_test, y_test))

    h5_path = os.path.join('model', 'tensorflow_mlp.h5')
    model.save(h5_path)

    onnx_path = os.path.join('onnx', 'tensorflow_mlp.onnx')
    onnx_model = keras2onnx.convert_keras(model, 'tensorflow_mlp', target_opset=qumico.SUPPORT_ONNX_OPSET)
    onnx.save_model(onnx_model, onnx_path)

    print('h5ファイルを生成しました。出力先:', h5_path)
    print('onnxファイルを生成しました。出力先:', onnx_path)

