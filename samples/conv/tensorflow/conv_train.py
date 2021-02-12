from tensorflow import keras
import os
import numpy as np
import onnx
import keras2onnx
import qumico

if __name__ == '__main__':
    # prepare the train date
    num_classes = 10
    input_shape = (28, 28, 1)

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            keras.layers.Conv2D(32, kernel_size=(5, 5), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(1024, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(num_classes, activation="softmax")
        ]
    )
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

    model_folder = 'model'
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)
    h5_path = os.path.join(model_folder, 'tensorflow_conv.h5')
    model.save(h5_path)

    onnx_folder = 'onnx'
    if not os.path.exists(onnx_folder):
        os.mkdir(onnx_folder)
    onnx_path = os.path.join(onnx_folder, 'tensorflow_conv.onnx')
    onnx_model = keras2onnx.convert_keras(model, 'tensorflow_conv', target_opset=qumico.SUPPORT_ONNX_OPSET)
    onnx.save_model(onnx_model, onnx_path)

    print('h5ファイルを生成しました。出力先:', h5_path)
    print('onnxファイルを生成しました。出力先:', onnx_path)


