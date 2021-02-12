import os
import numpy as np
from tensorflow import keras
import samples.utils.common_tool as common

if __name__ == '__main__':
    num_classes = 10
    input_shape = (28, 28, 1)

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_test = x_test[:10, ...]

    # Scale images to the [0, 1] range
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_test = np.expand_dims(x_test, -1)
    image_data = x_test

    h5_model = os.path.join('model', 'tensorflow_conv.h5')
    model = keras.models.load_model(h5_model)

    results = model.predict(image_data)
    # output the result
    print('Predict Index ', common.onehot_decoding(results))
