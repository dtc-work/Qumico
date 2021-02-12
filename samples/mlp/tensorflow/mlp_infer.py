import os
from tensorflow import keras
import samples.utils.common_tool as common

if __name__ == '__main__':
    # prepare the infer date 28px * 28px image
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_test = x_test.reshape(10000, 784) / 255.
    image_data = x_test[:10, ...]

    h5_model = os.path.join('model', 'tensorflow_mlp.h5')
    model = keras.models.load_model(h5_model)

    results = model.predict(image_data)

    # output the result
    print('Predict Index ', common.onehot_decoding(results))
