from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import numpy as np
import os

from samples.vgg16.keras import vgg16_generate_data


classes = vgg16_generate_data.extract_classes
file_name = 'test.jpg'
root = os.getcwd()
img_file = os.path.join(root, 'test_data', 'flowers', file_name)


def main():
    # load model
    model_folder = 'model'
    json_file = os.path.join(model_folder, 'sample.json')
    h5_file = os.path.join(model_folder, 'sample.hdf5')
    model = model_from_json(open(json_file).read())
    model.load_weights(h5_file)

    img = load_img(img_file, grayscale=False, color_mode='rgb', target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)

    # predict image
    result = model.predict(img, verbose=0)
    result_index = np.argmax(result, axis=-1)

    # print out result
    for i in result_index:
        print(classes[i])


if __name__ == "__main__":
    main()
