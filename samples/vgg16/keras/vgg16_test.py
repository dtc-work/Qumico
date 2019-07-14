from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import SGD
import numpy as np
import os

from samples.vgg16.keras import vgg16_generate_data

classes = vgg16_generate_data.extract_classes

# load model
model_folder = 'model'
json_file = os.path.join(model_folder, 'sample.json')
h5_file = os.path.join(model_folder, 'sample.hdf5')
model = model_from_json(open(json_file).read())

# read test image datasets
root = os.getcwd()
test_dir = os.path.join(root, 'test_data', 'flowers')

predict_datagen = ImageDataGenerator(
        rescale=1.0 / 255)

predict_generator = predict_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)


def main():
    model.load_weights(h5_file)
    # predict test data
    result = model.predict_generator(predict_generator, verbose=0, steps=1)
    result_index = np.argmax(result, axis=-1)

    # print out test result
    for i in result_index:
        print('image : ', classes[i])


if __name__ == "__main__":
    main()
