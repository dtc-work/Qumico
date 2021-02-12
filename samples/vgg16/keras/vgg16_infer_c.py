
import ctypes
from os import path

import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from samples.vgg16.keras import vgg16_generate_data

def init(so_lib_path, input_info, output_info):
    ModelDLL = ctypes.CDLL(so_lib_path)
    ModelDLL.qumico.argtypes = [input_info, output_info]
    ModelDLL.qumico.restype = ctypes.c_int
    return ModelDLL


classes = vgg16_generate_data.extract_classes


if __name__ == "__main__":
    # read infer image file
    file_name = "test.jpg"
    img_file = path.join(path.dirname(__file__), "test_data", 'flowers', file_name)
    
    img = load_img(img_file, grayscale=False, color_mode='rgb', target_size=(224, 224))
    img = img_to_array(img)

    # load model
    so_lib_path= path.join(path.dirname(__file__), 'out_c', 'qumico.so')

    # Load
    input_info = np.ctypeslib.ndpointer(dtype=np.float32, ndim=4,
                                    shape=(1,224, 224, 3), flags='CONTIGUOUS')

    output_info = np.ctypeslib.ndpointer(dtype=np.float32, ndim=2,
                                    shape=(1,5), flags='CONTIGUOUS')

    dll = init(so_lib_path, input_info, output_info)

    # predict image
    input = np.expand_dims(img, axis=0)
    output =np.zeros(dtype=np.float32, shape=(1,5))
    dll.qumico(input, output)

    result_index = np.argmax(output, axis=-1)
    
    # print out result
    for i in result_index:
        print(classes[i])

    # result
    # tf_infer [[0.14487912 0.0035579  0.00689958 0.06223369 0.7824297 ]] 
    # c_infer  [[0.14487956 0.00355789 0.00689957 0.06223398 0.782429  ]]
    # Daisy
