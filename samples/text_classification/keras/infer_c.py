from os import path
import ctypes
import time
import sys

import numpy as np
import pandas as pd

from tensorflow.python.keras.preprocessing.sequence import pad_sequences

dir_path = path.dirname(__file__)
if dir_path not in sys.path:
    sys.path.append(dir_path)

from utils import load_tokenizer_keras, CleanTokenize, labeling

TEST_CSV_PATH = path.join(path.dirname(__file__),"data", "test.csv")
MAX_LENGTH = 10

def init(so_lib_path, input_info, output_info):
    ModelDLL = ctypes.CDLL(so_lib_path)
    ModelDLL.qumico.argtypes = [input_info, output_info]
    ModelDLL.qumico.restype = ctypes.c_int
    ModelDLL.run.argtypes = [input_info, output_info]
    ModelDLL.run.restype = ctypes.c_int
    
    return ModelDLL


def infer_c(hdf5_path, json_path, tokenizer_path, max_length=10, infer_count=20):
    # load tokenizer
    tokenizer = load_tokenizer_keras(tokenizer_path) 
    test = pd.read_csv(TEST_CSV_PATH)
    test_lines = CleanTokenize(test["text"].values.tolist())       
    
    test_sequences = tokenizer.texts_to_sequences(test_lines)
    test_review_pad = pad_sequences(test_sequences, maxlen=MAX_LENGTH, padding='post')    

    # model path
    so_lib_path= path.join(path.dirname(__file__), "out_c", "qumico.so")

    # load & config
    input_info = np.ctypeslib.ndpointer(dtype=np.float32, ndim=2,
                                    shape=(1,10), flags='CONTIGUOUS')
    output_info = np.ctypeslib.ndpointer(dtype=np.float32, ndim=2,
                                    shape=(1,1), flags='CONTIGUOUS')
    dll = init(so_lib_path, input_info, output_info)

    dll.init()
    dll.load_initializers()

    # infer
    for i, text in enumerate(test_review_pad[:infer_count]):
        input = np.ascontiguousarray(np.expand_dims(text, axis=0).astype(np.float32))
        output =np.zeros(dtype=np.float32, shape=(1, 1)) 

        start = time.time()
        dll.run(input, output)
        end = time.time()

        print("prediction:",labeling(output[0][0]),'({:.3f})'.format(output[0][0]),
              " :", test["text"][i].replace("\n",""))
        #print("output", output[0],"elapsed_time:{0}".format(end - start) + "[sec]")

    print("finish")


if __name__ == '__main__':

    # path config
    model_path = path.join(path.dirname(__file__),"model")
    hdf5_path = path.join(model_path, "TweetDisaster.hdf5")
    json_path = path.join(model_path, "TweetDisaster.json")
    tokenizer_path = path.join(model_path, "tokenizer.json")
    
    infer_c(hdf5_path, json_path, tokenizer_path,infer_count=100)

