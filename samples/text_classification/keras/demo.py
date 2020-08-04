import json
import os
import sys
import time
import numpy as np
import ctypes
import cv2
import random

MAX_LENGTH = 10

def labeling(value):
    if 0.4 <= value:
        return "Real"
    else:
        return "Fake"

def init_dll(so_lib_path, input_info, output_info):
    ModelDLL = ctypes.CDLL(so_lib_path)
    ModelDLL.qumico.argtypes = [input_info, output_info]
    ModelDLL.qumico.restype = ctypes.c_int
    ModelDLL.run.argtypes = [input_info, output_info]
    ModelDLL.run.restype = ctypes.c_int

    return ModelDLL

def init():

    so_lib_path = os.path.join('out_c', 'qumico.so')

    # load & config
    input_info = np.ctypeslib.ndpointer(dtype=np.float32, ndim=2,
                                        shape=(1, 10), flags='CONTIGUOUS')
    output_info = np.ctypeslib.ndpointer(dtype=np.float32, ndim=2,
                                         shape=(1, 1), flags='CONTIGUOUS')
    dll = init_dll(so_lib_path, input_info, output_info)

    dll.init()
    dll.load_initializers()

    return dll


def load_dictionary():

    json_file = os.path.join('model', 'tokenizer.json')

    with open(json_file) as fp:
        dic = json.load(fp)
        dic2 = json.loads(dic)

    print(len(dic2), type(dic2))

    print(dic2["config"]["index_word"][1])
    obj = json.loads(dic2["config"]["word_index"])

    return obj


def infer(dll, enc_text):
    # infer
    input = np.ascontiguousarray(np.expand_dims(enc_text, axis=0).astype(np.float32))
    output = np.zeros(dtype=np.float32, shape=(1, 1))

    start = time.time()
    dll.run(input, output)
    end = time.time()

    return output[0][0], end - start

def encode(dict, text):
    enc = [0] * MAX_LENGTH
    text = text.lower()

    #特殊記号を削除
    for c in ",.?:/":
        text = text.replace(c, " ")

    words = text.split(" ")

    si = 0
    for word in words:
        word_index = int(dict.get(word, "0"))
        if word_index == 0:
            continue

        enc[si] = word_index
        si = si + 1
        if si == MAX_LENGTH:
            break

    print(enc)

    return enc


def main():
    dll = init()
    dict = load_dictionary()

    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    cv2.moveWindow("result", 80, 80)

    imr = cv2.imread(os.path.join(os.path.dirname(__file__), "images", "real.jpg"), cv2.IMREAD_COLOR)

    imf = cv2.imread(os.path.join(os.path.dirname(__file__), "images", "fake.jpg"), cv2.IMREAD_COLOR)


    while(True):
        text = sys.stdin.readline()

        if text == ".\n":
            break
        enc_text = encode(dict, text)
        rate, run_time = infer(dll, enc_text)
        print(labeling(rate), rate, text,  run_time, "sec")

        if labeling(rate) == "Real":
            cv2.resizeWindow("result", imr.shape[1], imr.shape[0])
            cv2.imshow("result", imr)
            cv2.waitKey(1)
            cv2.imshow("result", imr)
        else:
            cv2.resizeWindow("result", imf.shape[1], imf.shape[0])
            cv2.imshow("result", imf)
            cv2.waitKey(1)
            cv2.imshow("result", imf)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    print("finish.")

if __name__ == '__main__':
    main()