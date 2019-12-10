import numpy as np
import os
import sys
import tensorflow as tf
from PIL import Image
from datetime import datetime

from samples.mobilenet.tensorflow.mobilenet_common import WIDTH, HEIGHT, LABELS


def infer(tflite_model_path, image_path):

    image = Image.open(image_path)
    resized_image = image.resize((WIDTH, HEIGHT), Image.BICUBIC)

    image_data = np.array(resized_image, dtype='uint8')
    inputs = np.ascontiguousarray(np.expand_dims(image_data, axis=0))

    # infer
    print("init:start", datetime.now())
    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
    interpreter.allocate_tensors()
    print("load:start", datetime.now())
    interpreter.set_tensor(interpreter.get_input_details()[0]["index"], inputs)
    print("run:start", datetime.now())
    interpreter.invoke()
    predictions = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
    print("run:end", datetime.now())
    np.set_printoptions(threshold=sys.maxsize)
    label_index = np.argmax(predictions)
    print(LABELS[label_index])


def main():
    infer(tflite_model_path="./model/mobilenet_v1_0.25_128_quant.tflite",
          image_path=os.path.join(os.path.dirname(__file__), "images", "tiger.jpeg"))


if __name__ == "__main__":
    main()
