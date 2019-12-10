import sys
from os import path

dir_path = path.dirname(__file__)
if dir_path not in sys.path:
    sys.path.append(dir_path)


from tiny_yolo_v2_yad2k_model import tiny_yolo_model

MODEL_H5_FNAME  = "tiny_yolo_v2_yad2k.h5"
MODEL_H5_PATH = path.join(path.dirname(__file__), "model", MODEL_H5_FNAME)

MODEL_JSON_FNAME = "tiny_yolo_v2_yad2k.json"
MODEL_JSON_PATH = path.join(path.dirname(__file__), "model", MODEL_JSON_FNAME)

if __name__ == "__main__":
    tiny_yolo_model = tiny_yolo_model()
    tiny_yolo_model.load_weights(MODEL_H5_PATH)

    with open(MODEL_JSON_PATH, 'w') as fp:
        json_string = tiny_yolo_model.to_json()
        fp.write(json_string)

    tiny_yolo_model.summary()
