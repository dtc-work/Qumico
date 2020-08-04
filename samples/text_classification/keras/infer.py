from os import path
import sys
import pandas as pd

from tensorflow.keras import backend as K
from tensorflow.keras.models import model_from_json
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

dir_path = path.dirname(__file__)
if dir_path not in sys.path:
    sys.path.append(dir_path)

from utils import load_tokenizer_keras, CleanTokenize, labeling

TEST_CSV_PATH = path.join(path.dirname(__file__),"data", "test.csv")

def infer(hdf5_path, json_path, tokenizer_path, max_length=10, infer_count=20):

    K.clear_session()

    model = model_from_json(open(json_path).read())
    model.load_weights(hdf5_path)

    # load tokenizer
    tokenizer = load_tokenizer_keras(tokenizer_path)
    
    test = pd.read_csv(TEST_CSV_PATH)
    test_lines = CleanTokenize(test["text"].values.tolist())
    
    test_sequences = tokenizer.texts_to_sequences(test_lines)
    test_review_pad = pad_sequences(test_sequences, maxlen=max_length, padding='post')

    predictions = model.predict(test_review_pad[:infer_count])

    for i, p in enumerate(predictions):
        print("prediction:", labeling(p),'({:.3f})'.format(p[0]),
              " :", test["text"][i].replace("\n",""))


if __name__ == "__main__":
    model_path = path.join(path.dirname(__file__),"model")
    hdf5_path = path.join(model_path, "TweetDisaster.hdf5")
    json_path = path.join(model_path, "TweetDisaster.json")
    tokenizer_path = path.join(model_path, "tokenizer.json")

    infer(hdf5_path, json_path, tokenizer_path, infer_count=100)
