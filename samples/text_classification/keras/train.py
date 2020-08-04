from itertools import chain
from os import path

import numpy as np

from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
import pandas as pd

from samples.text_classification.keras.utils import(CleanTokenize,save_model,
                                              save_tokenizer, save_label_encoder)
from samples.text_classification.keras.model import model_lstm, model_gru,model_rnn


def train_sklearn_encoder(train_csv_path, export_path,
          batch_size=10, epochs=2,
          max_length=10,
          validation_split=0.2):

    data = pd.read_csv(train_csv_path)

    texts = CleanTokenize(data["text"].values.tolist())
    words =list(chain.from_iterable(texts))

    encoder = LabelEncoder()
    encoder.fit(words)
    
    encoded = np.zeros(shape=(len(texts), max_length),dtype=np.int)

    for index,text in enumerate(texts):
        trans = encoder.transform(text)
        encoded[index, :len(trans)] = encoder.transform(text)[:max_length]

    vocab_size = len(encoder.classes_) + 1
    
    # prepare train/test data
    test_labels =  data['target'].values

    indices = np.arange(encoded.shape[0])
    np.random.shuffle(indices)
    review = encoded[indices]
    sentiment = test_labels[indices]
    num_validation_samples = int(validation_split * review.shape[0])

    X_train_pad = review[:-num_validation_samples]
    y_train = sentiment[:-num_validation_samples]
    X_test_pad = review[-num_validation_samples:]
    y_test = sentiment[-num_validation_samples:]

    # train
    model = model_lstm(vocab_size, max_length=max_length)

    model.compile(loss='binary_crossentropy', 
                  optimizer=Adam(1e-4), 
                  metrics=['accuracy'])
    
    model.fit(X_train_pad, y_train, batch_size=batch_size, epochs=epochs, 
              validation_data=(X_test_pad, y_test), verbose=2)

    # save
    save_model(model, output_folder=export_path)
    save_label_encoder(encoder)


def train_keras_tokenizer(train_csv_path, export_path,
          batch_size=10, epochs=2,
          max_length=10,
          validation_split=0.2):
    data = pd.read_csv(train_csv_path)
    texts = CleanTokenize(data["text"].values.tolist())
    # keras
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    sequences_pad = pad_sequences(sequences, maxlen=max_length, padding='post')
    vocab_size = len(tokenizer.word_index) + 1
        
    # prepare train/test data
    test_labels =  data['target'].values

    indices = np.arange(sequences_pad.shape[0])
    np.random.shuffle(indices)
    review = sequences_pad[indices]
    sentiment = test_labels[indices]
    num_validation_samples = int(validation_split * review.shape[0])
    X_train_pad = review[:-num_validation_samples]
    y_train = sentiment[:-num_validation_samples]
    X_test_pad = review[-num_validation_samples:]
    y_test = sentiment[-num_validation_samples:]
    
    # train
    model = model_lstm(vocab_size, max_length=max_length)

    model.compile(loss='binary_crossentropy', 
                  optimizer=Adam(1e-4), 
                  metrics=['accuracy'])
    
    model.fit(X_train_pad, y_train, batch_size=batch_size, epochs=epochs, 
              validation_data=(X_test_pad, y_test), verbose=2)

    # save
    save_model(model, output_folder=export_path)
    save_tokenizer(tokenizer, output_folder=export_path)
    

if __name__ == "__main__":
    DATA_PATH = path.join(path.dirname(__file__),"data")
    TRAIN_CSV_PATH = path.join(DATA_PATH, 'train.csv')
    EXPORT_PATH = path.join(path.dirname(__file__),"model")

    # sklearn 
    train_sklearn_encoder(TRAIN_CSV_PATH, EXPORT_PATH, epochs=5)
    # keras
    # train_keras_tokenizer(TRAIN_CSV_PATH, EXPORT_PATH
    
