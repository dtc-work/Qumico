import tensorflow
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, GRU, LSTM, SimpleRNN

def model_lstm(vocab_size, embedding_dim=120, max_length=10):
    
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
    model.add(LSTM(units=32,  dropout=0.4, recurrent_dropout=0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    return model

def model_gru(vocab_size, embedding_dim=120, max_length=10):

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
    model.add(GRU(units=32,  dropout=0.4, recurrent_dropout=0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    return model

def model_rnn(vocab_size, embedding_dim=120, max_length=10):

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
    model.add(SimpleRNN(units=32,  dropout=0.4, recurrent_dropout=0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    return model


if __name__ == "__main__":
    model = model_rnn(14345,100)
    print(model)
