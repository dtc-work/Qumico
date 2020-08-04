import os
import io
import json
import re
import string
import pickle

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from keras_preprocessing.text import tokenizer_from_json

def labeling(value):
    if 0.5 <= value:
        return "Real" 
    else:
        return "Fake"

def save_label_encoder(encoder, output_folder="model"):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    with open(os.path.join(output_folder, 'label_encoder.pickle'), 'wb') as file:
        pickle.dump(encoder, file, pickle.HIGHEST_PROTOCOL)

def load_label_encoder_sklearn(fpath):

    with open(fpath, 'rb') as f:
        return pickle.load(f)  


def save_tokenizer(tokenizer, output_folder="model"):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    tokenizer_json = tokenizer.to_json()
    with io.open(os.path.join(output_folder, 'tokenizer.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))
    print("tokenizer export complete")


def load_tokenizer_keras(fpath):
    with open(fpath) as f:
        data = json.load(f)
    tokenizer = tokenizer_from_json(data)
    return tokenizer



def save_model(model, output_folder="model"):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    json_file = os.path.join(output_folder, 'TweetDisaster.json')
    yaml_file = os.path.join(output_folder, 'TweetDisaster.yaml')
    h5_file = os.path.join(output_folder, 'TweetDisaster.hdf5')

    json_string = model.to_json()
    open(json_file, 'w').write(json_string)
    yaml_string = model.to_yaml()
    open(yaml_file, 'w').write(yaml_string)
    model.save_weights(h5_file)

    print("model export complete")
        

def CleanTokenize(texts):
    tweet_lines = list()
#     lines = df["text"].values.tolist()

    for line in texts:
        line = clean_text(line)
        # tokenize the text
        tokens = word_tokenize(line)
        #print("tokens", tokens)
    #     tokens = [w.lower() for w in tokens]
        # remove puntuations
        table = str.maketrans('', '', string.punctuation)
        
        stripped = [w.translate(table) for w in tokens]
        #print("stripped", stripped)
        # remove non alphabetic characters
        words = [word for word in stripped if word.isalpha()]
        stop_words = set(stopwords.words("english"))

        words = [w for w in words if not w in stop_words]
        tweet_lines.append(words)
    return tweet_lines


def clean_text(text):
    text = text.lower()
    
    pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    text = pattern.sub('', text)
    text = " ".join(filter(lambda x:x[0]!='@', text.split()))
    emoji = re.compile("["
                           u"\U0001F600-\U0001FFFF"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    
    text = emoji.sub(r'', text)
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)        
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text) 
    text = re.sub(r"\'ll", " will", text)  
    text = re.sub(r"\'ve", " have", text)  
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"did't", "did not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"couldn't", "could not", text)
    text = re.sub(r"have't", "have not", text)
    text = re.sub(r"[,.\"\'!@#$%^&*(){}?/;`~:<>+=-]", "", text)
    return text

    