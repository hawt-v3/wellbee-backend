import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_data():
    df = pd.read_csv("data/data.csv")
    return df

def lower_case(text_list):
    lower_phrases = []
    for phrase in text_list:
        lower_phrases.append(phrase.lower())

    return lower_phrases

def tokenize_data(text_list, num_words=1000, oov_token="<UNK>", pad_type="post", trunc_type="post"):
    tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
    tokenizer.fit_on_texts(text_list)
    word_index = tokenizer.word_index
    train_sequences = tokenizer.texts_to_sequences(text_list)




    maxlen = max([len(x) for x in train_sequences])

    train_padded = pad_sequences(train_sequences, padding=pad_type, truncating=trunc_type, maxlen=maxlen)



    return tokenizer, train_sequences, word_index, maxlen, train_padded