from flask import Flask
import os
from nltk import tokenize
import numpy as np
from tensorflow.keras.models import load_model
import openai
import pandas as pd
import nltk
import pandas as pd
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 
import json
import pickle
from gpt import ask
import string
import collections
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint
import json
import random
nltk.download('stopwords')
nltk.download('punkt')


def process_text(text, stem=True):
    texts = text.translate(str.maketrans('','',string.punctuation))
    tokens = word_tokenize(texts)

    if not stem:
        return tokens
    else:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]

def cluster_texts(texts, clusters):
    print(texts)
    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
    tfidf = vectorizer.fit_transform(texts)
    print(tfidf)
    kmeans_model = KMeans(n_clusters=clusters)
    kmeans_model.fit(tfidf)

    clustering = collections.defaultdict(list)

    for index, label in enumerate(kmeans_model.labels_):
        clustering[label].append(index)

    return clustering

def grouping():
  with open("bios.json", "r") as f:
    request_json = json.load(f)

  bios = request_json["bios"]

  # process_text(bios)

  clusters = len(bios) / 2
  

  clusters  = cluster_texts(bios, int(clusters))

  pprint(dict(clusters))


app = Flask(__name__)

nltk.download("punkt")

sentiment_model = load_model("sentiment_analysis.h5")
emotion_model = load_model("emotion_analysis.h5")
word_model = load_model("word_analysis.h5")

with open("tokenizer.pickle", "rb") as f:
  sentiment_tokenizer = pickle.load(f)

with open("emotion_tokenizer.pickle", "rb") as f:
  emotion_tokenizer = pickle.load(f)

with open("word_tokenizer.pickle", "rb") as f:
  word_tokenizer = pickle.load(f)

@app.route("/")
def index():
    return "Hello World"

@app.route("/sentiment", methods=["POST"])
def sentiment_analysis():
    request_json = flask.request.json
    secret = request_json["secret"]

    if secret != "CONGPilDnoMinEThonYAnkoLViTypOlmStOd":
        return {"code": 401, "error": "Unauthorized"}
    
    text = request_json["text"]
    if text == "":
        return {"code": 401, "error": "Missing text"}
    
    sentiment_classes = ["negative", "neutral", "positive"]

    # converting our text to tokens
    tokenized_sentences = sentiment_tokenizer.texts_to_sequences([text])

    sentiment_prediction = sentiment_model.predict(tokenized_sentences)
    sentiment = sentiment_classes[np.argmax(sentiment_prediction)]

    tokenized_sentences = word_tokenizer.texts_to_sequences([text])

    sentiment_prediction = word_model.predict(tokenized_sentences)
    word_sentiment = sentiment_classes[np.argmax(sentiment_prediction)]

        
    
    if word_sentiment == sentiment:
        return ({"sentiment" : sentiment})
    elif sentiment == "Positive" or sentiment == "Positive":
        return ({"sentiment" : sentiment})
    else:
        return ({"sentiment" : word_sentiment})


    
@app.route("/emotion", methods=["POST"])
def emotion_analysis():
    request_json = flask.request.json
    secret = request_json["secret"]

    if secret != "CONGPilDnoMinEThonYAnkoLViTypOlmStOd":
        return {"code": 401, "error": "Unauthorized"}
    
    text = request_json["text"]
    if text == "":
        return {"code": 401, "error": "Missing text"}

    with open('emotion_tokenizer.pickle', 'rb') as f:
        emotion_tokenizer = pickle.load(f)


    sentiment_classes = ["negative", "neutral", "positive"]
    # converting our text to tokens
    sentiments = []
    tokenized_sentences = emotion_tokenizer.texts_to_sequences([text])

    sentiment_prediction = emotion_model.predict(tokenized_sentences)
    emotion = sentiment_classes[np.argmax(sentiment_prediction)]

    return ({"emotion" : emotion})



    
@app.route("/gpt3-response", methods=["POST"])
def gpt3_responses():
    request_json = flask.request.json
    secret = request_json["secret"]

    if secret != "CONGPilDnoMinEThonYAnkoLViTypOlmStOd":
        return {"code": 401, "error": "Unauthorized"}

    user_inputs = request_json["inputs"]
    bot_outputs = ["outputs"]

    if len(user_inputs) > 1:
        before = user_inputs[:-1]
        current = user_inputs[-1]
        chatlog = """Human: Hey, I am looking for some advice. Any chance you can help?
AI: Yes, I can definitely help you. What's on your mind? 
"""         

        print(before)

        i = 0
        for question in before:
            i += 1
            if i != len(before):
              chatlog += f"""Human: {question}\nAI: {bot_outputs[i]}\n"""
        
        response = ask(current, chatlog)

        return {"response" : response}
        
    elif len(user_inputs) == 1:
        current = user_inputs[0]

        chatlog = """Human: Hey, I am looking for some advice. Any chance you can help?
AI: Yes, I can definitely help you. What's on your mind? 
"""  
        response = ask(current, chatlog)

        return {"response" : response}
    
    else:
        return {"code" : 401, "error" : "request missing arguments"}


@app.route("/group-profiles", methods=["POST"])
def grouping():
    request_json = flask.request.json
    
    user = request_json["user"]
    all_users = request_json["allUsers"]

    ids = []
    bios = []

    for users in all_users:
        ids.append(users["id"])
        bios.append(users["bio"])
    

    ids.append(user["id"])
    bios.append(user["bio"])
    to_predict = user["id"]
    to_predict = ids.index(to_predict)

    print(ids)

    clusters = len(bios) / 2


    clusters  = cluster_texts(bios, int(clusters))

    clusters = dict(clusters)


    for item in clusters:
        for value in clusters[item]:
          if value == to_predict:
              to_predict_key = item
          else:
              pass

    profile_index = random.choice(clusters[to_predict_key])

    while profile_index == to_predict:
        profile_index = random.choice(clusters[to_predict_key])


    profile_id = ids[profile_index]


    return {"id" : profile_id}




if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))