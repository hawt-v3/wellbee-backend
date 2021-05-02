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



if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))