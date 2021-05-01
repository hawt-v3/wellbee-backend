from flask import Flask
import os
from data_preprocess import load_data, lower_case, tokenize_data
from nltk import tokenize
import numpy as np
from gpt import ask
from tensorflow.keras.models import load_model

app = Flask(__name__)


# load our machine learning models
sentiment_model = load_model("models/sentiment_analysis.h5")
emotion_model = load_model("models/emotion_analysis.h5")


# setting up our tokenizer to format our data properly
df = load_data()
sentences = df["sentences"]
sentences = lower_case(sentences)
tokenizer, _, _, _, _ = tokenize_data(sentences)


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

    
    # splitting our paragraph into sentences
    text_sentences = tokenize.sent_tokenize(text)
    
    # converting our text to tokens
    sentiments = []
    for sentence in text_sentences:
        tokenized_sentence = tokenizer.texts_to_sequence(sentence)
        sentiment_prediction = sentiment_prediction.predict(tokenized_sentence)

        sentiment_classes = ["negative", "neutral", "positive"]
        prediction_index = np.argmax(sentiment_prediction)
        sentiments.append(sentiment_classes[prediction_index])
    
    p = 0
    n = 0
    ne = 0
    for sentiment in sentiment_classes:
        if sentiment == "positive":
            p += 1
        elif sentiment == "nuetral":
            n += 1
        elif sentiment == "negative":
            ne += 1

    if len(text_sentences) > 1:
        if n > 2: # set a threshold for negativity in a paragraph
            return {"Sentiment" : "Negative"}
        else:
            if p > ne:
                return {"Sentiment" : "Positive"}
            else:
                return {"Sentiment" : "Neutral"}
    else:
        if n == 1: # set a threshold for negativity in a paragraph
            return {"Sentiment" : "Negative"}
        else:
            if p > ne:
                return {"Sentiment" : "Positive"}
            else:
                return {"Sentiment" : "Neutral"}

    
@app.route("/emotion", methods=["POST"])
def emotion_analysis():
    request_json = flask.request.json
    secret = request_json["secret"]

    if secret != "CONGPilDnoMinEThonYAnkoLViTypOlmStOd":
        return {"code": 401, "error": "Unauthorized"}
    
    text = request_json["text"]
    if text == "":
        return {"code": 401, "error": "Missing text"}
    
        # setting up our tokenizer to format our data properly
    df = load_data()
    sentences = df["sentences"]
    sentences = lower_case(sentences)
    tokenizer, _, _, _, _ = tokenize_data(sentences)
    
    # splitting our paragraph into sentences
    text_sentences = tokenize.sent_tokenize(text)
    
    # converting our text to tokens
    emotions = []
    for sentence in text_sentences:
        tokenized_sentence = tokenizer.texts_to_sequence(sentence)
        emotion_prediction = emotion_model.predict(tokenized_sentence)

        """
        emotion mappings:
            [neutral, joy, sadness, fear, anger,  surprise, disgust, non-neutral]
        """

        emotion_classes = ["neutral", "joy", "sadness", "fear", "anger", "surprise", "disgust", "non-neutral"]
        prediction_index = np.argmax(emotion_prediction)
        emotions.append(emotion_classes[prediction_index])

        n, j, s, f, a, su, d, nn = 0, 0, 0, 0, 0, 0, 0, 0

        for emotion in emotions:
            if emotion == "neutral":
                n += 1
            elif emotion == "joy":
                j += 1
            elif emotion == "sadness":
                s += 1
            elif emotion == "fear":
                f += 1
            elif emotion == "anger":
                a += 1
            elif emotion == "surprise":
                su += 1
            elif emotion == "disgust":
                d += 1
            elif emotion == "non-neutral":
                nn += 1
        
        emotion = max([n, j, s, f, a, su, d, nn])

        return {"emotion" : emotion}


    
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
        i = 0
        for question in before:
            i += 1
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
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))