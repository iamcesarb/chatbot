import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from keras.models import load_model

#import rospy
import openai
import os
import requests






def n_context_response(pre)->str:
    openai.api_key = "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"
    pregunta = pre + ' respondeme de manera corta como si fuera una conversacion'
    res = openai.Completion.create(engine = "text-davinci-002", prompt = pregunta, max_tokens = 35)
    return res["choices"][0]["text"]



lemmatizer = WordNetLemmatizer()

#Importamos los archivos generados en el código anterior
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

#Pasamos las palabras de oración a su forma raíz
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

#Convertimos la información a unos y ceros según si están presentes en los patrones
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i]=1
    return np.array(bag)

#Predecimos la categoría a la que pertenece la oración
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    max_index = np.where(res ==np.max(res))[0][0]
    category = classes[max_index]
    prob = np.max(res) 
    print('Categoria =', category, 'Probability = ', np.max(res))
    return category, prob

#Obtenemos una respuesta aleatoria
def get_response(tag, intents_json):
    list_of_intents = intents_json['intents']
    result = ""
    for i in list_of_intents:
        if i["tag"]==tag:
            result = random.choice(i['responses'])
            break
    return result

threshold = 0.4
#Ejecutamos el chat en bucle
while True:
    message=input(">>")
    ints, prob = predict_class(message)
    if prob > threshold:
        res = get_response(ints, intents)
    else:
        res = n_context_response(message)
    print(res)
    