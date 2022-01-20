# Team Infinity: AI bot that speaks with you when you speek to it.
# Author: Masood Ahmed
# References: Online documentations and youtube videos
# It can only work if you have all the libraries


# Importing all the important libraries
import random
import json
import pickle
import numpy as np
import gtts
from playsound import playsound
import speech_recognition as sr
#from io import BytesIO

# lemmatizer will reduce the word to stem
import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer =  WordNetLemmatizer()

# Loading the training data that is available in the same directory named as intents.json
intents = json.loads(open('./intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.model')

# A function that cleans up e
def cleanUpSentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]

    return sentence_words

def bagOfWords(sentence):
    sentence_words = cleanUpSentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    
    return np.array(bag)

def predictClass(sentence):
    bow = bagOfWords(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key = lambda x: x[1], reverse = True)
    return_list = []
    for r in results:
        return_list.append({'intent' : classes[r[0]], 'probability' : str(r[1])})
    
    return return_list

def getResponse( intents_list, intents_json):
    tag = intents_list[0]["intent"]
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    
    return result



def speak(text):
    voice = gtts.gTTS(text, lang='en')
    voice.save("Dora.mp3")
    playsound("Dora.mp3")



def getAudio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
        said = ""

        try:
            said = r.recognize_google(audio)
            return said
        except Exception as e:
            exception = "Exception: " + str(e)
            return exception
    

print("Everything Running fine")

while True:
    #message = input("")
    message = getAudio()
    ints = predictClass(message)
    res = getResponse(ints, intents)
    #print(res)
    speak(res)



#print("Running")

#print("Done")



