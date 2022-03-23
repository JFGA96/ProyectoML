from copyreg import pickle

import pickle
import pandas as pd
import nltk
import pickle
from nltk.corpus import stopwords
import string
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

punctuation = set(string.punctuation)
all_stopwords = stopwords.words('english')
all_stopwords.append('im')
all_stopwords.append('ive')

def Stopwords(sentencia):
  filtered_word = []
  for word in sentencia:
    if word not in all_stopwords:
      filtered_word.append(word)
  return filtered_word

def tokenizacion(sentencia):
    tokens = []
    for token in sentencia.split():
        new_token = []
        for character in token:
            if character not in punctuation:
                new_token.append(character.lower())
        if new_token:
            tokens.append("".join(new_token))
    salidaok=Stopwords(tokens)
    return salidaok
    
def modeloSVC(mensajes):
    print("este es el modelo SVC ")
    for mensaje in mensajes:
        loaded_vectorizer = pickle.load(open('vectorizer', 'rb'))

        # Cargar el modelo
        loaded_model = pickle.load(open('svc_lin_model', 'rb'))

        # hacer prediccion
        respuesta = loaded_model.predict(loaded_vectorizer.transform([mensaje]))
        print("Para el texto : '{}' ".format(str(mensaje)))
        print("su prediccion de 1-5 fue = {}".format(str(respuesta[0])))

def modeloTree(mensajes):
    print("este es el modelo clasification Tree")
    for mensaje in mensajes:
        loaded_vectorizer = pickle.load(open('vectorizer', 'rb'))

        # Cargar el modelo
        loaded_model = pickle.load(open('tree_model', 'rb'))

        # hacer prediccion
        respuesta = loaded_model.predict(loaded_vectorizer.transform([mensaje]))
        print("Para el texto : '{}' ".format(str(mensaje)))
        print("su prediccion de 1-5 fue = {}".format(str(respuesta[0])))

def modeloRandom(mensajes):
    print("este es el modelo RandomForestClassifier")
    for mensaje in mensajes:
        loaded_vectorizer = pickle.load(open('vectorizer', 'rb'))

        # Cargar el modelo
        loaded_model = pickle.load(open('random__model', 'rb'))

        # hacer prediccion
        respuesta = loaded_model.predict(loaded_vectorizer.transform([mensaje]))
        print("Para el texto : '{}' ".format(str(mensaje)))
        print("su prediccion de 1-5 fue = {}".format(str(respuesta[0])))
    

muyBueno = "this is fantastic"
neutro = "I think that it can be bad"
muyMalo = "this product was the worst"

examples = [
    muyBueno,
    neutro,
    muyMalo
]

modeloRandom(examples)
modeloTree(examples)
modeloSVC(examples)