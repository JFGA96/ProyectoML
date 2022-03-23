
import pandas as pd
import nltk
import pickle
from nltk.corpus import stopwords
import string
from nltk.stem import WordNetLemmatizer
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

lemmatizer = WordNetLemmatizer()

punctuation = set(string.punctuation)
all_stopwords = stopwords.words('english')
all_stopwords.append('im')
all_stopwords.append('ive')

#def models(X_train,Y_train,X_test,Y_test):
  
  

#Eliminar palabras comunes
def Stopwords(sentencia):
  filtered_word = []
  for word in sentencia:
    if word not in all_stopwords:
      filtered_word.append(word)
  return filtered_word

#dividir frases en palabras por tokens
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

#lemmatizar o quitar plurales de palabras o terminaciones 
def lematizar(sentencia):
  lema_word=[]
  for word in sentencia:
    palabra=lemmatizer.lemmatize(word)
    lema_word.append(palabra)
  return lema_word

 
df = pd.read_csv("Reviews.csv", sep=',')
 #print(df)

 #verificar si hay valores duplicados que cumplan con la mayoria de las columnas
df = df.drop_duplicates(df.columns[~df.columns.isin(['UserId','ProductId', 'Profilename', 'Time', 'Score'])],
                        keep='first')

df_check=df.isna()


#Eliminar valores nulos
check_for_any_nan= df.isna().values.any()
check_for_any_nan= df.isna().any().any()
total_nan_values = df.isna().sum().sum()
print(df_check)
print("NaN Presence:"+str(check_for_any_nan))
print ("Total Number of NaN values:"+str(total_nan_values))


#Eliminar valores duplicados
df=df.dropna()
print(len(df))


#Unificar datos a los Score
cuenta = df['Score'].value_counts()
minimo = min(cuenta)
df1=df[df.Score == 5]
df1=df1.sample(minimo)

valor4=df[df.Score == 4]
valor4=valor4.sample(minimo)

valor3=df[df.Score == 3]
valor3=valor3.sample(minimo)

valor2=df[df.Score == 2]
valor2=valor2.sample(minimo)

valor1=df[df.Score == 1]
valor1=valor1.sample(minimo)
#unirtodo nuevamente en dataframe
df1 = pd.concat([valor1, valor2, valor3, valor4, df1])
df_aleatorio=df1.sample(frac=1).reset_index(drop=True)

#Realizar una particion del 0.7 train y 0.3 test
from sklearn.model_selection import train_test_split
train_text,test_text, train_labels, test_labels = train_test_split(df_aleatorio["Text"], df_aleatorio["Score"], stratify=df_aleatorio["Score"],train_size=0.7, random_state=True)
print(f"Training examples: {len(train_text)}, testing examples {len(test_text)}")



from sklearn.feature_extraction.text import CountVectorizer
#Realizar vectorizado
real_vectorizer = CountVectorizer(tokenizer = tokenizacion, binary=False)

X_train = real_vectorizer.fit_transform(train_text)
X_test = real_vectorizer.transform(test_text)

print(X_train)

""" vec_file = 'vectorizer'
pickle.dump(real_vectorizer, open(vec_file, 'wb'))
Y_train = train_labels
Y_test = test_labels

#Usar el algoritmo Support Vector Machine Algorithm
from sklearn.svm import SVC
svc_lin = LinearSVC(max_iter=10000)
svc_lin.fit(X_train, Y_train)
#rendimiento de clasificador en testeo
predicciones = svc_lin.predict(X_test)
accuracy3 = accuracy_score(Y_test, predicciones)
print(f"Accuracy svc lin: {accuracy3:.4%}")


#Usar el algoritmo Decision Tree Algorithm
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
tree.fit(X_train, Y_train)
#rendimiento de clasificador en testeo
predicciones = tree.predict(X_test)
accuracy6 = accuracy_score(Y_test, predicciones)
print(f"Accuracy Desicion Tree: {accuracy6:.4%}")

#usar el algorimo Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
forest.fit(X_train, Y_train)
#rendimiento de clasificador en testeo
predicciones = forest.predict(X_test)
accuracy7 = accuracy_score(Y_test, predicciones)
print(f"Accuracy Random Forest: {accuracy7:.4%}") 

#Rendimiento de clasificador en entrenamiento
print('[1]Support Vector Machine (Linear Classifier) Training Accuracy:', svc_lin.score(X_train, Y_train))
print('[2]Decision Tree Classifier Training Accuracy:', tree.score(X_train, Y_train))
print('[3]Random Forest Classifier Training Accuracy:', forest.score(X_train, Y_train)) """


#Guardar modelos
""" mod_file = 'svc_lin_model1'
pickle.dump(svc_lin , open(mod_file, 'wb'))

mod_file = 'tree_model1'
pickle.dump(tree, open(mod_file, 'wb'))

mod_file = 'random__model1'
pickle.dump(forest, open(mod_file, 'wb'))
 """






