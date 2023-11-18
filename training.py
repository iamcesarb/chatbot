import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer #Para pasar las palabras a su forma raíz

#Para crear la red neuronal
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers.legacy import SGD as sgd_experimental

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '¿', '.', ',']

#Clasifica los patrones y las categorías
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))


training_x = []
training_y = []
output_empty = [0]*len(classes)


for document in documents:
    
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    
    
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    
    # Append each to the corresponding list
    training_x.append(bag)
    training_y.append(output_row)


combined = list(zip(training_x, training_y))
random.shuffle(combined)
training_x[:], training_y[:] = zip(*combined)


training_x = np.array(training_x)
training_y = np.array(training_y)

train_x = training_x

train_y = training_y 




#Creamos la red neuronal
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

#Creamos el optimizador y lo compilamos
sgd = sgd_experimental(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])

#Entrenamos el modelo y lo guardamos
train_process = model.fit(np.array(train_x), np.array(train_y), epochs=1000, batch_size=5, verbose=1)
model.save("chatbot_model.h5", train_process)