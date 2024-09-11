import pandas as pd

df = pd.read_csv('IMDB Dataset.csv')

vocab =  set()

import re

for text in df['review']:
    words = re.findall('[a-zA-Z0-9]+', text.lower())
    vocab.update(words)
   
vocab_dict = {word: index for index, word in enumerate(vocab)}

"""
vocab_dict = {} 

for index, word in enumerate(vocab):
    vocab_dict[word] = index
"""  
      
import numpy as np

dataset_x = np.zeros((len(df), len(vocab)), dtype='uint8')  

for row, text in enumerate(df['review']):
    words = re.findall('[a-zA-Z0-9]+', text.lower())
    word_numbers = [vocab_dict[word] for word in words]
    dataset_x[row, word_numbers] = 1

dataset_y = (df['sentiment'] == 'positive').to_numpy(dtype='uint8') 

from sklearn.model_selection import train_test_split

training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y, test_size=0.2)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense

model = Sequential(name='IMDB')

model.add(Input((training_dataset_x.shape[1],)))
model.add(Dense(128, activation='relu', name='Hidden-1'))
model.add(Dense(128, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='sigmoid', name='Output'))
model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['binary_accuracy'])
hist = model.fit(training_dataset_x, training_dataset_y, batch_size=32, epochs=5, validation_split=0.2)

import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))
plt.title('Epoch - Loss Graph', pad=10, fontsize=14)
plt.xticks(range(0, 300, 10))
plt.plot(hist.epoch, hist.history['loss'])
plt.plot(hist.epoch, hist.history['val_loss'])
plt.legend(['Loss', 'Validation Loss'])
plt.show()

plt.figure(figsize=(14, 6))
plt.title('Epoch - Binary Accuracy Graph', pad=10, fontsize=14)
plt.xticks(range(0, 300, 10))
plt.plot(hist.epoch, hist.history['binary_accuracy'])
plt.plot(hist.epoch, hist.history['val_binary_accuracy'])
plt.legend(['Accuracy', 'Validation Accuracy'])
plt.show()

eval_result = model.evaluate(test_dataset_x, test_dataset_y, batch_size=32)

for i in range(len(eval_result)):
    print(f'{model.metrics_names[i]}: {eval_result[i]}')
 
# prediction
predict_df = pd.read_csv('predict-imdb.csv')

predict_dataset_x = np.zeros((len(predict_df), len(vocab)))
for row, text in enumerate(predict_df['review']):
    words = re.findall('[a-zA-Z0-9]+', text.lower())
    word_numbers = [vocab_dict[word] for word in words]
    predict_dataset_x[row, word_numbers] = 1
    
predict_result = model.predict(predict_dataset_x)

for presult in predict_result[:, 0]:
    if (presult > 0.5):
        print('Positive')
    else:
        print('Negative')

# dataset_x'teki birinci yorumun yazı haline getirilmesi
rev_vocab_dict = {index: word for word, index in vocab_dict.items()}

word_indices = np.argwhere(dataset_x[0] == 1).flatten()
words = [rev_vocab_dict[index] for index in word_indices]
text = ' '.join(words)
print(text)
