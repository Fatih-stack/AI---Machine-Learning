import pandas as pd

df = pd.read_csv('Iris.csv')

dataset_x = df.iloc[:, 1:-1].to_numpy(dtype='float32')

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse_output=False)
dataset_y = ohe.fit_transform(df.iloc[:, -1].to_numpy().reshape(-1, 1))

from sklearn.model_selection import train_test_split

training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y, test_size=0.1)

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(training_dataset_x)
scaled_training_dataset_x = ss.transform(training_dataset_x)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

model = Sequential(name='Iris')
model.add(Input((training_dataset_x.shape[1], ), name='Input'))
model.add(Dense(64, activation='relu', name='Hidden-1'))
model.add(Dense(64, activation='relu', name='Hidden-2'))
model.add(Dense(dataset_y.shape[1], activation='softmax', name='Output'))
model.summary()

model.compile('rmsprop', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
hist = model.fit(scaled_training_dataset_x, training_dataset_y, batch_size=32, epochs=100, validation_split=0.2)

import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))
plt.title('Epoch - Loss Graph', fontsize=14, pad=10)
plt.plot(hist.epoch, hist.history['loss'])
plt.plot(hist.epoch, hist.history['val_loss'])
plt.legend(['Loss', 'Validation Loss'])
plt.show()

plt.figure(figsize=(14, 6))
plt.title('Categorcal Accuracy - Validation Categorical Accuracy', fontsize=14, pad=10)
plt.plot(hist.epoch, hist.history['categorical_accuracy'])
plt.plot(hist.epoch, hist.history['val_categorical_accuracy'])
plt.legend(['Categorical Accuracy', 'Validation Categorical Accuracy'])
plt.show()

scaled_test_dataset_x = ss.transform(test_dataset_x)
eval_result = model.evaluate(scaled_test_dataset_x , test_dataset_y, batch_size=32)
for i in range(len(eval_result)):
    print(f'{model.metrics_names[i]}: {eval_result[i]}')

predict_dataset_x = pd.read_csv('predict-iris.csv').to_numpy(dtype='float32')
scaled_predict_dataset_x = ss.transform(predict_dataset_x)

import numpy as np

predict_result = model.predict(scaled_predict_dataset_x)
predict_indexes = np.argmax(predict_result, axis=1)

for pi in predict_indexes:
    print(ohe.categories_[0][pi])

"""
predict_categories = ohe.categories_[0][predict_indexes]
print(predict_categories)
"""
