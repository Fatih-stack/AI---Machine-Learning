'''
IMDB veri kümesi üzerinde xxx_on_batch metotlariyla parçali biçimde işlemler yapilmasina bir örnek verilmiştir. 
Bu örnekte tüm veri kümesi tek hamlede DataFrame nesnesi olarak okunmuştur. Aslinda burada parçali işlemler için daha önce 
yapmiş olduğmuz işlemlerin uygulanmasi daha uygundur. Ancak biz örneği karmaşik hale getirmemek için tüm veri kümesini tek 
hamlede okuyup xxx_on_batch metotlarina onu parçalara ayirarak verdik.
'''

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, TextVectorization

EPOCHS = 5
BATCH_SIZE = 32
TEST_SPLIT_RATIO = .20
VALIDATION_SPLIT_RATIO = .20

def create_model(df):
    tv = TextVectorization(output_mode='count')
    tv.adapt(df['review'])
    
    model = Sequential(name='IMDB')
    
    model.add(Input((1, ), dtype='string'))
    model.add(tv)
    model.add(Dense(128, activation='relu', name='Hidden-1'))
    model.add(Dense(128, activation='relu', name='Hidden-2'))
    model.add(Dense(1, activation='sigmoid', name='Output'))
    model.summary()
    
    return model
    
def train_test_model(training_df, validation_df, epochs, verbose = 1):
    history_loss = []
    history_accuracy = []
    val_history_loss = []
    val_history_accuracy = []
    
    for epoch in range(epochs):
        training_df = training_df.sample(frac=1)
        
        print('-' * 30)
        
        mean_loss, mean_accuracy = batch_train(training_df, int(np.ceil(len(training_df) / BATCH_SIZE)), model.train_on_batch, verbose=1)
        history_loss.append(mean_loss)
        history_accuracy.append(mean_accuracy)
             
        if verbose == 1:
            print(f'Epoch: {epoch + 1}')       
            print(f'Epoch mean loss: {mean_loss}, Epoch Binary Accuracy: {mean_accuracy}')
            
        val_mean_loss, val_mean_accuracy = batch_train(validation_df, int(np.ceil(len(validation_df) / BATCH_SIZE)), model.test_on_batch)
        
        val_history_loss.append(val_mean_loss)
        val_history_accuracy.append(val_mean_accuracy)
                
        if verbose == 1:
            print(f'Validation Loss: {val_mean_loss}, Validation Binary Accuracy: {val_mean_accuracy}')
        
    return history_loss, history_accuracy, val_history_loss, val_history_accuracy
  
def batch_train(df, nbatches, batch_method, verbose=0):   
    loss_list = []
    accuracy_list = []
    
    for batch_no in range(nbatches):
        x = tf.convert_to_tensor(df['review'].iloc[batch_no * BATCH_SIZE: batch_no * BATCH_SIZE + BATCH_SIZE], dtype='string')
        y = tf.convert_to_tensor(df['sentiment'].iloc[batch_no * BATCH_SIZE: batch_no * BATCH_SIZE + BATCH_SIZE])
        rd = batch_method(x, y, return_dict=True)
        
        loss_list.append(rd['loss'])
        accuracy_list.append(rd['binary_accuracy'])
        
        if verbose:
            print(f'Batch No: {batch_no}')
            if verbose == 2:
                print(f"Batch Loss: {rd['loss']}, Batch Binary Accuracy: {rd['accuracy']}")
               
    mean_loss = np.mean(loss_list)
    mean_accuracy = np.mean(accuracy_list)
      
    return mean_loss, mean_accuracy
  
df = pd.read_csv('IMDB Dataset.csv').iloc[:10000, :]
df['sentiment'] = (df['sentiment'] == 'positive').astype(dtype='uint8')

df = df.sample(frac=1)
test_zone = int(len(df) * (1 - TEST_SPLIT_RATIO))
training_validation_df = df.iloc[:test_zone, :]
test_df = df.iloc[test_zone:, :]
validation_zone = int(len(training_validation_df) * (1 - VALIDATION_SPLIT_RATIO))
training_df = training_validation_df.iloc[:validation_zone, :]
validation_df = training_validation_df.iloc[validation_zone:, :]
    
model = create_model(df)
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['binary_accuracy'])
   
history_loss, history_accuracy, val_history_loss, val_history_accuracy = train_test_model(training_df, validation_df, EPOCHS)
            
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))
plt.title('Epoch - Loss Graph', pad=10, fontsize=14)
plt.xticks(range(0, EPOCHS))
plt.plot(range(EPOCHS), history_loss)
plt.plot(range(EPOCHS), val_history_loss)
plt.legend(['Loss', 'Validation Loss'])
plt.show()

plt.figure(figsize=(14, 6))
plt.title('Epoch - Binary Accuracy Graph', pad=10, fontsize=14)
plt.xticks(range(0, EPOCHS))
plt.plot(range(EPOCHS), history_accuracy)
plt.plot(range(EPOCHS), val_history_accuracy)
plt.legend(['Accuracy', 'Validation Accuracy'])
plt.show()

# evaluation
eval_mean_loss, eval_accuracy = batch_train(test_df, int(np.ceil(len(test_df) / BATCH_SIZE)), model.test_on_batch)   
print(f'Test Loss: {eval_mean_loss}, Test Binary Accuracy: {eval_accuracy}')

# prediction
predict_df = pd.read_csv('predict-imdb.csv') 

for i in range(int(np.ceil(len(predict_df) / BATCH_SIZE))):
    predict_result = model.predict_on_batch(predict_df)
    for presult in predict_result[:, 0]:
        if (presult > 0.5):
            print('Positive')
        else:
            print('Negative')