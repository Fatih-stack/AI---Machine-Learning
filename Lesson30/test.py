from tensorflow.keras.metrics import categorical_accuracy
import numpy as np

# elma, armut, kayısı
y = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 0]])
y_hat = np.array([[0.2, 0.7, 0.1], [0.2, 0.1, 0.7], [0.8, 0.1, 0.1], [0.7, 0.2, 0.1], [0.6, 0.2, 0.2]])

result = categorical_accuracy(y, y_hat) 
result_ratio = np.sum(result) / len(result)
print(result)


