# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 15:44:21 2023

@author: Fatih Durmaz
Neuron class
"""

import numpy as np

class Neuron:
    def __init__(self, w, b):
        self.w= w
        self.b = b
        
    def output(self, x, activation):
        return activation(np.dot(x, self.w) + self.b)

def sigmoid(x):
    return np.e ** x / (1 + np.e ** x)
        
w = np.array([1, 2, 3])
b = 1.2

n = Neuron(w, b)

x = np.array([1, 3, 4])
result = n.output(x, sigmoid)
print(result)