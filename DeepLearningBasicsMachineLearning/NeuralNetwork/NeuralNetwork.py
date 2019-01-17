'''
Created on 2018��12��21��

@author: TGL
'''
import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1.0 - np.tanh(x)*np.tanh(x)

def logistic(x):
    return 1/(1 + np.exp(-x))

def logistic_derivative(x):
    return logistic(x)*(1 - logistic(x))


class NeuralNetwork:
    def __init__(self, layers, activation = 'tanh'):
        
        
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh'
            self.activation = tanh
            self.activation_deriv = tanh_deriv
        
        self.weights[]
        for i in range(1, len(layers) - 1):
            self.weights.append((2*np.random(layers.[i - 1] + 1, layers[i] + 1)) - 1) * 0.25)
            self.weights.append(2*np.random.random((layers[i] + 1, layers[i + 1])) - 1) * 0.25)
    
    def fit(self, X, y, learning_rate=0.2, epochs=10000):
        X = np.atleast_2d(x)
        temp = np.ones([X.shape[0], X.shape[i]+1])
        temp[:, 0:-1] = X
        X = temp
        y = np.array(y)
        
        for k in range(epochs)
        