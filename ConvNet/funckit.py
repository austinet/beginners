# funckit :: activation & loss functions module #
# Based on Numpy #
# Austin Hyeon, 2019. "Sociology meets Computer Science" #

import numpy as np


def identity(x):
    return x


def step(x):
    if x > 0: return 1
    else: return 0


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def relu(x):
    return np.maximum(0, x)
    # if x >= 0: return x
    # else: return 0


def leakyrelu(x, alpha):
    # alpha must be a value between 0 and 1
	return max(alpha * x, x)


def elu(x, alpha):
    if x >= 0: return x
    else: return alpha * (np.exp(x) - 1)


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # to avoid overflow
    return np.exp(x) / np.sum(np.exp(x))


def mean_squared_error(y, t):
    '''
    parameters
    ---
        y: predicted values / np.array()
        t: actual values / np.array(), one-hot encoding required.
    '''
    MSE = 0.5 * np.sum((y - t) ** 2)
    return MSE


def cross_entropy_error(y, t):
    '''
    parameters
    ---
        y: predicted values / np.array()
        t: actual values / np.array(), one-hot encoding required.
    '''
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # if training set is one-hot encoded:
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    CEE = -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

    return CEE