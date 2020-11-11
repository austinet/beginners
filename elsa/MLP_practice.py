# Multilayer Perceptron #
# Available on Numpy #
# Austin Hyeon, 2019. "Sociology meets Computer Science" #

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    c = np.max(x)
    return np.exp(x - c) / np.sum(np.exp(x - c))


def init_param():
    # Initailize parameters 'weight' & 'bias'
    network = {}

    network['W1'] = np.array([[0.2, 0.3, 0.4],
                              [0.2, 0.3, 0.4]], dtype=np.float32)
    network['b1'] = np.array([0.1, 0.1, 0.1], dtype=np.float32)

    network['W2'] = np.array([[0.1, 0.2],
                              [0.1, 0.2],
                              [0.1, 0.2]], dtype=np.float32)
    network['b2'] = np.array([0.2, 0.2], dtype=np.float32)

    network['W3'] = np.array([[0.1, 0.3],
                              [0.1, 0.3]], dtype=np.float32)
    network['b3'] = np.array([0.1, 0.1], dtype=np.float32)

    return network


def neural_network(param, X):
    # Set hyperparameters
    W1, W2, W3 = param['W1'], param['W2'], param['W3']
    b1, b2, b3 = param['b1'], param['b2'], param['b3']

    # Hidden Layer 1
    tmp = np.dot(X, W1) + b1
    node1 = sigmoid(tmp)

    # Hidden Layer 2
    tmp = np.dot(node1, W2) + b2
    node2 = relu(tmp)

    # Output Layer
    tmp = np.dot(node2, W3) + b3
    y = softmax(tmp)

    return y


'''
Operate
'''
param = init_param()

in_tensor = np.array([0.1, 0.9], dtype=np.float32)
out_tensor = neural_network(param, in_tensor)
print(out_tensor)