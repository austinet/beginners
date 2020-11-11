# MLP :: multilayer perceptron module #
# Based on Numpy #
# Austin Hyeon, 2020. "Sociology meets Computer Science" #

import sys, os
sys.path.append(os.pardir)
from collections import OrderedDict

import numpy as np
from layerkit import *

class MLP:
    '''
    Multilayer Perceptron

    parameters
    ---
        depth: the number of hidden and output layers; the input layer falls outside of the count.
        input_nodes: the number of nodes in the input layer
        hidden_nodes: the number of nodes in each of the hidden layers
        output_nodes: the number of nodes in the output layer
    '''

    def __init__(self, depth, input_nodes, hidden_nodes, output_nodes, weight_init_std=0.01):

        ''' variables '''

        self.depth = depth
        params = {}
        layers = OrderedDict()
        W, b, affine, activ = [], [], [], []

        ''' set key names '''

        for i in range(self.depth):
            W.append('W%d' % i)
            b.append('b%d' % i)
            affine.append('affine%d' % i)
            activ.append('activ%d' % i)

        ''' weights & biases '''

        # codes below equals to: params['W0'], params['b0'], params['W1'], params['b1'], ...
        params[W[0]] = weight_init_std * np.random.randn(input_nodes, hidden_nodes)
        params[b[0]] = np.zeros(hidden_nodes)
        for i in range(1, self.depth-1):
            params[W[i]] = weight_init_std * np.random.randn(hidden_nodes, hidden_nodes)
            params[b[i]] = np.zeros(hidden_nodes)
        params[W[-1]] = weight_init_std * np.random.randn(hidden_nodes, output_nodes)
        params[b[-1]] = np.zeros(output_nodes)

        ''' layers '''

        for i in range(self.depth):
            layers[affine[i]] = AffineLayer(params[W[i]], params[b[i]])
            layers[activ[i]] = ReluLayer()
        layers[affine[-1]] = AffineLayer(params[W[-1]], params[b[-1]])
        lastlayer = SoftmaxLossLayer()

        ''' selfify '''

        self.params = params
        self.layers = layers
        self.lastlayer = lastlayer
        self.W, self.b, self.affine, self.activ = W, b, affine, activ


    def predict(self, x):
        for layer in self.layers.values(): # layer <- layers[]
            x = layer.flow(x)

        return x

    
    def loss(self, x, t):
        y = self.predict(x) # return predicted value
        return self.lastlayer.flow(y, t) # calculate loss by layerkit.SoftmaxLoss.flow(y, t)


    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)

        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    '''
    # only for validation
    def derivative(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        for i in range(self.depth):
            grads[self.W[i]] = derivative(loss_W, self.params[self.W[i]])
            grads[self.b[i]] = derivative(loss_W, self.params[self.b[i]])

        return grads
    '''

    
    def echo(self, x, t):
        '''
        echo: operate forward & backprops

        Parameters
        ---
            x: predicted values
            t: actual values for accuracy test
        '''
        # forward propagation
        self.loss(x, t)

        # backpropagation
        dstream = 1
        dstream = self.lastlayer.backflow(dstream)

        layers = list(self.layers.values())
        layers.reverse()

        for layer in layers: # layer <- reversed layers[]
            dstream = layer.backflow(dstream)

        grads = {}
        for i in range(self.depth):
            grads[self.W[i]] = self.layers[self.affine[i]].dW
            grads[self.b[i]] = self.layers[self.affine[i]].db

        return grads
        