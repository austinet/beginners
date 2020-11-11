import sys, os
sys.path.append(os.pardir)
from collections import OrderedDict

import numpy as np
from layerkit import * # neural layers


class SampleNet:
    '''
    the example net for understanding codes
    hidden layers: 2
    '''
    def __init__(self, input_nodes, hidden_nodes, output_nodes, weight_init_std=0.01):
        params = {}
        params['W1'] = weight_init_std * np.random.randn(input_nodes, hidden_nodes)
        params['b1'] = np.zeros(hidden_nodes)
        params['W2'] = weight_init_std * np.random.randn(hidden_nodes, output_nodes)
        params['b2'] = np.zeros(output_nodes)
        # initialize layer classes
        layers = OrderedDict()
        layers['Affine1'] = Affine(params['W1'], params['b1']) # layerkit.Affine
        layers['Relu1'] = Relu() # layerkit.Relu
        layers['Affine2'] = Affine(params['W2'], params['b2']) # layerkit.Affine
        lastlayer = SoftmaxLoss() # layerkit.SoftmaxLoss

        self.params = params
        self.layers = layers
        self.lastlayer = lastlayer


    def predict(self, x):
        # layer = { layers['Affine1'], layers['Relu1'], layers['Affine2'] }
        for layer in self.layers.values():
            x = layer.forward(x)
            # this loop operates like:
            # x = layers['Affine1'].forward(x)
            # x = layers['Relu1'].forward(x)
            # x = layers['Affine2'].forward(x)

        return x


    def loss(self, x, t):
        y = self.predict(x) # return predicted value
        return self.lastlayer.forward(y, t) # calculate loss by layerkit.SoftmaxLoss.forward(y, t)

    
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

        gradients = {}
        gradients['W1'] = derivative(loss_W, self.params['W1'])
        gradients['W2'] = derivative(loss_W, self.params['W2'])
        gradients['b1'] = derivative(loss_W, self.params['b1'])
        gradients['b2'] = derivative(loss_W, self.params['b2'])

        return gradients
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
        dstream = self.lastlayer.backward(dstream)

        layers = list(self.layers.values())
        layers.reverse() # layers = list( layers['Affine2'], layers['Relu1'], layers['Affine1'] )

        # layer = { layers['Affine2'], layers['Relu1'], layers['Affine1'] }
        for layer in layers:
            dstream = layer.backward(dstream)
            # this loop operates like:
            # dstream = layers['Affine2'].backward(dstream)
            # dstream = layers['Relu1'].backward(dstream)
            # dstream = layers['Affine1'].backward(dstream)

        # recording gradient outputs
        gradients = {}
        gradients['W1'] = self.layers['Affine1'].dW
        gradients['b1'] = self.layers['Affine1'].db
        gradients['W2'] = self.layers['Affine2'].dW
        gradients['b2'] = self.layers['Affine2'].db

        return gradients

