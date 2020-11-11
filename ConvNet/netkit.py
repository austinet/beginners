# netkit :: neural networks module #
# Based on Numpy #
# Austin Hyeon, 2020. "Sociology meets Computer Science" #

import sys, os
sys.path.append(os.pardir)
from collections import OrderedDict
import pickle

import numpy as np
from layerkit import *


class ConvNet:
    '''
    a simple convolutional net for sketch

    '''
    def __init__(self, input_dim=(1, 28, 28), 
                    conv_params={'filter_num': 30, 'filter_size': 5, 'padding': 0, 'stride': 1},
                    hidden_size=100, output_size=10, weight_init_std=0.01):
        
        ''' basic variables '''

        filter_num = conv_params['filter_num']
        filter_size = conv_params['filter_size']
        filter_padding = conv_params['padding']
        filter_stride = conv_params['stride']
        input_size = input_dim[1] # input_dim[1] = 28
        conv_output_size = (input_size - filter_size + (2 * filter_padding)) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size / 2) * (conv_output_size / 2))

        ''' filter :: weights and biases '''

        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)
        self.params['gamma1'] = np.ones(hidden_size)
        self.params['beta1'] = np.zeros(hidden_size)

        ''' layers '''

        self.layers = OrderedDict()

        # the 1st layer :: convolutional
        self.layers['conv1'] = CONVOLUTIONAL(self.params['W1'], self.params['b1'],
                                                conv_params['stride'], conv_params['padding'])
        self.layers['relu1'] = RELU()
        self.layers['maxpool1'] = MAXPOOLING(pool_h=2, pool_w=2, stride=2)

        # the 2nd layer :: affine
        self.layers['affine1'] = AFFINE(self.params['W2'], self.params['b2'])
        self.layers['batchnorm1'] = BATCHNORM(self.params['gamma1'], self.params['beta1'])
        self.layers['relu2'] = RELU()
        self.layers['dropout1'] = DROPOUT(0.5)

        # the last layer :: get losses
        self.layers['affine2'] = AFFINE(self.params['W3'], self.params['b3'])
        self.lastlayer = SOFTMAXLOSS()

    
    def predict(self, x, train_flg=False):

        ''' activate all layers.forward() in order '''

        for layer in self.layers.values():
            if isinstance(layer, DROPOUT):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

        return x


    def get_loss(self, x, t):

        ''' activate lastlayer.forward() '''

        y = self.predict(x, train_flg=True)
        return self.lastlayer.forward(y, t)


    def get_accuracy(self, x, t, batch_size=100):

        ''' get accuracies '''

        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size: (i+1)*batch_size]
            tt = t[i*batch_size: (i+1)*batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]


    def echo(self, x, t):

        ''' the main trigger of the neural net class '''

        # forward
        self.get_loss(x, t)

        # backprop
        dstream = self.lastlayer.backward()

        layers = list(self.layers.values())
        layers.reverse()
        
        ''' activate layers.backward() in reversed order '''

        for layer in layers:
            dstream = layer.backward(dstream)

        ''' record parameters '''
        
        grads = {}
        grads['W1'], grads['b1'] = self.layers['conv1'].dW, self.layers['conv1'].db
        grads['W2'], grads['b2'] = self.layers['affine1'].dW, self.layers['affine1'].db
        grads['W3'], grads['b3'] = self.layers['affine2'].dW, self.layers['affine2'].db
        grads['gamma1'], grads['beta1'] = \
            self.layers['batchnorm1'].dgamma, self.layers['batchnorm1'].dbeta

        return grads


    def pkl_save_params(self, pickle_file='records.pkl'):

        ''' save the last optimal parameters in a pickle file '''

        params = {} # weights & biases
        for key, val in self.params.items():
            params[key] = val

        with open(pickle_file, 'wb') as f:
            pickle.dump(params, f)


    def pkl_load_params(self, pickle_file='records.pkl'):

        ''' load the last optimal parameters in a pickle file '''   

        with open(pickle_file, 'rb') as f:
            params = pickle.load(f) # weights & biases
        
        for key, val in params.items():
            self.params[key] = val
        
        for i, key in enumerate(['conv1', 'affine1', 'affine2']):
            self.layers[key].W = self.params['W' + str(i+1)]
            self.layers[key].b = self.params['b' + str(i+1)]
            
        self.layers['bachnorm1'].dgamma = self.params['gamma1']
        self.layers['bachnorm1'].dbeta = self.params['beta1']