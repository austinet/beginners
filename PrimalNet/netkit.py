# netkit :: neural networks module #
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
        input_nodes: the number of nodes in the input layer
        hidden_nodes: the number of nodes in each of the hidden layers. Must be a 1D array.
        output_nodes: the number of nodes in the output layer
        activ: the activation functions
        weight_init_std: the standard deviation of random weight values
        use_batch_norm: bool type. the batch normalization layer
        use_dropout: bool type. applying 50% dropout
    '''

    def __init__(self, input_nodes, hidden_nodes, output_nodes, activ, 
                        weight_init_std=0.01, weight_decay_lambda=0, use_batch_norm=False, use_dropout=False):

        nodes_list = [input_nodes] + hidden_nodes + [output_nodes]
        self.nodes_list = nodes_list
        self.hidden_nodes = hidden_nodes
        self.weight_decay_lambda = weight_decay_lambda
        self.depth = len(nodes_list)
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout

        ''' initialize weights & biases '''

        params = self.__init_params(weight_init_std)
        self.params = params

        ''' initialize layers '''

        layers, lastlayer = self.__init_layers(activ)
        self.layers, self.lastlayer = layers, lastlayer


    def __init_params(self, weight_init_std):

        ''' initialize weights & biases '''

        params = {}

        for i in range(1, self.depth):

            # ===== initializing methods for weights ======
            if str(weight_init_std).lower() in ('he'):
                scale = np.sqrt(2.0 / self.nodes_list[i - 1])  # He method for ReLU
            elif str(weight_init_std).lower() in ('xavier'):
                scale = np.sqrt(1.0 / self.nodes_list[i - 1])  # Xavier method for sigmoid or tanh
            else:
                scale = weight_init_std
            # =============================================

            # set weights
            params['W' + str(i)] = scale * np.random.randn(self.nodes_list[i-1], self.nodes_list[i])
            # set biases
            params['b' + str(i)] = np.zeros(self.nodes_list[i])

        # the FOR statement equals to:

        # if self.depth = nodes_list = 3
        # for i in range(1, self.depth):
            # params['W1'] = ...randn(nodes_list[0], nodes_list[1])
            # params['b1'] = np.zeros(nodes_list[1])
            # params['W2'] = ...randn(nodes_list[1], nodes_list[2])
            # params['b2'] = np.zeros(nodes_list[2])

        return params


    def __init_layers(self, activ):

        ''' initialize layers '''

        layers = OrderedDict()
        activ_switcher = {'sigmoid': SigmoidLayer(), 'tanh': HyperTangentLayer(), \
            'relu': ReluLayer(), 'leakyrelu': LeakyReluLayer(), 'elu': EluLayer()}

        for i in range(1, self.depth-1):

            # set affine layers ======================
            layers['affine' + str(i)] = AffineLayer(self.params['W' + str(i)], self.params['b' + str(i)])
            # ========================================

            # set batch normalization layers =========
            if self.use_batch_norm == True:
                self.params['gamma' + str(i)] = np.ones(self.hidden_nodes[i-1])
                self.params['beta' + str(i)] = np.zeros(self.hidden_nodes[i-1])
                layers['batchnorm' + str(i)] = BatchNormLayer(self.params['gamma' + str(i)], self.params['beta' + str(i)])
            # ========================================

            # set activation layers ==================
            layers['activ' + str(i)] = activ_switcher[activ.lower()]
            # ========================================

            # applying dropout to each layer =========
            if self.use_dropout:
                layers['dropout' + str(i)] = Dropout()
            # ========================================

        ''' initialize THE LAST LAYER of the net '''

        # set the last affine layer
        i = self.depth-1
        layers['affine' + str(i)] = AffineLayer(self.params['W' + str(i)], self.params['b' + str(i)])
        # set the last activation layer
        lastlayer = SoftmaxLossLayer()

        # the FOR statement equals to:

        # if self.depth-1 = nodes_list-1 = 2
        # for i in range(1, self.depth):
            # layers['affine1'] = params['W1'], params['b1']
            # if: layers['batchnorm1'] = batchparams['gamma1'], batchparams['beta1']
            # layers['activ1'] = ActivLayer()
            # if: layers['dropout1'] = Dropout()
        # i = self.depth-1
        # layers['affine2'] = params['W2'], params['b2']
        # lastlayer = SotfmaxLossLayer()

        return layers, lastlayer
        

    def predict(self, x, train_flg=False):
        for key, layer in self.layers.items(): # layer <- layers[]
            if 'dropout' in key or 'batchnorm' in key:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)

        return x

    
    def get_loss(self, x, t, train_flg=False):
        y = self.predict(x, train_flg) # return predicted value

        weight_decay = 0
        for i in range(1, self.depth):
            W = self.params['W' + str(i)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)

        return self.lastlayer.forward(y, t) + weight_decay # calculate loss by layerkit.SoftmaxLoss.forward(y, t)


    def get_accuracy(self, x, t):
        y = self.predict(x, train_flg=False)
        y = np.argmax(y, axis=1)

        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    
    def echo(self, x, t):
        '''
        echo: operate forward & backprops \n
        the main trigger of the net class

        Parameters
        ---
            x: predicted values
            t: actual values for accuracy test
        '''
        # forward propagation
        self.get_loss(x, t, train_flg=True)

        # backpropagation
        dstream = self.lastlayer.backward()

        layers = list(self.layers.values())
        layers.reverse()

        for layer in layers: # layer <- reversed layers[]
            dstream = layer.backward(dstream)

        # record all gradients of weights & biases
        grads = {}
        for i in range(1, self.depth):
            grads['W' + str(i)] = self.layers['affine' + str(i)].dW + \
                self.weight_decay_lambda * self.layers['affine' + str(i)].W
            grads['b' + str(i)] = self.layers['affine' + str(i)].db

            if self.use_batch_norm == True and i != self.depth-1:
                grads['gamma' + str(i)] = self.layers['batchnorm' + str(i)].dgamma
                grads['beta' + str(i)] = self.layers['batchnorm' + str(i)].dbeta

        return grads
        