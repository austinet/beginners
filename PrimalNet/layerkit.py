# layerkit :: neural layers module #
# Based on Numpy #
# Austin Hyeon, 2020. "Sociology meets Computer Science" #

'''
layers for common use
---
    1. AffineLayer
    2. SigmoidLayer
    3. HyperTangentLayer
    4. ReluLayer
    5. LeakyReluLayer
    6. EluLayer
    7. SoftmaxLossLayer
    8. BatchNormLayer
    9. Dropout
'''

import numpy as np
from funckit import *


class AffineLayer:

    ''' affine layer: fully connected neural strands '''

    def __init__(self, W, b):
        self.W = W # weights
        self.b = b # biases
        self.x = None # input tensors
        self.original_x_shape = None
        self.dW = None # derivative of weights
        self.db = None # derivative of biases

    
    # x: predicted values
    def forward(self, x):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        result = np.dot(x, self.W) + self.b
        return result


    # dstream: stream of derivatives from SoftmaxLossLayer
    def backward(self, dstream):
        dx = np.dot(dstream, self.W.T)
        self.dW = np.dot(self.x.T, dstream)
        self.db = np.sum(dstream, axis=0)

        dx = dx.reshape(*self.original_x_shape)
        return dx


class SigmoidLayer:

    ''' sigmoid layer '''

    def __init__(self):
        self.dup = None


    # x: predicted values
    def forward(self, x):
        y = sigmoid(x)
        self.dup = y.copy()

        return y


    # dstream: stream of derivatives from SoftmaxLossLayer
    def backward(self, dstream):
        y = self.dup
        dx = dstream * (y * (1 - y))
        
        return dx


class HyperTangentLayer:

    ''' tanh, hyperbolic tagent layer '''

    def __init__(self):
        self.dup = None


    # x: predicted values
    def forward(self, x):
        y = tanh(x)
        self.dup = y.copy()

        return y


    # dstream: stream of derivatives from SoftmaxLossLayer
    def backward(self, dstream):
        y = self.dup
        dx = dstream * (1 - np.power(y, 2))
        
        return dx


class ReluLayer:

    ''' ReLU, rectified linear unit layer '''

    def __init__(self):
        self.tag = None 


    # x: predicted values
    def forward(self, x):
        self.tag = (x <= 0) # tag: tagging values under 0 as 'True' in np.array
        y = x.copy() # copy input tensor 'x' to 'result'
        y[self.tag] = 0 # return True-tagged values into '0' in 'result'

        return y


    # dstream: stream of derivatives from SoftmaxLossLayer
    def backward(self, dstream):
        dstream[self.tag] = 0 # return True-tagged values into '0' in 'dstream'
        dx = dstream

        return dx


class LeakyReluLayer:
    '''
    leakyReLU layer

    parameter
    ---
        alpha: default is 0.01
    '''

    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.tag = None


    # x: predicted values
    def forward(self, x):
        self.tag = (x <= 0)
        y = x.copy()
        y[self.tag] = y[self.tag] * self.alpha

        return y


    # dstream: stream of derivatives from SoftmaxLossLayer
    def backward(self, dstream):
        dstream[self.tag] = dstream[self.tag] * self.alpha
        dx = dstream

        return dx


class EluLayer:
    '''
    ELU, exponential linear unit layer

    parameter
    ---
        alpha: default is 0.01
    '''

    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.tag = None


    # x: predicted values
    def forward(self, x):
        self.tag = (x <= 0)
        y = x.copy()
        y[self.tag] = self.alpha * (np.exp(y[self.tag]) - 1)

        return y


    # dstream: stream of derivatives from SoftmaxLossLayer
    def backward(self, dstream):
        dstream[self.tag] =self.alpha * (np.exp(dstream[self.tag]) - 1)
        dx = dstream

        return dx


class SoftmaxLossLayer:

    ''' softmax-with-loss layer '''
    
    def __init__(self):
        self.y = None # predicted values
        self.t = None # actual values


    def forward(self, x, t):
        '''
        parameters
        ---
            x: predicted values
            t: actual values
        '''
        self.t = t
        self.y = softmax(x)
        loss = cross_entropy_error(self.y, self.t)

        return loss

    
    # generate streams of derivatives
    def backward(self):
        batch_size = self.t.shape[0]

        # if "t" is one-hot encoded
        if self.t.size == self.y.size:
            dstream = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dstream = dx / batch_size 

        return dstream


class BatchNormLayer:
    
    ''' Batch Normalization Layer http://arxiv.org/abs/1502.03167 '''

    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None # 합성곱 계층은 4차원, 완전연결 계층은 2차원  

        # 시험할 때 사용할 평균과 분산
        self.running_mean = running_mean
        self.running_var = running_var  
        
        # backward 시에 사용할 중간 데이터
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None


    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        result = self.__forward(x, train_flg)
        
        return result.reshape(*self.input_shape)


    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)
                        
        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std
            
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var            
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))
            
        reslut = self.gamma * xn + self.beta 
        return reslut


    def backward(self, dstream):
        if dstream.ndim != 2:
            N, C, H, W = dstream.shape
            dstream = dstream.reshape(N, -1)

        dx = self.__backward(dstream)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dstream):
        dbeta = dstream.sum(axis=0)
        dgamma = np.sum(self.xn * dstream, axis=0)
        dxn = self.gamma * dstream
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size
        
        self.dgamma = dgamma
        self.dbeta = dbeta
        
        return dx


class Dropout:
    
    ''' applying 50% dropout to a layer http://arxiv.org/abs/1207.0580 '''

    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None


    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)


    def backward(self, dstream):
        return dstream * self.mask