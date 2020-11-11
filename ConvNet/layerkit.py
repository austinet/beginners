# layerkit :: neural layers module #
# Based on Numpy #
# Austin Hyeon, 2020. "Sociology meets Computer Science" #

import numpy as np
from funckit import *
from utilitykit import *


'''
layers
---
    1. AFFINE
    2. CONVOLUTIONAL
    3. MAXPOOLING
    4. SIGMOID
    5. TANH
    6. RELU
    7. LEAKYRELU
    8. ELU
    9. SOFTMAXLOSS
    10. BATCHNORM
    11. DROPOUT
'''


class AFFINE:

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


    # dstream: stream of derivatives from SOFTMAXLOSS
    def backward(self, dstream):
        dx = np.dot(dstream, self.W.T)
        self.dW = np.dot(self.x.T, dstream)
        self.db = np.sum(dstream, axis=0)

        dx = dx.reshape(*self.original_x_shape)
        return dx


class CONVOLUTIONAL:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        
        # 중간 데이터（backward 시 사용）
        self.x = None   
        self.col = None
        self.col_W = None
        
        # 가중치와 편향 매개변수의 기울기
        self.dW = None
        self.db = None


    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        result = np.dot(col, col_W) + self.b
        result = result.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return result


    def backward(self, dstream):
        FN, C, FH, FW = self.W.shape
        dstream = dstream.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dstream, axis=0)
        self.dW = np.dot(self.col.T, dstream)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dstream, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


class MAXPOOLING:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        
        self.x = None
        self.arg_max = None


    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=1)
        result = np.max(col, axis=1)
        result = result.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return result


    def backward(self, dstream):
        dstream = dstream.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dstream.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dstream.flatten()
        dmax = dmax.reshape(dstream.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx


class SIGMOID:

    ''' sigmoid layer '''

    def __init__(self):
        self.dup = None


    # x: predicted values
    def forward(self, x):
        y = sigmoid(x)
        self.dup = y.copy()

        return y


    # dstream: stream of derivatives from SOFTMAXLOSS
    def backward(self, dstream):
        y = self.dup
        dx = dstream * (y * (1 - y))
        
        return dx


class TANH:

    ''' tanh, hyperbolic tagent layer '''

    def __init__(self):
        self.dup = None


    # x: predicted values
    def forward(self, x):
        y = tanh(x)
        self.dup = y.copy()

        return y


    # dstream: stream of derivatives from SOFTMAXLOSS
    def backward(self, dstream):
        y = self.dup
        dx = dstream * (1 - np.power(y, 2))
        
        return dx


class RELU:

    ''' ReLU, rectified linear unit layer '''

    def __init__(self):
        self.tag = None 


    # x: predicted values
    def forward(self, x):
        self.tag = (x <= 0) # tag: tagging values under 0 as 'True' in np.array
        y = x.copy() # copy input tensor 'x' to 'result'
        y[self.tag] = 0 # return True-tagged values into '0' in 'result'

        return y


    # dstream: stream of derivatives from SOFTMAXLOSS
    def backward(self, dstream):
        dstream[self.tag] = 0 # return True-tagged values into '0' in 'dstream'
        dx = dstream

        return dx


class LEAKYRELU:
    '''
    LeakyReLU layer

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


    # dstream: stream of derivatives from SOFTMAXLOSS
    def backward(self, dstream):
        dstream[self.tag] = dstream[self.tag] * self.alpha
        dx = dstream

        return dx


class ELU:
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


    # dstream: stream of derivatives from SOFTMAXLOSS
    def backward(self, dstream):
        dstream[self.tag] =self.alpha * (np.exp(dstream[self.tag]) - 1)
        dx = dstream

        return dx


class SOFTMAXLOSS:

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


class BATCHNORM:
    
    ''' Batch Normalization Layer http://arxiv.org/abs/1502.03167 '''

    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None

        ''' statistics values '''

        self.running_mean = running_mean # mean
        self.running_var = running_var # variance
        
        ''' variables for backward() '''

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
            
        result = self.gamma * xn + self.beta 
        return result


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


class DROPOUT:
    
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