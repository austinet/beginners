# cellkit :: recurrent cells module #
# Based on Numpy #
# Austin Hyeon, 2020. "Sociology meets Computer Science" #

import numpy as np
from funckit import *

class VanillaCell:

    ''' the vanilla cell '''

    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wh), np.zeros_like(Wx), np.zeros_like(b)]
        self.cache = None


    def forward(self, x, h_prev):
        Wx, Wh, b = self.params
        t = np.matmul(h_prev, Wh) + np.matmul(x, Wx) + b
        h_next = np.tanh(t)

        self.cache = (x, h_prev, h_next) # tuple: the unamendable list
        
        return h_next


    def backward(self, dh_next):
        Wx, Wh, b = self.params
        x, h_prev, h_next = self.cache

        dt = dh_next * (1 - h_next ** 2) # derivative of t; h before tanh()
        db = np.sum(dt, axis=0) # derivative of bias
        dWh = np.matmul(h_prev.T, dt) # derivative of Wh
        dh_prev = np.matmul(dt, Wh.T) # derivative of h_prev
        dWx = np.matmul(x.T, dt) # derivative of Wx
        dx = np.matmul(dt, Wx.T) # derivative of x

        self.grads[0][...] = dWx # input dWx into grads[0][0] to [0][n]
        self.grads[1][...] = dWh # input dWh into grads[1][0] to [1][n]
        self.grads[2][...] = db # input db into grads[2][0] to [2][n]

        # grads = 3 X n matrix

        return dx, dh_prev


class LSTMCell:

    ''' the long short-term memory cell '''

    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wh), np.zeros_like(Wx), np.zeros_like(b)]
        self.cache = None

    
    def forward(self, x, h_prev, c_prev):
        Wx, Wh, b = self.params
        N, H = h_prev.shape

        f = np.matmul(x, Wx) + np.matmul(h_prev, Wh) + b

        ''' slice 'ƒ' into four pieces for each gate '''

        f1 = f[:, :H]
        f2 = f[:, H:2*H]
        f3 = f[:, 2*H:3*H]
        f4 = f[:, 3*H:]

        ''' set gates '''

        sig1 = sigmoid(f1) # sigmoid dam 1
        tanh_f = np.tanh(f2) # tanh(ƒ) dam
        sig2 = sigmoid(f3) # sigmoid dam 2
        sig3 = sigmoid(f4) # sigmoid dam 3

        ''' set the flow '''

        c_next = (c_prev * sig1) + (tanh_f * sig2)
        h_next = np.tanh(c_next) * sig3

        # cache for bptt
        self.cache = (x, h_prev, c_prev, sig1, tanh_f, sig2, sig3, c_next)

        return h_next, c_next


    def backward(self, dh_next, dc_next):
        Wx, Wh, b = self.params
        x, h_prev, c_prev, sig1, tanh_f, sig2, sig3, c_next = self.cache

        tanh_c_next = np.tanh(c_next)

        ds = dc_next + (dh_next * sig3) * (1 - tanh_c_next ** 2)

        dc_prev = ds * sig1

        d_sig1 = ds * c_prev # derivative of sigmoid dam 1
        d_tanh_f = ds * sig2 # derivative of tanh(ƒ) dam
        d_sig2 = ds * tanh_f # derivative of sigmoid dam 2
        d_sig3 = dh_next * tanh_c_next # derivative of sigmoid dam3

        d_sig1 *= sig1 * (1 - sig1)
        d_tanh_f *= (1 - tanh_f ** 2)
        d_sig2 *= sig2 * (1 - sig2)
        d_sig3 *= sig3 * (1 - sig3)

        df = np.hstack((d_sig1, d_tanh_f, d_sig2, d_sig3)) # merge all derivatives(= gradients) of each gate

        ''' distribute dƒ '''

        dWh = np.dot(h_prev.T, df) # derivative of Wh
        dWx = np.dot(x.T, df) # derivative of Wx
        db = df.sum(axis=0) # derivative of b

        ''' record gradients '''

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        dx = np.dot(df, Wx.T)
        dh_prev = np.dot(df, Wh.T)

        return dx, dh_prev, dc_prev


class GRUCell:

    ''' the gated recurrent unit cell '''

    def __init__(self):
        None


class EmbeddingCell:

    ''' the embedding recurrent cell '''

    def __init__(self, EM):
        self.params = [EM]
        self.grads = [np.zeros_like(EM)]
        self.idx = None

    def forward(self, idx):
        EM, = self.params
        self.idx = idx
        out = EM[idx]

        return out

    def backward(self, dstream):
        dEM, = self.grads
        dEM[...] = 0
        np.add.at(dEM, self.idx, dstream)

        return None